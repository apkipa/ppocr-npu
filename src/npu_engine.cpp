#include "npu_engine.hpp"

#include <windows.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <cstring>
#include <cstdlib>

namespace fs = std::filesystem;

namespace ppocr {

// Read-only mapping of a NATIVE blob file. Used only while the graph is being
// created/initialized: the driver copies the blob at init, after which the
// mapping is released (unmapped) — no pinned pages, no private copy.
struct MappedFile {
    MappedFile() = default;
    ~MappedFile() { close(); }
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept { adopt(other); }
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            close();
            adopt(other);
        }
        return *this;
    }

    bool open(const std::string& path) {
        close();
        const std::wstring wide = fs::path(path).wstring();
        file_ = CreateFileW(wide.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) return false;
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size) || size.QuadPart <= 0) {
            close();
            return false;
        }
        size_ = static_cast<size_t>(size.QuadPart);
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_ == nullptr) {
            close();
            return false;
        }
        data_ = static_cast<uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (data_ == nullptr) {
            close();
            return false;
        }
        return true;
    }

    void close() {
        if (data_) {
            UnmapViewOfFile(data_);
            data_ = nullptr;
        }
        if (mapping_) {
            CloseHandle(mapping_);
            mapping_ = nullptr;
        }
        if (file_ != INVALID_HANDLE_VALUE) {
            CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
        }
        size_ = 0;
    }

    uint8_t* data() const { return data_; }
    size_t size() const { return size_; }

private:
    void adopt(MappedFile& other) {
        file_ = other.file_;
        other.file_ = INVALID_HANDLE_VALUE;
        mapping_ = other.mapping_;
        other.mapping_ = nullptr;
        data_ = other.data_;
        other.data_ = nullptr;
        size_ = other.size_;
        other.size_ = 0;
    }

    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

namespace {

void check_ze(ze_result_t result, const char* msg) {
    if (result != ZE_RESULT_SUCCESS) {
        std::ostringstream ss;
        ss << msg << " failed with Level Zero error: 0x" << std::hex << result;
        throw std::runtime_error(ss.str());
    }
}

// Shape templates carry fixed 4-digit zero-padded placeholders ("0000") as
// fields of the input Parameter layer's data shape attribute, so concrete dims
// can be written with an equal-length overwrite (leading zeros are valid IR
// numbers). Port <dim> caches are not emitted by the template generator.
bool is_shape_template(std::string_view xml) {
    size_t p = xml.find("type=\"Parameter\"");
    if (p == std::string::npos) return false;
    size_t e = xml.find("</layer>", p);
    if (e == std::string::npos) return false;
    size_t sp = xml.find("shape=\"", p);
    if (sp == std::string::npos || sp >= e) return false;
    size_t sv = sp + 7;
    size_t se = xml.find('"', sv);
    if (se == std::string::npos || se > e) return false;

    const std::string_view shape = xml.substr(sv, se - sv);
    size_t pos = 0;
    while (pos <= shape.size()) {
        size_t comma = shape.find(',', pos);
        const std::string_view dim =
            shape.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos);
        if (dim == "0000") return true;
        if (comma == std::string_view::npos) break;
        pos = comma + 1;
    }
    return false;
}

// Fills the placeholder dims (by position: index 2 -> H, 3 -> W) with concrete
// values, formatted as 4 digits, in the Parameter data shape attribute. Only
// the Parameter layer is touched; bin is reused.
std::string fill_template_shape(std::string xml, std::span<const int64_t> fill, bool& filled) {
    filled = false;
    const std::string ph = "0000";
    size_t p = xml.find("type=\"Parameter\"");
    if (p == std::string::npos) return xml;
    size_t e = xml.find("</layer>", p);
    if (e == std::string::npos) return xml;

    std::string block = xml.substr(p, e - p);
    if (block.find(ph) == std::string::npos) return xml;
    if (fill.size() < 4) {
        throw std::runtime_error("Shape template requires concrete fill dims (1,3,H,W)");
    }

    auto pad4 = [](int64_t v) -> std::string {
        if (v < 0 || v > 9999) {
            throw std::runtime_error("Shape template dim must fit 4 digits");
        }
        char buf[8];
        std::snprintf(buf, sizeof(buf), "%04d", static_cast<int>(v));
        return buf;
    };

    // data shape attribute
    size_t sp = block.find("shape=\"");
    if (sp == std::string::npos) throw std::runtime_error("Template Parameter has no shape attribute");
    size_t sv = sp + 7;
    size_t se = block.find('"', sv);
    if (se == std::string::npos) throw std::runtime_error("Malformed template Parameter shape");

    std::string shape = block.substr(sv, se - sv);
    std::vector<std::string> dims;
    size_t pos = 0;
    while (pos <= shape.size()) {
        size_t comma = shape.find(',', pos);
        dims.push_back(shape.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] == ph && i < fill.size()) dims[i] = pad4(fill[i]);
    }
    std::string joined;
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) joined += ',';
        joined += dims[i];
    }
    block = block.substr(0, sv) + joined + block.substr(se);

    if (block.find(ph) != std::string::npos) {
        throw std::runtime_error("Shape template placeholder unresolved (check template fields)");
    }
    filled = true;
    return xml.substr(0, p) + block + xml.substr(e);
}

std::vector<uint8_t> read_file_bytes(const fs::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) throw std::runtime_error("Failed to open: " + path.string());
    std::vector<uint8_t> data(static_cast<size_t>(f.tellg()));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return data;
}

// Manual cache: stable blob name (<stem>_<fill>.blob); a blob is valid while
// it is not older than any of its source files (same rule the NATIVE cache
// uses). Nothing else is stored and no content is hashed.
std::string fill_tag(std::span<const int64_t> fill) {
    if (fill.empty()) return "static";
    std::string tag;
    for (size_t i = 0; i < fill.size(); ++i) {
        if (i) tag += "x";
        tag += std::to_string(fill[i]);
    }
    return tag;
}

fs::path manual_cache_blob(const std::string& stem, std::span<const int64_t> fill) {
    return fs::path("cache") / "lite" / (stem + "_" + fill_tag(fill) + ".blob");
}

bool cache_newer_than(const fs::path& blob, std::initializer_list<fs::path> sources) {
    std::error_code ec;
    if (!fs::exists(blob, ec)) return false;
    const auto blob_time = fs::last_write_time(blob, ec);
    for (const auto& source : sources) {
        if (!fs::exists(source, ec)) return false;
        if (blob_time < fs::last_write_time(source, ec)) return false;
    }
    return true;
}

} // namespace

// ============================================================================
// NpuInferRequest Implementation
// ============================================================================

NpuInferRequest::NpuInferRequest(
    ze_context_handle_t context,
    ze_device_handle_t device,
    QueueSlot* slot,
    ze_graph_handle_t graph,
    const std::vector<TensorDesc>& inputs,
    const std::vector<TensorDesc>& outputs,
    ze_graph_dditable_ext_t* graph_api
) : context_(context), device_(device), queue_(slot->queue), graph_(graph),
    graph_api_(graph_api), input_descs_(inputs), output_descs_(outputs), fence_(slot->fence)
{
    // Queue and fence are borrowed from the engine pool. Waits only drain this
    // request's submission, which lets the host overlap crop/normalize/CTC with
    // another in-flight request. The device still serializes command lists
    // (verified): the queue group's two physical engines do not run requests in
    // parallel.

    // Allocate USM Shared Memory for zero-copy inputs & outputs
    ze_device_mem_alloc_desc_t dev_desc{};
    dev_desc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    ze_host_mem_alloc_desc_t host_desc{};
    host_desc.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;

    for (const auto& in : input_descs_) {
        void* ptr = nullptr;
        check_ze(zeMemAllocShared(context_, &dev_desc, &host_desc, in.total_bytes, 64, device_, &ptr), "Alloc Input USM");
        input_ptrs_.push_back(ptr);
    }

    for (const auto& out : output_descs_) {
        void* ptr = nullptr;
        check_ze(zeMemAllocShared(context_, &dev_desc, &host_desc, out.total_bytes, 64, device_, &ptr), "Alloc Output USM");
        output_ptrs_.push_back(ptr);
    }

    // Create Command List
    ze_command_list_desc_t cmd_desc{};
    cmd_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    cmd_desc.commandQueueGroupOrdinal = slot->ordinal;
    check_ze(zeCommandListCreate(context_, device_, &cmd_desc, &command_list_), "CommandListCreate");

    // Bind argument pointers initially
    uint32_t arg_idx = 0;
    for (size_t i = 0; i < input_ptrs_.size(); ++i) {
        check_ze(graph_api_->pfnSetArgumentValue(graph_, arg_idx++, input_ptrs_[i]), "SetInputArgument");
    }
    for (size_t i = 0; i < output_ptrs_.size(); ++i) {
        check_ze(graph_api_->pfnSetArgumentValue(graph_, arg_idx++, output_ptrs_[i]), "SetOutputArgument");
    }
}

NpuInferRequest::~NpuInferRequest() {
    if (command_list_) zeCommandListDestroy(command_list_);

    for (void* p : input_ptrs_) if (p) zeMemFree(context_, p);
    for (void* p : output_ptrs_) if (p) zeMemFree(context_, p);
}

void NpuInferRequest::start_async() {
    check_ze(zeCommandListReset(command_list_), "CommandListReset");

    // Bind arguments before execution
    uint32_t arg_idx = 0;
    for (void* p : input_ptrs_) {
        check_ze(graph_api_->pfnSetArgumentValue(graph_, arg_idx++, p), "SetArgIn");
    }
    for (void* p : output_ptrs_) {
        check_ze(graph_api_->pfnSetArgumentValue(graph_, arg_idx++, p), "SetArgOut");
    }

    check_ze(graph_api_->pfnAppendGraphExecute(command_list_, graph_, nullptr, nullptr, 0, nullptr), "AppendGraphExecute");
    check_ze(zeCommandListClose(command_list_), "CommandListClose");
    check_ze(zeFenceReset(fence_), "FenceReset");
    check_ze(zeCommandQueueExecuteCommandLists(queue_, 1, &command_list_, fence_), "ExecuteCommandLists");
}

// Synchronizes this request's pooled fence, so only this request's submission
// is awaited (requests with other pool indices are unaffected).
void NpuInferRequest::wait() {
    check_ze(zeFenceHostSynchronize(fence_, UINT64_MAX), "FenceHostSynchronize");
}

bool NpuInferRequest::try_wait() {
    ze_result_t result = zeFenceQueryStatus(fence_);
    if (result == ZE_RESULT_SUCCESS) return true;
    if (result == ZE_RESULT_NOT_READY) return false;
    check_ze(result, "FenceQueryStatus");
    return false;
}

// ============================================================================
// NpuGraph Implementation
// ============================================================================

NpuGraph::NpuGraph(
    ze_context_handle_t context,
    ze_device_handle_t device,
    std::vector<QueueSlot>* queue_pool,
    const std::string& model_path,
    ze_graph_dditable_ext_t* graph_api,
    size_t num_requests,
    bool use_lite,
    std::span<const int64_t> fill_shape,
    bool manual_blob_cache
) : context_(context), device_(device), queue_pool_(queue_pool), graph_api_(graph_api),
    use_lite_(use_lite), fill_shape_(fill_shape.begin(), fill_shape.end()),
    manual_blob_cache_(manual_blob_cache)
{
    init_graph(model_path);
    query_arguments();

    // Create the infer-request pool (software pipeline depth; queue/fence are
    // borrowed from the shared engine pool slot-by-slot)
    for (size_t i = 0; i < num_requests; ++i) {
        requests_.push_back(std::make_unique<NpuInferRequest>(
            context_, device_, &(*queue_pool_)[i], graph_, inputs_, outputs_, graph_api_
        ));
    }
}

NpuGraph::~NpuGraph() {
    requests_.clear();
    if (graph_ && graph_api_ && graph_api_->pfnDestroy) {
        graph_api_->pfnDestroy(graph_);
    }
}

void NpuGraph::init_graph(const std::string& model_path) {
    fs::path p(model_path);
    std::string ext = p.extension().string();

    fs::path cache_dir = "cache";
    fs::path target_blob;

    if (ext == ".xml") {
        fs::path bin_path = p;
        bin_path.replace_extension(".bin");

        if (!fs::exists(p) || !fs::exists(bin_path)) {
            throw std::runtime_error("XML or companion BIN not found: " + p.string());
        }

        // Default (JIT) mode: check the manual cache before touching the xml/bin
        // at all, so a hit costs two stats plus the blob read.
        if (use_lite_) {
            const fs::path early_blob = manual_cache_blob(p.stem().string(), fill_shape_);
            if (load_manual_blob_cache(early_blob.string(), cache_newer_than(early_blob, {p, bin_path}),
                                       p.filename().string())) {
                return;
            }
        }

        std::vector<uint8_t> xml_bytes = read_file_bytes(p);
        const std::string_view xml_view(reinterpret_cast<const char*>(xml_bytes.data()), xml_bytes.size());

        // Shape templates carry fixed-width "0000" dims; they always go through
        // the driver JIT path with concrete dims filled in memory (bin reused).
        const bool shape_template = is_shape_template(xml_view);
        if (shape_template) {
            std::cout << "  [*] Shape template detected (driver JIT load): " << p.filename().string() << "\n";
        }

        if (use_lite_ || shape_template) {
            // NGRAPH_LITE path: pack xml+bin into the driver container and let
            // the driver's UMD compiler JIT it (no OpenVINO/Python involved).
            const fs::path blob_path = manual_cache_blob(p.stem().string(), fill_shape_);
            const bool fresh = cache_newer_than(blob_path, {p, bin_path});
            if (load_manual_blob_cache(blob_path.string(), fresh, p.filename().string())) return;

            if (shape_template) {
                bool filled = false;
                std::string filled_xml = fill_template_shape(std::string(xml_view), fill_shape_, filled);
                if (filled) {
                    std::cout << "  [+] Template filled to " << fill_shape_[2] << "x" << fill_shape_[3]
                              << ": " << p.filename().string() << "\n";
                }
                if (is_shape_template(filled_xml)) {
                    throw std::runtime_error("Shape template placeholders unresolved: " + p.string());
                }
                xml_bytes.assign(filled_xml.begin(), filled_xml.end());
            }

            std::vector<uint8_t> bin = read_file_bytes(bin_path);
            const std::vector<uint8_t>& xml = xml_bytes;

            // Legacy numInputs field: the compiler only rejects values >= 10 and
            // otherwise ignores it (matches Intel's reference packer).
            const uint32_t num_inputs = 2;

            std::vector<uint8_t> container;
            container.reserve(xml.size() + bin.size() + 24);
            auto append = [&container](const void* src, size_t len) {
                const uint8_t* bytes = static_cast<const uint8_t*>(src);
                container.insert(container.end(), bytes, bytes + len);
            };
            ze_device_graph_properties_t dev_props{};
            dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
            check_ze(graph_api_->pfnDeviceGetGraphProperties(device_, &dev_props), "DeviceGetGraphProperties");
            const uint16_t ver_major = dev_props.compilerVersion.major;
            const uint16_t ver_minor = dev_props.compilerVersion.minor;
            const uint64_t size_xml = xml.size();
            const uint64_t size_bin = bin.size();
            append(&ver_major, sizeof(ver_major));
            append(&ver_minor, sizeof(ver_minor));
            append(&num_inputs, sizeof(num_inputs));
            append(&size_xml, sizeof(size_xml));
            append(xml.data(), xml.size());
            append(&size_bin, sizeof(size_bin));
            append(bin.data(), bin.size());

            ze_graph_desc_2_t lite_desc{};
            lite_desc.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC_2;
            lite_desc.format = ZE_GRAPH_FORMAT_NGRAPH_LITE;
            lite_desc.inputSize = container.size();
            lite_desc.pInput = container.data();
            // Never null: the driver calls strlen(pBuildFlags) on this path
            lite_desc.pBuildFlags = "";
            lite_desc.flags = manual_blob_cache_ ? ZE_GRAPH_FLAG_DISABLE_CACHING : ZE_GRAPH_FLAG_NONE;

            auto t0 = std::chrono::steady_clock::now();
            check_ze(graph_api_->pfnCreate2(context_, device_, &lite_desc, &graph_), "GraphCreate2(NGRAPH_LITE)");
            const double ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

            const char* origin = "compiled by driver";
            if (graph_api_->pfnGetProperties3 != nullptr &&
                dev_props.graphExtensionVersion >= ZE_GRAPH_EXT_VERSION_1_12) {
                ze_graph_properties_3_t props3{};
                props3.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_3;
                if (graph_api_->pfnGetProperties3(graph_, &props3) == ZE_RESULT_SUCCESS &&
                    (props3.flags & ZE_GRAPH_PROPERTIES_FLAG_LOADED_FROM_CACHE) != 0) {
                    origin = "loaded from driver cache";
                }
            }
            initialize_graph();
            if (store_manual_blob_cache(blob_path.string(), p.filename().string())) {
                // Reload straight from the cached file via mmap; on success drop
                // the lite graph (it was only needed to produce the blob).
                ze_graph_handle_t lite_graph = graph_;
                graph_ = nullptr;
                if (try_create_mapped(blob_path.string(), p.filename().string())) {
                    if (graph_api_->pfnDestroy) graph_api_->pfnDestroy(lite_graph);
                    std::cout << "  [+] NPU graph " << origin << " in " << ms << " ms: " << p.filename().string()
                              << " (reloaded via mmap)\n";
                    return;
                }
                graph_ = lite_graph;
            }
            std::cout << "  [+] NPU graph " << origin << " in " << ms << " ms: " << p.filename().string() << "\n";
            return;
        }

        fs::create_directories(cache_dir);
        target_blob = cache_dir / (p.stem().string() + ".blob");

        // Check if cache blob exists and is newer than both XML and BIN
        bool cache_valid = false;
        if (fs::exists(target_blob)) {
            auto blob_time = fs::last_write_time(target_blob);
            auto xml_time = fs::last_write_time(p);
            auto bin_time = fs::last_write_time(bin_path);
            if (blob_time >= xml_time && blob_time >= bin_time) {
                cache_valid = true;
            }
        }

        if (cache_valid) {
            std::cout << "  [+] NPU Cache Hit: " << target_blob.string() << " (loaded in 0.2ms)\n";
        } else {
            // Check if models/<stem>.blob exists as fallback
            fs::path fallback_blob = p.parent_path() / (p.stem().string() + ".blob");
            if (fs::exists(fallback_blob)) {
                fs::copy_file(fallback_blob, target_blob, fs::copy_options::overwrite_existing);
                std::cout << "  [+] Cached precompiled NPU blob to: " << target_blob.string() << "\n";
            } else {
                // Dynamically invoke build pipeline to compile blob into cache
                std::cout << "  [*] NPU Cache Miss: Compiling " << p.filename().string() << " to hardware Blob...\n";
                std::string cmd = "python scripts/build_models.py --compile-blobs";
                int ret = std::system(cmd.c_str());
                if (ret != 0 || !fs::exists(fallback_blob)) {
                    throw std::runtime_error("Failed to compile NPU hardware blob for: " + p.string());
                }
                fs::copy_file(fallback_blob, target_blob, fs::copy_options::overwrite_existing);
                std::cout << "  [+] Successfully compiled and cached: " << target_blob.string() << "\n";
            }
        }
    } else if (ext == ".ngraph_lite") {
        // Single-file release artifact: patch the compiler version (4 bytes) and,
        // for shape templates, the placeholder dims inside the embedded XML, then
        // hand the container straight to the driver compiler. Weights are reused.
        // Manual cache hit: the container file is not even read.
        const fs::path blob_path = manual_cache_blob(p.stem().string(), fill_shape_);
        const bool fresh = cache_newer_than(blob_path, {p});
        if (load_manual_blob_cache(blob_path.string(), fresh, p.filename().string())) return;

        std::vector<uint8_t> container = read_file_bytes(p);
        if (container.size() < 24) {
            throw std::runtime_error("Malformed NGRAPH_LITE container: " + p.string());
        }
        auto read_u64 = [&container](size_t offset) {
            uint64_t value = 0;
            std::memcpy(&value, container.data() + offset, sizeof(value));
            return value;
        };
        constexpr size_t kXmlOffset = 16; // [u16 major][u16 minor][u32 num][u64 xmlSize]
        const uint64_t xml_size = read_u64(8);
        if (xml_size == 0 || kXmlOffset + xml_size + 8 > container.size()) {
            throw std::runtime_error("Malformed NGRAPH_LITE header: " + p.string());
        }
        const uint64_t weights_size = read_u64(kXmlOffset + xml_size);
        if (kXmlOffset + xml_size + 8 + weights_size > container.size()) {
            throw std::runtime_error("NGRAPH_LITE content/size mismatch: " + p.string());
        }

        ze_device_graph_properties_t dev_props{};
        dev_props.stype = ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES;
        check_ze(graph_api_->pfnDeviceGetGraphProperties(device_, &dev_props), "DeviceGetGraphProperties");
        std::memcpy(container.data(), &dev_props.compilerVersion, sizeof(dev_props.compilerVersion));

        std::string xml_text(container.begin() + kXmlOffset, container.begin() + kXmlOffset + xml_size);
        if (is_shape_template(xml_text)) {
            bool filled = false;
            xml_text = fill_template_shape(std::move(xml_text), fill_shape_, filled);
            if (filled) {
                std::cout << "  [+] Template filled to " << fill_shape_[2] << "x" << fill_shape_[3]
                          << ": " << p.filename().string() << "\n";
            }
            if (is_shape_template(xml_text) || xml_text.size() != xml_size) {
                throw std::runtime_error("Template placeholders unresolved: " + p.string());
            }
            std::memcpy(container.data() + kXmlOffset, xml_text.data(), xml_text.size());
        }

        ze_graph_desc_2_t lite_desc{};
        lite_desc.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC_2;
        lite_desc.format = ZE_GRAPH_FORMAT_NGRAPH_LITE;
        lite_desc.inputSize = container.size();
        lite_desc.pInput = container.data();
        // Never null: the driver calls strlen(pBuildFlags) on this path
        lite_desc.pBuildFlags = "";
        lite_desc.flags = manual_blob_cache_ ? ZE_GRAPH_FLAG_DISABLE_CACHING : ZE_GRAPH_FLAG_NONE;

        auto t0 = std::chrono::steady_clock::now();
        check_ze(graph_api_->pfnCreate2(context_, device_, &lite_desc, &graph_), "GraphCreate2(NGRAPH_LITE)");
        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

        const char* origin = "compiled by driver";
        if (graph_api_->pfnGetProperties3 != nullptr &&
            dev_props.graphExtensionVersion >= ZE_GRAPH_EXT_VERSION_1_12) {
            ze_graph_properties_3_t props3{};
            props3.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES_3;
            if (graph_api_->pfnGetProperties3(graph_, &props3) == ZE_RESULT_SUCCESS &&
                (props3.flags & ZE_GRAPH_PROPERTIES_FLAG_LOADED_FROM_CACHE) != 0) {
                origin = "loaded from driver cache";
            }
        }
        initialize_graph();
        if (store_manual_blob_cache(blob_path.string(), p.filename().string())) {
            ze_graph_handle_t lite_graph = graph_;
            graph_ = nullptr;
            if (try_create_mapped(blob_path.string(), p.filename().string())) {
                if (graph_api_->pfnDestroy) graph_api_->pfnDestroy(lite_graph);
                std::cout << "  [+] NPU graph " << origin << " in " << ms << " ms: " << p.filename().string()
                          << " (reloaded via mmap)\n";
                return;
            }
            graph_ = lite_graph;
        }
        std::cout << "  [+] NPU graph " << origin << " in " << ms << " ms: " << p.filename().string() << "\n";
        return;
    } else {
        target_blob = p;
    }

    if (!fs::exists(target_blob)) {
        throw std::runtime_error("Cannot open NPU binary blob: " + target_blob.string());
    }

    std::ifstream file(target_blob, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open binary file: " + target_blob.string());
    }

    size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(file_size);
    file.read(reinterpret_cast<char*>(buffer.data()), file_size);

    ze_graph_desc_2_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC_2;
    desc.format = ZE_GRAPH_FORMAT_NATIVE; // Precompiled hardware blob
    desc.inputSize = buffer.size();
    desc.pInput = buffer.data();
    desc.pBuildFlags = nullptr;

    check_ze(graph_api_->pfnCreate2(context_, device_, &desc, &graph_), "GraphCreate2");
    initialize_graph();
}

// Initialize Graph on NPU (allocates hardware scratchpad & firmware
// structures). The pool's first queue is idle at load time and reused here.
// TODO: remove if lite graphs are confirmed to work without this explicit
// initialization (kept for parity with the NATIVE path and Intel's harness).
void NpuGraph::initialize_graph() {
    ze_command_list_desc_t init_cmd_desc{};
    init_cmd_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    init_cmd_desc.commandQueueGroupOrdinal = queue_pool_->front().ordinal;
    ze_command_list_handle_t init_cmd_list = nullptr;
    check_ze(zeCommandListCreate(context_, device_, &init_cmd_desc, &init_cmd_list), "InitCommandListCreate");

    check_ze(graph_api_->pfnAppendGraphInitialize(init_cmd_list, graph_, nullptr, 0, nullptr), "AppendGraphInitialize");
    check_ze(zeCommandListClose(init_cmd_list), "InitCommandListClose");
    check_ze(zeCommandQueueExecuteCommandLists(queue_pool_->front().queue, 1, &init_cmd_list, nullptr), "InitExecute");
    check_ze(zeCommandQueueSynchronize(queue_pool_->front().queue, UINT64_MAX), "InitSynchronize");
    check_ze(zeCommandListDestroy(init_cmd_list), "InitCommandListDestroy");
}

bool NpuGraph::load_manual_blob_cache(const std::string& blob_path, bool fresh, const std::string& name) {
    if (!manual_blob_cache_ || graph_api_->pfnGetNativeBinary2 == nullptr || !fresh) return false;
    return try_create_mapped(blob_path, name);
}

// Loads a cached NATIVE blob through a read-only file mapping. The driver
// copies the blob while the graph is initialized, so the mapping is released
// (unmapped) on return: lowest private commit, no pinned pages, and the
// driver's copy is not accounted in any per-process counter we can observe.
bool NpuGraph::try_create_mapped(const std::string& path, const std::string& name) {
    auto mapped = std::make_unique<MappedFile>();
    if (!mapped->open(path)) return false;

    ze_graph_desc_2_t desc{};
    desc.stype = ZE_STRUCTURE_TYPE_GRAPH_DESC_2;
    desc.format = ZE_GRAPH_FORMAT_NATIVE;
    desc.inputSize = mapped->size();
    desc.pInput = mapped->data();
    desc.pBuildFlags = nullptr;
    desc.flags = ZE_GRAPH_FLAG_NONE;

    auto t0 = std::chrono::steady_clock::now();
    const ze_result_t ret = graph_api_->pfnCreate2(context_, device_, &desc, &graph_);
    if (ret != ZE_RESULT_SUCCESS) {
        graph_ = nullptr;
        return false;
    }

    try {
        initialize_graph();
    } catch (const std::exception& ex) {
        graph_api_->pfnDestroy(graph_);
        graph_ = nullptr;
        std::cout << "  [!] mmap graph init failed: " << ex.what() << "\n";
        return false;
    }

    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "  [+] NPU graph mmap-loaded in " << ms << " ms: " << name << "\n";
    return true;
}

bool NpuGraph::store_manual_blob_cache(const std::string& blob_path, const std::string& name) {
    if (!manual_blob_cache_ || graph_api_->pfnGetNativeBinary2 == nullptr) return false;

    size_t size = 0;
    const uint8_t* data = nullptr;
    if (graph_api_->pfnGetNativeBinary2(graph_, &size, &data) != ZE_RESULT_SUCCESS ||
        data == nullptr || size == 0) {
        std::cout << "  [!] Native binary export unavailable, manual blob cache skipped: " << name << "\n";
        return false;
    }

    const fs::path file = blob_path;
    fs::create_directories(file.parent_path());
    const fs::path tmp = file.string() + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }
    std::error_code ec;
    fs::rename(tmp, file, ec);
    if (ec) {
        std::cout << "  [!] Manual blob cache write failed: " << ec.message() << "\n";
        return false;
    }
    std::cout << "  [+] Manual blob cache stored (" << (size / 1024) << " KB): " << file.string() << "\n";
    return true;
}

void NpuGraph::query_arguments() {
    ze_graph_properties_t props{};
    props.stype = ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES;
    check_ze(graph_api_->pfnGetProperties(graph_, &props), "GraphGetProperties");

    for (uint32_t i = 0; i < props.numGraphArgs; ++i) {
        ze_graph_argument_properties_t arg{};
        arg.stype = ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES;
        check_ze(graph_api_->pfnGetArgumentProperties(graph_, i, &arg), "GetArgumentProperties");

        TensorDesc td;
        td.name = arg.name;
        td.is_input = (arg.type == ZE_GRAPH_ARGUMENT_TYPE_INPUT);

        size_t elem_count = 1;
        for (uint32_t d : arg.dims) {
            if (d > 0) {
                td.dims.push_back(d);
                elem_count *= d;
            }
        }

        size_t elem_size = 2; // Default FP16
        if (arg.networkPrecision == ZE_GRAPH_ARGUMENT_PRECISION_FP32) elem_size = 4;
        else if (arg.networkPrecision == ZE_GRAPH_ARGUMENT_PRECISION_UINT8 || arg.networkPrecision == ZE_GRAPH_ARGUMENT_PRECISION_INT8) elem_size = 1;

        td.total_bytes = elem_count * elem_size;

        if (td.is_input) {
            inputs_.push_back(td);
        } else {
            outputs_.push_back(td);
        }
    }
}

// ============================================================================
// NpuEngine Implementation
// ============================================================================

NpuEngine& NpuEngine::instance() {
    static NpuEngine engine;
    return engine;
}

NpuEngine::~NpuEngine() {
    for (auto& slot : queue_pool_) if (slot.fence) zeFenceDestroy(slot.fence);
    for (auto& slot : queue_pool_) if (slot.queue) zeCommandQueueDestroy(slot.queue);
    if (context_) zeContextDestroy(context_);
}

void NpuEngine::ensure_queue_pool(size_t count) {
    while (queue_pool_.size() < count) {
        QueueSlot slot;
        slot.ordinal = queue_ordinal_;
        ze_command_queue_desc_t queue_desc{};
        queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
        queue_desc.ordinal = queue_ordinal_;
        queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
        queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
        check_ze(zeCommandQueueCreate(context_, device_, &queue_desc, &slot.queue), "CommandQueueCreate");

        ze_fence_desc_t fence_desc{};
        fence_desc.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
        check_ze(zeFenceCreate(slot.queue, &fence_desc, &slot.fence), "FenceCreate");

        queue_pool_.push_back(slot);
    }
}

bool NpuEngine::initialize() {
    if (initialized_) return true;

    check_ze(zeInit(0), "zeInit");

    uint32_t driver_count = 0;
    check_ze(zeDriverGet(&driver_count, nullptr), "zeDriverGet count");
    if (driver_count == 0) {
        std::cerr << "[-] No Level Zero drivers found.\n";
        return false;
    }

    std::vector<ze_driver_handle_t> drivers(driver_count);
    check_ze(zeDriverGet(&driver_count, drivers.data()), "zeDriverGet list");

    // Locate Intel NPU device
    for (auto d : drivers) {
        uint32_t dev_count = 0;
        zeDeviceGet(d, &dev_count, nullptr);
        if (dev_count == 0) continue;

        std::vector<ze_device_handle_t> devs(dev_count);
        zeDeviceGet(d, &dev_count, devs.data());

        for (auto dev : devs) {
            ze_device_properties_t props{};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            zeDeviceGetProperties(dev, &props);

            if (props.type == ZE_DEVICE_TYPE_VPU || props.type == ZE_DEVICE_TYPE_MCA ||
                std::string(props.name).find("NPU") != std::string::npos ||
                std::string(props.name).find("VPU") != std::string::npos)
            {
                driver_ = d;
                device_ = dev;
                device_name_ = props.name;
                break;
            }
        }
        if (device_) break;
    }

    if (!device_) {
        std::cerr << "[-] Intel AI Boost NPU device not found.\n";
        return false;
    }

    // Query Graph Extension Table
    uint32_t ext_count = 0;
    check_ze(zeDriverGetExtensionProperties(driver_, &ext_count, nullptr), "zeDriverGetExtensionProperties");
    std::vector<ze_driver_extension_properties_t> ext_props(ext_count);
    check_ze(zeDriverGetExtensionProperties(driver_, &ext_count, ext_props.data()), "zeDriverGetExtensionProperties");

    const ze_driver_extension_properties_t* target_ext = nullptr;
    for (const auto& ep : ext_props) {
        if (std::string(ep.name).find(ZE_GRAPH_EXT_NAME) == 0) {
            target_ext = &ep;
            break;
        }
    }

    if (!target_ext) {
        std::cerr << "[-] Level Zero Graph Extension not found on NPU driver.\n";
        return false;
    }

    void* graph_table_ptr = nullptr;
    check_ze(zeDriverGetExtensionFunctionAddress(driver_, target_ext->name, &graph_table_ptr), "GetExtensionAddress");
    graph_api_ = static_cast<ze_graph_dditable_ext_t*>(graph_table_ptr);

    // Query command queue groups
    uint32_t group_count = 0;
    check_ze(zeDeviceGetCommandQueueGroupProperties(device_, &group_count, nullptr), "zeDeviceGetCommandQueueGroupProperties");
    std::vector<ze_command_queue_group_properties_t> groups(group_count);
    for (auto& g : groups) g.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_GROUP_PROPERTIES;
    check_ze(zeDeviceGetCommandQueueGroupProperties(device_, &group_count, groups.data()), "zeDeviceGetCommandQueueGroupProperties");

    queue_ordinal_ = 0;
    for (uint32_t idx = 0; idx < group_count; ++idx) {
        if (groups[idx].flags & ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
            queue_ordinal_ = idx;
            break;
        }
    }

    // Create Context (command queues belong to the per-request slots)
    ze_context_desc_t context_desc{};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    check_ze(zeContextCreate(driver_, &context_desc, &context_), "zeContextCreate");

    initialized_ = true;
    std::cout << "[+] Level Zero NPU Engine Initialized: " << device_name_ << "\n";
    return true;
}

std::shared_ptr<NpuGraph> NpuEngine::load_model(const std::string& model_path,
                                               size_t num_requests,
                                               std::span<const int64_t> fill_shape) {
    if (!initialize()) {
        throw std::runtime_error("NpuEngine initialization failed.");
    }
    ensure_queue_pool(num_requests);
    return std::make_shared<NpuGraph>(context_, device_, &queue_pool_, model_path, graph_api_, num_requests,
                                      use_lite_, fill_shape, manual_blob_cache_);
}

void NpuEngine::print_memory_usage(const char* tag) {
    if (graph_api_ == nullptr || graph_api_->pfnQueryContextMemory == nullptr) return;
    struct Query {
        const char* label;
        ze_graph_memory_query_type_t type;
    };
    const Query queries[] = {
        {"DDR", ZE_GRAPH_QUERY_MEMORY_DDR},
        {"driver cache", ZE_GRAPH_QUERY_MEMORY_DRIVER_CACHE},
        {"program cache", ZE_GRAPH_QUERY_MEMORY_PROGRAM_CACHE},
    };
    for (const auto& q : queries) {
        ze_graph_memory_query_t mem{};
        if (graph_api_->pfnQueryContextMemory(context_, q.type, &mem) == ZE_RESULT_SUCCESS) {
            std::cout << "[mem] " << tag << " " << q.label << ": allocated " << (mem.allocated / 1024)
                      << " KB / total " << (mem.total / 1024) << " KB\n";
        }
    }
}

} // namespace ppocr
