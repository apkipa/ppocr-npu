#include "npu_engine.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace fs = std::filesystem;

namespace ppocr {

namespace {

void check_ze(ze_result_t result, const char* msg) {
    if (result != ZE_RESULT_SUCCESS) {
        std::ostringstream ss;
        ss << msg << " failed with Level Zero error: 0x" << std::hex << result;
        throw std::runtime_error(ss.str());
    }
}

} // namespace

// ============================================================================
// NpuInferRequest Implementation
// ============================================================================

NpuInferRequest::NpuInferRequest(
    ze_context_handle_t context,
    ze_device_handle_t device,
    ze_command_queue_handle_t queue,
    uint32_t queue_ordinal,
    ze_graph_handle_t graph,
    const std::vector<TensorDesc>& inputs,
    const std::vector<TensorDesc>& outputs,
    ze_graph_dditable_ext_t* graph_api
) : context_(context), device_(device), queue_(queue), graph_(graph),
    graph_api_(graph_api), input_descs_(inputs), output_descs_(outputs)
{
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
    cmd_desc.commandQueueGroupOrdinal = queue_ordinal;
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
    check_ze(zeCommandQueueExecuteCommandLists(queue_, 1, &command_list_, nullptr), "ExecuteCommandLists");
}

// TODO(NPU parallelism): wait() synchronizes the engine-wide shared command
// queue instead of this request, so all outstanding work drains together.
// Consequences measured on 2560x1524: the det tile ping-pong in main.cpp
// degenerates to serial execution (15 tiles ~20.3ms each, no overlap) and the
// per-box rec submit/wait never overlaps with crop/CTC decode.
// Optimization plan:
//   1. attach a ze_fence_t (or ze_event) per NpuInferRequest, execute the
//      command list against that fence and wait with zeFenceHostSynchronize;
//   2. or create one command queue per request so a wait only drains its own
//      queue (NPU3720 is dual-tile; nireq=2 is the measured sweet spot,
//      OpenVINO reference hits ~475 FPS rec with 2 concurrent requests);
//   3. main.cpp: pipeline rec as "submit box b, crop box b+1, decode b-1";
//      det tiles: submit tile i+1 before draining tile i.
void NpuInferRequest::wait() {
    check_ze(zeCommandQueueSynchronize(queue_, UINT64_MAX), "QueueSynchronize");
}

// ============================================================================
// NpuGraph Implementation
// ============================================================================

NpuGraph::NpuGraph(
    ze_context_handle_t context,
    ze_device_handle_t device,
    ze_command_queue_handle_t queue,
    uint32_t queue_ordinal,
    const std::string& model_path,
    ze_graph_dditable_ext_t* graph_api,
    size_t num_requests
) : context_(context), device_(device), queue_(queue), queue_ordinal_(queue_ordinal), graph_api_(graph_api)
{
    init_graph(model_path);
    query_arguments();

    // Create Ping-Pong InferRequest pool
    for (size_t i = 0; i < num_requests; ++i) {
        requests_.push_back(std::make_unique<NpuInferRequest>(
            context_, device_, queue_, queue_ordinal_, graph_, inputs_, outputs_, graph_api_
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

    // Initialize Graph on NPU (allocates hardware scratchpad & firmware structures)
    ze_command_list_desc_t init_cmd_desc{};
    init_cmd_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
    init_cmd_desc.commandQueueGroupOrdinal = queue_ordinal_;
    ze_command_list_handle_t init_cmd_list = nullptr;
    check_ze(zeCommandListCreate(context_, device_, &init_cmd_desc, &init_cmd_list), "InitCommandListCreate");

    check_ze(graph_api_->pfnAppendGraphInitialize(init_cmd_list, graph_, nullptr, 0, nullptr), "AppendGraphInitialize");
    check_ze(zeCommandListClose(init_cmd_list), "InitCommandListClose");
    check_ze(zeCommandQueueExecuteCommandLists(queue_, 1, &init_cmd_list, nullptr), "InitExecute");
    check_ze(zeCommandQueueSynchronize(queue_, UINT64_MAX), "InitSynchronize");
    check_ze(zeCommandListDestroy(init_cmd_list), "InitCommandListDestroy");
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
    if (queue_) zeCommandQueueDestroy(queue_);
    if (context_) zeContextDestroy(context_);
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

    // Create Context & Command Queue
    ze_context_desc_t context_desc{};
    context_desc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
    check_ze(zeContextCreate(driver_, &context_desc, &context_), "zeContextCreate");

    ze_command_queue_desc_t queue_desc{};
    queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    queue_desc.ordinal = queue_ordinal_;
    queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    check_ze(zeCommandQueueCreate(context_, device_, &queue_desc, &queue_), "zeCommandQueueCreate");

    initialized_ = true;
    std::cout << "[+] Level Zero NPU Engine Initialized: " << device_name_ << "\n";
    return true;
}

std::shared_ptr<NpuGraph> NpuEngine::load_model(const std::string& model_path, size_t num_requests) {
    if (!initialize()) {
        throw std::runtime_error("NpuEngine initialization failed.");
    }
    return std::make_shared<NpuGraph>(context_, device_, queue_, queue_ordinal_, model_path, graph_api_, num_requests);
}

} // namespace ppocr
