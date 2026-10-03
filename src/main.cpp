#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "npu_engine.hpp"
#include "postprocess.hpp"
#include "profiler.hpp"
#include "wic_decoder.hpp"

#include <windows.h>
#define PSAPI_VERSION 2
#include <psapi.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct ProcessMemorySnapshot {
    SIZE_T working_set = 0;
    SIZE_T private_bytes = 0;
    bool valid = false;
};

ProcessMemorySnapshot print_process_memory_usage(const char* tag) {
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                              sizeof(counters))) {
        std::cerr << "[mem] Process " << tag << ": GetProcessMemoryInfo failed ("
                  << GetLastError() << ")\n";
        return {};
    }

    std::cout << "[mem] Process " << tag << ": working set "
              << (counters.WorkingSetSize / 1024) << " KB, private bytes "
              << (counters.PrivateUsage / 1024) << " KB\n";
    return {counters.WorkingSetSize, counters.PrivateUsage, true};
}

void print_process_memory_delta(const char* tag, const ProcessMemorySnapshot& before,
                                const ProcessMemorySnapshot& after) {
    if (!before.valid || !after.valid) return;
    const auto delta_kb = [](SIZE_T after_bytes, SIZE_T before_bytes) -> int64_t {
        return (static_cast<int64_t>(after_bytes) - static_cast<int64_t>(before_bytes)) / 1024;
    };
    std::cout << "[mem] Process " << tag << " delta: working set "
              << delta_kb(after.working_set, before.working_set) << " KB, private bytes "
              << delta_kb(after.private_bytes, before.private_bytes) << " KB\n";
}

void print_npu_memory_delta(const char* tag, const std::optional<uint64_t>& before_kb,
                            const std::optional<uint64_t>& after_kb) {
    if (!before_kb || !after_kb) return;
    const auto delta_kb = static_cast<int64_t>(*after_kb) - static_cast<int64_t>(*before_kb);
    std::cout << "[mem] NPU DDR " << tag << " delta: " << delta_kb << " KB\n";
}

void print_usage(const char* prog) {
    std::cout << "PP-OCRv6 Intel NPU (Level Zero Native) Inference Engine\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " --image <path> [options]\n";
    std::cout << "  " << prog << " --clipboard [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --image <path>         Path to input image file (JPG, PNG, BMP, TIFF, WebP)\n";
    std::cout << "  --clipboard            Read image directly from Windows Clipboard\n";
    std::cout << "  --det-model <path>     Path to Det model (.xml, .ngraph_lite, or .blob) [default: models/ppocrv6_small_det_u8_template.ngraph_lite]\n";
    std::cout << "  --no-det-input-u8      Use FP16 NCHW Det input instead of the default BGR/NHWC U8\n";
    std::cout << "  --det-mode <mode>      Detection mode: resize or tile (default resize)\n";
    std::cout << "  --tile-size <n>        Tile edge length in pixels (multiple of 32, 32-960; default 960)\n";
    std::cout << "  --rec-model-160 <path> Path to Rec W=160 model (.xml or .ngraph_lite) [default: models/ppocrv6_small_rec_u8_template.ngraph_lite]\n";
    std::cout << "  --rec-model-320 <path> Path to Rec W=320 model (.xml or .ngraph_lite) [default: models/ppocrv6_small_rec_u8_template.ngraph_lite]\n";
    std::cout << "  --rec-model-640 <path> Path to Rec W=640 model (.xml or .ngraph_lite) [default: models/ppocrv6_small_rec_u8_template.ngraph_lite]\n";
    std::cout << "  --rec-model-1280 <path> Path to Rec W=1280 model (.xml or .ngraph_lite) [default: models/ppocrv6_small_rec_u8_template.ngraph_lite]\n";
    std::cout << "  --no-rec-input-u8      Use FP16 NCHW Rec input instead of the default BGR/NHWC U8\n";
    std::cout << "  --dictionary <path>    Path to dictionary [default: models/ppocr_keys_v1.txt]\n";
    std::cout << "  --rec-scheduling <mode> Request selection: round-robin or poll (poll requires inflight=2)\n";
    std::cout << "  --rec-inflight <n>     In-flight requests per recognition graph (default 2, range 1-8)\n";
    std::cout << "  --graph-format <name>  XML compilation format: ngraph-lite or native-blob (default ngraph-lite)\n";
    std::cout << "  --cache-backend <name> Compilation cache: app or driver (default app)\n";
    std::cout << "  --npu-low-latency      Request low-latency command queues (higher power)\n";
    std::cout << "  --profile              Print detailed end-to-end performance breakdown\n";
    std::cout << "  --trace <path>         Export host/NPU task timeline as Chrome Trace JSON\n";
    std::cout << "  --trace-mode <mode>    Trace detail: tasks or timestamps (default tasks)\n";
    std::cout << "  --trace-raw-dir <path> Dump RAW profiles and profiled graph blobs (tasks mode only)\n";
    std::cout << "  --rec-dump-dir <path>  Dump completed W640 Rec inputs/outputs for diagnostics\n";
    std::cout << "  --det-heatmap-dir <path>  Save detector probability heatmap as PGM (diagnostic)\n";
    std::cout << "  --cache-stats          Print driver/program cache statistics (slow diagnostic query)\n";
    std::cout << "  --help                 Show this help message\n";
}

std::wstring to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size);
    if (!wstr.empty() && wstr.back() == 0) wstr.pop_back();
    return wstr;
}

bool save_probability_pgm(const float* probabilities, int width, int height, const fs::path& path) {
    std::error_code directory_error;
    fs::create_directories(path.parent_path(), directory_error);
    if (directory_error) return false;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << "P5\n" << width << ' ' << height << "\n255\n";

    std::vector<uint8_t> row(static_cast<size_t>(width));
    for (int y = 0; y < height; ++y) {
        const size_t offset = static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            const float probability = std::clamp(probabilities[offset + x], 0.0f, 1.0f);
            row[static_cast<size_t>(x)] = static_cast<uint8_t>(probability * 255.0f + 0.5f);
        }
        out.write(reinterpret_cast<const char*>(row.data()), static_cast<std::streamsize>(row.size()));
    }
    return static_cast<bool>(out);
}

bool save_heatmap_pgm(const ppocr::ProbHeatmap& heatmap, const fs::path& path) {
    return save_probability_pgm(heatmap.data().data(), heatmap.width(), heatmap.height(), path);
}

} // namespace

int main(int argc, char* argv[]) {
    // Set console to UTF-8
    SetConsoleOutputCP(CP_UTF8);

    try {
        std::string image_path;
        bool from_clipboard = false;

        auto resolve_model = [](const std::string& base_no_ext) -> std::string {
            if (fs::exists(base_no_ext + ".ngraph_lite")) return base_no_ext + ".ngraph_lite";
            if (fs::exists(base_no_ext + ".xml")) return base_no_ext + ".xml";
            if (fs::exists("cache/" + fs::path(base_no_ext).stem().string() + ".blob")) return "cache/" + fs::path(base_no_ext).stem().string() + ".blob";
            if (fs::exists(base_no_ext + ".blob")) return base_no_ext + ".blob";
            return base_no_ext + ".xml";
        };

        bool rec_input_u8 = true;
        bool det_input_u8 = true;
        std::string det_model;
        std::string rec_160;
        std::string rec_320;
        std::string rec_640;
        std::string rec_1280;
        bool det_model_explicit = false;
        bool rec_model_160_explicit = false;
        bool rec_model_320_explicit = false;
        bool rec_model_640_explicit = false;
        bool rec_model_1280_explicit = false;
        std::string dict_path = "models/ppocr_keys_v1.txt";
        std::string trace_path;
        std::string trace_raw_dir;
        std::string rec_dump_dir;
        std::string det_heatmap_dir;
        bool task_trace = true;
        bool rec_poll = false;
        bool use_lite = true;
        bool blob_cache = true;
        enum class DetMode { Resize, Tile };
        DetMode det_mode = DetMode::Resize;
        bool tile_size_explicit = false;
        int det_size = 960;
        bool low_latency = false;
        bool cache_stats = false;
        int rec_depth = 2;
        ppocr::ProfileMetrics metrics;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--image") {
                if (i + 1 >= argc) throw std::invalid_argument("--image requires a path");
                image_path = argv[++i];
            } else if (arg == "--clipboard") {
                from_clipboard = true;
            } else if (arg == "--det-model") {
                if (i + 1 >= argc) throw std::invalid_argument("--det-model requires a path");
                det_model = argv[++i];
                det_model_explicit = true;
            } else if (arg == "--det-mode") {
                if (i + 1 >= argc) throw std::invalid_argument("--det-mode requires 'resize' or 'tile'");
                const std::string value = argv[++i];
                if (value == "resize") det_mode = DetMode::Resize;
                else if (value == "tile") det_mode = DetMode::Tile;
                else throw std::invalid_argument("--det-mode must be 'resize' or 'tile'");
            } else if (arg == "--tile-size") {
                if (i + 1 >= argc) throw std::invalid_argument("--tile-size requires a pixel size");
                const std::string value = argv[++i];
                const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), det_size);
                if (error != std::errc{} || end != value.data() + value.size())
                    throw std::invalid_argument("--tile-size must be an integer");
                tile_size_explicit = true;
            } else if (arg == "--rec-model-160") {
                if (i + 1 >= argc) throw std::invalid_argument("--rec-model-160 requires a path");
                rec_160 = argv[++i];
                rec_model_160_explicit = true;
            } else if (arg == "--rec-model-320") {
                if (i + 1 >= argc) throw std::invalid_argument("--rec-model-320 requires a path");
                rec_320 = argv[++i];
                rec_model_320_explicit = true;
            } else if (arg == "--rec-model-640") {
                if (i + 1 >= argc) throw std::invalid_argument("--rec-model-640 requires a path");
                rec_640 = argv[++i];
                rec_model_640_explicit = true;
            } else if (arg == "--rec-model-1280") {
                if (i + 1 >= argc) throw std::invalid_argument("--rec-model-1280 requires a path");
                rec_1280 = argv[++i];
                rec_model_1280_explicit = true;
            } else if (arg == "--rec-input-u8") {
                rec_input_u8 = true;
            } else if (arg == "--no-rec-input-u8") {
                rec_input_u8 = false;
            } else if (arg == "--det-input-u8") {
                det_input_u8 = true;
            } else if (arg == "--no-det-input-u8") {
                det_input_u8 = false;
            } else if (arg == "--dictionary") {
                if (i + 1 >= argc) throw std::invalid_argument("--dictionary requires a path");
                dict_path = argv[++i];
            } else if (arg == "--rec-dump-dir") {
                if (i + 1 >= argc) throw std::invalid_argument("--rec-dump-dir requires a path");
                rec_dump_dir = argv[++i];
            } else if (arg == "--rec-scheduling") {
                if (i + 1 >= argc) throw std::invalid_argument("--rec-scheduling requires 'round-robin' or 'poll'");
                const std::string value = argv[++i];
                if (value == "round-robin") rec_poll = false;
                else if (value == "poll") rec_poll = true;
                else throw std::invalid_argument("--rec-scheduling must be 'round-robin' or 'poll'");
            } else if (arg == "--rec-inflight") {
                if (i + 1 >= argc) throw std::invalid_argument("--rec-inflight requires a number from 1 through 8");
                const std::string value = argv[++i];
                const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), rec_depth);
                if (error != std::errc{} || end != value.data() + value.size() || rec_depth < 1 || rec_depth > 8)
                    throw std::invalid_argument("--rec-inflight must be an integer from 1 through 8");
            } else if (arg == "--graph-format") {
                if (i + 1 >= argc) throw std::invalid_argument("--graph-format requires 'ngraph-lite' or 'native-blob'");
                const std::string value = argv[++i];
                if (value == "ngraph-lite") use_lite = true;
                else if (value == "native-blob") use_lite = false;
                else throw std::invalid_argument("--graph-format must be 'ngraph-lite' or 'native-blob'");
            } else if (arg == "--cache-backend") {
                if (i + 1 >= argc) throw std::invalid_argument("--cache-backend requires 'app' or 'driver'");
                const std::string value = argv[++i];
                if (value == "app") blob_cache = true;
                else if (value == "driver") blob_cache = false;
                else throw std::invalid_argument("--cache-backend must be 'app' or 'driver'");
            } else if (arg == "--npu-low-latency") {
                low_latency = true;
            } else if (arg == "--profile") {
                metrics.enabled = true;
            } else if (arg == "--trace") {
                if (i + 1 >= argc) throw std::invalid_argument("--trace requires an output path");
                trace_path = argv[++i];
            } else if (arg == "--trace-mode") {
                if (i + 1 >= argc) throw std::invalid_argument("--trace-mode requires 'tasks' or 'timestamps'");
                const std::string value = argv[++i];
                if (value == "tasks") task_trace = true;
                else if (value == "timestamps") task_trace = false;
                else throw std::invalid_argument("--trace-mode must be 'tasks' or 'timestamps'");
            } else if (arg == "--trace-raw-dir") {
                if (i + 1 >= argc) throw std::invalid_argument("--trace-raw-dir requires a directory path");
                trace_raw_dir = argv[++i];
            } else if (arg == "--det-heatmap-dir") {
                if (i + 1 >= argc) throw std::invalid_argument("--det-heatmap-dir requires a directory path");
                det_heatmap_dir = argv[++i];
            } else if (arg == "--cache-stats") {
                cache_stats = true;
            } else if (arg == "--help") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::invalid_argument("unknown option: " + arg);
            }
        }

        if (!det_model_explicit) {
            det_model = resolve_model(det_input_u8
                ? "models/ppocrv6_small_det_u8_template"
                : "models/ppocrv6_small_det_template");
        }
        const std::string rec_template = resolve_model(rec_input_u8
            ? "models/ppocrv6_small_rec_u8_template"
            : "models/ppocrv6_small_rec_template");
        const std::string rec_prefix = rec_input_u8 ? "models/ppocrv6_small_rec_u8_w" : "models/ppocrv6_small_rec_w";
        if (!rec_model_160_explicit) rec_160 = fs::exists(rec_template) ? rec_template : resolve_model(rec_prefix + "160");
        if (!rec_model_320_explicit) rec_320 = fs::exists(rec_template) ? rec_template : resolve_model(rec_prefix + "320");
        if (!rec_model_640_explicit) rec_640 = fs::exists(rec_template) ? rec_template : resolve_model(rec_prefix + "640");
        if (!rec_model_1280_explicit) rec_1280 = fs::exists(rec_template) ? rec_template : resolve_model(rec_prefix + "1280");

        if (tile_size_explicit && det_mode != DetMode::Tile)
            throw std::invalid_argument("--tile-size requires --det-mode tile");
        if (det_mode == DetMode::Resize) det_size = 960;
        if (det_size < 32 || det_size > 960 || det_size % 32 != 0)
            throw std::invalid_argument("--tile-size must be a multiple of 32 from 32 through 960");
        if (!task_trace && trace_path.empty())
            throw std::invalid_argument("--trace-mode requires --trace");
        if (!trace_raw_dir.empty() && !task_trace)
            throw std::invalid_argument("--trace-raw-dir requires --trace-mode tasks");
        if (!rec_dump_dir.empty()) fs::create_directories(rec_dump_dir);
        if (rec_poll && rec_depth != 2)
            throw std::invalid_argument("--rec-scheduling poll requires --rec-inflight 2");

        if (image_path.empty() && !from_clipboard) {
            std::cerr << "[-] Error: Please specify an image via --image <path> or use --clipboard.\n\n";
            print_usage(argv[0]);
            return 1;
        }

        if (!from_clipboard && !fs::exists(image_path)) {
            std::cerr << "[-] Error: Input image file does not exist: " << image_path << "\n";
            return 1;
        }

        if (!fs::exists(det_model)) {
            std::cerr << "[-] Error: Detection model not found at '" << det_model << "'\n";
            std::cerr << "    Please generate models first using: uv run python scripts/build_models.py --compile-blobs\n";
            return 1;
        }

        if (!fs::exists(dict_path)) {
            std::cerr << "[-] Warning: Character dictionary not found at '" << dict_path << "'\n";
            std::cerr << "    Please run 'uv run python scripts/build_models.py' to generate the dictionary.\n";
        }

        std::unique_ptr<ppocr::TraceRecorder> trace;
        if (!trace_path.empty() || !trace_raw_dir.empty()) {
            trace = std::make_unique<ppocr::TraceRecorder>();
            trace->set_task_profiling_enabled(task_trace);
            if (!trace_raw_dir.empty()) trace->set_raw_dump_directory(trace_raw_dir);
            metrics.trace = trace.get();
        }

        auto start_total = ppocr::TraceRecorder::Clock::now();

        // 1. Image Acquisition via WIC
        ppocr::ImageBGR src_image;
        {
            ppocr::ScopedTimer timer(metrics.image_decode_ms, metrics.trace, "Image decode");
            if (from_clipboard) {
                std::cout << "[*] Reading image from Windows Clipboard...\n";
                if (!ppocr::WicDecoder::decode_clipboard(src_image)) {
                    return 1;
                }
            } else {
                if (!ppocr::WicDecoder::decode_file(to_wstring(image_path), src_image)) {
                    return 1;
                }
            }
        }
        std::cout << "[+] Input image acquired: " << src_image.width << "x" << src_image.height << " (24bpp BGR)" << std::endl;

        // 2. Initialize NPU Level Zero Engine
        auto& engine = ppocr::NpuEngine::instance();
        {
            const auto started = ppocr::TraceRecorder::Clock::now();
            if (!engine.initialize()) {
                std::cerr << "[-] NPU initialization failed.\n";
                return 1;
            }
            const auto ended = ppocr::TraceRecorder::Clock::now();
            metrics.engine_init_ms = std::chrono::duration<double, std::milli>(ended - started).count();
            if (trace) trace->add_host_event("NPU runtime initialization", "Model Load", started, ended);
        }
        engine.set_trace_recorder(trace.get());
        engine.set_ngraph_lite(use_lite);
        engine.set_manual_blob_cache(blob_cache);
        engine.set_turbo(low_latency);
        if (metrics.enabled) {
            engine.print_memory_usage("[baseline]");
            print_process_memory_usage("[baseline]");
        }
        if (cache_stats) engine.print_cache_usage("[baseline]");

        // 3. Load Models
        const auto det_fill_dims = det_input_u8
            ? std::array<int64_t, 4>{1, det_size, det_size, 3}
            : std::array<int64_t, 4>{1, 3, det_size, det_size};
        constexpr std::array<int64_t, 4> kRec160Fill{1, 3, 48, 160};
        constexpr std::array<int64_t, 4> kRec320Fill{1, 3, 48, 320};
        constexpr std::array<int64_t, 4> kRec640Fill{1, 3, 48, 640};
        constexpr std::array<int64_t, 4> kRec1280Fill{1, 3, 48, 1280};
        constexpr std::array<int64_t, 4> kRecU8_160Fill{1, 48, 160, 3};
        constexpr std::array<int64_t, 4> kRecU8_320Fill{1, 48, 320, 3};
        constexpr std::array<int64_t, 4> kRecU8_640Fill{1, 48, 640, 3};
        constexpr std::array<int64_t, 4> kRecU8_1280Fill{1, 48, 1280, 3};

        auto load_profiled = [&](const char* label, const std::string& path, size_t requests,
                                 std::span<const int64_t> fill, double& elapsed_total_ms) {
            const auto start = ppocr::TraceRecorder::Clock::now();
            auto graph = engine.load_model(path, requests, fill);
            const auto end = ppocr::TraceRecorder::Clock::now();
            const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
            elapsed_total_ms += elapsed;
            if (trace) trace->add_host_event(std::string(label) + " graph load", "Model Load", start, end);
            if (metrics.enabled) {
                const auto& stats = graph->load_stats();
                std::cout << "[profile] " << label << " load=" << elapsed
                          << " ms (graph_init=" << stats.graph_init_ms
                          << ", profile_blob_export=" << stats.profiling_blob_export_ms
                          << ", args=" << stats.argument_query_ms
                          << ", request_pool=" << stats.request_pool_ms
                          << ", requests=" << requests << ")\n";
                for (size_t i = 0; i < graph->outputs().size(); ++i) {
                    const auto& output = graph->outputs()[i];
                    std::cout << "[profile] " << label << " output[" << i << "] name=" << output.name
                              << " precision=" << static_cast<int>(output.precision) << " dims=";
                    for (size_t d = 0; d < output.dims.size(); ++d) {
                        if (d) std::cout << 'x';
                        std::cout << output.dims[d];
                    }
                    std::cout << " bytes=" << output.total_bytes << "\n";
                }
            }
            return graph;
        };
        const std::span<const int64_t> det_fill(det_fill_dims);
        std::cout << "[*] Loading Detection model onto NPU: " << det_model << "..." << std::endl;
        if (det_input_u8) {
            std::cout << "[*] Det input experiment: BGR/NHWC U8 with model-side preprocessing\n";
        }
        auto det_graph = load_profiled("det", det_model, 2, det_fill, metrics.det_model_load_ms); // 2-deep tile pipeline (submit-ahead)
        const size_t det_input_bytes = static_cast<size_t>(det_size) * det_size * 3 *
            (det_input_u8 ? sizeof(uint8_t) : sizeof(uint16_t));
        if (det_graph->inputs().empty() || det_graph->inputs()[0].total_bytes != det_input_bytes) {
            throw std::runtime_error("Detection graph input does not match the selected " +
                                     std::to_string(det_size) + "x" + std::to_string(det_size) +
                                     " bucket; use a matching shape template or --tile-size");
        }
        if (metrics.enabled) {
            engine.print_memory_usage("[after det]");
            print_process_memory_usage("[after det]");
        }
        if (cache_stats) engine.print_cache_usage("[after det]");

        auto fill_det_input = [&](const ppocr::ImageBGR& image, void* destination) {
            if (det_input_u8) {
                std::memcpy(destination, image.data.data(), image.data.size());
            } else {
                ppocr::WicDecoder::normalize_to_nchw_fp16(image, destination);
            }
        };

        std::cout << "[*] Loading Recognition Multi-Buckets onto NPU...\n";
        if (rec_input_u8) {
            std::cout << "[*] Rec input experiment: BGR/NHWC U8 with model-side preprocessing\n";
        }
        const auto& rec160_fill = rec_input_u8 ? kRecU8_160Fill : kRec160Fill;
        const auto& rec320_fill = rec_input_u8 ? kRecU8_320Fill : kRec320Fill;
        const auto& rec640_fill = rec_input_u8 ? kRecU8_640Fill : kRec640Fill;
        const auto& rec1280_fill = rec_input_u8 ? kRecU8_1280Fill : kRec1280Fill;
        auto rec_graph_160 = fs::exists(rec_160) ? load_profiled("rec160", rec_160, rec_depth, rec160_fill, metrics.rec_model_load_ms) : nullptr;
        if (metrics.enabled) {
            engine.print_memory_usage("[after rec160]");
            print_process_memory_usage("[after rec160]");
        }
        if (cache_stats) engine.print_cache_usage("[after rec160]");
        auto rec_graph_320 = fs::exists(rec_320) ? load_profiled("rec320", rec_320, rec_depth, rec320_fill, metrics.rec_model_load_ms) : nullptr;
        if (metrics.enabled) {
            engine.print_memory_usage("[after rec320]");
            print_process_memory_usage("[after rec320]");
        }
        if (cache_stats) engine.print_cache_usage("[after rec320]");
        auto rec_graph_640 = fs::exists(rec_640) ? load_profiled("rec640", rec_640, rec_depth, rec640_fill, metrics.rec_model_load_ms) : nullptr;
        ProcessMemorySnapshot after_rec640;
        std::optional<uint64_t> npu_after_rec640;
        if (metrics.enabled) {
            npu_after_rec640 = engine.print_memory_usage("[after rec640]");
            after_rec640 = print_process_memory_usage("[after rec640]");
        }
        if (cache_stats) engine.print_cache_usage("[after rec640]");
        auto rec_graph_1280 = fs::exists(rec_1280) ? load_profiled("rec1280", rec_1280, rec_depth, rec1280_fill, metrics.rec_model_load_ms) : nullptr;
        if (metrics.enabled) {
            const auto npu_after_rec1280 = engine.print_memory_usage("[after rec1280]");
            const auto after_rec1280 = print_process_memory_usage("[after rec1280]");
            print_npu_memory_delta("rec1280 graph + request pool", npu_after_rec640, npu_after_rec1280);
            print_process_memory_delta("rec1280 graph + request pool", after_rec640, after_rec1280);
        }
        if (cache_stats) engine.print_cache_usage("[after rec1280]");
        metrics.model_load_ms = metrics.det_model_load_ms + metrics.rec_model_load_ms;

        auto validate_rec_shape = [rec_input_u8](const std::shared_ptr<ppocr::NpuGraph>& graph, int width,
                                                  const char* label) {
            if (!graph) return;
            const std::vector<uint32_t> expected_dims = rec_input_u8
                ? std::vector<uint32_t>{1, 48, static_cast<uint32_t>(width), 3}
                : std::vector<uint32_t>{1, 3, 48, static_cast<uint32_t>(width)};
            if (graph->inputs().empty() || graph->inputs()[0].dims != expected_dims) {
                throw std::runtime_error(std::string(label) + " graph input does not match W=" +
                                         std::to_string(width) +
                                         "; use the Rec shape template or a matching static model");
            }
        };
        validate_rec_shape(rec_graph_160, 160, "rec160");
        validate_rec_shape(rec_graph_320, 320, "rec320");
        validate_rec_shape(rec_graph_640, 640, "rec640");
        validate_rec_shape(rec_graph_1280, 1280, "rec1280");

        const auto dict_load_start = ppocr::TraceRecorder::Clock::now();
        ppocr::CtcDecoder ctc_decoder(dict_path);
        const auto dict_load_end = ppocr::TraceRecorder::Clock::now();
        metrics.dict_load_ms = std::chrono::duration<double, std::milli>(dict_load_end - dict_load_start).count();
        if (trace) trace->add_host_event("Dictionary load", "Model Load", dict_load_start, dict_load_end);
        const auto post_load_start = ppocr::TraceRecorder::Clock::now();

        // 4. Detection Stage: Single-Tile or Multi-Tile Slicing
        const int tile_size = det_size;
        const bool need_slice = src_image.width > tile_size || src_image.height > tile_size;
        ppocr::ProbHeatmap heatmap(src_image.width, src_image.height);

        if (det_mode == DetMode::Resize) {
            if (det_graph->inputs().empty() ||
                det_graph->inputs()[0].total_bytes != det_input_bytes) {
                throw std::runtime_error("960 resize mode requires a 960x960 det graph (use the shape template)");
            }

            int tw = src_image.width;
            int th = src_image.height;
            {
                const int limit = 960;
                int max_side = std::max(tw, th);
                double ratio = (max_side > limit) ? static_cast<double>(limit) / max_side : 1.0;
                auto round32 = [](int v) {
                    return std::max(32, static_cast<int>(std::floor(v / 32.0 + 0.5)) * 32);
                };
                tw = round32(static_cast<int>(src_image.width * ratio));
                th = round32(static_cast<int>(src_image.height * ratio));
            }

            metrics.det_tile_count = 1;
            auto& req = det_graph->get_request(0);
            {
                ppocr::ScopedTimer timer(metrics.det_preprocess_ms, metrics.trace, "Det preprocess");
                // TODO: Resize/pad Det U8 directly into req USM instead of staging resized and bucket images.
                ppocr::ImageBGR resized = ppocr::WicDecoder::resize_linear(src_image, tw, th);

                ppocr::ImageBGR bucket;
                bucket.width = 960;
                bucket.height = 960;
                bucket.channels = 3;
                bucket.data.assign(static_cast<size_t>(960) * 960 * 3, 0);
                for (int y = 0; y < th; ++y) {
                    std::memcpy(bucket.data.data() + static_cast<size_t>(y) * 960 * 3,
                                resized.data.data() + static_cast<size_t>(y) * tw * 3,
                                static_cast<size_t>(tw) * 3);
                }
                fill_det_input(bucket, req.get_input_ptr(0));
            }
            {
                ppocr::ScopedTimer timer(metrics.det_inference_ms, metrics.trace, "Det host submit + wait");
                req.start_async();
                req.wait();
            }

            {
                ppocr::ScopedTimer timer(metrics.det_output_decode_ms, metrics.trace, "Det output decode/copy");
                const uint16_t* p_out_fp16 = reinterpret_cast<const uint16_t*>(req.get_output_ptr(0));
                ppocr::ProbHeatmap full(960, 960);
                for (int p = 0; p < 960 * 960; ++p) {
                    full.data()[p] = ppocr::fp16_to_float(p_out_fp16[p]);
                }

                heatmap = ppocr::ProbHeatmap(tw, th);
                for (int y = 0; y < th; ++y) {
                    std::memcpy(heatmap.data().data() + static_cast<size_t>(y) * tw,
                                full.data().data() + static_cast<size_t>(y) * 960,
                                static_cast<size_t>(tw) * sizeof(float));
                }
                if (!det_heatmap_dir.empty()) {
                    const fs::path tile_path = fs::path(det_heatmap_dir) / "det-tile-00-x0000-y0000.pgm";
                    if (!save_probability_pgm(full.data().data(), full.width(), full.height(), tile_path)) {
                        throw std::runtime_error("failed to save detector tile heatmap: " + tile_path.string());
                    }
                }
            }

            std::cout << "[*] Detection resize: " << src_image.width << "x" << src_image.height
                      << " -> " << tw << "x" << th << " (padded to 960x960)\n";
        } else if (need_slice) {
            // Preserve a positive stride for small experimental tile buckets.
            const int min_overlap = std::min(80, tile_size / 2);
            const int eff_step = tile_size - min_overlap;

            int nx = (src_image.width <= tile_size) ? 1 : static_cast<int>(std::ceil(static_cast<float>(src_image.width - min_overlap) / eff_step));
            int ny = (src_image.height <= tile_size) ? 1 : static_cast<int>(std::ceil(static_cast<float>(src_image.height - min_overlap) / eff_step));
            metrics.det_tile_count = nx * ny;

            float step_x = (nx > 1) ? static_cast<float>(src_image.width - tile_size) / (nx - 1) : 0.0f;
            float step_y = (ny > 1) ? static_cast<float>(src_image.height - tile_size) / (ny - 1) : 0.0f;

            std::cout << "[*] Large image detected. Dispatching tiled NPU detection: "
                      << nx << "x" << ny << " = " << metrics.det_tile_count << " tiles ("
                      << tile_size << "x" << tile_size << ", uniform step)...\n";

            struct TileTask {
                int x, y;
                ppocr::ImageBGR tile_img;
            };

            // TODO: After each slot is free, extract its tile directly into U8 USM instead of storing tile_img buffers.
            std::vector<TileTask> tasks;
            {
                ppocr::ScopedTimer timer(metrics.tile_prep_ms, metrics.trace, "Det tile preparation");
                for (int iy = 0; iy < ny; ++iy) {
                    int ty = (ny > 1) ? static_cast<int>(std::round(iy * step_y)) : 0;
                    for (int ix = 0; ix < nx; ++ix) {
                        int tx = (nx > 1) ? static_cast<int>(std::round(ix * step_x)) : 0;
                        tasks.push_back({ tx, ty, ppocr::WicDecoder::extract_tile(src_image, tx, ty, tile_size, tile_size) });
                    }
                }
            }

            // Two-slot tile pipeline: keep submitting ahead while each slot is
            // awaited only on its own queue (device execution itself is serial)
            std::vector<std::vector<float>> tile_outputs(tasks.size(), std::vector<float>(tile_size * tile_size));
            metrics.det_submission_wall_includes_components = true;
            {
                ppocr::ScopedTimer timer(metrics.det_inference_ms, metrics.trace, "Det submission pipeline",
                                         "Host", 2002);
                for (size_t i = 0; i < tasks.size(); ++i) {
                    auto& req = det_graph->get_request(i % 2);
                    if (i >= 2) {
                        req.wait();
                        {
                            ppocr::ScopedTimer timer(metrics.det_output_decode_ms, metrics.trace, "Det output decode/copy");
                            const uint16_t* p_out_fp16 = reinterpret_cast<const uint16_t*>(req.get_output_ptr(0));
                            for (int p = 0; p < tile_size * tile_size; ++p)
                                tile_outputs[i - 2][p] = ppocr::fp16_to_float(p_out_fp16[p]);
                        }
                    }

                    {
                        ppocr::ScopedTimer timer(metrics.det_preprocess_ms, metrics.trace, "Det preprocess");
                        fill_det_input(tasks[i].tile_img, req.get_input_ptr(0));
                    }
                    req.start_async();
                }

                for (size_t i = tasks.size(); i < tasks.size() + 2; ++i) {
                    if (tasks.size() >= 1) {
                        size_t slot = (i - 2);
                        if (slot < tasks.size()) {
                            auto& req = det_graph->get_request(slot % 2);
                            req.wait();
                            {
                            ppocr::ScopedTimer timer(metrics.det_output_decode_ms, metrics.trace, "Det output decode/copy");
                                const uint16_t* p_out_fp16 = reinterpret_cast<const uint16_t*>(req.get_output_ptr(0));
                                for (int p = 0; p < tile_size * tile_size; ++p)
                                    tile_outputs[slot][p] = ppocr::fp16_to_float(p_out_fp16[p]);
                            }
                        }
                    }
                }
            }

            // Element-wise Max Stitching
            if (!det_heatmap_dir.empty()) {
                for (size_t i = 0; i < tasks.size(); ++i) {
                    const auto& task = tasks[i];
                    const fs::path tile_path = fs::path(det_heatmap_dir) /
                        ("det-tile-" + std::to_string(i) + "-x" + std::to_string(task.x) +
                         "-y" + std::to_string(task.y) + ".pgm");
                    if (!save_probability_pgm(tile_outputs[i].data(), tile_size, tile_size, tile_path)) {
                        throw std::runtime_error("failed to save detector tile heatmap: " + tile_path.string());
                    }
                }
            }
            for (size_t i = 0; i < tasks.size(); ++i) {
                heatmap.stitch_tile_max(tile_outputs[i].data(), tile_size, tile_size, tasks[i].x, tasks[i].y);
            }
        } else {
            // Single-shot Detection
            metrics.det_tile_count = 1;
            // TODO: Extract the single Det U8 tile directly into the request USM.
            ppocr::ImageBGR single_tile;
            {
                ppocr::ScopedTimer timer(metrics.tile_prep_ms, metrics.trace, "Det tile preparation");
                single_tile = ppocr::WicDecoder::extract_tile(src_image, 0, 0, tile_size, tile_size);
            }

            auto& req = det_graph->get_request(0);
            {
                ppocr::ScopedTimer timer(metrics.det_preprocess_ms, metrics.trace, "Det preprocess");
                fill_det_input(single_tile, req.get_input_ptr(0));
            }

            {
                ppocr::ScopedTimer timer(metrics.det_inference_ms, metrics.trace, "Det host submit + wait");
                req.start_async();
                req.wait();
            }

            std::vector<float> single_prob(tile_size * tile_size);
            {
                ppocr::ScopedTimer timer(metrics.det_output_decode_ms, metrics.trace, "Det output decode/copy");
                const uint16_t* p_out_fp16 = reinterpret_cast<const uint16_t*>(req.get_output_ptr(0));
                for (int p = 0; p < tile_size * tile_size; ++p)
                    single_prob[p] = ppocr::fp16_to_float(p_out_fp16[p]);
            }
            if (!det_heatmap_dir.empty()) {
                const fs::path tile_path = fs::path(det_heatmap_dir) / "det-tile-00-x0000-y0000.pgm";
                if (!save_probability_pgm(single_prob.data(), tile_size, tile_size, tile_path)) {
                    throw std::runtime_error("failed to save detector tile heatmap: " + tile_path.string());
                }
            }
            heatmap.stitch_tile_max(single_prob.data(), tile_size, tile_size, 0, 0);
        }

        // 5. DBNet Post-processing
        ppocr::DbNetPostProcessor post_proc(0.2f, 0.45f, 1.4f);
        std::vector<ppocr::TextBox> boxes;
        {
            ppocr::ScopedTimer timer(metrics.dbnet_postprocess_ms, metrics.trace, "DBNet postprocess");
            boxes = post_proc.extract_boxes(heatmap, src_image.width, src_image.height);
        }
        metrics.detected_box_count = static_cast<int>(boxes.size());
        std::cout << "[+] Text Detection complete: found " << boxes.size() << " text lines.\n";
        if (!det_heatmap_dir.empty()) {
            fs::create_directories(det_heatmap_dir);
            const fs::path output = fs::path(det_heatmap_dir) / "detector-heatmap.pgm";
            if (!save_heatmap_pgm(heatmap, output)) {
                throw std::runtime_error("failed to save detector heatmap: " + output.string());
            }
            std::cout << "[*] Detector heatmap saved: " << output.string()
                      << " (" << heatmap.width() << "x" << heatmap.height() << ")\n";
        }

        // TODO(CLS): this pipeline has no text-line orientation classification (0/180).
        // Upside-down lines (scans, rotated screenshots) are fed to rec as-is and CTC
        // decodes them right-to-left, producing garbage. The 90-degree quad reordering
        // in crop_quad_natural() only handles vertical text geometry; it is NOT a
        // 0/180 classifier.
        // Plan: build a PP-LCNet textline-orientation IR + NPU blob (approx. 1ms/line
        // on NPU), run per crop before rec, flip the crop 180 degrees when cls says so.
        // Reference: official pipeline is det -> crop -> cls -> rec; the C# engine
        // measures cls_graph 10.7ms per ~10-line image (145 lines: 142ms).

        // 6. Text Recognition Stage with Aspect-Ratio Preserving Tiling & Grid Logits Stitching
        if (!boxes.empty()) {
            if (!rec_graph_160 && !rec_graph_320 && !rec_graph_640 && !rec_graph_1280) {
                std::cerr << "[-] Notice: No recognition models found. Detection boxes output only.\n";
            } else {
                ppocr::ScopedTimer rec_pipeline_timer(
                    metrics.rec_pipeline_wall_ms, metrics.trace, "Recognition pipeline wall", "Host", 2003);
                std::cout << "[*] Executing Recognition on NPU across Multi-Bucket Grid Pipeline...\n";

                // Recognition pipeline: submit box b, prepare b+1, decode b-2. Each
                // slot waits only on the request it submitted to (pooled per-slot
                // queue), so while the host crops, normalizes and CTC-decodes, the
                // other in-flight requests stay queued on the NPU (the device runs
                // them serially; depth=2 keeps it busy).
                struct PendingRec {
                    ppocr::NpuInferRequest* req = nullptr;
                    ppocr::TextBox* box = nullptr;
                    int vocab_size = 0;
                    int valid_seq_len = 0;
                    int graph_width = 0;
                    ze_graph_argument_precision_t output_precision = ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN;
                };
                std::vector<PendingRec> pending(static_cast<size_t>(rec_depth));
                size_t rec_dump_count = 0;

                auto dump_rec_request = [&](ppocr::NpuInferRequest& req, int width) {
                    if (rec_dump_dir.empty() || width != 640) return;
                    const std::string prefix = (fs::path(rec_dump_dir) /
                        ("rec640_" + std::to_string(rec_dump_count++))).string();
                    auto write_bytes = [](const fs::path& path, const void* data, size_t size) {
                        std::ofstream file(path, std::ios::binary);
                        if (!file || !file.write(static_cast<const char*>(data),
                                                 static_cast<std::streamsize>(size))) {
                            throw std::runtime_error("Failed to write Rec diagnostic dump: " + path.string());
                        }
                    };
                    write_bytes(prefix + "_input.u8", req.get_input_ptr(), req.get_input_size());
                    write_bytes(prefix + "_output.bin", req.get_output_ptr(), req.get_output_size());
                };

                auto is_index_precision = [](ze_graph_argument_precision_t precision) {
                    return precision == ZE_GRAPH_ARGUMENT_PRECISION_INT32 ||
                           precision == ZE_GRAPH_ARGUMENT_PRECISION_INT64;
                };
                auto rec_vocab_size = [&](const ppocr::TensorDesc& output, int seq_len) {
                    if (output.precision == ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
                        const size_t row_bytes = static_cast<size_t>(seq_len) * sizeof(uint16_t);
                        if (row_bytes == 0 || output.total_bytes % row_bytes != 0) {
                            throw std::runtime_error("Rec FP16 output size is not divisible by its sequence length");
                        }
                        return static_cast<int>(output.total_bytes / row_bytes);
                    }
                    if (is_index_precision(output.precision)) {
                        return static_cast<int>(ctc_decoder.vocab_size());
                    }
                    throw std::runtime_error("Unsupported Rec output precision for CTC decoding");
                };
                auto decode_rec_output = [&](const void* output, ze_graph_argument_precision_t precision,
                                             int seq_len, int vocab_size, float& confidence) {
                    if (precision == ZE_GRAPH_ARGUMENT_PRECISION_FP16) {
                        return ctc_decoder.decode(static_cast<const uint16_t*>(output), seq_len,
                                                  vocab_size, confidence);
                    }
                    if (precision == ZE_GRAPH_ARGUMENT_PRECISION_INT32) {
                        return ctc_decoder.decode_indices(static_cast<const int32_t*>(output),
                                                          seq_len, confidence);
                    }
                    if (precision == ZE_GRAPH_ARGUMENT_PRECISION_INT64) {
                        return ctc_decoder.decode_indices(static_cast<const int64_t*>(output),
                                                          seq_len, confidence);
                    }
                    throw std::runtime_error("Unsupported Rec output precision for CTC decoding");
                };

                auto consume_rec_slot = [&](size_t slot) {
                    PendingRec& p = pending[slot];
                    dump_rec_request(*p.req, p.graph_width);
                    {
                        ppocr::ScopedTimer timer(metrics.ctc_decode_ms, metrics.trace, "CTC decode");
                        p.box->text = decode_rec_output(p.req->get_output_ptr(0), p.output_precision,
                                                        p.valid_seq_len, p.vocab_size,
                                                        p.box->text_confidence);
                    }
                    p.req = nullptr;
                };

                auto flush_rec_slot = [&](size_t slot) {
                    PendingRec& p = pending[slot];
                    if (!p.req) return;
                    {
                        ppocr::ScopedTimer timer(metrics.rec_wait_ms, metrics.trace, "Rec fence wait");
                        p.req->wait();
                    }
                    consume_rec_slot(slot);
                };

                for (size_t b = 0; b < boxes.size(); ++b) {
                    auto& box = boxes[b];
                    // TODO: Derive natural width first, reclaim the bucket request, then warp and pad U8 directly into its USM input.
                    // Crop quad normalized to height 48, with natural uncompressed width
                    ppocr::ImageBGR natural_crop;
                    {
                        ppocr::ScopedTimer timer(metrics.rec_crop_prep_ms, metrics.trace, "Text crop and warp");
                        natural_crop = ppocr::WicDecoder::crop_quad_natural(src_image, box.quad, 48);
                    }

                    int natural_w = natural_crop.width;

                    // Use a single bucket through W=1280; preserve the W=640 sliding fallback above it.
                    if (natural_w <= 640 || (natural_w <= 1280 && rec_graph_1280)) {
                        int target_w = 320;
                        std::shared_ptr<ppocr::NpuGraph> target_graph = rec_graph_320;

                        if (natural_w <= 160 && rec_graph_160) {
                            target_w = 160;
                            target_graph = rec_graph_160;
                        } else if (natural_w > 320 && natural_w <= 640 && rec_graph_640) {
                            target_w = 640;
                            target_graph = rec_graph_640;
                        } else if (natural_w > 640 && natural_w <= 1280 && rec_graph_1280) {
                            target_w = 1280;
                            target_graph = rec_graph_1280;
                        }

                        if (!target_graph) target_graph = rec_graph_320 ? rec_graph_320 : (rec_graph_160 ? rec_graph_160 : (rec_graph_640 ? rec_graph_640 : rec_graph_1280));
                        if (!target_graph) continue;

                        ppocr::ImageBGR padded_crop;
                        padded_crop.width = target_w;
                        padded_crop.height = 48;
                        padded_crop.channels = 3;
                        padded_crop.data.assign(target_w * 48 * 3, 0);

                        int valid_w = std::min(target_w, natural_w);
                        {
                            ppocr::ScopedTimer timer(metrics.rec_padding_ms, metrics.trace, "Rec padding/copy");
                            for (int y = 0; y < 48; ++y) {
                                std::memcpy(
                                    padded_crop.data.data() + (y * target_w) * 3,
                                    natural_crop.data.data() + (y * natural_w) * 3,
                                    valid_w * 3
                                );
                            }
                        }

                        // Default: reuse slot b%depth only after its previous submission
                        // (box b-depth) is done; the other slots' requests are untouched.
                        // Poll mode (depth 2 only): consume whichever slot finished.
                        // first, then reuse that slot.
                        size_t slot = b % static_cast<size_t>(rec_depth);
                        if (rec_poll && rec_depth == 2) {
                            if (pending[0].req && pending[1].req) {
                                size_t completed_slot = pending.size();
                                {
                                    ppocr::ScopedTimer timer(metrics.rec_wait_ms, metrics.trace, "Rec poll wait");
                                    for (;;) {
                                        if (pending[0].req->try_wait()) { completed_slot = 0; break; }
                                        if (pending[1].req->try_wait()) { completed_slot = 1; break; }
                                        Sleep(0);
                                    }
                                }
                                slot = completed_slot;
                                consume_rec_slot(completed_slot);
                            } else {
                                slot = pending[0].req ? 1 : 0;
                            }
                        } else {
                            flush_rec_slot(slot);
                        }

                        auto& req = target_graph->get_request(slot);
                        {
                            ppocr::ScopedTimer timer(metrics.rec_preprocess_ms, metrics.trace, "Rec preprocess");
                            if (rec_input_u8) {
                                std::memcpy(req.get_input_ptr(0), padded_crop.data.data(), padded_crop.data.size());
                            } else {
                                ppocr::WicDecoder::normalize_to_nchw_fp16(
                                    padded_crop, req.get_input_ptr(0),
                                    127.5f, 127.5f, 127.5f, 127.5f, 127.5f, 127.5f
                                );
                            }
                        }

                        {
                            ppocr::ScopedTimer timer(metrics.rec_submit_ms, metrics.trace, "Rec host submit");
                            req.start_async();
                        }

                        int seq_len = target_w / 8;
                        const auto& output = target_graph->outputs().at(0);
                        int vocab_size = rec_vocab_size(output, seq_len);
                        pending[slot].req = &req;
                        pending[slot].box = &box;
                        pending[slot].vocab_size = vocab_size;
                        pending[slot].valid_seq_len =
                            std::min(seq_len, static_cast<int>(std::ceil(static_cast<float>(valid_w) / 8.0f)));
                        pending[slot].graph_width = target_w;
                        pending[slot].output_precision = output.precision;
                    } else {
                        // Ultra-long lines act as a pipeline barrier: the slice loop
                        // below reuses the 640-graph request slots.
                        for (size_t s = 0; s < pending.size(); ++s) flush_rec_slot(s);
                        // CASE 2: Ultra-long line (> 640px) -> Grid Sliding Window with Midpoint Merging
                        const int win_w = 640;
                        const int overlap_px = 128;
                        const int cell_stride = 8;
                        const int overlap_cells = overlap_px / cell_stride; // 16 cells
                        const int half_overlap_cells = overlap_cells / 2;    // 8 cells
                        const int step_px = win_w - overlap_px;             // 512px

                        int num_slices = static_cast<int>(std::ceil(static_cast<float>(natural_w - overlap_px) / step_px));
                        auto target_graph = rec_graph_640 ? rec_graph_640 : rec_graph_320;
                        if (!target_graph) continue;

                        int win_cells = win_w / cell_stride; // 80 cells
                        const auto& output = target_graph->outputs().at(0);
                        int vocab_size = rec_vocab_size(output, win_cells);
                        const bool index_output = is_index_precision(output.precision);

                        // Keep stitched class indices in their graph output width.
                        std::vector<int32_t> merged_indices32;
                        std::vector<int64_t> merged_indices64;
                        std::vector<uint16_t> merged_max_values;
                        if (index_output) {
                            if (output.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT64) {
                                merged_indices64.reserve(static_cast<size_t>(num_slices) * win_cells);
                            } else {
                                merged_indices32.reserve(static_cast<size_t>(num_slices) * win_cells);
                            }
                        } else {
                            merged_indices32.reserve(static_cast<size_t>(num_slices) * win_cells);
                            merged_max_values.reserve(static_cast<size_t>(num_slices) * win_cells);
                        }

                        for (int s = 0; s < num_slices; ++s) {
                            int sx = s * step_px;
                            int valid_slice_w = std::min(win_w, natural_w - sx);

                            ppocr::ImageBGR slice_img;
                            {
                                ppocr::ScopedTimer timer(metrics.rec_padding_ms, metrics.trace, "Rec padding/copy");
                                slice_img.width = win_w;
                                slice_img.height = 48;
                                slice_img.channels = 3;
                                slice_img.data.assign(win_w * 48 * 3, 0);

                                for (int y = 0; y < 48; ++y) {
                                    std::memcpy(
                                        slice_img.data.data() + (y * win_w) * 3,
                                        natural_crop.data.data() + (y * natural_w + sx) * 3,
                                        valid_slice_w * 3
                                    );
                                }
                            }

                            auto& req = target_graph->get_request(s % 2);
                            {
                                ppocr::ScopedTimer timer(metrics.rec_preprocess_ms, metrics.trace, "Rec preprocess");
                                if (rec_input_u8) {
                                    std::memcpy(req.get_input_ptr(0), slice_img.data.data(), slice_img.data.size());
                                } else {
                                    ppocr::WicDecoder::normalize_to_nchw_fp16(
                                        slice_img, req.get_input_ptr(0),
                                        127.5f, 127.5f, 127.5f, 127.5f, 127.5f, 127.5f
                                    );
                                }
                            }

                            {
                                {
                                    ppocr::ScopedTimer timer(metrics.rec_submit_ms, metrics.trace, "Rec host submit");
                                    req.start_async();
                                }
                                ppocr::ScopedTimer timer(metrics.rec_wait_ms, metrics.trace, "Rec fence wait");
                                req.wait();
                            }

                            dump_rec_request(req, win_w);

                            int valid_slice_cells = std::min(win_cells, static_cast<int>(std::ceil(static_cast<float>(valid_slice_w) / cell_stride)));

                            // Keep only per-frame winners after the midpoint split; the full vocabulary is never stitched.
                            int cell_start = (s == 0) ? 0 : half_overlap_cells;
                            int cell_end = (s == num_slices - 1) ? valid_slice_cells : (win_cells - half_overlap_cells);

                            ppocr::ScopedTimer merge_timer(metrics.rec_grid_output_merge_ms);
                            if (cell_end > cell_start) {
                                if (!index_output) {
                                    const auto* p_slice_out = static_cast<const uint16_t*>(req.get_output_ptr(0));
                                    for (int t = cell_start; t < cell_end; ++t) {
                                        const uint16_t* step = p_slice_out + static_cast<size_t>(t) * vocab_size;
                                        int best_idx = 0;
                                        float best_val = -1e9f;
                                        for (int c = 0; c < vocab_size; ++c) {
                                            const float value = ppocr::fp16_to_float(step[c]);
                                            if (value > best_val) {
                                                best_val = value;
                                                best_idx = c;
                                            }
                                        }
                                        merged_indices32.push_back(best_idx);
                                        merged_max_values.push_back(step[best_idx]);
                                    }
                                } else if (output.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT64) {
                                    const auto* p_slice_out = static_cast<const int64_t*>(req.get_output_ptr(0));
                                    merged_indices64.insert(merged_indices64.end(), p_slice_out + cell_start,
                                                            p_slice_out + cell_end);
                                } else {
                                    const auto* p_slice_out = static_cast<const int32_t*>(req.get_output_ptr(0));
                                    merged_indices32.insert(merged_indices32.end(), p_slice_out + cell_start,
                                                            p_slice_out + cell_end);
                                }
                            }
                        }

                        int total_seq_len = output.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT64
                            ? static_cast<int>(merged_indices64.size())
                            : static_cast<int>(merged_indices32.size());
                        {
                            ppocr::ScopedTimer timer(metrics.ctc_decode_ms, metrics.trace, "CTC decode");
                            if (output.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT64) {
                                box.text = ctc_decoder.decode_indices(merged_indices64.data(), total_seq_len,
                                                                      box.text_confidence);
                            } else if (output.precision == ZE_GRAPH_ARGUMENT_PRECISION_INT32) {
                                box.text = ctc_decoder.decode_indices(merged_indices32.data(), total_seq_len,
                                                                      box.text_confidence);
                            } else if (!index_output) {
                                box.text = ctc_decoder.decode_indices(merged_indices32.data(),
                                                                      merged_max_values.data(), total_seq_len,
                                                                      box.text_confidence);
                            } else {
                                throw std::runtime_error("Unsupported Rec output precision for grid CTC decoding");
                            }
                        }
                    }
                }

                // Flush all remaining in-flight recognition slots
                for (size_t s = 0; s < pending.size(); ++s) flush_rec_slot(s);
            }
        }

        // Official drop_score filtering: drop low confidence boxes (< 0.50)
        const float drop_score = 0.50f;
        std::vector<ppocr::TextBox> valid_boxes;
        for (const auto& b : boxes) {
            // Note: In unnormalized logits, text_confidence is unnormalized logits sum or softmax average
            if (!b.text.empty()) {
                valid_boxes.push_back(b);
            }
        }
        boxes = std::move(valid_boxes);

        auto end_total = ppocr::TraceRecorder::Clock::now();
        metrics.total_end_to_end_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();
        metrics.post_load_pipeline_ms = std::chrono::duration<double, std::milli>(
            end_total - post_load_start).count();
        if (trace) {
            trace->add_host_event("Post-load OCR pipeline", "Pipeline", post_load_start, end_total, 2000);
            trace->add_host_event("End-to-end", "Pipeline", start_total, end_total, 2001);
        }

        // 7. Formatted Terminal Output
        std::cout << "\n============================================================\n";
        std::cout << "                 OCR Recognition Results                    \n";
        std::cout << "============================================================\n";
        for (size_t i = 0; i < boxes.size(); ++i) {
            const auto& b = boxes[i];
            std::cout << "[" << std::setw(2) << (i + 1) << "] "
                      << "Conf: " << std::fixed << std::setprecision(2) << b.score << " | "
                      << "Box: [(" << static_cast<int>(b.quad[0].x) << "," << static_cast<int>(b.quad[0].y) << "), "
                      << "(" << static_cast<int>(b.quad[2].x) << "," << static_cast<int>(b.quad[2].y) << ")] | "
                      << "Text: " << b.text << "\n";
        }
        std::cout << "============================================================\n";

        // 8. Output Detailed Profile if Requested
        metrics.print_report();
        if (trace) {
            const size_t traced_requests = trace->collect_pending_profiles();
            if (!trace_path.empty()) {
                std::string error;
                if (!trace->write_chrome_trace(trace_path, error)) {
                    throw std::runtime_error("Failed to export trace to '" + trace_path + "': " + error);
                }
                std::cout << "[+] NPU/host trace exported: " << trace_path
                          << " (" << traced_requests << " requests)\n";
            }
            if (!trace_raw_dir.empty()) {
                std::cout << "[+] Raw NPU profile buffers and graph blobs exported to: " << trace_raw_dir << "\n";
            }
        }

    } catch (const std::exception& ex) {
        std::cerr << "[-] Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
