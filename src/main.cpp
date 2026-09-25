#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "npu_engine.hpp"
#include "postprocess.hpp"
#include "profiler.hpp"
#include "wic_decoder.hpp"

#include <windows.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void print_usage(const char* prog) {
    std::cout << "PP-OCRv6 Intel NPU (Level Zero Native) Inference Engine\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog << " --image <path> [options]\n";
    std::cout << "  " << prog << " --clipboard [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --image <path>         Path to input image file (JPG, PNG, BMP, TIFF, WebP)\n";
    std::cout << "  --clipboard            Read image directly from Windows Clipboard\n";
    std::cout << "  --det-model <path>     Path to Det model (.xml, .ngraph_lite, or .blob) [default: models/ppocrv6_small_det_640x640.xml]\n";
    std::cout << "  --rec-160 <path>       Path to Rec W=160 model [default: models/ppocrv6_small_rec_w160.xml]\n";
    std::cout << "  --rec-320 <path>       Path to Rec W=320 model [default: models/ppocrv6_small_rec_w320.xml]\n";
    std::cout << "  --rec-640 <path>       Path to Rec W=640 model [default: models/ppocrv6_small_rec_w640.xml]\n";
    std::cout << "  --dict <path>          Path to dictionary [default: models/ppocr_keys_v1.txt]\n";
    std::cout << "  --slice                Force tiling slicing for large images\n";
    std::cout << "  --profile, -p          Print detailed end-to-end performance breakdown\n";
    std::cout << "  --help, -h             Show this help message\n";
}

std::wstring to_wstring(const std::string& str) {
    if (str.empty()) return std::wstring();
    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, nullptr, 0);
    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), -1, &wstr[0], size);
    if (!wstr.empty() && wstr.back() == 0) wstr.pop_back();
    return wstr;
}

} // namespace

int main(int argc, char* argv[]) {
    // Set console to UTF-8
    SetConsoleOutputCP(CP_UTF8);

    try {
        std::string image_path;
        bool from_clipboard = false;

        auto resolve_model = [](const std::string& base_no_ext) -> std::string {
            if (fs::exists(base_no_ext + ".xml")) return base_no_ext + ".xml";
            if (fs::exists("cache/" + fs::path(base_no_ext).stem().string() + ".blob")) return "cache/" + fs::path(base_no_ext).stem().string() + ".blob";
            if (fs::exists(base_no_ext + ".blob")) return base_no_ext + ".blob";
            if (fs::exists(base_no_ext + ".ngraph_lite")) return base_no_ext + ".ngraph_lite";
            return base_no_ext + ".xml";
        };

        std::string det_model = resolve_model("models/ppocrv6_small_det_640x640");
        std::string rec_160 = resolve_model("models/ppocrv6_small_rec_w160");
        std::string rec_320 = resolve_model("models/ppocrv6_small_rec_w320");
        std::string rec_640 = resolve_model("models/ppocrv6_small_rec_w640");
        std::string dict_path = "models/ppocr_keys_v1.txt";
        bool force_slice = false;
        ppocr::ProfileMetrics metrics;

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--image" && i + 1 < argc) {
                image_path = argv[++i];
            } else if (arg == "--clipboard") {
                from_clipboard = true;
            } else if (arg == "--det-model" && i + 1 < argc) {
                det_model = argv[++i];
            } else if (arg == "--rec-160" && i + 1 < argc) {
                rec_160 = argv[++i];
            } else if (arg == "--rec-320" && i + 1 < argc) {
                rec_320 = argv[++i];
            } else if (arg == "--rec-640" && i + 1 < argc) {
                rec_640 = argv[++i];
            } else if (arg == "--dict" && i + 1 < argc) {
                dict_path = argv[++i];
            } else if (arg == "--slice") {
                force_slice = true;
            } else if (arg == "--profile" || arg == "-p") {
                metrics.enabled = true;
            } else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            }
        }

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

        auto start_total = std::chrono::high_resolution_clock::now();

        // 1. Image Acquisition via WIC
        ppocr::ImageBGR src_image;
        {
            ppocr::ScopedTimer timer(metrics.image_decode_ms);
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
        if (!engine.initialize()) {
            std::cerr << "[-] NPU initialization failed.\n";
            return 1;
        }

        // 3. Load Models
        std::cout << "[*] Loading Detection model onto NPU: " << det_model << "..." << std::endl;
        auto det_graph = engine.load_model(det_model, 2); // 2-concurrency Ping-Pong

        std::cout << "[*] Loading Recognition Multi-Buckets onto NPU...\n";
        auto rec_graph_160 = fs::exists(rec_160) ? engine.load_model(rec_160, 2) : nullptr;
        auto rec_graph_320 = fs::exists(rec_320) ? engine.load_model(rec_320, 2) : nullptr;
        auto rec_graph_640 = fs::exists(rec_640) ? engine.load_model(rec_640, 2) : nullptr;

        ppocr::CtcDecoder ctc_decoder(dict_path);

        // 4. Detection Stage: Single-Tile or Multi-Tile Slicing
        const int tile_size = 640;
        bool need_slice = force_slice || (src_image.width > tile_size || src_image.height > tile_size);
        ppocr::ProbHeatmap heatmap(src_image.width, src_image.height);

        if (need_slice) {
            // Uniformly distributed tiling with guaranteed min_overlap=80
            const int min_overlap = 80;
            const int eff_step = tile_size - min_overlap;

            int nx = (src_image.width <= tile_size) ? 1 : static_cast<int>(std::ceil(static_cast<float>(src_image.width - min_overlap) / eff_step));
            int ny = (src_image.height <= tile_size) ? 1 : static_cast<int>(std::ceil(static_cast<float>(src_image.height - min_overlap) / eff_step));
            metrics.det_tile_count = nx * ny;

            float step_x = (nx > 1) ? static_cast<float>(src_image.width - tile_size) / (nx - 1) : 0.0f;
            float step_y = (ny > 1) ? static_cast<float>(src_image.height - tile_size) / (ny - 1) : 0.0f;

            std::cout << "[*] Large image detected. Dispatching 2-stream NPU slicing: "
                      << nx << "x" << ny << " = " << metrics.det_tile_count << " tiles (640x640, uniform step)...\n";

            struct TileTask {
                int x, y;
                ppocr::ImageBGR tile_img;
            };

            std::vector<TileTask> tasks;
            {
                ppocr::ScopedTimer timer(metrics.tile_prep_ms);
                for (int iy = 0; iy < ny; ++iy) {
                    int ty = (ny > 1) ? static_cast<int>(std::round(iy * step_y)) : 0;
                    for (int ix = 0; ix < nx; ++ix) {
                        int tx = (nx > 1) ? static_cast<int>(std::round(ix * step_x)) : 0;
                        tasks.push_back({ tx, ty, ppocr::WicDecoder::extract_tile(src_image, tx, ty, tile_size, tile_size) });
                    }
                }
            }

            // Execute Ping-Pong asynchronous queue across tiles
            std::vector<std::vector<float>> tile_outputs(tasks.size(), std::vector<float>(tile_size * tile_size));
            {
                ppocr::ScopedTimer timer(metrics.det_inference_ms);
                for (size_t i = 0; i < tasks.size(); ++i) {
                    auto& req = det_graph->get_request(i % 2);
                    if (i >= 2) {
                        req.wait();
                        const uint16_t* p_out_fp16 = reinterpret_cast<const uint16_t*>(req.get_output_ptr(0));
                        for (int p = 0; p < tile_size * tile_size; ++p) {
                            tile_outputs[i - 2][p] = ppocr::fp16_to_float(p_out_fp16[p]);
                        }
                    }

                    ppocr::WicDecoder::normalize_to_nchw_fp16(tasks[i].tile_img, req.get_input_ptr(0));
                    req.start_async();
                }

                for (size_t i = tasks.size(); i < tasks.size() + 2; ++i) {
                    if (tasks.size() >= 1) {
                        size_t slot = (i - 2);
                        if (slot < tasks.size()) {
                            auto& req = det_graph->get_request(slot % 2);
                            req.wait();
                            const uint16_t* p_out_fp16 = reinterpret_cast<const uint16_t*>(req.get_output_ptr(0));
                            for (int p = 0; p < tile_size * tile_size; ++p) {
                                tile_outputs[slot][p] = ppocr::fp16_to_float(p_out_fp16[p]);
                            }
                        }
                    }
                }
            }

            // Element-wise Max Stitching
            for (size_t i = 0; i < tasks.size(); ++i) {
                heatmap.stitch_tile_max(tile_outputs[i].data(), tile_size, tile_size, tasks[i].x, tasks[i].y);
            }
        } else {
            // Single-shot Detection
            metrics.det_tile_count = 1;
            ppocr::ImageBGR single_tile;
            {
                ppocr::ScopedTimer timer(metrics.tile_prep_ms);
                single_tile = ppocr::WicDecoder::extract_tile(src_image, 0, 0, tile_size, tile_size);
            }

            auto& req = det_graph->get_request(0);
            ppocr::WicDecoder::normalize_to_nchw_fp16(single_tile, req.get_input_ptr(0));

            {
                ppocr::ScopedTimer timer(metrics.det_inference_ms);
                req.start_async();
                req.wait();
            }

            const uint16_t* p_out_fp16 = reinterpret_cast<const uint16_t*>(req.get_output_ptr(0));
            std::vector<float> single_prob(tile_size * tile_size);
            for (int p = 0; p < tile_size * tile_size; ++p) {
                single_prob[p] = ppocr::fp16_to_float(p_out_fp16[p]);
            }
            heatmap.stitch_tile_max(single_prob.data(), tile_size, tile_size, 0, 0);
        }

        // 5. DBNet Post-processing
        ppocr::DbNetPostProcessor post_proc(0.2f, 0.45f, 1.4f);
        std::vector<ppocr::TextBox> boxes;
        {
            ppocr::ScopedTimer timer(metrics.dbnet_postprocess_ms);
            boxes = post_proc.extract_boxes(heatmap, src_image.width, src_image.height);
        }
        metrics.detected_box_count = static_cast<int>(boxes.size());
        std::cout << "[+] Text Detection complete: found " << boxes.size() << " text lines.\n";

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
            if (!rec_graph_160 && !rec_graph_320 && !rec_graph_640) {
                std::cerr << "[-] Notice: No recognition models found. Detection boxes output only.\n";
            } else {
                std::cout << "[*] Executing Recognition on NPU across Multi-Bucket Grid Pipeline...\n";

                for (size_t b = 0; b < boxes.size(); ++b) {
                    auto& box = boxes[b];
                    // Crop quad normalized to height 48, with natural uncompressed width
                    ppocr::ImageBGR natural_crop;
                    {
                        ppocr::ScopedTimer timer(metrics.rec_crop_prep_ms);
                        natural_crop = ppocr::WicDecoder::crop_quad_natural(src_image, box.quad, 48);
                    }

                    int natural_w = natural_crop.width;

                    // CASE 1: Standard single-bucket inference (natural_w <= 640)
                    if (natural_w <= 640) {
                        int target_w = 320;
                        std::shared_ptr<ppocr::NpuGraph> target_graph = rec_graph_320;

                        if (natural_w <= 160 && rec_graph_160) {
                            target_w = 160;
                            target_graph = rec_graph_160;
                        } else if (natural_w > 320 && rec_graph_640) {
                            target_w = 640;
                            target_graph = rec_graph_640;
                        }

                        if (!target_graph) target_graph = rec_graph_320 ? rec_graph_320 : (rec_graph_160 ? rec_graph_160 : rec_graph_640);
                        if (!target_graph) continue;

                        ppocr::ImageBGR padded_crop;
                        padded_crop.width = target_w;
                        padded_crop.height = 48;
                        padded_crop.channels = 3;
                        padded_crop.data.assign(target_w * 48 * 3, 0);

                        int valid_w = std::min(target_w, natural_w);
                        for (int y = 0; y < 48; ++y) {
                            std::memcpy(
                                padded_crop.data.data() + (y * target_w) * 3,
                                natural_crop.data.data() + (y * natural_w) * 3,
                                valid_w * 3
                            );
                        }

                        // TODO(NPU parallelism): blocking per-box submit/wait serializes rec;
                        // see npu_engine.cpp wait() TODO. Pipeline: submit box b, crop b+1, decode b-1.
                        auto& req = target_graph->get_request(b % 2);
                        ppocr::WicDecoder::normalize_to_nchw_fp16(
                            padded_crop, req.get_input_ptr(0),
                            127.5f, 127.5f, 127.5f, 127.5f, 127.5f, 127.5f
                        );

                        {
                            ppocr::ScopedTimer timer(metrics.rec_inference_ms);
                            req.start_async();
                            req.wait();
                        }

                        const uint16_t* p_rec_out = reinterpret_cast<const uint16_t*>(req.get_output_ptr(0));
                        int seq_len = target_w / 8;
                        int vocab_size = static_cast<int>(target_graph->outputs()[0].total_bytes / (seq_len * 2));
                        int valid_seq_len = std::min(seq_len, static_cast<int>(std::ceil(static_cast<float>(valid_w) / 8.0f)));

                        {
                            ppocr::ScopedTimer timer(metrics.ctc_decode_ms);
                            box.text = ctc_decoder.decode(p_rec_out, valid_seq_len, vocab_size, box.text_confidence);
                        }
                    } else {
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
                        int vocab_size = static_cast<int>(target_graph->outputs()[0].total_bytes / (win_cells * 2));

                        // Vector to accumulate merged Logits across slices
                        std::vector<uint16_t> merged_logits;

                        for (int s = 0; s < num_slices; ++s) {
                            int sx = s * step_px;
                            int valid_slice_w = std::min(win_w, natural_w - sx);

                            ppocr::ImageBGR slice_img;
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

                            auto& req = target_graph->get_request(s % 2);
                            ppocr::WicDecoder::normalize_to_nchw_fp16(
                                slice_img, req.get_input_ptr(0),
                                127.5f, 127.5f, 127.5f, 127.5f, 127.5f, 127.5f
                            );

                            {
                                ppocr::ScopedTimer timer(metrics.rec_inference_ms);
                                req.start_async();
                                req.wait();
                            }

                            const uint16_t* p_slice_out = reinterpret_cast<const uint16_t*>(req.get_output_ptr(0));
                            int valid_slice_cells = std::min(win_cells, static_cast<int>(std::ceil(static_cast<float>(valid_slice_w) / cell_stride)));

                            // Merge slice into merged_logits using Midpoint Split rule
                            int cell_start = (s == 0) ? 0 : half_overlap_cells;
                            int cell_end = (s == num_slices - 1) ? valid_slice_cells : (win_cells - half_overlap_cells);

                            if (cell_end > cell_start) {
                                size_t offset_cells = cell_start;
                                size_t count_cells = cell_end - cell_start;
                                size_t copy_elems = count_cells * vocab_size;

                                size_t curr_sz = merged_logits.size();
                                merged_logits.resize(curr_sz + copy_elems);
                                std::memcpy(
                                    merged_logits.data() + curr_sz,
                                    p_slice_out + offset_cells * vocab_size,
                                    copy_elems * sizeof(uint16_t)
                                );
                            }
                        }

                        // Run single CTC decode on the merged Logits
                        int total_seq_len = static_cast<int>(merged_logits.size() / vocab_size);
                        {
                            ppocr::ScopedTimer timer(metrics.ctc_decode_ms);
                            box.text = ctc_decoder.decode(merged_logits.data(), total_seq_len, vocab_size, box.text_confidence);
                        }
                    }
                }
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

        auto end_total = std::chrono::high_resolution_clock::now();
        metrics.total_end_to_end_ms = std::chrono::duration<double, std::milli>(end_total - start_total).count();

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

    } catch (const std::exception& ex) {
        std::cerr << "[-] Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
