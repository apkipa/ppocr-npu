#pragma once

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace ppocr {

struct ProfileMetrics {
    bool enabled = false;
    double image_decode_ms = 0.0;
    double tile_prep_ms = 0.0;
    double det_inference_ms = 0.0;
    int det_tile_count = 1;
    double dbnet_postprocess_ms = 0.0;
    int detected_box_count = 0;
    double rec_crop_prep_ms = 0.0;
    double rec_inference_ms = 0.0;
    double ctc_decode_ms = 0.0;
    double total_end_to_end_ms = 0.0;

    void print_report() const {
        if (!enabled) return;
        std::cout << "\n============================================================\n";
        std::cout << "          End-to-End Inference Profiling Metrics           \n";
        std::cout << "============================================================\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  1. Image Acquisition / WIC Decode  : " << std::setw(7) << image_decode_ms << " ms\n";
        std::cout << "  2. Tiling & Preprocessing          : " << std::setw(7) << tile_prep_ms << " ms (" << det_tile_count << " tiles)\n";
        std::cout << "  3. NPU Detection (Det) Inference   : " << std::setw(7) << det_inference_ms << " ms (avg " << (det_tile_count ? det_inference_ms / det_tile_count : 0.0) << " ms/tile)\n";
        std::cout << "  4. DBNet Heatmap Stitch & Polygon  : " << std::setw(7) << dbnet_postprocess_ms << " ms (" << detected_box_count << " boxes)\n";
        std::cout << "  5. Text Crop & Perspective Warp    : " << std::setw(7) << rec_crop_prep_ms << " ms\n";
        std::cout << "  6. NPU Recognition (Rec) Inference : " << std::setw(7) << rec_inference_ms << " ms (" << (detected_box_count ? rec_inference_ms / detected_box_count : 0.0) << " ms/box)\n";
        std::cout << "  7. CTC Greedy Decoding             : " << std::setw(7) << ctc_decode_ms << " ms\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << "  Total End-to-End Pipeline Latency  : " << std::setw(7) << total_end_to_end_ms << " ms\n";
        if (total_end_to_end_ms > 0.0) {
            std::cout << "  Page Throughput                    : " << std::setw(7) << (1000.0 / total_end_to_end_ms) << " FPS\n";
        }
        if (rec_inference_ms > 0.0 && detected_box_count > 0) {
            std::cout << "  Recognition Text Line Throughput   : " << std::setw(7) << (detected_box_count / (rec_inference_ms / 1000.0)) << " lines/sec\n";
        }
        std::cout << "============================================================\n";
    }
};

class ScopedTimer {
public:
    explicit ScopedTimer(double& target_ms) : target_(target_ms), start_(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        target_ += std::chrono::duration<double, std::milli>(end - start_).count();
    }
private:
    double& target_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

} // namespace ppocr
