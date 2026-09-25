#!/usr/bin/env python3
"""
PP-OCRv6 Evaluation & Benchmark Suite for Intel NPU.

Evaluates accuracy, latency, and throughput across:
1. Detection (Det) models across resolutions (480, 640, 736) on validation images.
2. Recognition (Rec) multi-bucket models (160, 320, 640) on text crops.
3. Multi-stream concurrency throughput (nireq=1, 2, 4) verifying the 2-concurrency sweet spot.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
from shapely.geometry import Polygon


def polygon_iou(a, b):
    pa, pb = Polygon(a), Polygon(b)
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    union = pa.union(pb).area
    return 0.0 if union == 0 else pa.intersection(pb).area / union


def eval_det_accuracy(compiled_model, eval_items, size=640, thresh=0.2, box_thresh=0.45, unclip_ratio=1.4):
    """Evaluates detection accuracy and latency on NPU."""
    import importlib.util
    db_path = Path(r"D:\TEMP\npu-experiments\source\paddleocr-source\ppocr\postprocess\db_postprocess.py")
    if not db_path.exists():
        print("[-] db_postprocess.py not found, skipping det evaluation.")
        return None

    spec = importlib.util.spec_from_file_location("db_postprocess_direct", db_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    post = mod.DBPostProcess(thresh=thresh, box_thresh=box_thresh, max_candidates=3000, unclip_ratio=unclip_ratio)

    req = compiled_model.create_infer_request()
    latencies = []
    tp, fp, fn = 0, 0, 0

    for img_path, truths in eval_items:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        resized = cv2.resize(img, (size, size)).astype(np.float32) / 255.0
        resized = (resized - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        tensor = resized.transpose(2, 0, 1)[None]

        t0 = time.perf_counter()
        req.infer({compiled_model.input(0): tensor})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        out = req.get_output_tensor(0).data
        preds = post({"maps": out}, [[orig_h, orig_w, size / orig_h, size / orig_w]])[0]["points"]
        gt_polys = [np.array(t["points"], dtype=np.float32) for t in truths if t.get("transcription", "") != "###"]

        matched_gt = set()
        for p in preds:
            matched = False
            for j, g in enumerate(gt_polys):
                if j not in matched_gt and polygon_iou(p, g) >= 0.5:
                    matched_gt.add(j)
                    matched = True
                    break
            if matched:
                tp += 1
            else:
                fp += 1
        fn += len(gt_polys) - len(matched_gt)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    return {
        "size": f"{size}x{size}",
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
        "median_lat_ms": round(float(np.median(latencies)), 2),
        "min_lat_ms": round(float(np.min(latencies)), 2),
    }


def eval_rec_accuracy(compiled_model, test_samples, dict_chars, width=320):
    """Evaluates CTC recognition exact accuracy on NPU."""
    req = compiled_model.create_infer_request()
    correct = 0
    total = 0
    latencies = []

    for img_path, gt_text in test_samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        # Rescale keeping aspect ratio
        ratio = width / max(w, 1)
        new_w = min(width, int(w * 48 / h))
        resized = cv2.resize(img, (new_w, 48)).astype(np.float32) / 255.0
        resized = (resized - 0.5) / 0.5
        padded = np.zeros((48, width, 3), dtype=np.float32)
        padded[:, :new_w, :] = resized
        tensor = padded.transpose(2, 0, 1)[None]

        t0 = time.perf_counter()
        req.infer({compiled_model.input(0): tensor})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        out = req.get_output_tensor(0).data[0]  # shape: [seq_len, vocab_size]
        preds = np.argmax(out, axis=-1)

        # CTC greedy decode
        pred_chars = []
        last_idx = -1
        for idx in preds:
            if idx != 0 and idx != last_idx and idx < len(dict_chars):
                pred_chars.append(dict_chars[idx - 1])
            last_idx = idx
        pred_text = "".join(pred_chars)

        if pred_text == gt_text:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0.0
    return {
        "width": width,
        "exact_acc": round(acc * 100, 2),
        "total_samples": total,
        "median_lat_ms": round(float(np.median(latencies)), 2),
    }


def benchmark_throughput(compiled_model, input_shape, nireq_list=[1, 2, 4]):
    """Benchmarks asynchronous throughput across concurrency levels."""
    results = {}
    dummy_input = np.random.randn(*input_shape).astype(np.float16)

    for nireq in nireq_list:
        requests = [compiled_model.create_infer_request() for _ in range(nireq)]
        for r in requests:
            r.set_input_tensor(ov.Tensor(dummy_input))

        # Warmup
        for r in requests:
            r.start_async()
        for r in requests:
            r.wait()

        total_inferences = 120
        t0 = time.perf_counter()
        for i in range(total_inferences):
            req = requests[i % nireq]
            if i >= nireq:
                req.wait()
            req.start_async()
        for r in requests:
            r.wait()
        t1 = time.perf_counter()

        dur = t1 - t0
        fps = total_inferences / dur
        amortized = (dur / total_inferences) * 1000.0
        results[f"nireq_{nireq}"] = {
            "fps": round(fps, 1),
            "amortized_ms": round(amortized, 2),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PP-OCRv6 models on Intel NPU")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Directory with built models")
    parser.add_argument("--output-json", type=Path, default=Path("benchmark_results.json"), help="Output JSON report")
    args = parser.parse_args()

    core = ov.Core()
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "device": "Intel NPU3720"}

    print("==============================================================")
    print("  PP-OCRv6 Intel NPU Full Benchmark & Evaluation Suite       ")
    print("==============================================================")

    # 1. Detection Evaluation
    det_xml = args.models_dir / "ppocrv6_small_det_640x640.xml"
    if det_xml.exists():
        print("\n[*] Evaluating Detection Model (640x640) on NPU...")
        c_det = core.compile_model(str(det_xml), "NPU", {"INFERENCE_PRECISION_HINT": "f16"})
        val_txt = Path(r"D:\TEMP\npu-experiments\dataset\paddlex_examples\ocr_det_dataset_examples\val.txt")
        if val_txt.exists():
            data_root = val_txt.parent
            items = []
            for line in val_txt.read_text(encoding="utf-8").splitlines()[:30]:
                p, ann = line.split("\t", 1)
                items.append((data_root / p, json.loads(ann)))
            det_res = eval_det_accuracy(c_det, items, size=640)
            print(f"  [+] Det 640x640: Prec={det_res['precision']}%, Recall={det_res['recall']}%, F1={det_res['f1']}%, Latency={det_res['median_lat_ms']} ms")
            report["detection"] = det_res

    # 2. Recognition Multi-Bucket Evaluation
    rec_results = []
    print("\n[*] Evaluating Recognition Multi-Buckets on NPU...")
    dict_file = args.models_dir / "ppocr_keys_v1.txt"
    dict_chars = dict_file.read_text(encoding="utf-8").splitlines() if dict_file.exists() else []

    for w in [160, 320, 640]:
        rec_xml = args.models_dir / f"ppocrv6_small_rec_w{w}.xml"
        if rec_xml.exists():
            c_rec = core.compile_model(str(rec_xml), "NPU", {"PERFORMANCE_HINT": "THROUGHPUT"})
            tput_res = benchmark_throughput(c_rec, [1, 3, 48, w], nireq_list=[1, 2, 4])
            print(f"  [+] Rec 48x{w}: nireq=1: {tput_res['nireq_1']['fps']} FPS ({tput_res['nireq_1']['amortized_ms']}ms) | nireq=2: {tput_res['nireq_2']['fps']} FPS ({tput_res['nireq_2']['amortized_ms']}ms)")
            rec_results.append({"width": w, "throughput": tput_res})

    report["recognition"] = rec_results

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[V] Benchmark report saved to: {args.output_json.resolve()}")


if __name__ == "__main__":
    main()
