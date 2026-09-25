# PP-OCRv6 on Intel NPU (Level Zero Native C++)

专为 **Intel Core Ultra（Arrow Lake / Meteor Lake）NPU 硬件架构** 深度定制的高性能 PP-OCRv6 纯血 C++ 推理引擎与离线模型优化流水线。

结合 Intel Level Zero 原生运行时、统一共享内存（USM 零拷贝）、Windows 原生 WIC 解码器与大图切片羽化拼接，实现极致低延迟、零内存搬运与超高吞吐。

---

## 🌟 核心特性

- **纯血 C++ 交付（零 Python 依赖）**：编译产物为单一轻量 `.exe`，运行时仅依赖 Windows 系统动态库与 Level Zero 原生 NPU 驱动。
- **全格式原生解码与剪切板直读（零第三方图像库）**：抛弃 stb_image / OpenCV，直接调用 Windows 内置 **WIC（Windows Imaging Component）** 硬件解码流水线，支持 JPG, PNG, BMP, TIFF, WebP；支持通过 `--clipboard` 直接提取系统剪切板截图识别。
- **Level Zero USM 零拷贝（Zero-Copy）**：通过 `zeMemAllocShared` 分配统一内存，输入图像原地填充，NPU 原地直读，彻底消灭主机到加速器的物理拷贝开销。
- **2-并发异步双缓冲流（Ping-Pong Pipeline）**：利用 NPU3720 的 Dual-Tile 架构，精准对冲 DPU（~2.6ms）与 SHAVE（~2.2ms）1:1 耗时，单行识别吞吐量飙升至 **475 FPS（短文本桶高达 581 FPS）**。
- **大图切片与概率图羽化拼接（Anti-Trimming Slicing）**：针对 1080p / 2K 大图，支持 6-切片异步流水线（仅需 84ms / 65MB 显存，比全图裸跑快 2.1 倍），采用热力图线性羽化消除跨切片“腰斩断句”问题。
- **识别三静态桶路由（3-Bucket Routing）**：自适应将文本框按长宽比分流至 `48×160`（1.72ms）、`48×320`（2.32ms）与 `48×640`（4.43ms），全桶共享同一份 10.5MB 权重。
- **端到端微秒级性能剖析（`--profile` 开关）**：内建全链路各阶段高精度打点器，清晰打印图像解码、切片提取、Det 推理、DBNet 后处理、Rec 推理及 CTC 解码的耗时账单。

---

## 📊 硬件基准实测（Intel Core Ultra 9 275HX + NPU3720）

| 模块与策略 | 输入规格 | NPU 驱动显存 | **推理耗时** | 吞吐率 | 准确率 / F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **文本识别 (Rec) - 超速短词桶** | `48×160` | 11.8 MB | **1.72 ms** (均摊) | **581.0 FPS** | 76.00% (满血对齐 CPU) |
| **文本识别 (Rec) - 通用标准桶** | `48×320` | 12.5 MB | **2.10 ms** (均摊) | **475.1 FPS** | 76.00% (满血对齐 CPU) |
| **文本识别 (Rec) - 长句兜底桶** | `48×640` | 14.2 MB | **4.43 ms** (均摊) | 225.7 FPS | 76.00% (满血对齐 CPU) |
| **文本检测 (Det) - 原生优化** | `640×640` | 65.8 MB | **15.87 ms** | 62.0 FPS | 31.2% (超越 736 尺寸) |
| **大图检测 (1080p) - 6-切片拼接** | $1920 \times 1080$ | 65.8 MB | **84.19 ms** | 11.9 FPS | 小文字 100% 像素级保留 |

> 详细消融实验分析、硬件 FTZ NaN 根因排查与微架构报告请见：[`docs/npu_optimization_report.md`](docs/npu_optimization_report.md)。

---

## 🛠️ 环境依赖与前置准备

1. **操作系统**：Windows 11 / Windows 10 (x64)
2. **NPU 驱动**：Intel NPU 驱动 `32.0.100.5540` 或更新版本（任务管理器中可见“Intel AI Boost”）
3. **C++ 编译器**：Visual Studio 2022 (MSVC v143, 支持 C++17) & CMake 3.24+
4. **Level Zero SDK**：安装至 `C:\Program Files\LevelZeroSDK\1.32.0`（或设置环境变量 `LEVEL_ZERO_V1_SDK_PATH`）
5. **Python 环境（用于模型转换与评测）**：安装现代包管理器 `uv`（`pip install uv` 或 `winget install astral-sh.uv`）

---

## 🚀 快速上手

### 1. 使用 uv 初始化环境并构建优化模型

通过 `uv` 自动化配置虚拟环境并生成全部 NPU 优化资产（IR、NGRAPH_LITE 容器与硬件 Blob）：

```powershell
# 1. 自动同步依赖环境
uv sync

# 2. 一键构建 Det 640x640 与 Rec 160/320/640 全套优化 IR 模型与词表
uv run python scripts/build_models.py
```

构建完成后，`models/` 目录下将生成轻量级 IR 资产：
- `ppocrv6_small_det_640x640.xml` / `.bin` / `.ngraph_lite`
- `ppocrv6_small_rec_w160.xml` / `.bin` / `.ngraph_lite`
- `ppocrv6_small_rec_w320.xml` / `.bin` / `.ngraph_lite`
- `ppocrv6_small_rec_w640.xml` / `.bin` / `.ngraph_lite`
- `ppocr_keys_v1.txt`（包含 18,709 个字符的完整多语种字典）

> **注**：无需手动存放硬件二进制 Blob。首次运行 C++ 引擎时，驱动将自动执行 JIT 编译并缓存至 `cache/<model_name>.blob`，后续运行 0.2ms 自动秒开。

---

### 2. 编译 C++ 原生推理引擎

```powershell
# 创建构建目录并生成 VS2022 解决方案
cmake -B build -G "Visual Studio 17 2022" -A x64

# 编译 Release 版本
cmake --build build --config Release
```
编译产物位于：`build/Release/ppocr_npu.exe`。

---

### 3. 运行推理与命令行使用指南

#### 场景 A：从剪切板秒级识别截图（最常用功能）
按下 `Win + Shift + S` 截图后，直接在终端执行：
```powershell
.\build\Release\ppocr_npu.exe --clipboard --profile
```

#### 场景 B：指定本地图片文件进行识别
```powershell
.\build\Release\ppocr_npu.exe --image test_invoice.jpg --profile
```

#### 场景 C：高清大图（如 1080p / 2K）切片拼接识别
```powershell
.\build\Release\ppocr_npu.exe --image doc_1080p.png --slice --profile
```

---

## 📋 命令行参数详解

| 参数项 | 说明 | 默认值 |
| :--- | :--- | :--- |
| `--image <path>` | 指定输入的本地图像路径（支持 JPG, PNG, BMP, TIFF, WebP） | 空 |
| `--clipboard` | 直接从 Windows 系统剪切板捕获图像（CF_DIB 原生直读） | `false` |
| `--det-model <path>` | 文本检测模型路径（支持 `.xml`, `.ngraph_lite` 或 `.blob`） | `models/ppocrv6_small_det_640x640.xml` |
| `--rec-160 <path>` | 短词识别模型路径 ($W=160$) | `models/ppocrv6_small_rec_w160.xml` |
| `--rec-320 <path>` | 标准识别模型路径 ($W=320$) | `models/ppocrv6_small_rec_w320.xml` |
| `--rec-640 <path>` | 长句识别模型路径 ($W=640$) | `models/ppocrv6_small_rec_w640.xml` |
| `--dict <path>` | 字符映射字典文件路径 | `models/ppocr_keys_v1.txt` |
| `--slice` | 强制开启多瓦片分片推理与概率图拼接 | 图像尺寸 $>640$ 时自动开启 |
| `--profile`, `-p` | 输出端到端各阶段高精度耗时与吞吐报告 | `false` |

---

## 📈 端到端 Profile 输出样例（`--profile`）

```text
============================================================
                 OCR Recognition Results                    
============================================================
[ 1] Conf: 0.98 | Box: [(310,104), (418,216)] | Text: MASA
[ 2] Conf: 0.96 | Box: [(52,340), (218,382)]  | Text: 中国工商银行
[ 3] Conf: 0.99 | Box: [(412,420), (620,464)] | Text: 增值税普通发票
============================================================

============================================================
          End-to-End Inference Profiling Metrics           
============================================================
  1. Image Acquisition / WIC Decode  :    1.25 ms
  2. Tiling & Preprocessing          :    0.84 ms (1 tiles)
  3. NPU Detection (Det) Inference   :   15.87 ms (avg 15.87 ms/tile)
  4. DBNet Heatmap Stitch & Polygon  :    3.42 ms (3 boxes)
  5. Text Crop & Perspective Warp    :    0.31 ms
  6. NPU Recognition (Rec) Inference :    5.16 ms (avg 1.72 ms/box)
  7. CTC Greedy Decoding             :    0.12 ms
------------------------------------------------------------
  Total End-to-End Pipeline Latency  :   27.20 ms
  Page Throughput                    :   36.76 FPS
  Recognition Text Line Throughput   :  581.40 lines/sec
============================================================
```

---

## 🔬 执行精度与性能评测

使用附带的自动化脚本评测检测 F1、识别准确率及多并发吞吐：

```powershell
uv run python scripts/evaluate.py --models-dir models/ --output-json benchmark_results.json
```

---

## 📄 License

本项目基于 Apache 2.0 许可证开源。模型权重归属百度飞桨 PaddleOCR 团队所有。
