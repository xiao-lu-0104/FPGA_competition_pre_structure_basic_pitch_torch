# Spleeter 分离 —— PC 参考实现

FPGA 方案的"标准答案"所在地：先用这里的脚本跑一遍，硬件的 RTL 逐层对拍就以本目录产出为准。

## 一次性准备

```powershell
# 1) 下载权重（约 75 MB，不入库；走 hf-mirror，国内可达）
python download_weights.py

# 2) 顺便导出 ONNX（推荐，给 RTL 做黄金参考）
python download_weights.py --onnx
```

需要的 Python 包：`torch librosa soundfile numpy`（可选 `onnx onnxruntime` 用于导出与自检）。

## 文件说明

| 文件 | 作用 |
|---|---|
| `unet.py` | Spleeter U-Net 网络定义（逐层与 `docs/FPGA_分离实现方案.md` §2 对应） |
| `run_spleeter.py` | 分离推理：44.1k 立体声 → STFT → 双模型 → 掩码 → iSTFT → 两轨 wav |
| `download_weights.py` | 下载 `vocals.pt` / `accompaniment.pt`（可加 `--onnx` 导出 ONNX） |
| `export_spleeter_onnx.py` | 导出两个 stem 的 ONNX（各 37.5 MB，opset 13）并做数值自检 |
| `layer_stats.py` | 打印逐层输入/输出尺寸、参数量、MAC、实时算力（文档 §2.1/§3.1 数据来源） |
| `eval_patch.py` | 不同分块大小的质量对比（文档 §4 数据来源） |

## 常用命令

```powershell
# 分离（官方默认 512 帧）
python run_spleeter.py --audio "..\..\audio\吉他+人声.mp3" --out-dir out

# 分离（FPGA 推荐配置：128 帧 = 3.0 s 延迟）
python run_spleeter.py --audio "..\..\audio\吉他+人声.mp3" --out-dir out_p128 --patch 128

# 逐层参数/MAC 表
python layer_stats.py

# 分块大小对比（需要 release/output/ 里的 Demucs 参考结果）
python eval_patch.py --ref-dir ..\..\output
```

## 实测基准（本机 RTX 5060，参考 = Demucs）

| 分块 | 延迟 | 人声保留 | 吉他泄漏 | 人声泄漏 | 人声 SI-SDR | 伴奏 SI-SDR |
|---|---|---|---|---|---|---|
| 512 帧 | 11.9 s | 0.976 | 0.099 | 0.042 | 12.98 dB | 8.74 dB |
| 256 帧 | 5.9 s | 0.974 | 0.097 | 0.049 | 12.73 dB | 8.51 dB |
| **128 帧** | **3.0 s** | **0.974** | **0.088** | **0.064** | **12.60 dB** | **8.44 dB** |
| 64 帧 | 1.5 s | 0.965 | 0.074 | 0.103 | 11.35 dB | 7.34 dB |

> 对比原 DSP 谐波掩码方案：人声保留 0.884 / SI-SDR 5.53 dB、伴奏 SI-SDR 0.20 dB。
