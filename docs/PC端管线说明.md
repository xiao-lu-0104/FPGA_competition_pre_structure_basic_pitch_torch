# PC 端管线说明（Demucs 分离 + Basic Pitch 转谱）

> 本文档记录**PC 端**实现与验证结果，供开发/报告使用。
> FPGA 部署请看仓库根目录 [README.md](../README.md) 与 `docs/` 下两份 FPGA 实现方案。

## 1. PyTorch 复现

本仓库包含 Spotify Basic Pitch（TensorFlow 官方实现）的 PyTorch 复现，网络结构与官方**逐层等价**，
官方预训练权重已预先转换为 PyTorch 格式（`models/basic_pitch_pytorch_icassp_2022.pth`），
使用者无需接触 TensorFlow。

### 框架与权重

| 项 | 原版（TensorFlow） | 本复现（PyTorch） |
|----|-------------------|------------------|
| 深度学习框架 | TensorFlow 2 / Keras | PyTorch |
| 权重格式 | ICASSP_2022 saved_model | `.pth`（从原版转换而来） |
| CQT 实现 | 自研 TensorFlow 层 | nnAudio `CQT2010v2` |
| 参数量 | 相同结构 | 16,782 |

### 逐层等价

- CQT、谐波堆叠、三分支卷积的**层数 / 通道数 / 卷积核大小 / 激活函数**与原版一致；
- BatchNorm 的 `eps=0.001` 与原版保持一致；
- `normalized_log` 中的除零处理用 `torch.nan_to_num` 实现，等价于 TensorFlow 的 `div_no_nan`。

### 关键适配点（TensorFlow → PyTorch）

1. **stride 卷积的 padding**：TensorFlow 的 `padding="same"` 在 stride ≠ 1 时与 PyTorch 语义不同，
   代码用 `F.pad` 手写 padding 公式逐层对齐（`model.py` 注释里有推导）；
2. **输入 / 输出形状**：输入 `(batch, 43844)`（22.05kHz × 2 秒窗口），输出三张热力图
   `(batch, 172, 88)`（note/onset）和 `(batch, 172, 264)`（contour）；
3. **后处理兼容性**：新版 scipy 移除了 `scipy.signal.gaussian`，本仓库改用
   `scipy.signal.windows.gaussian`；
4. **精度差异**：与原版输出的差异主要来自浮点除法（`normalized_log`）及误差在网络中的传播，
   数值极小，不影响转谱结果（见下方"测试结果"）。

## 2. PC 端快速开始

### 安装依赖

```powershell
pip install -r requirements.txt
```

（需要 Python 3.8+，建议使用带 CUDA 的 PyTorch 以获得 GPU 加速；CPU 也能跑。）

### 单轨转写（音频 → MIDI）

```powershell
python scripts/transcribe.py --audio 你的音频.wav
# 输出：你的音频.mid
```

适用于任意单乐器/多乐器音频：钢琴、吉他、人声、管乐……输出包含所有音符、力度和
音高弯曲（滑音）的 MIDI 文件，可直接导入 MuseScore、Cubase、Logic 等查看乐谱。

### 人声+吉他同步扒谱（混音 → 两份分轨谱）

```powershell
python scripts/separate_and_transcribe.py --audio 你的混音.mp3 --out-dir 输出目录
```

输出 4 个文件：

| 文件 | 内容 |
|------|------|
| `vocals.wav` | 分离出的人声轨 |
| `accompaniment.wav` | 分离出的伴奏轨（吉他等） |
| `<音频名>_vocals.mid` | 人声谱 |
| `<音频名>_guitar.mid` | 吉他/伴奏谱 |

> 提示：首次运行会自动下载 Demucs 权重（约 80MB，只需一次）。

| 模式 | 输入 | 输出 |
|------|------|------|
| 单轨转写 | 任意音频文件（wav/mp3/flac，自动重采样到 22.05kHz 单声道） | 1 个 MIDI（全部音符） |
| 分离转谱 | 人声+吉他（或带伴奏）混音 | 2 个分离音轨 wav + 2 个 MIDI |

## 3. 测试结果

### 验证 1：与原版 TensorFlow 模型输出对比

用官方测试音频（GuitarSet）对比 PyTorch 复现与 Spotify TensorFlow 原版：

| 输出 | 最大绝对误差 | 平均绝对误差 |
|------|-------------|-------------|
| contour | 0.000301 | 5.86e-06 |
| onset | 0.000271 | 1.43e-05 |
| note | 0.000230 | 6.60e-06 |

转写出的 MIDI 与原版**完全一致**（95 个音符逐项相同）。

### 验证 2：真实"吉他+人声"录音端到端测试

对 `audio/吉他+人声.mp3`（77 秒）运行 Demucs 分离转谱管线，分离耗时约 5 秒（GPU）：

| 输出 | 音符数 | 音高范围 | 说明 |
|------|--------|----------|------|
| 人声谱 `吉他+人声_vocals.mid` | 169 | 54~70 为主 | 集中在中音区，符合演唱音域 |
| 吉他谱 `吉他+人声_guitar.mid` | 429 | 42~54 密集 | 低频密集，符合分解和弦伴奏形态 |

分离出的 `vocals.wav` 与 `accompaniment.wav` 可直接试听核对分离质量。

## 4. 原理简述

- **Basic Pitch**（Bittner et al., ICASSP 2022）是一个乐器无关的复音转录模型：把音频做成
  对数频率的时频图（CQT），用全卷积网络同时预测音符、起始点和音高轮廓三张热力图，
  再经后处理转成 MIDI。它只回答"什么时间有什么音高"，不区分乐器。
- **分离转谱管线**：Demucs（Hybrid Transformer Demucs）先把混音按声源分离
  （人声/鼓/贝斯/其他），再对每条声源轨单独转写。因为转写是逐轨进行的，天然得到
  每个乐器独立的谱子。

## 5. PC 端已知限制

- Basic Pitch 对超出钢琴 88 键范围的音高（如极低贝斯、打击乐）不输出音符。
- Demucs 分离不是 100% 干净，两条轨之间可能有少量串音（如人声谱里偶见极低音）。
- MIDI 输出默认速度 120 BPM，节拍需要在 DAW 中自行对齐。
- 首次运行需要联网下载 Demucs 权重（约 80MB）。
- Demucs 有约 41M 参数，**无法部署到 FPGA**——FPGA 场景请使用 DSP 谐波掩码方案
  （见 `docs/FPGA_分离实现方案.md`）。

## 6. 参考

- [Spotify Basic Pitch（原版）](https://github.com/spotify/basic-pitch)
- [Demucs](https://github.com/facebookresearch/demucs)
- Bittner, R. M., et al. "A lightweight instrument-agnostic model for polyphonic note
  transcription and multipitch estimation." ICASSP 2022.
