# Basic Pitch (PyTorch) + 人声/吉他分离扒谱

本项目包含三部分：

1. **Basic Pitch 的 PyTorch 复现**——Spotify 的轻量级音频转 MIDI 模型，预训练权重来自官方
   （TensorFlow 版转换而来），可直接用于任意音频的复音转谱。
2. **人声+吉他同步扒谱管线**（PC 端）——先用 Demucs 把混音分离成人声轨和伴奏轨，再对每条轨
   分别跑 Basic Pitch 转写，一次得到**人声谱 + 吉他谱**两份 MIDI。
3. **纯 FPGA 实时部署方案**——面向安路 PH1A180 的"零参数 DSP 分离 + 轻量 CNN 转谱"方案，
   含完整实现文档与 ONNX 算子图（`docs/` 与 `models/`）。

## 文件结构

```
release/
├── README.md                        # 本文档
├── requirements.txt                 # 依赖列表
├── basic_pitch_torch/               # 模型与推理代码
│   ├── model.py                     # 神经网络定义（BasicPitchTorch）
│   ├── inference.py                 # 推理流程（分窗、拼接、predict 入口）
│   ├── note_creation.py             # 模型输出 → MIDI 后处理
│   └── constants.py                 # 常量（采样率、频率 bin 等）
├── models/
│   ├── basic_pitch_pytorch_icassp_2022.pth  # 预训练权重（ICASSP 2022）
│   ├── basic_pitch_heads.onnx       # 卷积主干 ONNX（含全部参数，70.8KB）★FPGA 用
│   └── basic_pitch_full.onnx        # 完整模型 ONNX（含 CQT，供对照）
├── docs/
│   ├── FPGA_分离实现方案.md          # DSP 谐波掩码分离：数据流/资源/延迟/参数表
│   └── FPGA_转谱实现方案.md          # Basic Pitch 转谱：CQT/卷积/量化/后处理方案
├── scripts/
│   ├── transcribe.py                # 单轨转写：音频 → MIDI
│   ├── separate_and_transcribe.py   # 两阶段：混音 → 人声 MIDI + 吉他 MIDI
│   ├── export_onnx.py               # 导出 ONNX 算子图 + 数值校验
│   └── show_structure.py            # 打印网络结构与参数量
├── audio/
│   └── 吉他+人声.mp3                 # 演示用真实录音（约 77 秒）
└── output/                          # 演示输出
    ├── vocals.wav                   # 分离出的人声轨
    ├── accompaniment.wav            # 分离出的伴奏轨（吉他）
    ├── 吉他+人声_vocals.mid          # 人声谱
    └── 吉他+人声_guitar.mid          # 吉他谱
```

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

（需要 Python 3.8+，建议使用带 CUDA 的 PyTorch 以获得 GPU 加速；CPU 也能跑。）

### 2. 单轨转写（输入音频 → 输出 MIDI）

```powershell
python scripts/transcribe.py --audio 你的音频.wav
# 输出：你的音频.mid
```

适用于任意单乐器/多乐器音频：钢琴、吉他、人声、管乐……输出包含所有音符、力度和
音高弯曲（滑音）的 MIDI 文件，可直接导入 MuseScore、Cubase、Logic 等查看乐谱。

### 3. 人声+吉他同步扒谱（输入混音 → 输出两份分轨谱）

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

## 输入 → 输出

| 模式 | 输入 | 输出 |
|------|------|------|
| 单轨转写 | 任意音频文件（wav/mp3/flac，自动重采样到 22.05kHz 单声道） | 1 个 MIDI（全部音符） |
| 分离转谱 | 人声+吉他（或带伴奏）混音 | 2 个分离音轨 wav + 2 个 MIDI |

## 模型结构

Basic Pitch 是一个"CQT + 谐波堆叠 + 三分支卷积"网络，总参数量 **16,782**：

```mermaid
flowchart TD
    A["音频窗口 22.05kHz x 2s = 43844 采样"] --> B["CQT 变换 fmin=27.5Hz, 每八度36 bins, hop=256"]
    B --> C["log 功率归一化 + BatchNorm"]
    C --> D["HarmonicStacking 谐波堆叠 8 层 (0.5,1,2,3,4,5,6,7 倍频)"]
    D --> F1["轮廓分支 conv_contour: Conv 8-8 (3x39) -> BN -> ReLU -> Conv 8-1 (5) -> Sigmoid"]
    D --> F2["起始点分支 conv_onset_pre: Conv 8-32 (5, stride 1x3) -> BN -> ReLU"]
    F1 --> G1["contour 输出: 时间x264, 音高轮廓, 1/3 半音分辨率"]
    F1 -->|padding| H["音符分支 conv_note: Conv 1-32 (7, stride 1x3) -> ReLU -> Conv 32-1 (7x3) -> Sigmoid"]
    H --> G3["note 输出: 时间x88, 88 个钢琴键的发音概率"]
    H --> F2
    F2 --> I["拼接 note+onset 特征: Conv 33-1 (3) -> Sigmoid"]
    I --> G2["onset 输出: 时间x88, 88 个钢琴键的起始点概率"]
```

三个输出分工：

- **note**（88 维）：每一帧"哪些音高正在发声"
- **onset**（88 维）：每一帧"哪些音高刚刚开始"
- **contour**（264 维）：每 1/3 半音级的精细音高，用于生成滑音/音高弯曲

前端细节：CQT 最低频率 27.5Hz（钢琴最低键），覆盖 88 个钢琴键；谐波堆叠把 0.5~7 倍频的
能量沿频率轴平移到基频带并堆叠，让低音也能利用其泛音能量，提升复音转写精度。

## 与原版模型的差异

本仓库是 Spotify Basic Pitch（TensorFlow 官方实现）的 PyTorch 复现，网络结构与官方**逐层等价**，
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
   代码用 `F.pad` 手写 padding 公式逐层对齐（`model.py` 注释里有推导）。
2. **输入 / 输出形状**：输入 `(batch, 43844)`（22.05kHz × 2 秒窗口），输出三张热力图
   `(batch, 172, 88)`（note/onset）和 `(batch, 172, 264)`（contour）。
3. **后处理兼容性**：新版 scipy 移除了 `scipy.signal.gaussian`，本仓库改用
   `scipy.signal.windows.gaussian`。
4. **精度差异**：与原版输出的差异主要来自浮点除法（`normalized_log`）及误差在网络中的传播，
   数值极小，不影响转谱结果（见下方"测试结果"）。

### 相对原版的新增内容

- **人声/吉他分离扒谱管线**（`scripts/separate_and_transcribe.py`）：原版只做转写，
  不含声源分离；本仓库在其前加了一层 Demucs 分离，支持"混音 → 分轨谱"。
- 分窗 / 重叠拼接的推理流程按原版 `predict` 逻辑复刻为 PyTorch 实现。

## 测试结果

### 验证 1：与原版 TensorFlow 模型输出对比

用官方测试音频（GuitarSet）对比 PyTorch 复现与 Spotify TensorFlow 原版：

| 输出 | 最大绝对误差 | 平均绝对误差 |
|------|-------------|-------------|
| contour | 0.000301 | 5.86e-06 |
| onset | 0.000271 | 1.43e-05 |
| note | 0.000230 | 6.60e-06 |

转写出的 MIDI 与原版**完全一致**（95 个音符逐项相同）。

### 验证 2：真实"吉他+人声"录音端到端测试

对 `audio/吉他+人声.mp3`（77 秒）运行分离转谱管线，分离耗时约 5 秒（GPU）：

| 输出 | 音符数 | 音高范围 | 说明 |
|------|--------|----------|------|
| 人声谱 `吉他+人声_vocals.mid` | 169 | 54~70 为主 | 集中在中音区，符合演唱音域 |
| 吉他谱 `吉他+人声_guitar.mid` | 429 | 42~54 密集 | 低频密集，符合分解和弦伴奏形态 |

分离出的 `vocals.wav` 与 `accompaniment.wav` 可直接试听核对分离质量。

## FPGA 实时部署方案

面向**纯 FPGA、流式实时、无 PC 参与**的场景（目标平台：安路 PH1A180-MLK-H10-CK203/204，
TangDynasty 工具链），本项目提供完整方案。
（PC 端管线的 Demucs 有 41M 参数，无法上 FPGA，因此分离改用**零参数 DSP 算法**。）

```
音频输入 → [DSP 谐波掩码分离] → [Basic Pitch 转谱] → [软核后处理] → MIDI
           零参数、纯 FFT          16,782 参数 CNN        RISC-V 软核
           延迟 < 100ms            0.5 秒块长
```

| 环节 | 方案 | 关键指标 |
|------|------|---------|
| 分离 | DSP 谐波掩码（FFT → 谐波筛法估 F0 → 掩码 → IFFT） | **零参数**；帧延迟 23ms；查找表 ROM 约 2.7KB |
| 转谱 | Basic Pitch（INT8 量化） | **16,782 参数**；484M MAC / 2 秒窗口 |
| 后处理 | 软核 CPU 跑 C 代码（移植自 `note_creation.py`） | 或简化为阈值 + 峰值状态机 |

详细实现文档：

- [`docs/FPGA_分离实现方案.md`](docs/FPGA_分离实现方案.md)——逐帧数据流、查找表设计、资源与延迟估算、参数速查表
- [`docs/FPGA_转谱实现方案.md`](docs/FPGA_转谱实现方案.md)——CQT 多速率实现、逐层卷积参数表、量化方案、后处理接口

ONNX 算子图（供硬件团队做 RTL 参考与量化仿真）：

- `models/basic_pitch_heads.onnx`——**卷积主干**（含全部参数，输入 `(1,8,T,264)`），与 PyTorch 数值误差 < 1e-6
- `models/basic_pitch_full.onnx`——完整模型（含 CQT，供对照）

重新导出与自检：

```powershell
python scripts/export_onnx.py --out-dir models
```

> 分离质量客观评分（以 Demucs 为参考的 SI-SDR）：人声轨 **3.57 dB**、伴奏轨 -0.71 dB。
> 零参数 DSP 分离存在物理上限（人声与吉他频率重叠处不可分），预期听感为"人声保留主旋律谐波、
> 伴奏偏挖空"，请以"可实时、零参数、可用"为目标评估。

## 原理简述

- **Basic Pitch**（Bittner et al., ICASSP 2022）是一个乐器无关的复音转录模型：把音频做成
  对数频率的时频图（CQT），用全卷积网络同时预测音符、起始点和音高轮廓三张热力图，
  再经后处理转成 MIDI。它只回答"什么时间有什么音高"，不区分乐器。
- **分离转谱管线**：Demucs（Hybrid Transformer Demucs）先把混音按声源分离
  （人声/鼓/贝斯/其他），再对每条声源轨单独转写。因为转写是逐轨进行的，天然得到
  每个乐器独立的谱子。

## 已知限制

- Basic Pitch 对超出钢琴 88 键范围的音高（如极低贝斯、打击乐）不输出音符。
- Demucs 分离不是 100% 干净，两条轨之间可能有少量串音（如人声谱里偶见极低音）。
- MIDI 输出默认速度 120 BPM，节拍需要在 DAW 中自行对齐。
- 首次运行需要联网下载 Demucs 权重（约 80MB）。

## 参考

- [Spotify Basic Pitch（原版）](https://github.com/spotify/basic-pitch)
- [Demucs](https://github.com/facebookresearch/demucs)
- Bittner, R. M., et al. "A lightweight instrument-agnostic model for polyphonic note
  transcription and multipitch estimation." ICASSP 2022.
