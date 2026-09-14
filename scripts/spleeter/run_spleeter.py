"""Spleeter (PyTorch 版) 分离推理 —— 用于可行性评估。

预处理按 Spleeter 官方设置：44.1kHz 立体声、STFT(n_fft=4096, hop=1024)、取前 1024 个频点、
分块 512 帧、全局归一化。模型输出 = mask × 归一化谱，这里反解出 mask 再应用到原始谱。

用法：
    python spleeter_test/run_spleeter.py --audio 吉他+人声.mp3 --out-dir spleeter_test/out
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import librosa
import numpy as np
import soundfile as sf
import torch

from unet import UNet

SR = 44100
N_FFT = 4096
HOP = 1024
F_BINS = 1024          # Spleeter 只建模前 1024 个频点（约 11kHz 以下）
PATCH = 512            # 每次推理 512 帧
OVERLAP = 0.25         # 分块重叠


def load_model(path: Path) -> UNet:
    m = UNet()
    m.load_state_dict(torch.load(str(path), map_location="cpu"))
    m.eval()
    return m


def run_model(model: UNet, x: np.ndarray) -> np.ndarray:
    """x: (T, F, 2) 归一化幅度谱 -> 输出 (T, F, 2)。"""
    T = x.shape[0]
    out = np.zeros_like(x)
    step = int(PATCH * (1 - OVERLAP))
    for t0 in range(0, T, step):
        seg = x[t0 : t0 + PATCH]
        L = seg.shape[0]
        if L == 0:
            break
        if L < PATCH:
            seg_p = np.concatenate([seg, np.zeros((PATCH - L, F_BINS, 2), np.float32)], axis=0)
        else:
            seg_p = seg
        xin = torch.from_numpy(seg_p.transpose(2, 0, 1)[None]).float()  # (1, 2, 512, 1024)
        with torch.no_grad():
            y = model(xin)[0].numpy().transpose(1, 2, 0)  # (512, F, 2)
        out[t0 : t0 + L] = y[:L]
        if t0 + PATCH >= T:
            break
    return out


def separate(audio_path: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print("加载模型...")
    model_v = load_model(Path(__file__).parent / "vocals.pt")
    model_a = load_model(Path(__file__).parent / "accompaniment.pt")

    y, _ = librosa.load(audio_path, sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    print(f"音频 {y.shape[1]/SR:.1f} 秒（{y.shape[0]} 声道）")

    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)          # (ch, 2049, T)
    D = D.transpose(1, 2, 0)                                  # (2049, T, ch)
    mag = np.abs(D)
    x = mag[:F_BINS].transpose(1, 0, 2).astype(np.float32)    # (T, F, ch)

    # 全局归一化（Spleeter 默认）
    mu, sd = float(x.mean()), float(x.std())
    xn = (x - mu) / (sd + 1e-8)

    print("推理中（512 帧/块，25% 重叠）...")
    out_v = run_model(model_v, xn)
    out_a = run_model(model_a, xn)

    # 反解掩码：model 输出 = mask × 归一化谱
    mask_v = np.clip(out_v / (xn + np.sign(xn) * 1e-6 + 1e-9), 0.0, 1.0)
    mask_a = np.clip(out_a / (xn + np.sign(xn) * 1e-6 + 1e-9), 0.0, 1.0)

    # 人声：掩码作用于原始谱（仅建模频段）；伴奏：其余部分（保证完整）
    full_mask_v = np.zeros_like(mag)                          # (2049, T, ch)
    full_mask_v[:F_BINS] = mask_v.transpose(1, 0, 2)
    D_v = D * full_mask_v
    D_a = D - D_v

    vocals = librosa.istft(D_v.transpose(2, 0, 1), hop_length=HOP, length=y.shape[1])
    accomp = librosa.istft(D_a.transpose(2, 0, 1), hop_length=HOP, length=y.shape[1])

    sf.write(str(out_dir / "vocals_spleeter.wav"), vocals.T, SR)
    sf.write(str(out_dir / "accompaniment_spleeter.wav"), accomp.T, SR)
    print(f"人声轨 -> {out_dir/'vocals_spleeter.wav'}")
    print(f"伴奏轨 -> {out_dir/'accompaniment_spleeter.wav'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "out"))
    ap.add_argument("--patch", type=int, default=PATCH, help="分块长度（帧），512=官方默认，越小延迟越低")
    ap.add_argument("--overlap", type=float, default=OVERLAP)
    args = ap.parse_args()
    PATCH = args.patch
    OVERLAP = args.overlap
    separate(args.audio, Path(args.out_dir))
