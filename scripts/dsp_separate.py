"""DSP 流式分离：谐波掩码法（FPGA 可实现版）。

全部算法都只用 FFT + 乘加，零参数、逐帧处理，对应 FPGA 数据流：
    音频帧 -> FFT -> 谐波筛法估计 F0 -> 生成谐波掩码 -> 掩码相乘 -> IFFT

改进点（均为 FPGA 友好的低成本逻辑）：
    1. F0 限制在人声音域（80-500Hz），谐波只取到 4kHz；
    2. 掩码带软权重 + 时间中值平滑。

用法：
    python finetune/dsp_separate.py --audio 混音.wav --out-dir 输出目录
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import soundfile as sf

SR = 22050
N_FFT = 2048
HOP = 512
F0_MIN = 80.0    # 人声基频下限 (Hz)
F0_MAX = 500.0   # 人声基频上限 (Hz)
F0_CANDIDATES = 48
N_HARMONICS = 14
MAX_HARMONIC_FREQ = 4500.0  # 谐波上限，避免拾取高频吉他泛音
VOICING_THRESHOLD = 0.3     # 经参数扫描调优


def estimate_f0_sieve(mag: np.ndarray, freqs: np.ndarray):
    """谐波筛法估计 F0（FPGA 可实现：对候选频率做谐波位置的幅度累加）。

    Returns:
        f0: (n_frames,) 每帧估计的基频（0 表示无音高）
        conf: (n_frames,) 置信度 0~1
    """
    n_frames = mag.shape[1]
    f0 = np.zeros(n_frames)
    conf = np.zeros(n_frames)

    # 对数间隔的候选 F0 网格（FPGA 上用查找表实现同样网格）
    candidates = F0_MIN * (F0_MAX / F0_MIN) ** (np.arange(F0_CANDIDATES) / (F0_CANDIDATES - 1))
    cand_bins = []
    for fc in candidates:
        bins = []
        for k in range(1, N_HARMONICS + 1):
            fh = fc * k
            if fh > MAX_HARMONIC_FREQ:
                break
            bins.append(int(round(fh / (SR / N_FFT))))
        cand_bins.append(np.array(bins))

    frame_energy = mag.sum(axis=0) + 1e-9
    for t in range(n_frames):
        col = mag[:, t]
        best_score, best_idx = -1.0, -1
        for ci, bins in enumerate(cand_bins):
            bins = bins[bins < len(col)]
            if len(bins) == 0:
                continue
            score = col[bins].sum() / len(bins)  # 平均谐波幅度
            if score > best_score:
                best_score, best_idx = score, ci
        if best_idx >= 0:
            conf[t] = min(1.0, (best_score / (frame_energy[t] / len(col))) / 20.0)
        f0[t] = candidates[best_idx] if best_idx >= 0 else 0.0
    return f0, conf


def smooth_f0(f0: np.ndarray, voiced: np.ndarray, win: int = 5) -> np.ndarray:
    """时间中值平滑（FPGA 上用移位寄存器实现）。"""
    f0_s = f0.copy()
    n = len(f0)
    half = win // 2
    for t in range(n):
        if not voiced[t]:
            continue
        seg = f0[max(0, t - half) : min(n, t + half + 1)]
        seg = seg[seg > 0]
        if len(seg):
            f0_s[t] = np.median(seg)
    return f0_s


def harmonic_mask_separate(y: np.ndarray, f0: np.ndarray, voiced: np.ndarray,
                           n_harmonics: int = N_HARMONICS, soft: bool = False):
    """生成谐波掩码并分离，返回 (人声轨, 伴奏轨, 掩码)。"""
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mag = np.abs(D)
    mask = np.zeros_like(mag)

    for t in range(mag.shape[1]):
        if not voiced[t] or f0[t] <= 0:
            continue
        for h in range(1, n_harmonics + 1):
            fh = f0[t] * h
            if fh > MAX_HARMONIC_FREQ:
                break
            # 谐波带宽：低次谐波稍宽，高次谐波更窄
            width = 0.5 if h <= 4 else 0.35
            lo = fh * 2 ** (-width / 12)
            hi = fh * 2 ** (width / 12)
            band = (freqs >= lo) & (freqs < hi)
            # 软权重：谐波次数越高权重越低（人声能量集中在低次谐波）
            w = 1.0 if not soft else max(0.35, 1.0 - 0.05 * (h - 1))
            mask[band, t] = np.maximum(mask[band, t], w)

    vocal = librosa.istft(D * mask, hop_length=HOP, length=len(y))
    accomp = librosa.istft(D * (1.0 - mask), hop_length=HOP, length=len(y))
    return vocal, accomp, mask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out-dir", default="out_dsp")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(args.audio, sr=SR, mono=True)
    print(f"音频 {len(y)/sr:.1f} 秒")

    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    f0, conf = estimate_f0_sieve(np.abs(D), freqs)
    voiced = conf > VOICING_THRESHOLD
    f0_s = smooth_f0(f0, voiced)
    print(f"F0 估计完成：有声帧 {np.sum(voiced)}/{len(voiced)}，平均 F0={np.mean(f0_s[voiced]):.1f} Hz")

    vocal, accomp, mask = harmonic_mask_separate(y, f0_s, voiced)

    sf.write(str(out_dir / "vocals_dsp.wav"), vocal, SR)
    sf.write(str(out_dir / "accompaniment_dsp.wav"), accomp, SR)
    print(f"人声轨 -> {out_dir / 'vocals_dsp.wav'}")
    print(f"伴奏轨 -> {out_dir / 'accompaniment_dsp.wav'}")


if __name__ == "__main__":
    main()
