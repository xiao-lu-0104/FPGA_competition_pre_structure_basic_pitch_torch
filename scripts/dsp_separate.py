"""DSP 流式分离：平滑梳状掩码 + 不对称抑制（FPGA 可实现版）。

全部算法只用 FFT + 查表 + 乘加，零参数、逐帧处理。FPGA 数据流：
    音频帧 -> FFT -> 谐波筛法估计 F0 -> 生成两个梳状掩码 -> 掩码相乘 -> IFFT

相对初版的三项改进（均保持零参数、逐帧、低延迟）：
    1. 硬掩码(0/1) -> **平滑梳状掩码**（按半音距离高斯衰减），人声更饱满；
    2. 单掩码 -> **不对称掩码**：人声轨用窄梳状（少混吉他），伴奏轨用宽抑制（更狠挖人声），
       代价是两轨之和不等于原混音，换取两轨互相泄漏显著下降；
    3. 输出**响度归一化**，解决人声轨听起来偏小的问题。

用法：
    python scripts/dsp_separate.py --audio 混音.wav --out-dir 输出目录
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
N_HARMONICS = 20
MAX_HARMONIC_FREQ = 4500.0   # 人声轨保留的谐波上限
SUPPRESS_FREQ = 6000.0       # 伴奏轨抑制人声的频段上限
SIGMA_VOCAL = 0.40           # 人声轨梳状宽度（半音，越小越干净但越薄）
SIGMA_SUPPRESS = 1.00        # 伴奏轨抑制宽度（半音，1.0=温和保吉他，1.5=强抑制更干净）
VOICING_THRESHOLD = 0.15     # 经参数扫描调优
VOCAL_GAIN_DB = -1.5         # 人声轨响度（相对原始混音 RMS，dB）


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


def comb_mask(freqs: np.ndarray, f0: np.ndarray, voiced: np.ndarray,
              sigma: float = SIGMA_VOCAL, max_hz: float = MAX_HARMONIC_FREQ) -> np.ndarray:
    """平滑梳状掩码：谐波中心权重 1，按半音距离做高斯衰减。

    FPGA 实现：把高斯曲线量化为 8 级查找表（按 bin 偏移查表加权），仍是查表 + 乘法。
    """
    n_bins, n_frames = len(freqs), len(f0)
    mask = np.zeros((n_bins, n_frames), dtype=np.float32)
    for t in range(n_frames):
        if not voiced[t] or f0[t] <= 0:
            continue
        for h in range(1, N_HARMONICS + 1):
            fh = f0[t] * h
            if fh > max_hz:
                break
            lo = fh * 2 ** (-3.0 * sigma / 12)
            hi = fh * 2 ** (3.0 * sigma / 12)
            idx = (freqs >= lo) & (freqs <= hi)
            if not np.any(idx):
                continue
            d = 12.0 * np.log2(freqs[idx] / fh)
            w = np.exp(-0.5 * (d / sigma) ** 2)
            mask[idx, t] = np.maximum(mask[idx, t], w)
    return mask


def separate(y: np.ndarray):
    """返回 (人声轨, 伴奏轨)。"""
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    f0_raw, conf = estimate_f0_sieve(np.abs(D), freqs)
    voiced = conf > VOICING_THRESHOLD
    f0_s = smooth_f0(f0_raw, voiced)

    mask_vocal = comb_mask(freqs, f0_s, voiced, sigma=SIGMA_VOCAL, max_hz=MAX_HARMONIC_FREQ)
    mask_suppress = comb_mask(freqs, f0_s, voiced, sigma=SIGMA_SUPPRESS, max_hz=SUPPRESS_FREQ)

    vocal = librosa.istft(D * mask_vocal, hop_length=HOP, length=len(y))
    accomp = librosa.istft(D * (1.0 - mask_suppress), hop_length=HOP, length=len(y))

    # 响度归一化：人声轨对齐到目标电平，解决"听起来很小"的问题
    target_rms = np.sqrt(np.mean(y**2)) * (10 ** (VOCAL_GAIN_DB / 20))
    cur = np.sqrt(np.mean(vocal**2)) + 1e-9
    vocal = vocal * (target_rms / cur)
    peak = np.max(np.abs(vocal))
    if peak > 0.99:  # 防削顶
        vocal = vocal * (0.99 / peak)
    return vocal, accomp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out-dir", default="out_dsp")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(args.audio, sr=SR, mono=True)
    print(f"音频 {len(y)/sr:.1f} 秒")

    vocal, accomp = separate(y)

    sf.write(str(out_dir / "vocals_dsp.wav"), vocal, SR)
    sf.write(str(out_dir / "accompaniment_dsp.wav"), accomp, SR)
    print(f"人声轨 -> {out_dir / 'vocals_dsp.wav'}")
    print(f"伴奏轨 -> {out_dir / 'accompaniment_dsp.wav'}")


if __name__ == "__main__":
    main()
