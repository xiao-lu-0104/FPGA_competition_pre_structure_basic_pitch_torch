"""DSP 分离参数扫描 v3：不对称掩码（人声轨窄梳状 / 伴奏轨宽抑制）。

人声轨与伴奏轨使用不同的掩码（不再是互补 1-mask），用"过度抑制"换取更低的互相泄漏：
    vocals = ISTFT(X * comb(σ_v))            # 窄：少混入吉他
    accomp = ISTFT(X * (1 - comb(σ_a)))      # 宽：更狠地挖掉人声

用法：
    python scripts/tune_dsp2.py --audio 混音.mp3 --ref-dir out_吉他人声
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import soundfile as sf

from dsp_separate import HOP, N_FFT, SR, estimate_f0_sieve, smooth_f0


def comb_mask(freqs, f0, voiced, sigma=0.7, floor=0.15, max_hz=6000.0,
              n_harmonics=20, band_lo=60.0):
    """平滑梳状掩码：谐波中心权重 1，按半音距离高斯衰减，整体有底噪 floor。"""
    n_bins, n_frames = len(freqs), len(f0)
    mask = np.zeros((n_bins, n_frames), dtype=np.float32)
    for t in range(n_frames):
        if voiced[t] and f0[t] > 0:
            for h in range(1, n_harmonics + 1):
                fh = f0[t] * h
                if fh > max_hz:
                    break
                lo = fh * 2 ** (-3.0 * sigma / 12)
                hi = fh * 2 ** (3.0 * sigma / 12)
                idx = (freqs >= lo) & (freqs <= hi)
                d = 12.0 * np.log2(freqs[idx] / fh)
                w = np.exp(-0.5 * (d / sigma) ** 2)
                mask[idx, t] = np.maximum(mask[idx, t], w)
            # 人声频带底噪（保留共振峰裙边、辅音，代价是少量吉他）
            band = (freqs >= band_lo) & (freqs <= max_hz)
            mask[band, t] = np.maximum(mask[band, t], floor)
    return mask


def separate_with(mask, y, D):
    vocal = librosa.istft(D * mask, hop_length=HOP, length=len(y))
    accomp = librosa.istft(D * (1.0 - mask), hop_length=HOP, length=len(y))
    return vocal, accomp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref-dir", required=True)
    args = ap.parse_args()

    y, _ = librosa.load(args.audio, sr=SR, mono=True)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    f0_raw, conf = estimate_f0_sieve(np.abs(D), freqs)

    ref = Path(args.ref_dir)
    dv, _ = librosa.load(str(ref / "vocals.wav"), sr=SR, mono=True)
    da, _ = librosa.load(str(ref / "accompaniment.wav"), sr=SR, mono=True)
    n = min(len(y), len(dv))
    dv, da, y_s = dv[:n], da[:n], y[:n]

    def corr(x, z):
        return float(np.corrcoef(x[:n], z)[0, 1])

    results = []
    for v_thresh in [0.15, 0.3]:
        voiced = conf > v_thresh
        f0_s = smooth_f0(f0_raw, voiced)
        for sig_v in [0.4, 0.5, 0.6]:          # 人声轨：窄 -> 干净
            for sig_a in [1.0, 1.5, 2.0]:      # 伴奏轨：宽 -> 抑制人声
                mask_v = comb_mask(freqs, f0_s, voiced, sigma=sig_v, max_hz=4500.0)
                mask_supp = comb_mask(freqs, f0_s, voiced, sigma=sig_a, max_hz=6000.0)
                v = librosa.istft(D * mask_v, hop_length=HOP, length=len(y))[:n]
                a = librosa.istft(D * (1.0 - mask_supp), hop_length=HOP, length=len(y))[:n]
                keep_v, keep_a = corr(v, dv), corr(a, da)
                leak_v, leak_a = corr(v, da), corr(a, dv)
                score = keep_v + keep_a - leak_v - leak_a
                results.append((score, v_thresh, sig_v, sig_a, keep_v, keep_a, leak_v, leak_a))

    results.sort(key=lambda r: -r[0])
    print(f"{'得分':>7s} {'阈值':>5s} {'σ_人声':>7s} {'σ_伴奏':>7s} "
          f"{'人声保留':>8s} {'吉他保留':>8s} {'吉他泄漏':>8s} {'人声泄漏':>8s}")
    for r in results[:12]:
        score, vt, sv, sa, kv, ka, lv, la = r
        print(f"{score:7.3f} {vt:5.2f} {sv:7.2f} {sa:7.2f} {kv:8.3f} {ka:8.3f} {lv:8.3f} {la:8.3f}")


if __name__ == "__main__":
    main()
