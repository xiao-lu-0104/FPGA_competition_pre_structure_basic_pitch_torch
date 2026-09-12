"""改进实验：Viterbi 音高连续性跟踪 + 置信度加权掩码。

思路：
  1. 逐帧独立取最大（现方案）容易被吉他低音抢走 F0 -> 改为 Viterbi 全局路径搜索，
     对音高跳变施加惩罚（人声旋律连续，吉他伴奏跳变多）；
  2. 置信度低的帧（人声弱/无）-> 掩码乘以置信度权重，减少把吉他收进人声轨。

两者都是 FPGA 可实现的（DP 48x48/帧 + 查表加权），无需训练。

用法：
    python scripts/tune_dsp3.py --audio 吉他+人声.mp3 --ref-dir out_吉他人声
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np

from dsp_separate import (
    F0_CANDIDATES,
    F0_MAX,
    F0_MIN,
    HOP,
    MAX_HARMONIC_FREQ,
    N_FFT,
    N_HARMONICS,
    SR,
    comb_mask,
    smooth_f0,
)


def emission_scores(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """返回 (n_candidates, n_frames) 的发射分数（谐波位置平均幅度 / 帧平均幅度）。"""
    n_frames = mag.shape[1]
    candidates = F0_MIN * (F0_MAX / F0_MIN) ** (np.arange(F0_CANDIDATES) / (F0_CANDIDATES - 1))
    scores = np.zeros((F0_CANDIDATES, n_frames), dtype=np.float32)
    frame_mean = mag.mean(axis=0) + 1e-9
    for ci, fc in enumerate(candidates):
        bins = []
        for k in range(1, N_HARMONICS + 1):
            fh = fc * k
            if fh > MAX_HARMONIC_FREQ:
                break
            bins.append(int(round(fh / (SR / N_FFT))))
        bins = np.array([b for b in bins if b < mag.shape[0]])
        if len(bins) == 0:
            continue
        scores[ci] = mag[bins].mean(axis=0) / frame_mean
    return scores, candidates


def viterbi(scores: np.ndarray, candidates: np.ndarray, jump_penalty: float = 0.15):
    """带音高跳变惩罚的 Viterbi 路径搜索（对数频率距离）。"""
    C, T = scores.shape
    logf = np.log(candidates)
    trans = -jump_penalty * np.abs(logf[:, None] - logf[None, :])  # (prev, cur)
    dp = scores[:, 0].astype(np.float64).copy()
    back = np.zeros((C, T), dtype=np.int32)
    for t in range(1, T):
        cand = dp[:, None] + trans
        best = np.argmax(cand, axis=0)
        dp = scores[:, t] + cand[best, np.arange(C)]
        back[:, t] = best
    path = np.zeros(T, dtype=np.int32)
    path[-1] = int(np.argmax(dp))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return candidates[path], scores[path, np.arange(T)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--sigma-suppress", type=float, default=1.0)
    args = ap.parse_args()

    y, _ = librosa.load(args.audio, sr=SR, mono=True)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    mag = np.abs(D)
    scores, candidates = emission_scores(mag, freqs)
    frame_mean = mag.mean(axis=0) + 1e-9

    ref = Path(args.ref_dir)
    dv, _ = librosa.load(str(ref / "vocals.wav"), sr=SR, mono=True)
    da, _ = librosa.load(str(ref / "accompaniment.wav"), sr=SR, mono=True)
    n = min(len(y), len(dv), len(da))
    dv, da = dv[:n], da[:n]

    seg = 10 * SR
    weak_idx = [
        s for s in range(0, n - seg, seg)
        if np.sqrt(np.mean(dv[s:s+seg]**2)) < np.median([np.sqrt(np.mean(dv[k:k+seg]**2)) for k in range(0, n-seg, seg)])
    ]

    def evaluate(v, a, label, params):
        v, a = v[:n], a[:n]
        kv = np.corrcoef(v, dv)[0, 1]
        lv = np.corrcoef(v, da)[0, 1]
        ka = np.corrcoef(a, da)[0, 1]
        la = np.corrcoef(a, dv)[0, 1]
        kv_w = np.mean([np.corrcoef(v[s:s+seg], dv[s:s+seg])[0, 1] for s in weak_idx])
        lv_w = np.mean([np.corrcoef(v[s:s+seg], da[s:s+seg])[0, 1] for s in weak_idx])
        print(f"{label:26s} {params:22s} 保留 {kv:.3f} 泄漏 {lv:.3f} 伴奏保留 {ka:.3f} 伴奏泄漏 {la:.3f} | 弱段保留 {kv_w:.3f} 弱段泄漏 {lv_w:.3f}")

    print(f"{'方案':26s} {'参数':22s} {'人声':38s} {'伴奏':30s} 弱段表现")

    for jp in [0.0, 0.05, 0.15, 0.30]:
        if jp == 0.0:
            f0 = candidates[np.argmax(scores, axis=0)]
            conf = scores.max(axis=0)
        else:
            f0, conf = viterbi(scores, candidates, jump_penalty=jp)
        for gate in [None, (0.15, 0.45), (0.25, 0.60)]:
            voiced = conf > 0.10
            f0_s = smooth_f0(f0, voiced)
            mv = comb_mask(freqs, f0_s, voiced, sigma=0.40, max_hz=MAX_HARMONIC_FREQ)
            if gate is not None:
                c0, c1 = gate
                w = np.clip((conf - c0) / (c1 - c0), 0.0, 1.0)[None, :]
                mv = mv * w
            ms = comb_mask(freqs, f0_s, voiced, sigma=args.sigma_suppress, max_hz=6000.0)
            v = librosa.istft(D * mv, hop_length=HOP, length=len(y))
            a = librosa.istft(D * (1.0 - ms), hop_length=HOP, length=len(y))
            evaluate(v, a, f"Viterbi jp={jp}" if jp else "逐帧argmax(现方案)", f"gate={gate}", )


if __name__ == "__main__":
    main()
