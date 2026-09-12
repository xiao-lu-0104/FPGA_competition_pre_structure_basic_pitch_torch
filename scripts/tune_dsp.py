"""参数扫描：在 FPGA 可实现的前提下，寻找谐波掩码分离的最优配置。

评估方式：以 Demucs 分离结果为参考，计算 SI-SDR（人声轨 + 伴奏轨的平均值）。

用法：
    python scripts/tune_dsp.py --audio 混音.wav --ref-dir out_吉他人声
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import soundfile as sf

from dsp_separate import (
    HOP,
    N_FFT,
    SR,
    estimate_f0_sieve,
    harmonic_mask_separate,
    smooth_f0,
)
from eval_dsp import si_sdr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref-dir", required=True, help="Demucs 参考结果目录")
    args = ap.parse_args()

    y, _ = librosa.load(args.audio, sr=SR, mono=True)
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    f0_raw, conf = estimate_f0_sieve(np.abs(D), freqs)
    print(f"F0 估计完成，平均 F0={np.mean(f0_raw[f0_raw>0]):.1f} Hz")

    ref_dir = Path(args.ref_dir)
    ref_v, _ = librosa.load(str(ref_dir / "vocals.wav"), sr=SR, mono=True)
    ref_a, _ = librosa.load(str(ref_dir / "accompaniment.wav"), sr=SR, mono=True)

    results = []
    for v_thresh in [0.3, 0.4, 0.5, 0.6]:
        voiced = conf > v_thresh
        if voiced.sum() == 0:
            continue
        f0_s = smooth_f0(f0_raw, voiced)
        for soft in [False, True]:
            for max_freq in [3500.0, 4500.0]:
                for n_harm in [10, 14]:
                    # 临时覆盖模块常量
                    import dsp_separate as ds

                    ds.MAX_HARMONIC_FREQ = max_freq
                    vocal, accomp, _ = harmonic_mask_separate(
                        y, f0_s, voiced, n_harmonics=n_harm, soft=soft
                    )
                    s_v = si_sdr(vocal, ref_v)
                    s_a = si_sdr(accomp, ref_a)
                    results.append((v_thresh, soft, max_freq, n_harm, s_v, s_a, (s_v + s_a) / 2))

    results.sort(key=lambda r: -r[-1])
    print(f"{'阈值':>5s} {'软掩码':>6s} {'谐波上限':>8s} {'谐波数':>6s} {'人声dB':>8s} {'伴奏dB':>8s} {'平均':>8s}")
    for v_thresh, soft, max_freq, n_harm, s_v, s_a, avg in results[:10]:
        print(f"{v_thresh:5.1f} {str(soft):>6s} {max_freq:8.0f} {n_harm:6d} {s_v:8.2f} {s_a:8.2f} {avg:8.2f}")


if __name__ == "__main__":
    main()
