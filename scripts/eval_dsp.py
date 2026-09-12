"""以 Demucs 分离结果为参考，客观评分 DSP 分离质量（SI-SDR）。

用法：
    python finetune/eval_dsp.py --reference-dir out_吉他人声 --dsp-dir out_dsp
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np
import soundfile as sf


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    """尺度不变信噪比（dB），越高越接近参考。"""
    n = min(len(est), len(ref))
    est, ref = est[:n], ref[:n]
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-dir", required=True, help="Demucs 分离结果目录（vocals.wav/accompaniment.wav）")
    ap.add_argument("--dsp-dir", required=True, help="DSP 分离结果目录")
    args = ap.parse_args()

    ref_dir, dsp_dir = Path(args.reference_dir), Path(args.dsp_dir)

    pairs = [("vocals_dsp.wav", "vocals.wav", "人声轨"), ("accompaniment_dsp.wav", "accompaniment.wav", "伴奏轨")]
    print(f"{'音轨':8s} {'SI-SDR (dB)':>12s}")
    for dsp_name, ref_name, label in pairs:
        est, _ = librosa.load(str(dsp_dir / dsp_name), sr=22050, mono=True)
        ref, _ = librosa.load(str(ref_dir / ref_name), sr=22050, mono=True)
        score = si_sdr(est, ref)
        print(f"{label:8s} {score:12.2f}")


if __name__ == "__main__":
    main()
