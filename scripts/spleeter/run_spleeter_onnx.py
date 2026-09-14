"""用 ONNX（fp32 或 INT8 量化版）跑 Spleeter 分离，用于评估量化掉点。

用法：
    python run_spleeter_onnx.py --audio "吉他+人声.mp3" --out-dir out_int8 --int8 --patch 128
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import librosa
import numpy as np
import soundfile as sf

SR = 44100
N_FFT = 4096
HOP = 1024
F_BINS = 1024
PATCH = 512
OVERLAP = 0.25

HERE = Path(__file__).resolve().parent


def run_onnx(sess, x: np.ndarray) -> np.ndarray:
    """x: (T, F, 2) -> (T, F, 2)。"""
    T = x.shape[0]
    out = np.zeros_like(x)
    step = int(PATCH * (1 - OVERLAP))
    for t0 in range(0, T, step):
        seg = x[t0 : t0 + PATCH]
        L = seg.shape[0]
        if L == 0:
            break
        if L < PATCH:
            seg = np.concatenate([seg, np.zeros((PATCH - L, F_BINS, 2), np.float32)], axis=0)
        xin = seg.transpose(2, 0, 1)[None].astype(np.float32)  # (1,2,512,1024)
        y = sess.run(None, {"spectrogram": xin})[0][0].transpose(1, 2, 0)
        out[t0 : t0 + L] = y[:L]
        if t0 + PATCH >= T:
            break
    return out


def separate(audio_path: str, out_dir: Path, quant: str) -> None:
    import onnxruntime as ort

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = {"none": "", "qdq8": "_qdq8", "dyn8": "_int8"}[quant]
    sess = {}
    for name in ["vocals", "accompaniment"]:
        p = HERE / "onnx" / f"spleeter_{name}{suffix}.onnx"
        sess[name] = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        print(f"加载 {p.name}  ({p.stat().st_size/1024/1024:.1f} MB)")

    y, _ = librosa.load(audio_path, sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    print(f"音频 {y.shape[1]/SR:.1f} 秒（{y.shape[0]} 声道），分块 {PATCH} 帧，模型={quant}")

    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP).transpose(1, 2, 0)
    mag = np.abs(D)
    x = mag[:F_BINS].transpose(1, 0, 2).astype(np.float32)
    mu, sd = float(x.mean()), float(x.std())
    xn = (x - mu) / (sd + 1e-8)

    out_v = run_onnx(sess["vocals"], xn)
    out_a = run_onnx(sess["accompaniment"], xn)

    mask_v = np.clip(out_v / (xn + np.sign(xn) * 1e-6 + 1e-9), 0.0, 1.0)
    mask_a = np.clip(out_a / (xn + np.sign(xn) * 1e-6 + 1e-9), 0.0, 1.0)

    full_v = np.zeros_like(mag)
    full_v[:F_BINS] = mask_v.transpose(1, 0, 2)
    full_a = np.zeros_like(mag)
    full_a[:F_BINS] = mask_a.transpose(1, 0, 2)

    D_v = D * full_v
    D_a = D - D_v

    vocals = librosa.istft(D_v.transpose(2, 0, 1), hop_length=HOP, length=y.shape[1])
    accomp = librosa.istft(D_a.transpose(2, 0, 1), hop_length=HOP, length=y.shape[1])

    sf.write(str(out_dir / "vocals_spleeter.wav"), vocals.T, SR)
    sf.write(str(out_dir / "accompaniment_spleeter.wav"), accomp.T, SR)
    print(f"输出 -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out-dir", default=str(HERE / "out_onnx"))
    ap.add_argument("--quant", choices=["none", "qdq8", "dyn8"], default="none",
                    help="none=fp32 / qdq8=静态量化 / dyn8=动态量化")
    ap.add_argument("--patch", type=int, default=PATCH)
    args = ap.parse_args()
    PATCH = args.patch
    separate(args.audio, Path(args.out_dir), args.quant)
