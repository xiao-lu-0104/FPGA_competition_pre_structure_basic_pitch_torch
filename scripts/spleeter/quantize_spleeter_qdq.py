"""Spleeter ONNX 的 INT8 静态量化（QDQ 格式），用真实音频谱做校准。

比动态量化更接近 FPGA 的 INT8 数据通路（权重 + 激活都量化）。
输出：onnx/spleeter_{stem}_qdq8.onnx

注意：onnxruntime 在含非 ASCII 的路径上会失败，脚本自动复制到临时目录再处理。

用法：
    python quantize_spleeter_qdq.py --audio "吉他+人声.mp3" --patch 128
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import librosa
import numpy as np

from onnxruntime.quantization import (CalibrationDataReader, QuantFormat,
                                      QuantType, quantize_static)

HERE = Path(__file__).resolve().parent
ONNX = HERE / "onnx"

SR = 44100
N_FFT = 4096
HOP = 1024
F_BINS = 1024
N_CALIB = 16          # 校准用分块数
PATCH = 512


def _ascii_copy(p: Path) -> Path:
    if all(ord(c) < 128 for c in str(p)):
        return p
    tmp = Path(tempfile.gettempdir()) / "spleeter_quant"
    tmp.mkdir(exist_ok=True)
    dst = tmp / p.name
    shutil.copy2(p, dst)
    return dst


class SpecReader(CalibrationDataReader):
    """用真实音频的归一化幅度谱分块做校准。"""

    def __init__(self, patches: list[np.ndarray]) -> None:
        self.patches = patches
        self.i = 0

    def get_next(self):
        if self.i >= len(self.patches):
            return None
        x = self.patches[self.i][None]  # (1,2,T,1024)
        self.i += 1
        return {"spectrogram": x.astype(np.float32)}

    def rewind(self) -> None:
        self.i = 0


def make_patches(audio: str, patch: int, n: int) -> list[np.ndarray]:
    y, _ = librosa.load(audio, sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP).transpose(1, 2, 0)
    x = np.abs(D)[:F_BINS].transpose(1, 0, 2).astype(np.float32)
    xn = (x - x.mean()) / (x.std() + 1e-8)
    T = xn.shape[0]
    idx = np.linspace(0, max(0, T - patch), n).astype(int)
    return [xn[i : i + patch].transpose(2, 0, 1) for i in idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(HERE.parent / "吉他+人声.mp3"))
    ap.add_argument("--patch", type=int, default=PATCH, help="校准分块长度，需与实际使用一致")
    args = ap.parse_args()

    patches = make_patches(args.audio, args.patch, N_CALIB)
    print(f"校准数据：{len(patches)} 块，每块形状 {patches[0].shape}（(C, T, F)）")

    for name in ["vocals", "accompaniment"]:
        src = ONNX / f"spleeter_{name}.onnx"
        dst = ONNX / f"spleeter_{name}_qdq8.onnx"
        if not src.exists():
            raise SystemExit(f"缺少 {src}，先运行 export_spleeter_onnx.py")

        src_q, dst_q = _ascii_copy(src), None
        dst_q = src_q.with_name(src_q.stem + "_qdq8.onnx")
        quantize_static(
            str(src_q), str(dst_q), SpecReader(patches),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QUInt8,
        )
        shutil.move(str(dst_q), str(dst))
        print(f"{name:14s} fp32 {src.stat().st_size/1024/1024:5.1f} MB"
              f"  ->  INT8(QDQ) {dst.stat().st_size/1024/1024:5.1f} MB")


if __name__ == "__main__":
    main()
