"""导出 Spleeter (PyTorch) 两个 stem 模型为 ONNX，并做数值校验。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from unet import UNet

OUT = Path(__file__).resolve().parent / "onnx"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    dummy = torch.randn(1, 2, 512, 1024)

    for name in ["vocals", "accompaniment"]:
        model = UNet()
        model.load_state_dict(torch.load(str(Path(__file__).parent / f"{name}.pt"), map_location="cpu"))
        model.eval()

        path = OUT / f"spleeter_{name}.onnx"
        torch.onnx.export(
            model,
            (dummy,),
            str(path),
            input_names=["spectrogram"],
            output_names=["masked_spectrogram"],
            opset_version=13,
            do_constant_folding=True,
        )
        size_mb = path.stat().st_size / 1024 / 1024

        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = model(dummy).numpy()
        got = sess.run(None, {"spectrogram": dummy.numpy()})[0]
        max_diff = float(np.abs(ref - got).max())

        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:14s} 参数 {n_params:>12,} | ONNX {size_mb:6.1f} MB | 最大误差 {max_diff:.2e}")


if __name__ == "__main__":
    main()
