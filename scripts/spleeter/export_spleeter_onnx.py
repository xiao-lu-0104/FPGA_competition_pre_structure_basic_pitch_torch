"""导出 Spleeter (PyTorch) 两个 stem 模型为 ONNX，并做数值校验。

时间轴（dim=2）为动态维，可用任意 64 的倍数（64/128/256/512 帧）输入，
方便硬件团队按自己的分块长度对拍。fre 轴固定 1024。
"""

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
            dynamic_axes={"spectrogram": {0: "batch", 2: "frames"},
                          "masked_spectrogram": {0: "batch", 2: "frames"}},
        )
        size_mb = path.stat().st_size / 1024 / 1024

        import onnxruntime as ort

        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        # 校验 512 帧与 128 帧（FPGA 推荐）两种分块
        for T in (512, 128):
            d = torch.randn(1, 2, T, 1024)
            with torch.no_grad():
                ref = model(d).numpy()
            got = sess.run(None, {"spectrogram": d.numpy()})[0]
            print(f"    T={T:4d} 帧  最大误差 {np.abs(ref - got).max():.2e}")

        n_params = sum(p.numel() for p in model.parameters())
        print(f"{name:14s} 参数 {n_params:>12,} | ONNX {size_mb:6.1f} MB")



if __name__ == "__main__":
    main()
