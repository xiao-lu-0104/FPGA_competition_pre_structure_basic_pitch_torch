"""导出 Basic Pitch 的 ONNX 算子图（供 FPGA 硬件团队参考/量化仿真）。

两种导出：
1. 仅卷积主干（推荐）：输入谐波堆叠后的特征 (B, 8, T, 264)，输出 onset/contour/note 三张热力图；
   CQT 前端建议在 RTL 里用多速率 FFT 实现，不参与 ONNX。
2. 完整模型（含 CQT）：供对照，CQT 建议 RTL 实现。

用法：
    python scripts/export_onnx.py --out-dir models
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from basic_pitch_torch.model import BasicPitchTorch  # noqa: E402

DEFAULT_WEIGHTS = ROOT / "models" / "basic_pitch_pytorch_icassp_2022.pth"


class HeadOnly(nn.Module):
    """包装 BasicPitchTorch 的卷积部分（跳过 CQT/谐波堆叠）。"""

    def __init__(self, model: BasicPitchTorch) -> None:
        super().__init__()
        self.conv_contour = model.conv_contour
        self.conv_note = model.conv_note
        self.conv_onset_pre = model.conv_onset_pre
        self.conv_onset_post = model.conv_onset_post

    def forward(self, cqt: torch.Tensor):
        x_contour = self.conv_contour(cqt)
        x_contour_for_note = F.pad(x_contour, (2, 2, 3, 3))
        x_note = self.conv_note(x_contour_for_note)
        cqt_for_onset = F.pad(cqt, (1, 1, 2, 2))
        x_onset_pre = self.conv_onset_pre(cqt_for_onset)
        x_onset_pre = torch.cat([x_note, x_onset_pre], dim=1)
        x_onset = self.conv_onset_post(x_onset_pre)
        return x_onset, x_contour, x_note


def print_layer_table(model: BasicPitchTorch) -> None:
    print("\n=== 逐层参数表（供 RTL 实现参考）===")
    print(f"{'层':32s} {'权重形状':>22s} {'参数量':>10s}")
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            n = sum(p.numel() for p in module.parameters())
            total += n
            shape = tuple(module.weight.shape) if hasattr(module, "weight") else ()
            print(f"{name:32s} {str(shape):>22s} {n:10,d}")
    print(f"{'合计':32s} {'':>22s} {total:10,d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--out-dir", default=str(ROOT / "models"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = BasicPitchTorch()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    print_layer_table(model)

    # ---- 1) 导出卷积主干 ----
    head = HeadOnly(model).eval()
    dummy = torch.randn(1, 8, 172, 264)
    onnx_path = out_dir / "basic_pitch_heads.onnx"
    torch.onnx.export(
        head,
        (dummy,),
        str(onnx_path),
        input_names=["cqt_stacked"],
        output_names=["onset", "contour", "note"],
        dynamic_axes={"cqt_stacked": {2: "time"}, "onset": {2: "time"}, "contour": {2: "time"}, "note": {2: "time"}},
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"\n已导出卷积主干: {onnx_path}  ({onnx_path.stat().st_size/1024:.1f} KB)")

    # ---- 2) onnxruntime 数值验证 ----
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = head(dummy)
    got = sess.run(None, {"cqt_stacked": dummy.numpy()})
    print("\n=== ONNX 与 PyTorch 数值一致性 ===")
    for name, r, g in zip(["onset", "contour", "note"], ref, got):
        print(f"  {name:8s} 最大误差 {np.abs(r.numpy() - g).max():.2e}")

    # ---- 3) 完整模型导出（含 CQT）----
    try:
        full_path = out_dir / "basic_pitch_full.onnx"
        torch.onnx.export(
            model,
            (torch.randn(1, 43844),),
            str(full_path),
            input_names=["audio"],
            output_names=["onset", "contour", "note"],
            opset_version=13,
        )
        print(f"\n完整模型（含 CQT）已导出: {full_path}")
    except Exception as e:
        print(f"\n完整模型导出失败（可忽略，CQT 建议 RTL 实现）: {type(e).__name__}")


if __name__ == "__main__":
    main()
