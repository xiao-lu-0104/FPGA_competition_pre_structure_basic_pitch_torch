"""统计 Spleeter U-Net 逐层结构：输入/输出形状、参数量、MAC。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn

from unet import UNet

records = []


def hook(mod, inp, out):
    if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d)):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        cin = mod.in_channels
        if isinstance(mod, nn.Conv2d):
            macs = out.shape[1] * out.shape[2] * out.shape[3] * cin * k
        else:  # ConvTranspose2d：按输入元素 × 输出通道 × 核面积计
            macs = inp[0].shape[1] * inp[0].shape[2] * inp[0].shape[3] * mod.out_channels * k
        records.append((mod.__class__.__name__, tuple(inp[0].shape[1:]), tuple(out.shape[1:]),
                        sum(p.numel() for p in mod.parameters()), macs))


def main() -> None:
    m = UNet().eval()
    for mod in m.modules():
        mod.register_forward_hook(hook)
    with torch.no_grad():
        m(torch.randn(1, 2, 512, 1024))

    total_p = total_m = 0
    print(f"{'层':<10s}{'输入(C,H,W)':>20s}{'输出(C,H,W)':>20s}{'参数':>12s}{'MAC':>14s}")
    for name, i, o, p, mac in records:
        total_p += p
        total_m += mac
        print(f"{name:<10s}{str(i):>20s}{str(o):>20s}{p:>12,d}{mac/1e6:>12.1f}M")
    print(f"{'合计':<10s}{'':>20s}{'':>20s}{total_p:>12,d}{total_m/1e9:>12.2f}G")
    print()
    frames_per_patch = 512
    hop = 1024
    sr = 44100
    seconds = frames_per_patch * hop / sr
    print(f"一个 512 帧块 = {seconds:.1f} 秒音频，单模型 {total_m/1e9:.2f} GMAC")
    print(f"实时（含 25% 重叠）：单模型 {total_m/1e9/seconds*1.33:.2f} GMAC/s，双模型 {2*total_m/1e9/seconds*1.33:.2f} GMAC/s")


if __name__ == "__main__":
    main()
