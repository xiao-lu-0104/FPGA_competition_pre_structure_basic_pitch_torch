"""打印 Basic Pitch 模型的网络结构与参数量。

用法：
    python scripts/show_structure.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from basic_pitch_torch.model import BasicPitchTorch  # noqa: E402


def main() -> None:
    model = BasicPitchTorch()
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print(model)
    print()
    print("输入:  (batch, 43844)  22.05kHz × 2 秒音频窗口")
    print("输出:  contour (batch, 172, 264) 音高轮廓（1/3 半音分辨率）")
    print("       note    (batch, 172, 88)  88 个钢琴键的发音概率")
    print("       onset   (batch, 172, 88)  88 个钢琴键的起始点概率")


if __name__ == "__main__":
    main()
