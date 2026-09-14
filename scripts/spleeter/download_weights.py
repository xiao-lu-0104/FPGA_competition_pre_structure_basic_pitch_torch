"""下载 Spleeter 2stems PyTorch 权重（vocals.pt / accompaniment.pt）。

权重不进 git（两文件共 75 MB），用本脚本重新拉取即可。
默认走 hf-mirror（国内可达），失败自动回退 huggingface.co。

用法：
    python download_weights.py            # 只下载 .pt
    python download_weights.py --onnx     # 下载后顺便导出 ONNX（给 RTL 做黄金参考）
"""

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve

HERE = Path(__file__).resolve().parent
STEMS = ["vocals", "accompaniment"]

MIRRORS = [
    "https://hf-mirror.com/csukuangfj/spleeter-torch/resolve/main/2stems/{name}.pt",
    "https://hf-mirror.com/csukuangfj/spleeter-torch/resolve/main/{name}.pt",
    "https://huggingface.co/csukuangfj/spleeter-torch/resolve/main/2stems/{name}.pt",
    "https://huggingface.co/csukuangfj/spleeter-torch/resolve/main/{name}.pt",
]
EXPECT_MB = 37.5  # 单文件约 37.5 MB（9.82 M 参数 fp32）


def _report(block: int, block_size: int, total: int) -> None:
    if total > 0:
        done = min(block * block_size, total) / total
        print(f"\r  {done * 100:5.1f}%", end="", flush=True)


def fetch(name: str) -> Path:
    dst = HERE / f"{name}.pt"
    if dst.exists():
        mb = dst.stat().st_size / 1024 / 1024
        print(f"[跳过] {name}.pt 已存在（{mb:.1f} MB）")
        return dst
    last_err: Exception | None = None
    for url in MIRRORS:
        src = url.format(name=name)
        try:
            print(f"[下载] {name}.pt  <-  {src}")
            urlretrieve(src, dst, reporthook=_report)
            print()
            break
        except Exception as e:  # noqa: BLE001 - 逐个镜像回退
            last_err = e
            print(f"\n  [!] 失败：{e}")
            if dst.exists():
                dst.unlink()
    else:
        raise SystemExit(
            f"所有镜像都下载失败，请手动下载 {name}.pt 放到 {HERE}\n最后错误：{last_err}"
        )
    mb = dst.stat().st_size / 1024 / 1024
    flag = "OK" if abs(mb - EXPECT_MB) < 3 else "?? 大小异常，请检查！"
    print(f"[完成] {name}.pt  {mb:.1f} MB  [{flag}]")
    return dst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", action="store_true", help="下载后导出 ONNX（spleeter_test/onnx/）")
    args = ap.parse_args()

    for s in STEMS:
        fetch(s)

    print("\n权重就绪，可直接：")
    print(f'  python run_spleeter.py --audio "吉他+人声.mp3" --patch 128')

    if args.onnx:
        sys.path.insert(0, str(HERE))
        import export_spleeter_onnx

        print("\n导出 ONNX ...")
        export_spleeter_onnx.main()
        print("ONNX 在 scripts/spleeter/onnx/，可直接用于 RTL 逐层对拍。")


if __name__ == "__main__":
    main()
