"""诊断：分段评估分离质量 + 分析 F0 分布（定位主歌段效果差的原因）。

用法：
    python scripts/diag_dsp.py --audio 吉他+人声.mp3 --ref-dir out_吉他人声 --out-dir out_dsp_v3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import librosa
import numpy as np

from dsp_separate import HOP, N_FFT, SR, VOCAL_GAIN_DB


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seg", type=float, default=10.0, help="分段秒数")
    args = ap.parse_args()

    y, _ = librosa.load(args.audio, sr=SR, mono=True)
    ref = Path(args.ref_dir)
    out = Path(args.out_dir)
    dv, _ = librosa.load(str(ref / "vocals.wav"), sr=SR, mono=True)
    da, _ = librosa.load(str(ref / "accompaniment.wav"), sr=SR, mono=True)
    v, _ = librosa.load(str(out / "vocals_dsp.wav"), sr=SR, mono=True)
    a, _ = librosa.load(str(out / "accompaniment_dsp.wav"), sr=SR, mono=True)
    n = min(len(y), len(dv), len(v))
    dv, da, v, a, y = (x[:n] for x in (dv, da, v, a, y))

    # 用 Demucs 的人声轨声学活跃度标出"主歌/高音段"
    hop = SR  # 1 秒窗
    seg = int(args.seg * SR)
    print(f"{'时间段':>10s} {'人声能量占比':>10s} {'人声保留':>8s} {'吉他泄漏':>8s} {'人声泄漏':>8s}")
    rows = []
    for start in range(0, n - seg, seg):
        sl = slice(start, start + seg)
        ev = np.sqrt(np.mean(dv[sl] ** 2))
        ea = np.sqrt(np.mean(da[sl] ** 2))
        # 该段人声/伴奏能量比 -> 判断是否主歌（低音、人声弱）段
        ratio = ev / (ev + ea + 1e-9)
        kv = np.corrcoef(v[sl], dv[sl])[0, 1]
        lv = np.corrcoef(v[sl], da[sl])[0, 1]
        la = np.corrcoef(a[sl], dv[sl])[0, 1]
        rows.append((start / SR, ratio, kv, lv, la))
        print(f"{start/SR:8.0f}-{start/SR+args.seg:.0f}s {ratio:10.2f} {kv:8.3f} {lv:8.3f} {la:8.3f}")

    rows = np.array(rows)
    weak = rows[rows[:, 1] < np.median(rows[:, 1])]   # 人声能量占比低的段（主歌）
    strong = rows[rows[:, 1] >= np.median(rows[:, 1])]
    print()
    print(f"人声较弱段（类似主歌）: 人声保留 {weak[:,2].mean():.3f} | 吉他泄漏 {weak[:,3].mean():.3f} | 人声泄漏 {weak[:,4].mean():.3f}")
    print(f"人声较强段（类似副歌）: 人声保留 {strong[:,2].mean():.3f} | 吉他泄漏 {strong[:,3].mean():.3f} | 人声泄漏 {strong[:,4].mean():.3f}")


if __name__ == "__main__":
    main()
