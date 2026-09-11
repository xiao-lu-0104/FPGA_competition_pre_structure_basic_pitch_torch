"""单轨音频转 MIDI（Basic Pitch 预训练模型）。

用法：
    python scripts/transcribe.py --audio 你的音频.wav
    python scripts/transcribe.py --audio 你的音频.mp3 --out 输出.mid
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from basic_pitch_torch.inference import predict  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "basic_pitch_pytorch_icassp_2022.pth"


def main() -> None:
    ap = argparse.ArgumentParser(description="Basic Pitch 音频转 MIDI")
    ap.add_argument("--audio", required=True, help="输入音频（wav/mp3/flac 等）")
    ap.add_argument("--out", default=None, help="输出 MIDI 路径（默认与音频同名 .mid）")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()

    audio = Path(args.audio)
    out = Path(args.out) if args.out else audio.with_suffix(".mid")

    _, midi_data, note_events = predict(str(audio), str(args.model))
    midi_data.write(str(out))
    print(f"完成 ✔ 共 {len(note_events)} 个音符 -> {out}")


if __name__ == "__main__":
    main()
