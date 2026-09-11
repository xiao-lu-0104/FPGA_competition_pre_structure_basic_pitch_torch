"""人声+吉他同步扒谱：两阶段管线（Demucs 分离 + Basic Pitch 逐轨转写）。

输入：一段人声+吉他（或带伴奏）的混音
输出：
    1) vocals.wav        分离出的人声轨
    2) accompaniment.wav 分离出的伴奏轨（吉他等）
    3) *_vocals.mid      人声谱
    4) *_guitar.mid      伴奏谱

用法：
    python scripts/separate_and_transcribe.py --audio 混音.mp3 --out-dir 输出目录
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from demucs.apply import apply_model  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402

from basic_pitch_torch.inference import predict  # noqa: E402

DEFAULT_MODEL = ROOT / "models" / "basic_pitch_pytorch_icassp_2022.pth"


def separate(audio_path: str, out_dir: Path, device: str):
    """Demucs 分离，返回 (人声轨路径, 伴奏轨路径)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("加载 Demucs (htdemucs)...")
    model = get_model("htdemucs")
    model.to(device)
    model.eval()

    wav, _ = librosa.load(audio_path, sr=model.samplerate, mono=False)  # (ch, T)
    wav = torch.from_numpy(wav).float()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)  # htdemucs 需要立体声
    ref = wav.mean(0)
    wav_norm = (wav - ref.mean()) / (ref.std() + 1e-8)

    print("分离中...")
    with torch.no_grad():
        sources = apply_model(model, wav_norm.unsqueeze(0), device=device, progress=True)[0]
    sources = sources * ref.std() + ref.mean()
    stems = {name: src.mean(0) for name, src in zip(model.sources, sources)}

    vocals_path = out_dir / "vocals.wav"
    accomp_path = out_dir / "accompaniment.wav"
    accomp = sum(stems[n] for n in ["drums", "bass", "other"]) if "drums" in stems else stems.get("other", torch.zeros_like(stems["vocals"]))
    sf.write(str(vocals_path), stems["vocals"].cpu().numpy(), model.samplerate)
    sf.write(str(accomp_path), accomp.cpu().numpy(), model.samplerate)
    print(f"  人声轨 -> {vocals_path}")
    print(f"  伴奏轨 -> {accomp_path}")
    return vocals_path, accomp_path


def transcribe(audio_path: Path, out_midi: Path) -> int:
    _, midi_data, note_events = predict(str(audio_path), str(DEFAULT_MODEL))
    midi_data.write(str(out_midi))
    return len(note_events)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="人声+吉他混音文件")
    ap.add_argument("--out-dir", default="output")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    stem = Path(args.audio).stem

    vocals, accomp = separate(args.audio, out_dir, device)

    print("\n逐轨转写（Basic Pitch）...")
    n_vocal = transcribe(vocals, out_dir / f"{stem}_vocals.mid")
    n_guitar = transcribe(accomp, out_dir / f"{stem}_guitar.mid")
    print(f"\n完成 ✔ 人声谱 {n_vocal} 个音符，伴奏谱 {n_guitar} 个音符")
    print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
