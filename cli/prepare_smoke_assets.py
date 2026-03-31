from __future__ import annotations

import argparse
import math
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli.long_audio_core import list_frame_images, write_manifest


DEFAULT_AUDIO = Path("samples/input/HOWL AT THE HAIRPIN2.wav")
DEFAULT_FRAMES_DIR = Path("samples/input/momiji_studio")
DEFAULT_OUTPUT_DIR = Path("cli_output") / "smoke_assets" / "howl20_momiji"
DEFAULT_CLIP_SECONDS = 20.0
DEFAULT_RESIZE_WIDTH = 384
DEFAULT_RESIZE_HEIGHT = 216
DEFAULT_MANIFEST_NAME = "smoke_assets_manifest.json"


@dataclass(frozen=True)
class PreparedFrame:
    source_path: str
    output_path: str
    width: int
    height: int


@dataclass(frozen=True)
class PreparedAudio:
    source_path: str
    output_path: str
    start_seconds: float
    clip_seconds: float
    frame_rate: int
    channels: int
    sample_width: int
    frame_count: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a lightweight smoke-test dataset by clipping the sample HOWL audio "
            "and resizing momiji_studio frames."
        )
    )
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="Source WAV used for the smoke clip.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=DEFAULT_FRAMES_DIR,
        help="Source image folder used for the smoke-test frame set.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for prepared assets.")
    parser.add_argument(
        "--audio-start",
        type=float,
        default=0.0,
        help="Start offset for the clipped smoke audio in seconds.",
    )
    parser.add_argument(
        "--clip-seconds",
        type=float,
        default=DEFAULT_CLIP_SECONDS,
        help=f"Audio clip length in seconds. Defaults to {DEFAULT_CLIP_SECONDS:g}.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=DEFAULT_RESIZE_WIDTH,
        help=f"Output frame width. Defaults to {DEFAULT_RESIZE_WIDTH}.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=DEFAULT_RESIZE_HEIGHT,
        help=f"Output frame height. Defaults to {DEFAULT_RESIZE_HEIGHT}.",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help=f"Manifest filename. Defaults to {DEFAULT_MANIFEST_NAME}.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing a non-empty output directory.")
    return parser.parse_args(argv)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    resolved = output_dir.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a directory: {resolved}")
        if any(resolved.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {resolved}. "
                "Pass --overwrite to reuse it."
            )
    else:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _clip_wav_audio(
    source_audio: Path,
    output_path: Path,
    *,
    start_seconds: float,
    clip_seconds: float,
) -> PreparedAudio:
    resolved_input = source_audio.expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Audio file does not exist: {resolved_input}")
    if resolved_input.suffix.lower() != ".wav":
        raise ValueError(f"Only WAV input is supported for smoke-asset clipping: {resolved_input}")
    if clip_seconds <= 0:
        raise ValueError("--clip-seconds must be greater than 0.")
    if start_seconds < 0:
        raise ValueError("--audio-start must be 0 or greater.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(resolved_input), "rb") as reader:
        frame_rate = reader.getframerate()
        start_frame = max(int(math.floor(start_seconds * frame_rate)), 0)
        clip_frame_count = max(int(math.ceil(clip_seconds * frame_rate)), 1)
        reader.setpos(min(start_frame, reader.getnframes()))
        frames = reader.readframes(clip_frame_count)

        with wave.open(str(output_path), "wb") as writer:
            writer.setnchannels(reader.getnchannels())
            writer.setsampwidth(reader.getsampwidth())
            writer.setframerate(frame_rate)
            writer.setcomptype(reader.getcomptype(), reader.getcompname())
            writer.writeframes(frames)

        actual_frames = max(len(frames) // max(reader.getsampwidth() * reader.getnchannels(), 1), 0)
        return PreparedAudio(
            source_path=str(resolved_input),
            output_path=str(output_path.resolve()),
            start_seconds=float(start_seconds),
            clip_seconds=actual_frames / float(max(frame_rate, 1)),
            frame_rate=frame_rate,
            channels=reader.getnchannels(),
            sample_width=reader.getsampwidth(),
            frame_count=actual_frames,
        )


def _resize_image(
    source_path: Path,
    output_path: Path,
    *,
    width: int,
    height: int,
) -> PreparedFrame:
    if width <= 0 or height <= 0:
        raise ValueError("--resize-width and --resize-height must be greater than 0.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        source = image.convert("RGB")
        scale = min(width / float(source.width), height / float(source.height))
        resized_width = max(int(round(source.width * scale)), 1)
        resized_height = max(int(round(source.height * scale)), 1)
        resized = source.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        offset = ((width - resized_width) // 2, (height - resized_height) // 2)
        canvas.paste(resized, offset)
        canvas.save(output_path)

    return PreparedFrame(
        source_path=str(source_path.expanduser().resolve()),
        output_path=str(output_path.expanduser().resolve()),
        width=width,
        height=height,
    )


def prepare_smoke_assets(
    *,
    audio: Path,
    frames_dir: Path,
    output_dir: Path,
    audio_start: float = 0.0,
    clip_seconds: float = DEFAULT_CLIP_SECONDS,
    resize_width: int = DEFAULT_RESIZE_WIDTH,
    resize_height: int = DEFAULT_RESIZE_HEIGHT,
    manifest_name: str = DEFAULT_MANIFEST_NAME,
    overwrite: bool = False,
) -> dict[str, object]:
    target_dir = _prepare_output_dir(output_dir, overwrite=overwrite)
    frame_paths = list_frame_images(frames_dir)

    prepared_audio = _clip_wav_audio(
        audio,
        target_dir / "audio" / f"{audio.expanduser().resolve().stem}_{int(round(clip_seconds))}s.wav",
        start_seconds=audio_start,
        clip_seconds=clip_seconds,
    )
    prepared_frames = [
        _resize_image(
            frame_path,
            target_dir / "frames" / f"{frame_path.stem}.png",
            width=resize_width,
            height=resize_height,
        )
        for frame_path in frame_paths
    ]

    manifest = {
        "source_audio": prepared_audio.source_path,
        "prepared_audio": asdict(prepared_audio),
        "source_frames_dir": str(Path(frames_dir).expanduser().resolve()),
        "prepared_frames_dir": str((target_dir / "frames").resolve()),
        "resize_width": int(resize_width),
        "resize_height": int(resize_height),
        "frame_count": len(prepared_frames),
        "frames": [asdict(frame) for frame in prepared_frames],
    }
    manifest_path = write_manifest(manifest, target_dir / manifest_name)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = prepare_smoke_assets(
        audio=args.audio,
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        audio_start=args.audio_start,
        clip_seconds=args.clip_seconds,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        manifest_name=args.manifest_name,
        overwrite=args.overwrite,
    )
    print(f"Prepared audio: {manifest['prepared_audio']['output_path']}")
    print(f"Prepared frames: {manifest['prepared_frames_dir']}")
    print(f"Frame count: {manifest['frame_count']}")
    print(f"Manifest: {manifest['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
