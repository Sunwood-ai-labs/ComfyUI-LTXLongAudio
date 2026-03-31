from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli.prepare_smoke_assets import main, prepare_smoke_assets


def _write_wave_file(path: Path, *, frame_rate: int, frame_count: int, channels: int = 2) -> None:
    silence_frame = b"\x00\x00" * channels
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(frame_rate)
        writer.writeframes(silence_frame * frame_count)


def test_prepare_smoke_assets_clips_audio_and_resizes_frames(tmp_path: Path):
    source_audio = tmp_path / "source.wav"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    output_dir = tmp_path / "prepared"
    _write_wave_file(source_audio, frame_rate=4, frame_count=160, channels=2)

    from PIL import Image

    for index in range(2):
        image_path = frames_dir / f"frame_{index}.png"
        Image.new("RGB", (688, 384), color=(index * 10, 20, 30)).save(image_path)

    manifest = prepare_smoke_assets(
        audio=source_audio,
        frames_dir=frames_dir,
        output_dir=output_dir,
        audio_start=5.0,
        clip_seconds=20.0,
        resize_width=384,
        resize_height=216,
        overwrite=True,
    )

    prepared_audio = Path(str(manifest["prepared_audio"]["output_path"]))
    assert prepared_audio.exists()
    with wave.open(str(prepared_audio), "rb") as reader:
        assert reader.getframerate() == 4
        assert reader.getnchannels() == 2
        assert reader.getnframes() == 80

    prepared_frames = sorted((output_dir / "frames").glob("*.png"))
    assert len(prepared_frames) == 2
    for prepared_frame in prepared_frames:
        with Image.open(prepared_frame) as image:
            assert image.size == (384, 216)

    manifest_path = output_dir / "smoke_assets_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["frame_count"] == 2
    assert payload["resize_width"] == 384
    assert payload["resize_height"] == 216


def test_main_uses_requested_manifest_name(tmp_path: Path):
    source_audio = tmp_path / "source.wav"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    output_dir = tmp_path / "prepared"
    _write_wave_file(source_audio, frame_rate=8, frame_count=160, channels=2)

    from PIL import Image

    Image.new("RGB", (688, 384), color=(10, 20, 30)).save(frames_dir / "frame.png")

    exit_code = main(
        [
            "--audio",
            str(source_audio),
            "--frames-dir",
            str(frames_dir),
            "--output-dir",
            str(output_dir),
            "--clip-seconds",
            "10",
            "--resize-width",
            "320",
            "--resize-height",
            "180",
            "--manifest-name",
            "custom_manifest.json",
            "--overwrite",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "custom_manifest.json").exists()
