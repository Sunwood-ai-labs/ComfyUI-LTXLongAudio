from __future__ import annotations

import sys
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cli.long_audio_core as core
from cli.run_origin_workflow import load_origin_workflow_defaults


def _write_wave_file(path: Path, *, frame_rate: int, frame_count: int, channels: int = 1, sample_width: int = 2) -> None:
    silence_frame = b"\x00" * sample_width * channels
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(frame_rate)
        writer.writeframes(silence_frame * frame_count)


def test_load_origin_workflow_defaults_reads_sample_values():
    defaults = load_origin_workflow_defaults(
        REPO_ROOT / "samples" / "workflows" / "LTX_2.3_Image_or_Text_&_Audio_2_Video_App_Origin.json"
    )

    assert defaults.segment_seconds == 20
    assert defaults.fps == 24.0
    assert defaults.seed == 1234
    assert defaults.source_audio == "Grateful to be here.mp3"
    assert defaults.image_load_cap == 0
    assert defaults.skip_first_images == 0
    assert defaults.select_every_nth == 1


def test_plan_segments_matches_existing_quantization():
    image_paths = [Path("a.png"), Path("b.png"), Path("c.png")]

    segments = core.plan_segments(26.0, 10, 24.0, image_paths, 1234)

    assert len(segments) == 3
    assert segments[0].start_time == 0.0
    assert segments[0].exact_seconds == 10.0
    assert segments[1].start_time == 10.0
    assert segments[1].exact_seconds == 10.0
    assert segments[2].start_time == 20.0
    assert segments[2].current_seconds == 6.0
    assert segments[2].frames == 145
    assert segments[2].exact_seconds == 6.0


def test_list_frame_images_applies_skip_step_and_cap(tmp_path: Path):
    for name in ("001.png", "002.jpg", "003.txt", "004.webp", "005.bmp"):
        target = tmp_path / name
        target.write_bytes(b"demo")

    images = core.list_frame_images(tmp_path, image_load_cap=2, skip_first_images=1, select_every_nth=2)

    assert [path.name for path in images] == ["002.jpg", "005.bmp"]


def test_export_audio_segments_uses_wave_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    source = tmp_path / "source.wav"
    _write_wave_file(source, frame_rate=4, frame_count=180)
    image_paths = [tmp_path / "frame.png"]
    image_paths[0].write_bytes(b"frame")

    segments = core.plan_segments(45.0, 20, 24.0, image_paths, 1234)

    monkeypatch.setattr(core, "_ffmpeg_executable", lambda: None)
    exported = core.export_audio_segments(source, segments, tmp_path / "chunks", overwrite=True)

    assert len(exported) == 3
    durations = []
    for segment in exported:
        chunk_path = Path(segment.audio_chunk_path or "")
        assert chunk_path.exists()
        with wave.open(str(chunk_path), "rb") as reader:
            durations.append(reader.getnframes() / reader.getframerate())
    assert durations == [20.0, 20.0, 5.0]


def test_probe_audio_info_wave_fallback_and_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "source.wav"
    _write_wave_file(source, frame_rate=8, frame_count=32)

    monkeypatch.setattr(core, "_ffprobe_executable", lambda: None)
    info = core.probe_audio_info(source)
    segments = core.plan_segments(info.duration_seconds, 2, 24.0, [tmp_path / "frame.png"], 99)
    manifest = core.build_manifest(
        workflow_path=tmp_path / "workflow.json",
        audio_info=info,
        frames_dir=tmp_path,
        fps=24.0,
        segment_seconds=2,
        seed=99,
        image_load_cap=0,
        skip_first_images=0,
        select_every_nth=1,
        segments=segments,
    )

    assert info.duration_seconds == 4.0
    assert info.probe == "wave"
    assert manifest["segment_count"] == 2
    assert manifest["segments"][0]["selected_image_index"] == 0
