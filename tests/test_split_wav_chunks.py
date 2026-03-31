import importlib.util
import wave
from pathlib import Path

import pytest


def _load_split_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "split_wav_chunks.py"
    spec = importlib.util.spec_from_file_location("comfyui_ltx_long_audio.split_wav_chunks", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_wave_file(path: Path, *, frame_rate: int, frame_count: int, channels: int = 1, sample_width: int = 2) -> None:
    silence_frame = b"\x00" * sample_width * channels
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(frame_rate)
        writer.writeframes(silence_frame * frame_count)


def test_split_wav_file_writes_expected_chunk_durations(tmp_path: Path):
    module = _load_split_module()
    source_path = tmp_path / "demo.wav"
    _write_wave_file(source_path, frame_rate=4, frame_count=180)

    results = module.split_wav_file(source_path, chunk_seconds=20.0)

    assert [result.frame_count for result in results] == [80, 80, 20]
    assert [result.duration_seconds for result in results] == [20.0, 20.0, 5.0]
    assert [result.path.name for result in results] == ["demo_001.wav", "demo_002.wav", "demo_003.wav"]
    for result, expected_frames in zip(results, [80, 80, 20], strict=True):
        with wave.open(str(result.path), "rb") as reader:
            assert reader.getnframes() == expected_frames
            assert reader.getframerate() == 4


def test_parse_args_uses_twenty_second_default(tmp_path: Path):
    module = _load_split_module()
    input_path = tmp_path / "input.wav"

    args = module.parse_args([str(input_path)])

    assert args.input_wav == input_path
    assert args.chunk_seconds == 20.0
    assert args.output_dir is None
    assert args.suffix_width == 3
    assert args.overwrite is False


def test_split_wav_file_rejects_non_empty_output_dir_without_overwrite(tmp_path: Path):
    module = _load_split_module()
    source_path = tmp_path / "demo.wav"
    output_dir = tmp_path / "demo_chunks"
    _write_wave_file(source_path, frame_rate=4, frame_count=100)
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        module.split_wav_file(source_path, output_dir=output_dir)


def test_split_wav_file_rejects_output_path_that_is_a_file(tmp_path: Path):
    module = _load_split_module()
    source_path = tmp_path / "demo.wav"
    output_path = tmp_path / "not-a-directory"
    _write_wave_file(source_path, frame_rate=4, frame_count=100)
    output_path.write_text("occupied", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        module.split_wav_file(source_path, output_dir=output_path)
