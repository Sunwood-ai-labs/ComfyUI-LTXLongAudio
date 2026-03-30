import importlib.util
from pathlib import Path

import pytest


def _load_nodes_module():
    module_path = Path(__file__).resolve().parents[1] / "nodes.py"
    spec = importlib.util.spec_from_file_location("comfyui_ltx_long_audio.nodes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_node_module_exports():
    module = _load_nodes_module()

    assert "LTXLongAudioSegmentInfo" in module.NODE_CLASS_MAPPINGS
    assert "LTXRandomImageIndex" in module.NODE_CLASS_MAPPINGS
    assert "LTXLoadImageUpload" in module.NODE_CLASS_MAPPINGS
    assert "LTXBatchUploadedFrames" in module.NODE_CLASS_MAPPINGS
    assert "LTXForLoopStart" in module.NODE_CLASS_MAPPINGS
    assert "LTXForLoopEnd" in module.NODE_CLASS_MAPPINGS
    assert "LTXAudioSlice" in module.NODE_CLASS_MAPPINGS
    assert "LTXVideoCombine" in module.NODE_CLASS_MAPPINGS
    assert "LTXSimpleCalculator" in module.NODE_CLASS_MAPPINGS
    for legacy in ("easy forLoopStart", "VHS_VideoCombine", "SimpleCalculatorKJ"):
        assert legacy not in module.NODE_CLASS_MAPPINGS


def test_pure_node_behaviors():
    module = _load_nodes_module()

    segment = module.LTXLongAudioSegmentInfo()
    start_time, segment_seconds, frames, exact_seconds, is_last, segment_count = segment.segment_info(26.0, 10, 2, 24.0)
    assert start_time == 20.0
    assert segment_seconds == 6.0
    assert frames == 145
    assert exact_seconds == 6.0
    assert is_last is True
    assert segment_count == 3

    simple_math = module.NODE_CLASS_MAPPINGS["LTXSimpleMath"]()
    assert simple_math.execute("ceil(a / b)", a=21, b=10) == (3, 3.0, True)

    compare = module.NODE_CLASS_MAPPINGS["LTXCompare"]()
    assert compare.compare(5, 10, "a < b") == (True,)

    formula = module.NODE_CLASS_MAPPINGS["LTXSimpleCalculator"]()
    assert formula.calculate("a + b", a=2, b=3) == (5.0, 5, True)

    if module.torch is not None:
        frame_batch = module.NODE_CLASS_MAPPINGS["LTXBatchUploadedFrames"]()
        image_a = module.torch.zeros((1, 4, 4, 3), dtype=module.torch.float32)
        image_b = module.torch.ones((1, 4, 4, 3), dtype=module.torch.float32)
        batched_images, frame_count = frame_batch.batch_images(image_a, image_2=image_b)
        assert batched_images.shape == (2, 4, 4, 3)
        assert frame_count == 2

        audio_slice = module.NODE_CLASS_MAPPINGS["LTXAudioSlice"]()
        sample_rate = 8
        waveform = module.torch.arange(0, 32, dtype=module.torch.float32).reshape(1, 1, 32)
        sliced_audio, sliced_duration = audio_slice.slice_audio(
            {"waveform": waveform, "sample_rate": sample_rate},
            start_time=1.0,
            duration=1.5,
        )
        assert sliced_audio["waveform"].shape[-1] == 12
        assert sliced_duration == 1.5


def test_sample_inputs_are_listed_and_resolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_nodes_module()

    repo_root = tmp_path / "repo"
    sample_input = repo_root / "samples" / "input"
    sample_input.mkdir(parents=True)
    (sample_input / "demo.wav").write_bytes(b"demo")
    (sample_input / "demo.png").write_bytes(b"demo-image")
    (sample_input / "frames_pool").mkdir()

    comfy_input = tmp_path / "comfy_input"
    comfy_input.mkdir()
    (comfy_input / "root.wav").write_bytes(b"root")
    (comfy_input / "root.png").write_bytes(b"root-image")
    (comfy_input / "3d").mkdir()

    monkeypatch.setattr(module, "_repository_root", lambda: repo_root)
    monkeypatch.setattr(module, "_sample_input_directory", lambda: sample_input)
    monkeypatch.setattr(module, "_input_directory", lambda: str(comfy_input))
    monkeypatch.setattr(module, "folder_paths", None)

    assert module._list_input_audio_files() == ["root.wav", "samples/input/demo.wav"]
    assert module._list_input_image_files() == ["root.png", "samples/input/demo.png"]
    assert module._list_input_subdirectories() == ["3d", "samples/input/frames_pool"]
    assert module._resolve_input_path("root.wav") == str(comfy_input / "root.wav")
    assert module._resolve_input_path("samples/input/demo.wav") == str(sample_input / "demo.wav")
    assert module._resolve_input_path("samples/input/frames_pool") == str(sample_input / "frames_pool")


def test_upload_inputs_include_blank_and_upload_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_nodes_module()

    comfy_input = tmp_path / "comfy_input"
    comfy_input.mkdir()
    (comfy_input / "root.wav").write_bytes(b"root")
    (comfy_input / "root.png").write_bytes(b"root-image")

    monkeypatch.setattr(module, "_input_directory", lambda: str(comfy_input))
    monkeypatch.setattr(module, "_sample_input_directory", lambda: tmp_path / "missing_samples")
    monkeypatch.setattr(module, "folder_paths", None)

    audio_spec = module.NODE_CLASS_MAPPINGS["LTXLoadAudioUpload"].INPUT_TYPES()["required"]["audio"]
    image_spec = module.NODE_CLASS_MAPPINGS["LTXLoadImageUpload"].INPUT_TYPES()["required"]["image"]

    assert audio_spec[0][0] == ""
    assert audio_spec[1]["audio_upload"] is True
    assert "root.wav" in audio_spec[0]

    assert image_spec[0][0] == ""
    assert image_spec[1]["image_upload"] is True
    assert "root.png" in image_spec[0]


def test_ffmpeg_executable_uses_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_nodes_module()

    ffmpeg_path = tmp_path / "ffmpeg-custom.exe"
    ffmpeg_path.write_bytes(b"")

    monkeypatch.setenv("LTX_FFMPEG_EXE", str(ffmpeg_path))
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    assert module._ffmpeg_executable() == str(ffmpeg_path)
