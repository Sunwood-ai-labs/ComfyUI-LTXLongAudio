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
    assert "LTXForLoopStart" in module.NODE_CLASS_MAPPINGS
    assert "LTXForLoopEnd" in module.NODE_CLASS_MAPPINGS
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


def test_sample_inputs_are_listed_and_resolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_nodes_module()

    repo_root = tmp_path / "repo"
    sample_input = repo_root / "samples" / "input"
    sample_input.mkdir(parents=True)
    (sample_input / "demo.wav").write_bytes(b"demo")
    (sample_input / "frames_pool").mkdir()

    comfy_input = tmp_path / "comfy_input"
    comfy_input.mkdir()
    (comfy_input / "root.wav").write_bytes(b"root")
    (comfy_input / "3d").mkdir()

    monkeypatch.setattr(module, "_repository_root", lambda: repo_root)
    monkeypatch.setattr(module, "_sample_input_directory", lambda: sample_input)
    monkeypatch.setattr(module, "_input_directory", lambda: str(comfy_input))
    monkeypatch.setattr(module, "folder_paths", None)

    assert module._list_input_audio_files() == ["root.wav", "samples/input/demo.wav"]
    assert module._list_input_subdirectories() == ["3d", "samples/input/frames_pool"]
    assert module._resolve_input_path("root.wav") == str(comfy_input / "root.wav")
    assert module._resolve_input_path("samples/input/demo.wav") == str(sample_input / "demo.wav")
    assert module._resolve_input_path("samples/input/frames_pool") == str(sample_input / "frames_pool")
