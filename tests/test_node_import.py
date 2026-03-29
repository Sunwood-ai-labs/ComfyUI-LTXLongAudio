import importlib.util
from pathlib import Path


def test_node_module_exports():
    module_path = Path(__file__).resolve().parents[1] / "nodes.py"
    spec = importlib.util.spec_from_file_location("comfyui_ltx_long_audio.nodes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert "LTXLongAudioSegmentInfo" in module.NODE_CLASS_MAPPINGS
    assert "LTXRandomImageIndex" in module.NODE_CLASS_MAPPINGS
