import importlib.util
from pathlib import Path


def _load_api_smoke_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_comfyui_api_smoke.py"
    spec = importlib.util.spec_from_file_location("comfyui_ltx_long_audio.api_smoke", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_default_workflow_points_to_repo_sample():
    module = _load_api_smoke_module()

    expected_workflow = module.CUSTOM_NODE_REPO / "samples" / "workflows" / "LTXLongAudio_CustomNodes_SmokeTest.json"
    assert module.DEFAULT_WORKFLOW == expected_workflow
    assert module.DEFAULT_WORKFLOW.exists()
