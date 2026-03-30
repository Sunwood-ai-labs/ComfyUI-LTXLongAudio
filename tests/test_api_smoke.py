import argparse
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


def test_resolve_runtime_args_derives_paths_from_comfy_root(tmp_path: Path):
    module = _load_api_smoke_module()

    comfy_root = tmp_path / "ComfyUI"
    (comfy_root / ".venv" / "Scripts").mkdir(parents=True)
    (comfy_root / "main.py").write_text("", encoding="utf-8")
    (comfy_root / ".venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")

    args = argparse.Namespace(
        workflow=module.DEFAULT_WORKFLOW,
        comfy_root=comfy_root,
        comfy_main=None,
        python_exe=None,
        user_directory=None,
        input_directory=None,
        output_directory=None,
        temp_directory=None,
        database_url=None,
        startup_timeout=60.0,
        execution_timeout=120.0,
        port=0,
        ffmpeg_exe=None,
        auto_fill_missing=False,
    )

    resolved = module.resolve_runtime_args(args)

    assert resolved.comfy_main == comfy_root / "main.py"
    assert resolved.python_exe == comfy_root / ".venv" / "Scripts" / "python.exe"
    assert resolved.user_directory == comfy_root / "user"
    assert resolved.input_directory == comfy_root / "input"
    assert resolved.output_directory == comfy_root / "output"
    assert resolved.temp_directory == comfy_root
    assert resolved.database_url == f"sqlite:///{(comfy_root / 'user').as_posix()}/comfyui_ci.db"


def test_stage_default_input_assets_uses_tracked_audio_name(monkeypatch, tmp_path: Path):
    module = _load_api_smoke_module()

    audio_source = tmp_path / "optional-long.wav"
    frame_source = tmp_path / "frame.png"
    audio_source.write_bytes(b"audio")
    frame_source.write_bytes(b"frame")

    monkeypatch.setattr(module, "choose_sample_audio_file", lambda: audio_source)
    monkeypatch.setattr(module, "choose_sample_frame_files", lambda: [frame_source])

    staged = module.stage_default_input_assets(tmp_path / "input")

    assert Path(staged["audio"]).name == module.TRACKED_SAMPLE_AUDIO_NAME
