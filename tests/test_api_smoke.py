import argparse
import importlib.util
from pathlib import Path

import pytest


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


def test_prompt_widget_value_wraps_lists():
    module = _load_api_smoke_module()

    assert module.prompt_widget_value(["a.png", "b.png"]) == {"__value__": ["a.png", "b.png"]}
    assert module.prompt_widget_value("a.png") == "a.png"


def test_validate_workflow_defaults_accepts_batch_prompt_values():
    module = _load_api_smoke_module()

    workflow_prompt = {
        "1": {
            "class_type": "LTXLoadImageBatchUpload",
            "inputs": {"image": {"__value__": ["a.png", "b.png"]}},
            "_meta": {"title": "Images Upload"},
        }
    }

    module.validate_workflow_defaults(workflow_prompt)


def test_workflow_to_prompt_wraps_array_widget_values(tmp_path: Path):
    module = _load_api_smoke_module()
    workflow_path = tmp_path / "batch_workflow.json"
    workflow_path.write_text(
        """
{
  "nodes": [
    {
      "id": 1,
      "type": "LTXLoadImageBatchUpload",
      "title": "Images Upload",
      "inputs": [
        {
          "name": "image",
          "type": "COMBO",
          "link": null
        }
      ],
      "widgets_values": [["a.png", "b.png"]]
    }
  ],
  "links": []
}
""".strip(),
        encoding="utf-8",
    )

    prompt, _selected = module.workflow_to_prompt(workflow_path, node_registry=module.load_local_node_registry())

    assert prompt["1"]["inputs"]["image"] == {"__value__": ["a.png", "b.png"]}


def test_apply_smoke_overrides_fills_batch_upload_nodes(monkeypatch, tmp_path: Path):
    module = _load_api_smoke_module()
    frame_a = tmp_path / "a.png"
    frame_b = tmp_path / "b.png"
    frame_a.write_bytes(b"a")
    frame_b.write_bytes(b"b")

    monkeypatch.setattr(module, "choose_sample_frame_files", lambda: [frame_a, frame_b])
    monkeypatch.setattr(module, "upload_input_file", lambda _base_url, file_path: file_path.name)

    workflow_prompt = {
        "1": {
            "class_type": "LTXLoadImageBatchUpload",
            "inputs": {"image": ""},
            "_meta": {"title": "Images Upload"},
        }
    }

    upload_info = module.apply_smoke_overrides("http://127.0.0.1:9999", workflow_prompt)

    assert workflow_prompt["1"]["inputs"]["image"] == {"__value__": ["a.png", "b.png"]}
    assert upload_info["uploaded_frames"] == ["a.png", "b.png"]


def test_summarize_previews_accepts_preview_image_outputs():
    module = _load_api_smoke_module()
    history_entry = {
        "outputs": {
            "2": {
                "images": [{"filename": "preview_00001.png", "type": "output", "subfolder": ""}],
                "text": [],
            }
        }
    }
    workflow_prompt = {
        "2": {
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview Uploaded Images"},
            "inputs": {},
        }
    }

    previews = module.summarize_previews(history_entry, workflow_prompt)

    assert previews == [
        {
            "node_id": "2",
            "title": "Preview Uploaded Images",
            "class_type": "PreviewImage",
            "has_images": True,
            "animated": False,
            "text": [],
        }
    ]


def test_apply_smoke_overrides_fills_batch_image_upload(monkeypatch):
    module = _load_api_smoke_module()

    workflow_prompt = {
        "1": {
            "class_type": "LTXLoadImageBatchUpload",
            "inputs": {"image": ""},
            "_meta": {"title": "Images Upload"},
        }
    }

    monkeypatch.setattr(module, "choose_sample_frame_files", lambda: [Path("a.png"), Path("b.png")])
    monkeypatch.setattr(module, "upload_input_file", lambda base_url, file_path: file_path.name)

    upload_info = module.apply_smoke_overrides("http://127.0.0.1:8188", workflow_prompt)

    assert workflow_prompt["1"]["inputs"]["image"] == {"__value__": ["a.png", "b.png"]}
    assert upload_info["uploaded_frames"] == ["a.png", "b.png"]


def test_validate_workflow_defaults_rejects_blank_batch_image_upload():
    module = _load_api_smoke_module()

    workflow_prompt = {
        "1": {
            "class_type": "LTXLoadImageBatchUpload",
            "inputs": {"image": ""},
            "_meta": {"title": "Images Upload"},
        }
    }

    with pytest.raises(RuntimeError, match="missing default 'image'"):
        module.validate_workflow_defaults(workflow_prompt)


def test_summarize_previews_accepts_preview_image_output():
    module = _load_api_smoke_module()

    history_entry = {
        "outputs": {
            "2": {
                "images": [{"filename": "preview.png"}],
                "text": [],
            }
        }
    }
    workflow_prompt = {
        "2": {
            "class_type": "PreviewImage",
            "_meta": {"title": "Preview Uploaded Images"},
        }
    }

    summaries = module.summarize_previews(history_entry, workflow_prompt)

    assert summaries == [
        {
            "node_id": "2",
            "title": "Preview Uploaded Images",
            "class_type": "PreviewImage",
            "has_images": True,
            "animated": False,
            "text": [],
        }
    ]
