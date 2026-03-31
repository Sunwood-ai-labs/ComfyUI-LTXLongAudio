import importlib.util
import json
import sys
import wave
from pathlib import Path


def _load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "cli" / "origin_long_audio.py"
    spec = importlib.util.spec_from_file_location("comfyui_ltx_long_audio.cli.origin_long_audio", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_wave_file(path: Path, *, frame_rate: int, frame_count: int) -> None:
    silence_frame = b"\x00\x00"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(frame_rate)
        writer.writeframes(silence_frame * frame_count)


def test_load_workflow_defaults_reads_origin_values(tmp_path: Path):
    module = _load_cli_module()
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": 167, "widgets_values": ["frames_pool", 0, 0, 1]},
                    {"id": 285, "widgets_values": [24]},
                    {"id": 291, "widgets_values": [20]},
                    {"id": 372, "widgets_values": ["demo.wav", None, None]},
                    {"id": 399, "widgets_values": [1234]},
                ]
            }
        ),
        encoding="utf-8",
    )

    defaults = module.load_workflow_defaults(workflow_path)

    assert defaults.audio == "demo.wav"
    assert defaults.frames_dir == "frames_pool"
    assert defaults.segment_seconds == 20
    assert defaults.fps == 24.0
    assert defaults.seed == 1234


def test_plan_segments_matches_existing_node_math():
    module = _load_cli_module()

    plans = module.plan_segments(26.0, 10, 24.0)

    assert len(plans) == 3
    assert plans[2]["start_time"] == 20.0
    assert plans[2]["nominal_seconds"] == 6.0
    assert plans[2]["frames"] == 145
    assert plans[2]["exact_seconds"] == 6.0
    assert plans[2]["is_last_segment"] is True


def test_build_segment_plan_is_deterministic(tmp_path: Path):
    module = _load_cli_module()
    image_paths = []
    for index in range(3):
        path = tmp_path / f"frame_{index}.png"
        path.write_bytes(b"frame")
        image_paths.append(path)

    plans_a = module.build_segment_plan(25.0, image_paths, segment_seconds=10, fps=24.0, seed=1234)
    plans_b = module.build_segment_plan(25.0, image_paths, segment_seconds=10, fps=24.0, seed=1234)

    assert [plan.image_index for plan in plans_a] == [plan.image_index for plan in plans_b]
    assert [Path(plan.image_path).name for plan in plans_a] == [Path(plan.image_path).name for plan in plans_b]


def test_main_writes_manifest_without_comfyui(tmp_path: Path):
    module = _load_cli_module()
    workflow_path = tmp_path / "samples" / "workflows" / "workflow.json"
    workflow_path.parent.mkdir(parents=True)
    frames_dir = tmp_path / "samples" / "input" / "frames_pool"
    frames_dir.mkdir(parents=True)
    audio_path = tmp_path / "samples" / "input" / "demo.wav"
    output_dir = tmp_path / "cli_output"

    for index in range(2):
        (frames_dir / f"frame_{index}.png").write_bytes(b"frame")
    _write_wave_file(audio_path, frame_rate=4, frame_count=100)

    workflow_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": 167, "widgets_values": ["frames_pool", 0, 0, 1]},
                    {"id": 285, "widgets_values": [8]},
                    {"id": 291, "widgets_values": [10]},
                    {"id": 372, "widgets_values": ["demo.wav", None, None]},
                    {"id": 399, "widgets_values": [99]},
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = module.main(
        [
            "--workflow",
            str(workflow_path),
            "--audio",
            str(audio_path),
            "--frames-dir",
            str(frames_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    manifest = json.loads((output_dir / module.DEFAULT_MANIFEST_NAME).read_text(encoding="utf-8"))

    assert exit_code == 0
    assert manifest["segment_count"] == 3
    assert manifest["segment_seconds"] == 10
    assert manifest["fps"] == 8.0
    assert len(manifest["segments"]) == 3
    assert manifest["segments"][0]["audio_path"] is None
