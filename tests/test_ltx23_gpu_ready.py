from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cli.ltx23_gpu_ready as gpu_ready


def _write_wave_file(path: Path, *, frame_rate: int, frame_count: int) -> None:
    silence_frame = b"\x00\x00"
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(frame_rate)
        writer.writeframes(silence_frame * frame_count)


def test_load_ltx23_workflow_defaults_reads_sample_values():
    defaults = gpu_ready.load_ltx23_workflow_defaults(
        REPO_ROOT / "samples" / "workflows" / "LTX_2.3_Image_or_Text_&_Audio_2_Video_App_Origin.json"
    )

    assert defaults.segment_seconds == 20
    assert defaults.fps == 24.0
    assert defaults.image_selection_seed == 1234
    assert defaults.inference_seed == 420
    assert defaults.use_text_to_video is False
    assert defaults.use_only_vocals is True
    assert defaults.width == 832
    assert defaults.height == 480
    assert "fox-girl singer" in defaults.prompt


def test_build_segment_commands_for_image_conditioned_a2vid(tmp_path: Path):
    for name in ("checkpoint.safetensors", "distilled.safetensors", "upscaler.safetensors"):
        (tmp_path / name).write_bytes(b"model")
    (tmp_path / "gemma").mkdir()
    (tmp_path / "gemma" / "config.json").write_text("{}", encoding="utf-8")
    defaults = gpu_ready.LTX23WorkflowDefaults(
        workflow_path=tmp_path / "workflow.json",
        source_audio="demo.wav",
        frames_dir="frames",
        segment_seconds=20,
        fps=24.0,
        image_selection_seed=1234,
        inference_seed=420,
        prompt="demo prompt",
        negative_prompt="bad",
        use_text_to_video=False,
        use_only_vocals=False,
        width=832,
        height=480,
    )
    runtime = gpu_ready.LTX23RuntimeConfig(
        pipeline_module="ltx_pipelines.a2vid_two_stage",
        python_executable="python",
        ltx_repo_root=None,
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        distilled_lora_path=tmp_path / "distilled.safetensors",
        distilled_lora_strength=0.6,
        spatial_upsampler_path=tmp_path / "upscaler.safetensors",
        gemma_root=tmp_path / "gemma",
        output_dir=tmp_path / "out",
        num_inference_steps=8,
        image_strength=1.0,
        quantization="fp8-cast",
        enhance_prompt=True,
        run_commands=False,
        emit_run_script=False,
        overwrite=True,
    )
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"img")
    segments = gpu_ready.plan_segments(26.0, 10, 24.0, [image_path], 1234)

    commands = gpu_ready.build_segment_commands(
        defaults=defaults,
        runtime=runtime,
        source_audio=tmp_path / "source.wav",
        conditioning_audio=tmp_path / "conditioning.wav",
        prompt=defaults.prompt,
        negative_prompt=defaults.negative_prompt,
        width=defaults.width,
        height=defaults.height,
        fps=defaults.fps,
        use_text_to_video=False,
        ltx_seed_base=defaults.inference_seed,
        segments=segments,
    )

    first = commands[0].command
    assert first[:3] == ["python", "-m", "ltx_pipelines.a2vid_two_stage"]
    assert "--audio-path" in first
    assert "--audio-start-time" in first
    assert "--audio-max-duration" in first
    assert "--image" in first
    assert "--quantization" in first
    assert "--enhance-prompt" in first
    assert first[first.index("--num-frames") + 1] == "241"
    assert first[first.index("--seed") + 1] == "420"


def test_text_to_video_mode_omits_image_conditioning(tmp_path: Path):
    for name in ("checkpoint.safetensors", "distilled.safetensors", "upscaler.safetensors"):
        (tmp_path / name).write_bytes(b"model")
    (tmp_path / "gemma").mkdir()
    (tmp_path / "gemma" / "config.json").write_text("{}", encoding="utf-8")
    defaults = gpu_ready.LTX23WorkflowDefaults(
        workflow_path=tmp_path / "workflow.json",
        source_audio="demo.wav",
        frames_dir="frames",
        segment_seconds=20,
        fps=24.0,
        image_selection_seed=1234,
        inference_seed=420,
        prompt="demo prompt",
        negative_prompt="bad",
        use_text_to_video=True,
        use_only_vocals=False,
        width=832,
        height=480,
    )
    runtime = gpu_ready.LTX23RuntimeConfig(
        pipeline_module="ltx_pipelines.a2vid_two_stage",
        python_executable="python",
        ltx_repo_root=None,
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        distilled_lora_path=tmp_path / "distilled.safetensors",
        distilled_lora_strength=0.6,
        spatial_upsampler_path=tmp_path / "upscaler.safetensors",
        gemma_root=tmp_path / "gemma",
        output_dir=tmp_path / "out",
        num_inference_steps=8,
        image_strength=1.0,
        quantization=None,
        enhance_prompt=False,
        run_commands=False,
        emit_run_script=False,
        overwrite=True,
    )
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"img")
    segments = gpu_ready.plan_segments(10.0, 10, 24.0, [image_path], 1234)

    commands = gpu_ready.build_segment_commands(
        defaults=defaults,
        runtime=runtime,
        source_audio=tmp_path / "source.wav",
        conditioning_audio=tmp_path / "conditioning.wav",
        prompt=defaults.prompt,
        negative_prompt=defaults.negative_prompt,
        width=defaults.width,
        height=defaults.height,
        fps=defaults.fps,
        use_text_to_video=True,
        ltx_seed_base=defaults.inference_seed,
        segments=segments,
    )

    assert "--image" not in commands[0].command


def test_run_writes_manifest_and_script_without_running(tmp_path: Path):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"title": "Source Song Upload", "widgets_values": ["demo.wav", None, None]},
                    {"title": "Segment Image Folder", "widgets_values": ["frames", 0, 0, 1]},
                    {"title": "SEGMENT SECONDS", "widgets_values": [10]},
                    {"title": "FPS", "widgets_values": [8]},
                    {"title": "RANDOM IMAGE SEED", "widgets_values": [99]},
                    {"title": "PROMPT", "widgets_values": ["demo prompt"]},
                    {"title": "Text To Video (no image ref)", "widgets_values": [False]},
                    {"title": "USE ONLY VOCALS", "widgets_values": [False]},
                    {"title": "WIDTH", "widgets_values": [832]},
                    {"title": "HEIGHT", "widgets_values": [480]},
                    {"id": 110, "type": "CLIPTextEncode", "widgets_values": ["bad"]},
                    {"id": 114, "type": "RandomNoise", "widgets_values": [420, "fixed"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    audio_path = tmp_path / "demo.wav"
    _write_wave_file(audio_path, frame_rate=4, frame_count=100)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame.png").write_bytes(b"img")
    for name in ("checkpoint.safetensors", "distilled.safetensors", "upscaler.safetensors"):
        (tmp_path / name).write_bytes(b"model")
    (tmp_path / "gemma").mkdir()
    (tmp_path / "gemma" / "config.json").write_text("{}", encoding="utf-8")

    exit_code = gpu_ready.run(
        [
            "--workflow",
            str(workflow_path),
            "--audio",
            str(audio_path),
            "--frames-dir",
            str(frames_dir),
            "--checkpoint-path",
            str(tmp_path / "checkpoint.safetensors"),
            "--distilled-lora-path",
            str(tmp_path / "distilled.safetensors"),
            "--spatial-upsampler-path",
            str(tmp_path / "upscaler.safetensors"),
            "--gemma-root",
            str(tmp_path / "gemma"),
            "--output-dir",
            str(tmp_path / "out"),
            "--emit-run-script",
            "--overwrite",
        ]
    )

    manifest_path = tmp_path / "out" / "ltx23_gpu_ready_manifest.json"
    script_path = tmp_path / "out" / "run_segments.sh"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert manifest_path.exists()
    assert script_path.exists()
    assert payload["segment_commands"]
    assert payload["segment_commands"][0]["command"][2] == "ltx_pipelines.a2vid_two_stage"


def test_run_requires_real_gemma_snapshot_for_script_generation(tmp_path: Path):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"title": "Source Song Upload", "widgets_values": ["demo.wav", None, None]},
                    {"title": "Segment Image Folder", "widgets_values": ["frames", 0, 0, 1]},
                    {"title": "SEGMENT SECONDS", "widgets_values": [10]},
                    {"title": "FPS", "widgets_values": [8]},
                    {"title": "RANDOM IMAGE SEED", "widgets_values": [99]},
                    {"title": "PROMPT", "widgets_values": ["demo prompt"]},
                    {"title": "Text To Video (no image ref)", "widgets_values": [False]},
                    {"title": "USE ONLY VOCALS", "widgets_values": [False]},
                    {"title": "WIDTH", "widgets_values": [832]},
                    {"title": "HEIGHT", "widgets_values": [480]},
                    {"id": 110, "type": "CLIPTextEncode", "widgets_values": ["bad"]},
                    {"id": 114, "type": "RandomNoise", "widgets_values": [420, "fixed"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    audio_path = tmp_path / "demo.wav"
    _write_wave_file(audio_path, frame_rate=4, frame_count=100)
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame.png").write_bytes(b"img")
    for name in ("checkpoint.safetensors", "distilled.safetensors", "upscaler.safetensors"):
        (tmp_path / name).write_bytes(b"model")
    (tmp_path / "gemma").mkdir()

    with pytest.raises(FileNotFoundError):
        gpu_ready.run(
            [
                "--workflow",
                str(workflow_path),
                "--audio",
                str(audio_path),
                "--frames-dir",
                str(frames_dir),
                "--checkpoint-path",
                str(tmp_path / "checkpoint.safetensors"),
                "--distilled-lora-path",
                str(tmp_path / "distilled.safetensors"),
                "--spatial-upsampler-path",
                str(tmp_path / "upscaler.safetensors"),
                "--gemma-root",
                str(tmp_path / "gemma"),
                "--output-dir",
                str(tmp_path / "out"),
                "--emit-run-script",
                "--overwrite",
            ]
        )
