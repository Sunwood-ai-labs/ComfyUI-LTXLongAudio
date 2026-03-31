from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cli.ltx23_gpu_runner import (
    LTX23RuntimeConfig,
    LTX23WorkflowDefaults,
    Ltx23Assets,
    SegmentRenderRequest,
    build_segment_command,
    build_segment_commands,
    main,
)


def _write_wave_file(path: Path, *, frame_rate: int, frame_count: int) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(frame_rate)
        writer.writeframes(b"\x00\x00" * frame_count)


def test_build_segment_command_includes_audio_and_image():
    request = SegmentRenderRequest(
        segment_index=0,
        prompt="fox singer",
        image_path="/tmp/frame.png",
        audio_path="/tmp/audio.wav",
        fps=24.0,
        frame_count=33,
        width=832,
        height=448,
        seed=1234,
        output_path="/tmp/out.mp4",
        mode="image_to_video",
    )
    assets = Ltx23Assets(
        ltx_python="python",
        checkpoint_path="/weights/checkpoint.safetensors",
        distilled_lora_path="/weights/distilled.safetensors",
        distilled_lora_strength=0.8,
        spatial_upsampler_path="/weights/upscaler.safetensors",
        gemma_root="/weights/gemma",
        negative_prompt="bad",
        num_inference_steps=30,
        extra_args=("--video-cfg-guidance-scale", "3.5"),
    )

    command = build_segment_command(request, assets, image_strength=1.0, image_crf=33)

    assert command[:3] == ["python", "-m", "ltx_pipelines.a2vid_two_stage"]
    assert "--audio-path" in command
    assert "--image" in command
    assert "--num-frames" in command
    assert "--video-cfg-guidance-scale" in command


def test_build_segment_command_omits_image_for_text_to_video():
    request = SegmentRenderRequest(
        segment_index=0,
        prompt="fox singer",
        image_path=None,
        audio_path="/tmp/audio.wav",
        fps=24.0,
        frame_count=33,
        width=832,
        height=448,
        seed=1234,
        output_path="/tmp/out.mp4",
        mode="text_to_video",
    )
    assets = Ltx23Assets(
        ltx_python="python",
        checkpoint_path="/weights/checkpoint.safetensors",
        distilled_lora_path="/weights/distilled.safetensors",
        distilled_lora_strength=0.8,
        spatial_upsampler_path="/weights/upscaler.safetensors",
        gemma_root="/weights/gemma",
        negative_prompt="bad",
    )

    command = build_segment_command(request, assets, image_strength=1.0, image_crf=33)

    assert "--audio-path" in command
    assert "--image" not in command


def test_build_segment_commands_prefers_prepared_audio_chunks(tmp_path: Path):
    conditioning_audio = tmp_path / "conditioning.wav"
    prepared_audio = tmp_path / "conditioning_audio" / "segment_001.wav"
    image_path = tmp_path / "frame.png"
    conditioning_audio.write_bytes(b"wav")
    prepared_audio.parent.mkdir(parents=True)
    prepared_audio.write_bytes(b"wav")
    image_path.write_bytes(b"png")

    defaults = LTX23WorkflowDefaults(
        workflow_path=tmp_path / "workflow.json",
        source_audio=None,
        frames_dir=None,
    )
    runtime = LTX23RuntimeConfig(output_dir=tmp_path, python_executable="python")
    segments = [
        type(
            "Segment",
            (),
            {
                "index": 0,
                "start_time": 12.0,
                "current_seconds": 0.5,
                "exact_seconds": 0.0,
                "frames": 1,
                "selected_image_path": str(image_path),
            },
        )()
    ]

    commands = build_segment_commands(
        defaults=defaults,
        runtime=runtime,
        source_audio=conditioning_audio,
        conditioning_audio=conditioning_audio,
        prompt="fox singer",
        negative_prompt="bad",
        width=256,
        height=128,
        fps=8.0,
        use_text_to_video=False,
        ltx_seed_base=420,
        segments=segments,
        prepared_audio_paths={0: prepared_audio},
    )

    assert len(commands) == 1
    assert commands[0].audio_path == str(prepared_audio)
    assert commands[0].audio_start_time == 0.0
    assert commands[0].audio_max_duration == 0.125


def test_main_writes_gpu_manifest_without_running(tmp_path: Path):
    workflow_path = tmp_path / "workflow.json"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    audio_path = tmp_path / "demo.wav"
    output_dir = tmp_path / "gpu_output"

    (frames_dir / "frame_0.png").write_bytes(b"frame")
    (frames_dir / "frame_1.png").write_bytes(b"frame")
    _write_wave_file(audio_path, frame_rate=8, frame_count=32)

    workflow_path.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": 167, "widgets_values": [str(frames_dir), 0, 0, 1]},
                    {"id": 285, "widgets_values": [8]},
                    {"id": 291, "widgets_values": [2]},
                    {"id": 372, "widgets_values": [str(audio_path), None, None]},
                    {"id": 399, "widgets_values": [99]},
                    {"id": 352, "widgets_values": ["fox singer"]},
                    {"id": 290, "widgets_values": [False]},
                    {"id": 292, "widgets_values": [832]},
                    {"id": 293, "widgets_values": [480]},
                    {"id": 140, "widgets_values": [24, 0, "demo-output", "video/h264-mp4", "yuv420p", 19, True, True, False, True]},
                ]
            }
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--workflow",
            str(workflow_path),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ]
    )

    manifest = json.loads((output_dir / "ltx23_gpu_ready_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert manifest["segment_count"] == 2
    assert manifest["text_to_video"] is False
    assert manifest["height"] == 448
    assert manifest["width"] == 832
    assert len(manifest["command_preview"]) == 2
    assert manifest["rendered_segments"] == []
