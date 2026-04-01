from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cli.ltx23_gpu_ready as gpu_ready
import cli.ltx23_gpu_runner as gpu_runner


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
    assert float(first[first.index("--audio-max-duration") + 1]) == pytest.approx(241 / 24.0)


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


def test_prepared_conditioning_audio_rewrites_command_audio_path(tmp_path: Path):
    for name in ("checkpoint.safetensors", "distilled.safetensors", "upscaler.safetensors"):
        (tmp_path / name).write_bytes(b"model")
    (tmp_path / "gemma").mkdir()
    (tmp_path / "gemma" / "config.json").write_text("{}", encoding="utf-8")
    defaults = gpu_ready.LTX23WorkflowDefaults(
        workflow_path=tmp_path / "workflow.json",
        source_audio="demo.wav",
        frames_dir="frames",
        segment_seconds=1,
        fps=8.0,
        image_selection_seed=1234,
        inference_seed=420,
        prompt="demo prompt",
        negative_prompt="bad",
        use_text_to_video=False,
        use_only_vocals=False,
        width=256,
        height=128,
    )
    runtime = gpu_ready.LTX23RuntimeConfig(
        pipeline_module="ltx_pipelines.a2vid_two_stage",
        python_executable="python",
        checkpoint_path=tmp_path / "checkpoint.safetensors",
        distilled_lora_path=tmp_path / "distilled.safetensors",
        spatial_upsampler_path=tmp_path / "upscaler.safetensors",
        gemma_root=tmp_path / "gemma",
        output_dir=tmp_path / "out",
        overwrite=True,
    )
    image_path = tmp_path / "frame.png"
    image_path.write_bytes(b"img")
    prepared_audio = tmp_path / "prepared" / "segment_001.wav"
    prepared_audio.parent.mkdir()
    prepared_audio.write_bytes(b"wav")
    segments = gpu_ready.plan_segments(1.0, 1, 8.0, [image_path], 1234)

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
        prepared_audio_paths={0: prepared_audio},
    )

    first = commands[0]
    assert first.audio_path == str(prepared_audio)
    assert first.audio_start_time == 0.0
    assert float(first.command[first.command.index("--audio-max-duration") + 1]) == pytest.approx(9 / 8.0)


def test_run_writes_manifest_and_script_without_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

    real_subprocess_run = gpu_runner.subprocess.run

    def fake_subprocess_run(command, *args, **kwargs):  # type: ignore[no-untyped-def]
        if command[0] == "ffmpeg":
            output_path = Path(command[-1])
            duration_seconds = float(command[command.index("-t") + 1])
            frame_rate = 8
            frame_count = max(int(round(duration_seconds * frame_rate)), 1)
            silence_frame = b"\x00\x00\x00\x00"
            with wave.open(str(output_path), "wb") as writer:
                writer.setnchannels(2)
                writer.setsampwidth(2)
                writer.setframerate(frame_rate)
                writer.writeframes(silence_frame * frame_count)
            return None
        return real_subprocess_run(command, *args, **kwargs)

    monkeypatch.setattr(gpu_runner, "_ffmpeg_executable", lambda: "ffmpeg")
    monkeypatch.setattr(gpu_runner.subprocess, "run", fake_subprocess_run)

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
    conditioning_chunk = tmp_path / "out" / "conditioning_audio" / "segment_001.wav"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert manifest_path.exists()
    assert script_path.exists()
    assert conditioning_chunk.exists()
    assert payload["segment_commands"]
    assert payload["segment_commands"][0]["command"][2] == "ltx_pipelines.a2vid_two_stage"
    script_text = script_path.read_text(encoding="utf-8")
    assert "trim=duration=25.000000,setpts=PTS-STARTPTS" in script_text
    assert "atrim=duration=25.000000,asetpts=PTS-STARTPTS" in script_text
    with wave.open(str(conditioning_chunk), "rb") as reader:
        assert reader.getnchannels() == 2


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


def test_run_uses_single_in_process_pipeline_for_multiple_segments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
    (frames_dir / "frame_a.png").write_bytes(b"img")
    (frames_dir / "frame_b.png").write_bytes(b"img")
    for name in ("checkpoint.safetensors", "distilled.safetensors", "upscaler.safetensors"):
        (tmp_path / name).write_bytes(b"model")
    gemma_dir = tmp_path / "gemma"
    gemma_dir.mkdir()
    (gemma_dir / "config.json").write_text("{}", encoding="utf-8")
    (gemma_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    init_calls: list[dict[str, object]] = []
    render_calls: list[dict[str, object]] = []
    encoded_outputs: list[Path] = []

    class FakeParser:
        def add_argument(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            return None

        def parse_args(self, args: list[str]) -> SimpleNamespace:
            values = list(args)

            def _value(flag: str, default: object = None) -> object:
                return values[values.index(flag) + 1] if flag in values else default

            images: list[tuple[str, int, float, int]] = []
            if "--image" in values:
                idx = values.index("--image")
                images.append((values[idx + 1], int(values[idx + 2]), float(values[idx + 3]), int(values[idx + 4])))
            return SimpleNamespace(
                checkpoint_path=_value("--checkpoint-path"),
                distilled_lora=[(_value("--distilled-lora"), float(values[values.index("--distilled-lora") + 2]), "map")],
                spatial_upsampler_path=_value("--spatial-upsampler-path"),
                gemma_root=_value("--gemma-root"),
                lora=[],
                quantization=None,
                compile=False,
                prompt=_value("--prompt"),
                negative_prompt=_value("--negative-prompt"),
                seed=int(_value("--seed", 0)),
                height=int(_value("--height", 0)),
                width=int(_value("--width", 0)),
                num_frames=int(_value("--num-frames", 0)),
                frame_rate=float(_value("--frame-rate", 0)),
                num_inference_steps=int(_value("--num-inference-steps", 8)),
                images=images,
                output_path=str(_value("--output-path")),
                audio_path=str(_value("--audio-path")),
                audio_start_time=float(_value("--audio-start-time", 0.0)),
                audio_max_duration=float(_value("--audio-max-duration", 0.0)),
                enhance_prompt="--enhance-prompt" in values,
                streaming_prefetch_count=int(_value("--streaming-prefetch-count", 1)),
                max_batch_size=int(_value("--max-batch-size", 1)),
                video_cfg_guidance_scale=float(_value("--video-cfg-guidance-scale", 3.5)),
                video_stg_guidance_scale=0.0,
                video_rescale_scale=0.0,
                a2v_guidance_scale=1.0,
                video_skip_step=0,
                video_stg_blocks=[],
            )

    class FakePipeline:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)

        def __call__(self, **kwargs):
            render_calls.append(kwargs)
            return ("video", "audio")

    class FakeTilingConfig:
        @staticmethod
        def default() -> str:
            return "tiling"

    def fake_encode_video(*, video, fps, audio, output_path, video_chunks_number):  # type: ignore[no-untyped-def]
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp4")
        encoded_outputs.append(path)

    def fake_prepare_chunks(source_audio, segments, *, fps, output_dir, overwrite):  # type: ignore[no-untyped-def]
        prepared: dict[int, Path] = {}
        for segment in segments:
            path = output_dir / "conditioning_audio" / f"segment_{int(segment.index) + 1:03d}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"wav")
            prepared[int(segment.index)] = path
        return prepared

    def fake_concat(segment_paths, output_path, *, overwrite):  # type: ignore[no-untyped-def]
        output_path.write_bytes(b"video")
        return output_path

    def fake_mux(video_path, audio_path, output_path, *, overwrite):  # type: ignore[no-untyped-def]
        output_path.write_bytes(b"final")
        return output_path

    monkeypatch.setattr(gpu_runner, "_prepare_conditioning_audio_chunks", fake_prepare_chunks)
    monkeypatch.setattr(
        gpu_runner,
        "_load_ltx_inference_runtime",
        lambda ltx_repo_root: gpu_runner.LtxInProcessRuntime(
            parser_factory=FakeParser,
            pipeline_class=FakePipeline,
            multi_modal_guider_params=lambda **kwargs: kwargs,
            tiling_config_class=FakeTilingConfig,
            get_video_chunks_number=lambda num_frames, tiling_config: 1,
            encode_video=fake_encode_video,
        ),
    )
    monkeypatch.setattr(gpu_runner, "_concat_video_streams", fake_concat)
    monkeypatch.setattr(gpu_runner, "_mux_original_audio", fake_mux)

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
            str(gemma_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--run",
            "--overwrite",
        ]
    )

    manifest = json.loads((tmp_path / "out" / "ltx23_gpu_ready_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(init_calls) == 1
    assert len(render_calls) == 3
    assert len(encoded_outputs) == 3
    assert Path(manifest["final_video"]).exists()
    assert manifest["notes"][0] == "Backend target: official LTX-2 a2vid two-stage pipeline in-process."
