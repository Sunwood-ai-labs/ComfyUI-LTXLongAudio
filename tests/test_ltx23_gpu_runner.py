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

import cli.ltx23_gpu_runner as gpu_runner
from cli.ltx23_gpu_runner import (
    DEFAULT_DISTILLED_LORA_STRENGTH,
    LTX23RuntimeConfig,
    LTX23WorkflowDefaults,
    Ltx23Assets,
    RUNTIME_BACKEND_NOTEBOOK_REFERENCED,
    SegmentRenderRequest,
    _build_timing_summary,
    build_segment_command,
    build_segment_commands,
    main,
    parse_args,
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


def test_parse_args_uses_workflow_aligned_lora_strength_default():
    args = parse_args([])

    assert args.distilled_lora_strength == pytest.approx(DEFAULT_DISTILLED_LORA_STRENGTH)


def test_parse_args_defaults_to_official_runtime_backend():
    args = parse_args([])

    assert args.runtime_backend == "official-python"


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


def test_main_writes_debug_log_without_running(tmp_path: Path):
    workflow_path = tmp_path / "workflow.json"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    audio_path = tmp_path / "demo.wav"
    output_dir = tmp_path / "gpu_output"

    (frames_dir / "frame_0.png").write_bytes(b"frame")
    _write_wave_file(audio_path, frame_rate=8, frame_count=16)

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
            "--debug",
        ]
    )

    manifest = json.loads((output_dir / "ltx23_gpu_ready_manifest.json").read_text(encoding="utf-8"))
    debug_log_path = output_dir / "ltx23_debug.jsonl"
    log_events = [json.loads(line)["event"] for line in debug_log_path.read_text(encoding="utf-8").splitlines() if line]

    assert exit_code == 0
    assert manifest["debug_log"] == str(debug_log_path.resolve())
    assert log_events[:2] == ["run_start", "segments_planned"]
    assert "manifest_written" in log_events


def test_main_writes_notebook_referenced_manifest_without_running(tmp_path: Path):
    workflow_path = tmp_path / "workflow.json"
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    audio_path = tmp_path / "demo.wav"
    output_dir = tmp_path / "gpu_output"
    notebook_assets = {
        "gemma": tmp_path / "gemma_fp8.safetensors",
        "connectors": tmp_path / "connectors.safetensors",
        "video_vae": tmp_path / "video_vae.safetensors",
        "audio_vae": tmp_path / "audio_vae.safetensors",
    }

    (frames_dir / "frame_0.png").write_bytes(b"frame")
    _write_wave_file(audio_path, frame_rate=8, frame_count=16)
    for path in notebook_assets.values():
        path.write_bytes(b"model")

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
            "--runtime-backend",
            RUNTIME_BACKEND_NOTEBOOK_REFERENCED,
            "--notebook-gemma-fp8-path",
            str(notebook_assets["gemma"]),
            "--notebook-embeddings-connectors-path",
            str(notebook_assets["connectors"]),
            "--notebook-video-vae-path",
            str(notebook_assets["video_vae"]),
            "--notebook-audio-vae-path",
            str(notebook_assets["audio_vae"]),
        ]
    )

    manifest = json.loads((output_dir / "ltx23_gpu_ready_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert manifest["runtime_backend"] == RUNTIME_BACKEND_NOTEBOOK_REFERENCED
    assert manifest["notebook_referenced_assets"] == {
        "gemma_fp8_path": str(notebook_assets["gemma"].resolve()),
        "embeddings_connectors_path": str(notebook_assets["connectors"].resolve()),
        "video_vae_path": str(notebook_assets["video_vae"].resolve()),
        "audio_vae_path": str(notebook_assets["audio_vae"].resolve()),
        "unet_gguf_path": None,
        "mmproj_gguf_path": None,
        "melband_path": None,
        "tae_vae_path": None,
    }
    assert any("notebook-referenced split-asset pipeline" in note for note in manifest["notes"])


def test_mux_original_audio_trims_to_source_duration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured: list[str] = []
    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.wav"
    output_path = tmp_path / "final.mp4"
    video_path.write_bytes(b"video")
    _write_wave_file(audio_path, frame_rate=10, frame_count=123)

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        captured.extend(command)
        return None

    monkeypatch.setattr(gpu_runner, "_ffmpeg_executable", lambda: "ffmpeg")
    monkeypatch.setattr(gpu_runner.subprocess, "run", fake_run)

    result = gpu_runner._mux_original_audio(video_path, audio_path, output_path, overwrite=True)

    assert result == output_path
    assert "-vf" in captured
    assert captured[captured.index("-vf") + 1] == "trim=duration=12.300000,setpts=PTS-STARTPTS"
    assert "-af" in captured
    assert captured[captured.index("-af") + 1] == "atrim=duration=12.300000,asetpts=PTS-STARTPTS"
    assert captured[captured.index("-c:v") + 1] == "libx264"


def test_build_in_process_pipeline_can_override_prompt_encoder_device(monkeypatch: pytest.MonkeyPatch):
    init_calls: list[dict[str, object]] = []
    call_kwargs: list[dict[str, object]] = []

    class FakeTensor:
        def __init__(self, label: str):
            self.label = label

        def to(self, device):  # type: ignore[no-untyped-def]
            return f"{self.label}@{device}"

    class FakeOutput(tuple):
        def __new__(cls, video_encoding, audio_encoding, attention_mask):  # type: ignore[no-untyped-def]
            return super().__new__(cls, (video_encoding, audio_encoding, attention_mask))

        video_encoding = property(lambda self: self[0])
        audio_encoding = property(lambda self: self[1])
        attention_mask = property(lambda self: self[2])

    class FakeTorchModule:
        float32 = "float32"

        @staticmethod
        def device(raw: str) -> str:
            return str(raw)

    class DummyPromptEncoder:
        def __init__(self, *, checkpoint_path, gemma_root, dtype, device):  # type: ignore[no-untyped-def]
            init_calls.append(
                {
                    "checkpoint_path": checkpoint_path,
                    "gemma_root": gemma_root,
                    "dtype": dtype,
                    "device": str(device),
                }
            )

        def __call__(self, prompts, **kwargs):  # type: ignore[no-untyped-def]
            call_kwargs.append(dict(kwargs))
            return [FakeOutput(FakeTensor("video"), FakeTensor("audio"), FakeTensor("mask"))]

    class DummyPipeline:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs
            self.device = "cuda:0"
            self.dtype = FakeTorchModule.float32
            self.prompt_encoder = "original"

    original_import_module = gpu_runner.importlib.import_module

    def fake_import_module(name: str):  # type: ignore[no-untyped-def]
        if name == "torch":
            return FakeTorchModule
        return original_import_module(name)

    monkeypatch.setattr(gpu_runner.importlib, "import_module", fake_import_module)

    runtime = gpu_runner.LtxInProcessRuntime(
        parser_factory=lambda: None,
        pipeline_class=DummyPipeline,
        prompt_encoder_class=DummyPromptEncoder,
        multi_modal_guider_params=lambda **kwargs: kwargs,
        tiling_config_class=SimpleNamespace(default=lambda: None),
        get_video_chunks_number=lambda num_frames, tiling_config: 1,
        encode_video=lambda **kwargs: None,
    )
    namespace = SimpleNamespace(
        checkpoint_path="/weights/checkpoint.safetensors",
        distilled_lora=(),
        spatial_upsampler_path="/weights/upscaler.safetensors",
        gemma_root="/weights/gemma",
        lora=(),
        quantization=None,
        compile=False,
    )
    config = LTX23RuntimeConfig(prompt_encoder_device="cpu")

    pipeline = gpu_runner._build_in_process_pipeline(runtime, namespace, config)

    assert isinstance(pipeline.prompt_encoder, gpu_runner._PromptEncoderOutputDeviceAdapter)
    assert init_calls == [
        {
            "checkpoint_path": "/weights/checkpoint.safetensors",
            "gemma_root": "/weights/gemma",
            "dtype": pipeline.dtype,
            "device": "cpu",
        }
    ]
    outputs = pipeline.prompt_encoder(["hello"], streaming_prefetch_count=3)
    assert outputs[0].video_encoding == "video@cuda:0"
    assert outputs[0].audio_encoding == "audio@cuda:0"
    assert outputs[0].attention_mask == "mask@cuda:0"
    assert call_kwargs == [{"streaming_prefetch_count": None}]


def test_build_in_process_pipeline_can_override_prompt_streaming(monkeypatch: pytest.MonkeyPatch):
    call_kwargs: list[dict[str, object]] = []

    class FakeTensor:
        def __init__(self, label: str):
            self.label = label

        def to(self, device):  # type: ignore[no-untyped-def]
            return f"{self.label}@{device}"

    class FakeOutput(tuple):
        def __new__(cls, video_encoding, audio_encoding, attention_mask):  # type: ignore[no-untyped-def]
            return super().__new__(cls, (video_encoding, audio_encoding, attention_mask))

        video_encoding = property(lambda self: self[0])
        audio_encoding = property(lambda self: self[1])
        attention_mask = property(lambda self: self[2])

    class FakeTorchModule:
        float32 = "float32"

        @staticmethod
        def device(raw: str) -> str:
            return str(raw)

    class DummyPromptEncoder:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs

        def __call__(self, prompts, **kwargs):  # type: ignore[no-untyped-def]
            call_kwargs.append(dict(kwargs))
            return [FakeOutput(FakeTensor("video"), FakeTensor("audio"), FakeTensor("mask"))]

    class DummyPipeline:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs
            self.device = "cuda:0"
            self.dtype = FakeTorchModule.float32
            self.prompt_encoder = "original"

    original_import_module = gpu_runner.importlib.import_module

    def fake_import_module(name: str):  # type: ignore[no-untyped-def]
        if name == "torch":
            return FakeTorchModule
        return original_import_module(name)

    monkeypatch.setattr(gpu_runner.importlib, "import_module", fake_import_module)

    runtime = gpu_runner.LtxInProcessRuntime(
        parser_factory=lambda: None,
        pipeline_class=DummyPipeline,
        prompt_encoder_class=DummyPromptEncoder,
        multi_modal_guider_params=lambda **kwargs: kwargs,
        tiling_config_class=SimpleNamespace(default=lambda: None),
        get_video_chunks_number=lambda num_frames, tiling_config: 1,
        encode_video=lambda **kwargs: None,
    )
    namespace = SimpleNamespace(
        checkpoint_path="/weights/checkpoint.safetensors",
        distilled_lora=(),
        spatial_upsampler_path="/weights/upscaler.safetensors",
        gemma_root="/weights/gemma",
        lora=(),
        quantization=None,
        compile=False,
    )
    config = LTX23RuntimeConfig(prompt_streaming_prefetch_count=1)

    pipeline = gpu_runner._build_in_process_pipeline(runtime, namespace, config)
    outputs = pipeline.prompt_encoder(["hello"], streaming_prefetch_count=None)

    assert isinstance(pipeline.prompt_encoder, gpu_runner._PromptEncoderOutputDeviceAdapter)
    assert outputs[0].video_encoding == "video@cuda:0"
    assert call_kwargs == [{"streaming_prefetch_count": 1}]


def test_build_in_process_pipeline_supports_notebook_referenced_backend(monkeypatch: pytest.MonkeyPatch):
    build_calls: list[dict[str, object]] = []

    class DummyPipeline:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs
            self.device = "cuda:0"
            self.dtype = "float32"
            self._prompt_encoder_device = "cpu"
            self.prompt_encoder = "custom-prompt-encoder"

    fake_notebook_backend = SimpleNamespace(
        build_pipeline=lambda **kwargs: build_calls.append(dict(kwargs)) or DummyPipeline(**kwargs),
    )

    original_import_module = gpu_runner.importlib.import_module

    def fake_import_module(name: str):  # type: ignore[no-untyped-def]
        if name == "cli.ltx23_notebook_reference_backend":
            return fake_notebook_backend
        return original_import_module(name)

    monkeypatch.setattr(gpu_runner.importlib, "import_module", fake_import_module)

    runtime = gpu_runner.LtxInProcessRuntime(
        parser_factory=lambda: None,
        pipeline_class=DummyPipeline,
        prompt_encoder_class=lambda **kwargs: None,
        multi_modal_guider_params=lambda **kwargs: kwargs,
        tiling_config_class=SimpleNamespace(default=lambda: None),
        get_video_chunks_number=lambda num_frames, tiling_config: 1,
        encode_video=lambda **kwargs: None,
    )
    namespace = SimpleNamespace(
        checkpoint_path="/weights/checkpoint.safetensors",
        distilled_lora=(),
        spatial_upsampler_path="/weights/upscaler.safetensors",
        gemma_root="/weights/gemma",
        lora=(),
        quantization=None,
        compile=False,
    )
    config = LTX23RuntimeConfig(
        runtime_backend=RUNTIME_BACKEND_NOTEBOOK_REFERENCED,
        notebook_gemma_fp8_path=Path("/weights/gemma_fp8.safetensors"),
        notebook_embeddings_connectors_path=Path("/weights/connectors.safetensors"),
        notebook_video_vae_path=Path("/weights/video_vae.safetensors"),
        notebook_audio_vae_path=Path("/weights/audio_vae.safetensors"),
        prompt_streaming_prefetch_count=1,
    )

    pipeline = gpu_runner._build_in_process_pipeline(runtime, namespace, config)

    assert build_calls == [
        {
            "checkpoint_path": "/weights/checkpoint.safetensors",
            "distilled_lora": (),
            "spatial_upsampler_path": "/weights/upscaler.safetensors",
            "text_encoder_path": str(Path("/weights/gemma_fp8.safetensors")),
            "embeddings_connectors_path": str(Path("/weights/connectors.safetensors")),
            "video_vae_path": str(Path("/weights/video_vae.safetensors")),
            "audio_vae_path": str(Path("/weights/audio_vae.safetensors")),
            "tokenizer_root": "/weights/gemma",
            "loras": (),
            "quantization": None,
            "torch_compile": False,
            "prompt_encoder_device": "match",
            "prompt_streaming_prefetch_count": 1,
        }
    ]
    assert pipeline.prompt_encoder == "custom-prompt-encoder"


def test_build_in_process_pipeline_emits_phase_level_debug_events(tmp_path: Path) -> None:
    debug_log_path = tmp_path / "ltx23_debug.jsonl"
    debug_logger = gpu_runner.RunDebugLogger(enabled=True, log_path=debug_log_path, echo=False)

    class DummyPromptEncoder:
        def __call__(self, prompts, **kwargs):  # type: ignore[no-untyped-def]
            return [(prompts, kwargs)]

    class DummyStage:
        def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
            return ("video", "audio")

    class DummyPipeline:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs
            self.device = "cpu"
            self.dtype = "float32"
            self.prompt_encoder = DummyPromptEncoder()
            self.audio_conditioner = lambda callback: callback("audio-encoder")
            self.image_conditioner = lambda callback: callback("image-encoder")
            self.stage_1 = DummyStage()
            self.upsampler = lambda latent: latent
            self.stage_2 = DummyStage()
            self.video_decoder = lambda latent, tiling_config, generator=None: latent

    runtime = gpu_runner.LtxInProcessRuntime(
        parser_factory=lambda: None,
        pipeline_class=DummyPipeline,
        prompt_encoder_class=DummyPromptEncoder,
        multi_modal_guider_params=lambda **kwargs: kwargs,
        tiling_config_class=SimpleNamespace(default=lambda: None),
        get_video_chunks_number=lambda num_frames, tiling_config: 1,
        encode_video=lambda **kwargs: None,
    )
    namespace = SimpleNamespace(
        checkpoint_path="/weights/checkpoint.safetensors",
        distilled_lora=(),
        spatial_upsampler_path="/weights/upscaler.safetensors",
        gemma_root="/weights/gemma",
        lora=(),
        quantization=None,
        compile=False,
    )
    config = LTX23RuntimeConfig()

    pipeline = gpu_runner._build_in_process_pipeline(runtime, namespace, config, debug_logger)

    pipeline.prompt_encoder(["hello", "world"], streaming_prefetch_count=1)
    pipeline.stage_1(width=448, height=256, frames=361, fps=24.0, streaming_prefetch_count=1, max_batch_size=4)
    pipeline.video_decoder("latent", None)

    events = [json.loads(line) for line in debug_log_path.read_text(encoding="utf-8").splitlines() if line]
    event_names = [item["event"] for item in events]

    assert "pipeline_phase_start" in event_names
    assert "pipeline_phase_done" in event_names

    prompt_start = next(item for item in events if item["event"] == "pipeline_phase_start" and item["phase"] == "prompt_encoder")
    stage_done = next(item for item in events if item["event"] == "pipeline_phase_done" and item["phase"] == "stage_1")
    decoder_start = next(item for item in events if item["event"] == "pipeline_phase_start" and item["phase"] == "video_decoder")

    assert prompt_start["prompt_count"] == 2
    assert prompt_start["streaming_prefetch_count"] == 1
    assert stage_done["width"] == 448
    assert stage_done["max_batch_size"] == 4
    assert "tiling_config" in decoder_start


def test_build_in_process_pipeline_applies_stage2_default_batch_size(tmp_path: Path) -> None:
    stage_2_calls: list[dict[str, object]] = []

    class DummyStage:
        def __call__(self, **kwargs):  # type: ignore[no-untyped-def]
            stage_2_calls.append(dict(kwargs))
            return ("video", "audio")

    class DummyPipeline:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            self.kwargs = kwargs
            self.device = "cpu"
            self.dtype = "float32"
            self.prompt_encoder = lambda prompts, **inner_kwargs: [(prompts, inner_kwargs)]
            self.audio_conditioner = lambda callback: callback("audio-encoder")
            self.image_conditioner = lambda callback: callback("image-encoder")
            self.stage_1 = DummyStage()
            self.upsampler = lambda latent: latent
            self.stage_2 = DummyStage()
            self.video_decoder = lambda latent, tiling_config, generator=None: latent

    debug_logger = gpu_runner.RunDebugLogger(enabled=True, log_path=tmp_path / "debug.jsonl", echo=False)
    runtime = gpu_runner.LtxInProcessRuntime(
        parser_factory=lambda: None,
        pipeline_class=DummyPipeline,
        prompt_encoder_class=lambda **kwargs: None,
        multi_modal_guider_params=lambda **kwargs: kwargs,
        tiling_config_class=SimpleNamespace(default=lambda: None),
        get_video_chunks_number=lambda num_frames, tiling_config: 1,
        encode_video=lambda **kwargs: None,
    )
    namespace = SimpleNamespace(
        checkpoint_path="/weights/checkpoint.safetensors",
        distilled_lora=(),
        spatial_upsampler_path="/weights/upscaler.safetensors",
        gemma_root="/weights/gemma",
        lora=(),
        quantization=None,
        compile=False,
        max_batch_size=4,
    )
    config = LTX23RuntimeConfig()

    pipeline = gpu_runner._build_in_process_pipeline(runtime, namespace, config, debug_logger)
    pipeline.stage_2(width=448, height=256, frames=361, fps=24.0, streaming_prefetch_count=1)

    assert stage_2_calls == [
        {
            "width": 448,
            "height": 256,
            "frames": 361,
            "fps": 24.0,
            "streaming_prefetch_count": 1,
            "max_batch_size": 4,
        }
    ]


def test_build_timing_summary_collects_phase_and_segment_timings() -> None:
    summary = _build_timing_summary(
        [
            {"event": "runtime_loaded", "seconds": 7.9},
            {"event": "pipeline_build_done", "seconds": 0.15},
            {"event": "pipeline_phase_done", "phase": "prompt_encoder", "call_index": 1, "seconds": 54.9},
            {"event": "pipeline_phase_done", "phase": "image_conditioner", "call_index": 2, "seconds": 0.6},
            {"event": "segment_pipeline_done", "segment_index": 1, "seconds": 457.9},
            {"event": "segment_encode_done", "segment_index": 1, "seconds": 10.5},
            {"event": "video_concat_done", "seconds": 0.7},
            {"event": "final_mux_done", "seconds": 0.8},
        ]
    )

    assert summary["runtime_load_seconds"] == 7.9
    assert summary["pipeline_build_seconds"] == 0.15
    assert summary["pipeline_phases"]["prompt_encoder"]["seconds"] == 54.9
    assert summary["pipeline_phases"]["image_conditioner#2"]["seconds"] == 0.6
    assert summary["segment_pipeline_seconds"] == {"1": 457.9}
    assert summary["segment_encode_seconds"] == {"1": 10.5}
    assert summary["video_concat_seconds"] == 0.7
    assert summary["final_mux_seconds"] == 0.8
