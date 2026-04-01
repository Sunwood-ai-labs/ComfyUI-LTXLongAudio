from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from cli.long_audio_core import build_manifest as build_base_manifest
from cli.long_audio_core import list_frame_images, plan_segments, probe_audio_info, write_manifest
from cli.origin_long_audio import find_repo_root, resolve_input_path


DEFAULT_WORKFLOW = Path("samples/workflows/LTX_2.3_Image_or_Text_&_Audio_2_Video_App_Origin.json")
DEFAULT_MANIFEST_NAME = "ltx23_gpu_ready_manifest.json"
DEFAULT_PIPELINE_MODULE = "ltx_pipelines.a2vid_two_stage"
DEFAULT_OUTPUT_PREFIX = "LTX-2.3-longaudio-randomimg"
DEFAULT_VIDEO_GUIDANCE_SCALE = 3.5
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of field, "
    "background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent lighting direction, "
    "color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny valley effect, incorrect ethnicity, "
    "wrong gender, exaggerated expressions, wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, "
    "robotic voice, echo, background noise, off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, "
    "awkward pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, inconsistent tone, "
    "cinematic oversaturation, stylized filters, or AI artifacts."
)

WORKFLOW_NODE_IDS = {
    "frames_dir": 167,
    "fps": 285,
    "segment_seconds": 291,
    "source_audio": 372,
    "image_selection_seed": 399,
    "prompt": 352,
    "text_to_video": 290,
    "width": 292,
    "height": 293,
    "output": 140,
    "negative_prompt": 110,
    "inference_seed": 114,
    "use_only_vocals": 382,
}


@dataclass(frozen=True)
class LTX23WorkflowDefaults:
    workflow_path: Path
    source_audio: str | None
    frames_dir: str | None
    image_load_cap: int = 0
    skip_first_images: int = 0
    select_every_nth: int = 1
    segment_seconds: int = 20
    fps: float = 24.0
    image_selection_seed: int = 1234
    inference_seed: int = 420
    prompt: str = ""
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    use_text_to_video: bool = False
    use_only_vocals: bool = False
    width: int = 832
    height: int = 480
    output_prefix: str = DEFAULT_OUTPUT_PREFIX


@dataclass(frozen=True)
class LTX23RuntimeConfig:
    pipeline_module: str = DEFAULT_PIPELINE_MODULE
    python_executable: str = "python"
    ltx_repo_root: Path | None = None
    checkpoint_path: Path | None = None
    distilled_lora_path: Path | None = None
    distilled_lora_strength: float = 1.0
    spatial_upsampler_path: Path | None = None
    gemma_root: Path | None = None
    output_dir: Path = Path(".")
    num_inference_steps: int | None = None
    image_strength: float = 1.0
    image_crf: int = 33
    quantization: str | tuple[str, ...] | None = None
    enhance_prompt: bool = False
    video_cfg_guidance_scale: float | None = DEFAULT_VIDEO_GUIDANCE_SCALE
    prompt_encoder_device: str = "match"
    performance_profile: str = "manual"
    streaming_prefetch_count: int | None = None
    prompt_streaming_prefetch_count: int | None = None
    max_batch_size: int | None = None
    compile_transformer: bool = False
    debug: bool = False
    debug_log_path: Path | None = None
    run_commands: bool = False
    emit_run_script: bool = False
    overwrite: bool = False
    extra_ltx_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Ltx23Assets:
    ltx_python: str
    checkpoint_path: str
    distilled_lora_path: str
    distilled_lora_strength: float
    spatial_upsampler_path: str
    gemma_root: str
    negative_prompt: str
    pipeline_module: str = DEFAULT_PIPELINE_MODULE
    num_inference_steps: int | None = None
    quantization: tuple[str, ...] = ()
    enhance_prompt: bool = False
    video_cfg_guidance_scale: float | None = None
    streaming_prefetch_count: int | None = None
    prompt_streaming_prefetch_count: int | None = None
    max_batch_size: int | None = None
    compile_transformer: bool = False
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class SegmentRenderRequest:
    segment_index: int
    prompt: str
    image_path: str | None
    audio_path: str
    audio_start_time: float = 0.0
    audio_max_duration: float | None = None
    fps: float = 24.0
    frame_count: int = 1
    width: int = 832
    height: int = 512
    seed: int = 0
    output_path: str = ""
    mode: str = "image_to_video"


@dataclass(frozen=True)
class SegmentCommand:
    segment_index: int
    command: list[str]
    output_path: str
    image_path: str | None
    audio_path: str
    audio_start_time: float
    audio_max_duration: float | None
    frame_count: int
    fps: float
    width: int
    height: int
    seed: int
    mode: str


@dataclass(frozen=True)
class LtxInProcessRuntime:
    parser_factory: Any
    pipeline_class: Any
    prompt_encoder_class: Any
    multi_modal_guider_params: Any
    tiling_config_class: Any
    get_video_chunks_number: Any
    encode_video: Any


@dataclass
class RunDebugLogger:
    enabled: bool = False
    log_path: Path | None = None
    echo: bool = False

    def __post_init__(self) -> None:
        if not self.enabled or self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def event(self, name: str, **fields: Any) -> None:
        if not self.enabled:
            return
        payload = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "event": name,
            **{key: _json_safe(value) for key, value in fields.items()},
        }
        line = json.dumps(payload, ensure_ascii=True)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.write("\n")
        if self.echo:
            summary = ", ".join(f"{key}={payload[key]}" for key in sorted(fields))
            print(f"[debug] {name}" + (f": {summary}" if summary else ""))


def _normalize_dimension(value: int, divisor: int = 64) -> int:
    safe_value = max(int(value), divisor)
    return safe_value - (safe_value % divisor)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _resolve_debug_log_path(debug: bool, debug_log: str | Path | None, output_dir: Path) -> Path | None:
    if debug_log is not None:
        return Path(debug_log).expanduser().resolve()
    if debug:
        return (output_dir / "ltx23_debug.jsonl").resolve()
    return None


def _maybe_torch_cuda_synchronize(device: Any) -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if not getattr(torch.cuda, "is_available", lambda: False)():
        return
    device_text = str(device)
    if not device_text.startswith("cuda"):
        return
    with contextlib.suppress(Exception):
        torch.cuda.synchronize(device)


def _gpu_snapshot(device: Any) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"requested_device": str(device)}
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        snapshot["torch"] = "unavailable"
        return snapshot
    snapshot["cuda_available"] = bool(getattr(torch.cuda, "is_available", lambda: False)())
    if not snapshot["cuda_available"]:
        return snapshot
    device_text = str(device)
    if not device_text.startswith("cuda"):
        return snapshot
    try:
        torch_device = torch.device(device_text)
        index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
        snapshot["device_name"] = torch.cuda.get_device_name(index)
        snapshot["memory_allocated_mib"] = round(torch.cuda.memory_allocated(index) / (1024 * 1024), 2)
        snapshot["memory_reserved_mib"] = round(torch.cuda.memory_reserved(index) / (1024 * 1024), 2)
        snapshot["max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated(index) / (1024 * 1024), 2)
        with contextlib.suppress(Exception):
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            snapshot["memory_free_mib"] = round(free_bytes / (1024 * 1024), 2)
            snapshot["memory_total_mib"] = round(total_bytes / (1024 * 1024), 2)
    except Exception as exc:  # pragma: no cover - defensive snapshot path
        snapshot["snapshot_error"] = str(exc)
    return snapshot


def _default_output_dir(audio_path: Path) -> Path:
    return audio_path.expanduser().resolve().with_name(f"{audio_path.stem}_ltx23_gpu")


def _required_path(raw: str | Path | None, env_name: str) -> str:
    candidate = raw if raw is not None else os.environ.get(env_name)
    if not candidate:
        raise ValueError(f"Missing required path. Provide the flag or set {env_name}.")
    resolved = Path(candidate).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Required path does not exist: {resolved}")
    return str(resolved)


def _required_gemma_root(raw: str | Path | None, env_name: str) -> str:
    resolved = Path(_required_path(raw, env_name)).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"Gemma root must be a directory: {resolved}")
    required_markers = ("config.json", "tokenizer.json")
    if not any((resolved / marker).exists() for marker in required_markers):
        marker_list = ", ".join(required_markers)
        raise FileNotFoundError(
            f"Gemma root is missing the expected files ({marker_list}): {resolved}. "
            "Download the gated Gemma snapshot with cli/ltx23_download_models.py or set HF_TOKEN first."
        )
    return str(resolved)


def _optional_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    return Path(raw).expanduser().resolve()


def _prepend_pythonpath(path: Path) -> None:
    resolved = str(path.expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def _add_ltx_repo_src_paths(ltx_repo_root: Path | None) -> None:
    if ltx_repo_root is None:
        return
    packages_root = ltx_repo_root.expanduser().resolve() / "packages"
    if not packages_root.exists():
        return
    for src_dir in sorted(packages_root.glob("*/src")):
        if src_dir.is_dir():
            _prepend_pythonpath(src_dir)


def _load_ltx_inference_runtime(ltx_repo_root: Path | None) -> LtxInProcessRuntime:
    try:
        args_module = importlib.import_module("ltx_pipelines.utils.args")
        pipeline_module = importlib.import_module("ltx_pipelines.a2vid_two_stage")
        blocks_module = importlib.import_module("ltx_pipelines.utils.blocks")
        guiders_module = importlib.import_module("ltx_core.components.guiders")
        video_vae_module = importlib.import_module("ltx_core.model.video_vae")
        media_io_module = importlib.import_module("ltx_pipelines.utils.media_io")
    except ImportError:
        _add_ltx_repo_src_paths(ltx_repo_root)
        try:
            args_module = importlib.import_module("ltx_pipelines.utils.args")
            pipeline_module = importlib.import_module("ltx_pipelines.a2vid_two_stage")
            blocks_module = importlib.import_module("ltx_pipelines.utils.blocks")
            guiders_module = importlib.import_module("ltx_core.components.guiders")
            video_vae_module = importlib.import_module("ltx_core.model.video_vae")
            media_io_module = importlib.import_module("ltx_pipelines.utils.media_io")
        except ImportError as nested_exc:
            root_text = str(ltx_repo_root.expanduser().resolve()) if ltx_repo_root is not None else "<not provided>"
            raise ImportError(
                "Unable to import the official LTX-2 Python runtime in-process. "
                "Run this CLI inside the LTX-2 environment or provide --ltx-repo-root so its packages/*/src paths can be imported. "
                f"ltx_repo_root={root_text}"
            ) from nested_exc
    return LtxInProcessRuntime(
        parser_factory=args_module.default_2_stage_arg_parser,
        pipeline_class=pipeline_module.A2VidPipelineTwoStage,
        prompt_encoder_class=blocks_module.PromptEncoder,
        multi_modal_guider_params=guiders_module.MultiModalGuiderParams,
        tiling_config_class=video_vae_module.TilingConfig,
        get_video_chunks_number=video_vae_module.get_video_chunks_number,
        encode_video=media_io_module.encode_video,
    )


def _coerce_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _widget_value(node: dict[str, Any] | None, index: int, default: object) -> object:
    if node is None:
        return default
    values = node.get("widgets_values") or []
    if index >= len(values):
        return default
    value = values[index]
    return default if value is None else value


def _find_node(
    nodes: Sequence[dict[str, Any]],
    *,
    node_id: int | None = None,
    titles: Iterable[str] = (),
    node_types: Iterable[str] = (),
) -> dict[str, Any] | None:
    if node_id is not None:
        for node in nodes:
            if int(node.get("id", -1)) == int(node_id):
                return node
    title_set = set(titles)
    if title_set:
        for node in nodes:
            if node.get("title") in title_set:
                return node
    type_set = set(node_types)
    if type_set:
        for node in nodes:
            if node.get("type") in type_set:
                return node
    return None


def load_ltx23_workflow_defaults(workflow_path: Path = DEFAULT_WORKFLOW) -> LTX23WorkflowDefaults:
    resolved = workflow_path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", [])

    frames_node = _find_node(
        nodes,
        node_id=WORKFLOW_NODE_IDS["frames_dir"],
        titles=("Segment Image Folder",),
        node_types=("LTXLoadImages",),
    )
    fps_node = _find_node(nodes, node_id=WORKFLOW_NODE_IDS["fps"], titles=("FPS",))
    segment_node = _find_node(nodes, node_id=WORKFLOW_NODE_IDS["segment_seconds"], titles=("SEGMENT SECONDS",))
    source_audio_node = _find_node(
        nodes,
        node_id=WORKFLOW_NODE_IDS["source_audio"],
        titles=("Source Song Upload",),
        node_types=("LoadAudio", "LTXLoadAudioUpload"),
    )
    image_seed_node = _find_node(nodes, node_id=WORKFLOW_NODE_IDS["image_selection_seed"], titles=("RANDOM IMAGE SEED",))
    inference_seed_node = _find_node(nodes, node_id=WORKFLOW_NODE_IDS["inference_seed"], node_types=("RandomNoise",))
    prompt_node = _find_node(nodes, node_id=WORKFLOW_NODE_IDS["prompt"], titles=("PROMPT",))
    negative_prompt_node = _find_node(nodes, node_id=WORKFLOW_NODE_IDS["negative_prompt"], node_types=("CLIPTextEncode",))
    text_to_video_node = _find_node(
        nodes,
        node_id=WORKFLOW_NODE_IDS["text_to_video"],
        titles=("Text To Video (no image ref)",),
    )
    use_only_vocals_node = _find_node(
        nodes,
        node_id=WORKFLOW_NODE_IDS["use_only_vocals"],
        titles=("USE ONLY VOCALS",),
    )
    width_node = _find_node(nodes, node_id=WORKFLOW_NODE_IDS["width"], titles=("WIDTH",))
    height_node = _find_node(nodes, node_id=WORKFLOW_NODE_IDS["height"], titles=("HEIGHT",))
    output_node = _find_node(
        nodes,
        node_id=WORKFLOW_NODE_IDS["output"],
        titles=("Video Combine (20s Long Audio)",),
        node_types=("LTXVideoCombine",),
    )

    return LTX23WorkflowDefaults(
        workflow_path=resolved,
        source_audio=_coerce_text(_widget_value(source_audio_node, 0, None)),
        frames_dir=_coerce_text(_widget_value(frames_node, 0, None)),
        image_load_cap=int(_widget_value(frames_node, 1, 0)),
        skip_first_images=int(_widget_value(frames_node, 2, 0)),
        select_every_nth=int(_widget_value(frames_node, 3, 1)),
        segment_seconds=int(_widget_value(segment_node, 0, 20)),
        fps=float(_widget_value(fps_node, 0, 24.0)),
        image_selection_seed=int(_widget_value(image_seed_node, 0, 1234)),
        inference_seed=int(_widget_value(inference_seed_node, 0, 420)),
        prompt=str(_widget_value(prompt_node, 0, "") or ""),
        negative_prompt=str(_widget_value(negative_prompt_node, 0, DEFAULT_NEGATIVE_PROMPT) or DEFAULT_NEGATIVE_PROMPT),
        use_text_to_video=bool(_widget_value(text_to_video_node, 0, False)),
        use_only_vocals=bool(_widget_value(use_only_vocals_node, 0, False)),
        width=int(_widget_value(width_node, 0, 832)),
        height=int(_widget_value(height_node, 0, 480)),
        output_prefix=str(_widget_value(output_node, 2, DEFAULT_OUTPUT_PREFIX) or DEFAULT_OUTPUT_PREFIX),
    )


def _normalize_quantization(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        text = raw.strip()
        return (text,) if text else ()
    values = tuple(str(item).strip() for item in raw if str(item).strip())
    return values


def _resolve_runtime_tuning(
    *,
    performance_profile: str,
    streaming_prefetch_count: int | None,
    max_batch_size: int | None,
    compile_transformer: bool,
    extra_ltx_args: Sequence[str],
) -> tuple[int | None, int, bool, tuple[str, ...]]:
    passthrough: list[str] = []
    legacy_streaming_prefetch_count: int | None = None
    legacy_max_batch_size: int | None = None
    legacy_compile = False

    index = 0
    while index < len(extra_ltx_args):
        value = str(extra_ltx_args[index])
        if value == "--streaming-prefetch-count" and index + 1 < len(extra_ltx_args):
            legacy_streaming_prefetch_count = int(str(extra_ltx_args[index + 1]))
            index += 2
            continue
        if value == "--max-batch-size" and index + 1 < len(extra_ltx_args):
            legacy_max_batch_size = int(str(extra_ltx_args[index + 1]))
            index += 2
            continue
        if value == "--compile":
            legacy_compile = True
            index += 1
            continue
        passthrough.append(value)
        index += 1

    effective_streaming_prefetch_count = (
        streaming_prefetch_count if streaming_prefetch_count is not None else legacy_streaming_prefetch_count
    )
    effective_max_batch_size = max_batch_size if max_batch_size is not None else legacy_max_batch_size
    effective_compile = bool(compile_transformer or legacy_compile)

    if performance_profile == "throughput":
        if streaming_prefetch_count is None and legacy_streaming_prefetch_count is None:
            effective_streaming_prefetch_count = None
        if max_batch_size is None and legacy_max_batch_size is None:
            effective_max_batch_size = 4
    elif performance_profile == "low-vram":
        if streaming_prefetch_count is None and legacy_streaming_prefetch_count is None:
            effective_streaming_prefetch_count = 1
        if max_batch_size is None and legacy_max_batch_size is None:
            effective_max_batch_size = 4

    return (
        effective_streaming_prefetch_count,
        max(int(effective_max_batch_size or 1), 1),
        effective_compile,
        tuple(passthrough),
    )

def build_segment_command(
    request: SegmentRenderRequest,
    assets: Ltx23Assets,
    *,
    image_strength: float,
    image_crf: int,
) -> list[str]:
    command = [
        assets.ltx_python,
        "-m",
        assets.pipeline_module,
        "--checkpoint-path",
        assets.checkpoint_path,
        "--distilled-lora",
        assets.distilled_lora_path,
        str(assets.distilled_lora_strength),
        "--spatial-upsampler-path",
        assets.spatial_upsampler_path,
        "--gemma-root",
        assets.gemma_root,
        "--prompt",
        request.prompt,
        "--negative-prompt",
        assets.negative_prompt,
        "--output-path",
        request.output_path,
        "--seed",
        str(request.seed),
        "--height",
        str(request.height),
        "--width",
        str(request.width),
        "--num-frames",
        str(request.frame_count),
        "--frame-rate",
        str(request.fps),
        "--audio-path",
        request.audio_path,
        "--audio-start-time",
        f"{request.audio_start_time:.6f}",
    ]
    if request.audio_max_duration is not None:
        command.extend(["--audio-max-duration", f"{request.audio_max_duration:.6f}"])
    if assets.num_inference_steps is not None:
        command.extend(["--num-inference-steps", str(assets.num_inference_steps)])
    if request.mode == "image_to_video" and request.image_path:
        command.extend(["--image", request.image_path, "0", str(image_strength), str(image_crf)])
    if assets.quantization:
        command.extend(["--quantization", *assets.quantization])
    if assets.compile_transformer:
        command.append("--compile")
    if assets.enhance_prompt:
        command.append("--enhance-prompt")
    if assets.video_cfg_guidance_scale is not None:
        command.extend(["--video-cfg-guidance-scale", str(assets.video_cfg_guidance_scale)])
    if assets.streaming_prefetch_count is not None:
        command.extend(["--streaming-prefetch-count", str(assets.streaming_prefetch_count)])
    if assets.max_batch_size is not None and assets.max_batch_size > 1:
        command.extend(["--max-batch-size", str(assets.max_batch_size)])
    command.extend(list(assets.extra_args))
    return command


def build_segment_commands(
    *,
    defaults: LTX23WorkflowDefaults,
    runtime: LTX23RuntimeConfig,
    source_audio: Path,
    conditioning_audio: Path | None,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    fps: float,
    use_text_to_video: bool,
    ltx_seed_base: int,
    segments: Sequence[Any],
    prepared_audio_paths: dict[int, Path] | None = None,
) -> list[SegmentCommand]:
    (
        effective_streaming_prefetch_count,
        effective_max_batch_size,
        effective_compile_transformer,
        passthrough_extra_args,
    ) = _resolve_runtime_tuning(
        performance_profile=runtime.performance_profile,
        streaming_prefetch_count=runtime.streaming_prefetch_count,
        max_batch_size=runtime.max_batch_size,
        compile_transformer=runtime.compile_transformer,
        extra_ltx_args=runtime.extra_ltx_args,
    )
    assets = Ltx23Assets(
        ltx_python=str(runtime.python_executable),
        checkpoint_path=str(runtime.checkpoint_path or "<checkpoint-path>"),
        distilled_lora_path=str(runtime.distilled_lora_path or "<distilled-lora-path>"),
        distilled_lora_strength=float(runtime.distilled_lora_strength),
        spatial_upsampler_path=str(runtime.spatial_upsampler_path or "<spatial-upsampler-path>"),
        gemma_root=str(runtime.gemma_root or "<gemma-root>"),
        negative_prompt=negative_prompt,
        pipeline_module=runtime.pipeline_module,
        num_inference_steps=runtime.num_inference_steps,
        quantization=_normalize_quantization(runtime.quantization),
        enhance_prompt=runtime.enhance_prompt,
        video_cfg_guidance_scale=runtime.video_cfg_guidance_scale,
        streaming_prefetch_count=effective_streaming_prefetch_count,
        max_batch_size=effective_max_batch_size,
        compile_transformer=effective_compile_transformer,
        extra_args=passthrough_extra_args,
    )
    conditioning_path = str((conditioning_audio or source_audio).expanduser().resolve())
    normalized_width = _normalize_dimension(width)
    normalized_height = _normalize_dimension(height)
    output_root = runtime.output_dir.expanduser().resolve() / "segment_videos"

    commands: list[SegmentCommand] = []
    for segment in segments:
        duration_seconds = float(int(segment.frames) / max(float(fps), 1.0))
        prepared_audio_path = prepared_audio_paths.get(int(segment.index)) if prepared_audio_paths else None
        output_path = output_root / f"segment_{int(segment.index) + 1:03d}.mp4"
        request = SegmentRenderRequest(
            segment_index=int(segment.index),
            prompt=prompt,
            image_path=None if use_text_to_video else str(Path(segment.selected_image_path).expanduser().resolve()),
            audio_path=str(prepared_audio_path) if prepared_audio_path is not None else conditioning_path,
            audio_start_time=0.0 if prepared_audio_path is not None else float(segment.start_time),
            audio_max_duration=duration_seconds,
            fps=float(fps),
            frame_count=int(segment.frames),
            width=normalized_width,
            height=normalized_height,
            seed=int(ltx_seed_base) + int(segment.index),
            output_path=str(output_path),
            mode="text_to_video" if use_text_to_video else "image_to_video",
        )
        command = build_segment_command(
            request,
            assets,
            image_strength=runtime.image_strength,
            image_crf=runtime.image_crf,
        )
        commands.append(
            SegmentCommand(
                segment_index=request.segment_index,
                command=command,
                output_path=request.output_path,
                image_path=request.image_path,
                audio_path=request.audio_path,
                audio_start_time=request.audio_start_time,
                audio_max_duration=request.audio_max_duration,
                frame_count=request.frame_count,
                fps=request.fps,
                width=request.width,
                height=request.height,
                seed=request.seed,
                mode=request.mode,
            )
        )
    return commands


def _ffmpeg_executable() -> str:
    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved
    raise RuntimeError("ffmpeg is required for final segment concatenation and muxing.")


def _ffmpeg_reference() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def _segment_target_audio_duration(frame_count: int, fps: float) -> float:
    return float(max(int(frame_count), 1)) / max(float(fps), 1.0)


def _prepare_conditioning_audio_chunk(
    source_audio: Path,
    output_path: Path,
    *,
    start_time: float,
    target_duration: float,
    overwrite: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        _ffmpeg_executable(),
        "-y" if overwrite else "-n",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_audio.expanduser().resolve()),
        "-ss",
        f"{float(start_time):.6f}",
        "-t",
        f"{float(target_duration):.6f}",
        "-ac",
        "2",
        "-af",
        f"apad=whole_dur={float(target_duration):.6f}",
        "-c:a",
        "pcm_s16le",
        str(output_path.expanduser().resolve()),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path.expanduser().resolve()


def _prepare_conditioning_audio_chunks(
    source_audio: Path,
    segments: Sequence[Any],
    *,
    fps: float,
    output_dir: Path,
    overwrite: bool,
) -> dict[int, Path]:
    prepared: dict[int, Path] = {}
    chunk_dir = output_dir.expanduser().resolve() / "conditioning_audio"
    for segment in segments:
        segment_index = int(segment.index)
        target_duration = _segment_target_audio_duration(int(segment.frames), fps)
        chunk_path = chunk_dir / f"segment_{segment_index + 1:03d}.wav"
        prepared[segment_index] = _prepare_conditioning_audio_chunk(
            source_audio,
            chunk_path,
            start_time=float(segment.start_time),
            target_duration=target_duration,
            overwrite=overwrite,
        )
    return prepared


def _write_concat_file(segment_paths: Sequence[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"file '{path.as_posix()}'" for path in segment_paths]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _concat_video_streams(segment_paths: Sequence[Path], output_path: Path, *, overwrite: bool) -> Path:
    concat_path = _write_concat_file(segment_paths, output_path.parent / "segments_concat.txt")
    command = [
        _ffmpeg_executable(),
        "-y" if overwrite else "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


def _format_media_duration(duration_seconds: float) -> str:
    return f"{max(float(duration_seconds), 0.0):.6f}"


def _mux_original_audio(video_path: Path, audio_path: Path, output_path: Path, *, overwrite: bool) -> Path:
    source_duration = _format_media_duration(probe_audio_info(audio_path).duration_seconds)
    command = [
        _ffmpeg_executable(),
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        f"trim=duration={source_duration},setpts=PTS-STARTPTS",
        "-af",
        f"atrim=duration={source_duration},asetpts=PTS-STARTPTS",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


def _shell_join(items: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in items)


def _emit_run_script(
    runtime: LTX23RuntimeConfig,
    *,
    source_audio: Path,
    output_prefix: str,
    segment_commands: Sequence[SegmentCommand],
    include_final_concat: bool,
) -> Path:
    script_path = runtime.output_dir.expanduser().resolve() / "run_segments.sh"
    segment_paths = [Path(command.output_path).resolve() for command in segment_commands]
    concat_path = runtime.output_dir.expanduser().resolve() / "segments_concat.txt"
    video_only_path = runtime.output_dir.expanduser().resolve() / f"{output_prefix}_video_only.mp4"
    final_path = runtime.output_dir.expanduser().resolve() / f"{output_prefix}.mp4"
    source_duration = _format_media_duration(probe_audio_info(source_audio).duration_seconds)

    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    if runtime.ltx_repo_root is not None:
        lines.extend([f"cd {shlex.quote(str(runtime.ltx_repo_root.expanduser().resolve()))}", ""])
    for command in segment_commands:
        lines.append(_shell_join(command.command))
    if include_final_concat and segment_paths:
        lines.extend(
            [
                "",
                "cat > " + shlex.quote(str(concat_path)) + " <<'EOF'",
                *[f"file '{path.as_posix()}'" for path in segment_paths],
                "EOF",
                _shell_join(
                    [
                        _ffmpeg_reference(),
                        "-y" if runtime.overwrite else "-n",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_path),
                        "-map",
                        "0:v:0",
                        "-c:v",
                        "copy",
                        "-an",
                        str(video_only_path),
                    ]
                ),
                _shell_join(
                    [
                        _ffmpeg_reference(),
                        "-y" if runtime.overwrite else "-n",
                        "-i",
                        str(video_only_path),
                        "-i",
                        str(source_audio.expanduser().resolve()),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-vf",
                        f"trim=duration={source_duration},setpts=PTS-STARTPTS",
                        "-af",
                        f"atrim=duration={source_duration},asetpts=PTS-STARTPTS",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-movflags",
                        "+faststart",
                        str(final_path),
                    ]
                ),
            ]
        )
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return script_path


def _segment_command_cli_args(command: Sequence[str]) -> list[str]:
    if len(command) >= 3 and command[1] == "-m":
        return list(command[3:])
    return list(command)


def _parse_official_segment_args(runtime: LtxInProcessRuntime, command: Sequence[str]) -> Any:
    parser = runtime.parser_factory()
    parser.add_argument(
        "--audio-path",
        type=str,
        required=True,
        help="Path to the audio file to condition the video generation.",
    )
    parser.add_argument(
        "--audio-start-time",
        type=float,
        default=0.0,
        help="Start time in seconds to read audio from (default: 0.0).",
    )
    parser.add_argument(
        "--audio-max-duration",
        type=float,
        default=None,
        help="Maximum audio duration in seconds. Defaults to video duration (num_frames / frame_rate).",
    )
    try:
        return parser.parse_args(_segment_command_cli_args(command))
    except SystemExit as exc:  # pragma: no cover - argparse prints its own diagnostics
        raise ValueError(f"Failed to parse official LTX pipeline args from command: {command}") from exc


def _torch_inference_context() -> Any:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return contextlib.nullcontext()
    return torch.inference_mode()


def _build_video_guider_params(namespace: Any, runtime: LtxInProcessRuntime) -> Any:
    return runtime.multi_modal_guider_params(
        cfg_scale=namespace.video_cfg_guidance_scale,
        stg_scale=namespace.video_stg_guidance_scale,
        rescale_scale=namespace.video_rescale_scale,
        modality_scale=namespace.a2v_guidance_scale,
        skip_step=namespace.video_skip_step,
        stg_blocks=namespace.video_stg_blocks,
    )


def _resolve_torch_device(raw: str | None, default: Any) -> Any:
    if raw in (None, "", "match"):
        return default
    torch = importlib.import_module("torch")
    return torch.device(str(raw))


class _PromptEncoderOutputDeviceAdapter:
    def __init__(
        self,
        *,
        prompt_encoder_class: Any,
        checkpoint_path: str,
        gemma_root: str,
        dtype: Any,
        model_device: Any,
        output_device: Any,
        streaming_prefetch_count_override: int | None = None,
    ) -> None:
        self._inner = prompt_encoder_class(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            dtype=dtype,
            device=model_device,
        )
        self._output_device = output_device
        self._disable_streaming = str(model_device).startswith("cpu")
        self._streaming_prefetch_count_override = streaming_prefetch_count_override

    def __call__(self, prompts: list[str], **kwargs: Any) -> Any:
        if self._disable_streaming and "streaming_prefetch_count" in kwargs:
            kwargs = dict(kwargs)
            kwargs["streaming_prefetch_count"] = None
        elif self._streaming_prefetch_count_override is not None:
            kwargs = dict(kwargs)
            kwargs["streaming_prefetch_count"] = self._streaming_prefetch_count_override
        outputs = self._inner(prompts, **kwargs)
        if not outputs:
            return outputs
        moved = []
        for item in outputs:
            audio_encoding = item.audio_encoding.to(self._output_device) if item.audio_encoding is not None else None
            moved.append(
                type(item)(
                    item.video_encoding.to(self._output_device),
                    audio_encoding,
                    item.attention_mask.to(self._output_device),
                )
            )
        return moved


class _DebugPhaseCallable:
    def __init__(
        self,
        *,
        phase_name: str,
        inner: Any,
        device_getter: Callable[[], Any],
        debug_logger: RunDebugLogger,
        metadata_builder: Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self._phase_name = phase_name
        self._inner = inner
        self._device_getter = device_getter
        self._debug_logger = debug_logger
        self._metadata_builder = metadata_builder
        self._call_index = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self._call_index += 1
        device = self._device_getter()
        metadata = {"phase": self._phase_name, "call_index": self._call_index}
        if self._metadata_builder is not None:
            metadata.update(self._metadata_builder(args, kwargs))
        self._debug_logger.event("pipeline_phase_start", **metadata, gpu_snapshot=_gpu_snapshot(device))
        _maybe_torch_cuda_synchronize(device)
        started = time.perf_counter()
        result = self._inner(*args, **kwargs)
        _maybe_torch_cuda_synchronize(device)
        self._debug_logger.event(
            "pipeline_phase_done",
            **metadata,
            seconds=round(time.perf_counter() - started, 3),
            gpu_snapshot=_gpu_snapshot(device),
        )
        return result


def _wrap_pipeline_phase(
    pipeline: Any,
    *,
    attr_name: str,
    phase_name: str,
    debug_logger: RunDebugLogger | None,
    device_getter: Callable[[], Any],
    metadata_builder: Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]] | None = None,
) -> None:
    if debug_logger is None or not debug_logger.enabled:
        return
    inner = getattr(pipeline, attr_name, None)
    if inner is None or not callable(inner):
        return
    setattr(
        pipeline,
        attr_name,
        _DebugPhaseCallable(
            phase_name=phase_name,
            inner=inner,
            device_getter=device_getter,
            debug_logger=debug_logger,
            metadata_builder=metadata_builder,
        ),
    )


def _prompt_encoder_phase_metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    prompts = args[0] if args else None
    prompt_count = len(prompts) if isinstance(prompts, list) else None
    return {
        "prompt_count": prompt_count,
        "streaming_prefetch_count": kwargs.get("streaming_prefetch_count"),
        "enhance_first_prompt": bool(kwargs.get("enhance_first_prompt", False)),
    }


def _diffusion_phase_metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "width": kwargs.get("width"),
        "height": kwargs.get("height"),
        "frames": kwargs.get("frames"),
        "fps": kwargs.get("fps"),
        "streaming_prefetch_count": kwargs.get("streaming_prefetch_count"),
        "max_batch_size": kwargs.get("max_batch_size"),
    }


def _video_decoder_phase_metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    tiling_config = args[1] if len(args) > 1 else kwargs.get("tiling_config")
    return {"tiling_config": tiling_config}


def _instrument_pipeline_phases(
    pipeline: Any,
    *,
    debug_logger: RunDebugLogger | None,
    pipeline_device: Any,
) -> None:
    if debug_logger is None or not debug_logger.enabled:
        return

    def device_getter() -> Any:
        return getattr(pipeline, "device", pipeline_device)

    _wrap_pipeline_phase(
        pipeline,
        attr_name="prompt_encoder",
        phase_name="prompt_encoder",
        debug_logger=debug_logger,
        device_getter=device_getter,
        metadata_builder=_prompt_encoder_phase_metadata,
    )
    _wrap_pipeline_phase(
        pipeline,
        attr_name="audio_conditioner",
        phase_name="audio_conditioner",
        debug_logger=debug_logger,
        device_getter=device_getter,
    )
    _wrap_pipeline_phase(
        pipeline,
        attr_name="image_conditioner",
        phase_name="image_conditioner",
        debug_logger=debug_logger,
        device_getter=device_getter,
    )
    _wrap_pipeline_phase(
        pipeline,
        attr_name="stage_1",
        phase_name="stage_1",
        debug_logger=debug_logger,
        device_getter=device_getter,
        metadata_builder=_diffusion_phase_metadata,
    )
    _wrap_pipeline_phase(
        pipeline,
        attr_name="upsampler",
        phase_name="upsampler",
        debug_logger=debug_logger,
        device_getter=device_getter,
    )
    _wrap_pipeline_phase(
        pipeline,
        attr_name="stage_2",
        phase_name="stage_2",
        debug_logger=debug_logger,
        device_getter=device_getter,
        metadata_builder=_diffusion_phase_metadata,
    )
    _wrap_pipeline_phase(
        pipeline,
        attr_name="video_decoder",
        phase_name="video_decoder",
        debug_logger=debug_logger,
        device_getter=device_getter,
        metadata_builder=_video_decoder_phase_metadata,
    )


def _build_in_process_pipeline(
    runtime: LtxInProcessRuntime,
    namespace: Any,
    config: LTX23RuntimeConfig,
    debug_logger: RunDebugLogger | None = None,
) -> Any:
    build_started = time.perf_counter()
    if debug_logger is not None:
        debug_logger.event(
            "pipeline_build_start",
            checkpoint_path=namespace.checkpoint_path,
            gemma_root=namespace.gemma_root,
            quantization=getattr(namespace, "quantization", None),
            compile=bool(getattr(namespace, "compile", False)),
        )
    pipeline = runtime.pipeline_class(
        checkpoint_path=namespace.checkpoint_path,
        distilled_lora=namespace.distilled_lora,
        spatial_upsampler_path=namespace.spatial_upsampler_path,
        gemma_root=namespace.gemma_root,
        loras=tuple(namespace.lora) if getattr(namespace, "lora", None) else (),
        quantization=getattr(namespace, "quantization", None),
        torch_compile=bool(getattr(namespace, "compile", False)),
    )
    prompt_encoder_device = _resolve_torch_device(config.prompt_encoder_device, getattr(pipeline, "device", None))
    pipeline_device = getattr(pipeline, "device", prompt_encoder_device)
    prompt_streaming_override = config.prompt_streaming_prefetch_count
    if prompt_encoder_device != pipeline_device or prompt_streaming_override is not None:
        pipeline.prompt_encoder = _PromptEncoderOutputDeviceAdapter(
            prompt_encoder_class=runtime.prompt_encoder_class,
            checkpoint_path=namespace.checkpoint_path,
            gemma_root=namespace.gemma_root,
            dtype=getattr(pipeline, "dtype", None),
            model_device=prompt_encoder_device,
            output_device=pipeline_device,
            streaming_prefetch_count_override=prompt_streaming_override,
        )
    _instrument_pipeline_phases(
        pipeline,
        debug_logger=debug_logger,
        pipeline_device=pipeline_device,
    )
    if debug_logger is not None:
        debug_logger.event(
            "pipeline_build_done",
            seconds=round(time.perf_counter() - build_started, 3),
            pipeline_device=str(pipeline_device),
            prompt_encoder_device=str(prompt_encoder_device),
            prompt_streaming_prefetch_count=prompt_streaming_override,
            gpu_snapshot=_gpu_snapshot(pipeline_device),
        )
    return pipeline


def _render_segment_in_process(
    pipeline: Any,
    runtime: LtxInProcessRuntime,
    namespace: Any,
    debug_logger: RunDebugLogger | None = None,
    segment_index: int | None = None,
    segment_total: int | None = None,
) -> Path:
    output_path = Path(namespace.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiling_config = runtime.tiling_config_class.default()
    pipeline_device = getattr(pipeline, "device", "unknown")
    if debug_logger is not None:
        debug_logger.event(
            "segment_start",
            segment_index=segment_index,
            segment_total=segment_total,
            output_path=output_path,
            image_path=getattr(namespace, "images", None),
            audio_path=namespace.audio_path,
            audio_start_time=namespace.audio_start_time,
            audio_max_duration=namespace.audio_max_duration,
            num_frames=namespace.num_frames,
            frame_rate=namespace.frame_rate,
            width=namespace.width,
            height=namespace.height,
            seed=namespace.seed,
            gpu_snapshot=_gpu_snapshot(pipeline_device),
        )
    _maybe_torch_cuda_synchronize(pipeline_device)
    pipeline_started = time.perf_counter()
    video, audio = pipeline(
        prompt=namespace.prompt,
        negative_prompt=namespace.negative_prompt,
        seed=namespace.seed,
        height=namespace.height,
        width=namespace.width,
        num_frames=namespace.num_frames,
        frame_rate=namespace.frame_rate,
        num_inference_steps=namespace.num_inference_steps,
        video_guider_params=_build_video_guider_params(namespace, runtime),
        images=namespace.images,
        audio_path=namespace.audio_path,
        audio_start_time=namespace.audio_start_time,
        audio_max_duration=(
            namespace.audio_max_duration
            if namespace.audio_max_duration is not None
            else float(namespace.num_frames) / max(float(namespace.frame_rate), 1.0)
        ),
        tiling_config=tiling_config,
        enhance_prompt=bool(namespace.enhance_prompt),
        streaming_prefetch_count=namespace.streaming_prefetch_count,
        max_batch_size=namespace.max_batch_size,
    )
    _maybe_torch_cuda_synchronize(pipeline_device)
    if debug_logger is not None:
        debug_logger.event(
            "segment_pipeline_done",
            segment_index=segment_index,
            seconds=round(time.perf_counter() - pipeline_started, 3),
            gpu_snapshot=_gpu_snapshot(pipeline_device),
        )
    _maybe_torch_cuda_synchronize(pipeline_device)
    encode_started = time.perf_counter()
    runtime.encode_video(
        video=video,
        fps=namespace.frame_rate,
        audio=audio,
        output_path=str(output_path),
        video_chunks_number=runtime.get_video_chunks_number(namespace.num_frames, tiling_config),
    )
    _maybe_torch_cuda_synchronize(pipeline_device)
    if debug_logger is not None:
        debug_logger.event(
            "segment_encode_done",
            segment_index=segment_index,
            seconds=round(time.perf_counter() - encode_started, 3),
            output_path=output_path,
            gpu_snapshot=_gpu_snapshot(pipeline_device),
        )
    return output_path


def _run_segments_in_process(
    *,
    ltx_runtime: LtxInProcessRuntime,
    segment_commands: Sequence[SegmentCommand],
    runtime_config: LTX23RuntimeConfig,
    debug_logger: RunDebugLogger | None = None,
) -> list[Path]:
    if not segment_commands:
        return []
    parse_started = time.perf_counter()
    parsed_segments = [_parse_official_segment_args(ltx_runtime, segment.command) for segment in segment_commands]
    if debug_logger is not None:
        debug_logger.event(
            "segment_args_parsed",
            segment_count=len(parsed_segments),
            seconds=round(time.perf_counter() - parse_started, 3),
        )
    rendered: list[Path] = []
    with _torch_inference_context():
        pipeline = _build_in_process_pipeline(ltx_runtime, parsed_segments[0], runtime_config, debug_logger)
        for index, namespace in enumerate(parsed_segments, start=1):
            rendered.append(
                _render_segment_in_process(
                    pipeline,
                    ltx_runtime,
                    namespace,
                    debug_logger=debug_logger,
                    segment_index=index,
                    segment_total=len(parsed_segments),
                )
            )
    return rendered


def _build_runtime_config(args: argparse.Namespace, output_dir: Path) -> LTX23RuntimeConfig:
    return LTX23RuntimeConfig(
        pipeline_module=args.pipeline_module,
        python_executable=args.ltx_python,
        ltx_repo_root=_optional_path(args.ltx_repo_root),
        checkpoint_path=_optional_path(args.checkpoint_path),
        distilled_lora_path=_optional_path(args.distilled_lora_path),
        distilled_lora_strength=float(args.distilled_lora_strength),
        spatial_upsampler_path=_optional_path(args.spatial_upsampler_path),
        gemma_root=_optional_path(args.gemma_root),
        output_dir=output_dir,
        num_inference_steps=args.num_inference_steps,
        image_strength=float(args.image_strength),
        image_crf=int(args.image_crf),
        quantization=None if args.quantization is None else tuple(args.quantization),
        enhance_prompt=bool(args.enhance_prompt),
        video_cfg_guidance_scale=args.video_cfg_guidance_scale,
        prompt_encoder_device=str(args.prompt_encoder_device),
        performance_profile=str(args.performance_profile),
        streaming_prefetch_count=args.streaming_prefetch_count,
        prompt_streaming_prefetch_count=args.prompt_streaming_prefetch_count,
        max_batch_size=args.max_batch_size,
        compile_transformer=bool(args.compile_transformer),
        debug=bool(args.debug or args.debug_log is not None),
        debug_log_path=_resolve_debug_log_path(bool(args.debug), args.debug_log, output_dir),
        run_commands=bool(args.run),
        emit_run_script=bool(args.emit_run_script),
        overwrite=bool(args.overwrite),
        extra_ltx_args=tuple(args.extra_ltx_arg),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and optionally run official LTX-2.3 segment inference for the Origin workflow."
    )
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW)
    parser.add_argument("--audio", type=Path, default=None)
    parser.add_argument("--conditioning-audio", type=Path, default=None)
    parser.add_argument("--frames-dir", type=Path, default=None)
    parser.add_argument("--segment-seconds", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override deterministic frame-selection seed.")
    parser.add_argument("--ltx-seed", type=int, default=None, help="Override the LTX inference seed base.")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--text-to-video", action="store_true")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--emit-run-script", action="store_true")
    parser.add_argument("--skip-final-concat", action="store_true")
    parser.add_argument("--image-strength", type=float, default=1.0)
    parser.add_argument("--image-crf", type=int, default=33)
    parser.add_argument("--pipeline-module", default=DEFAULT_PIPELINE_MODULE)
    parser.add_argument("--ltx-python", default=os.environ.get("LTX2_PYTHON", "python"))
    parser.add_argument("--ltx-repo-root", default=os.environ.get("LTX2_REPO_ROOT"))
    parser.add_argument("--checkpoint-path", default=os.environ.get("LTX2_CHECKPOINT_PATH"))
    parser.add_argument("--distilled-lora-path", default=os.environ.get("LTX2_DISTILLED_LORA_PATH"))
    parser.add_argument(
        "--distilled-lora-strength",
        type=float,
        default=float(os.environ.get("LTX2_DISTILLED_LORA_STRENGTH", "1.0")),
    )
    parser.add_argument("--spatial-upsampler-path", default=os.environ.get("LTX2_SPATIAL_UPSAMPLER_PATH"))
    parser.add_argument("--gemma-root", default=os.environ.get("LTX2_GEMMA_ROOT"))
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--quantization", nargs="+", default=None)
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--video-cfg-guidance-scale", type=float, default=DEFAULT_VIDEO_GUIDANCE_SCALE)
    parser.add_argument("--prompt-encoder-device", default=os.environ.get("LTX2_PROMPT_ENCODER_DEVICE", "match"))
    parser.add_argument(
        "--performance-profile",
        choices=("manual", "throughput", "low-vram"),
        default="manual",
        help="throughput keeps models GPU-resident when possible; low-vram enables streaming defaults; manual preserves explicit tuning only.",
    )
    parser.add_argument("--streaming-prefetch-count", type=int, default=None, help="Official layer-streaming prefetch count.")
    parser.add_argument(
        "--prompt-streaming-prefetch-count",
        type=int,
        default=None,
        help="Optional prompt-encoder-only streaming prefetch count for in-process runs.",
    )
    parser.add_argument("--max-batch-size", type=int, default=None, help="Official guidance batching size per transformer call.")
    parser.add_argument("--compile-transformer", action="store_true", help="Enable official --compile transformer mode.")
    parser.add_argument("--debug", action="store_true", help="Emit step-by-step debug progress and write a JSONL debug log.")
    parser.add_argument("--debug-log", default=None, help="Optional path for the JSONL debug log.")
    parser.add_argument("--extra-ltx-arg", action="append", default=[])
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    defaults = load_ltx23_workflow_defaults(args.workflow)
    repo_root = find_repo_root(defaults.workflow_path)

    audio_value = args.audio if args.audio is not None else defaults.source_audio
    frames_value = args.frames_dir if args.frames_dir is not None else defaults.frames_dir
    conditioning_value = args.conditioning_audio if args.conditioning_audio is not None else None

    audio_path = resolve_input_path(audio_value, label="audio", workflow_path=defaults.workflow_path, repo_root=repo_root)
    frames_dir = resolve_input_path(frames_value, label="frames_dir", workflow_path=defaults.workflow_path, repo_root=repo_root)
    conditioning_audio = (
        resolve_input_path(conditioning_value, label="conditioning_audio", workflow_path=defaults.workflow_path, repo_root=repo_root)
        if conditioning_value is not None
        else audio_path
    )

    segment_seconds = max(int(args.segment_seconds if args.segment_seconds is not None else defaults.segment_seconds), 1)
    fps = max(float(args.fps if args.fps is not None else defaults.fps), 1.0)
    image_selection_seed = int(args.seed if args.seed is not None else defaults.image_selection_seed)
    ltx_seed_base = int(args.ltx_seed if args.ltx_seed is not None else defaults.inference_seed)
    prompt = str(args.prompt if args.prompt is not None else defaults.prompt)
    negative_prompt = str(args.negative_prompt if args.negative_prompt is not None else defaults.negative_prompt)
    use_text_to_video = bool(args.text_to_video or defaults.use_text_to_video)
    requested_width = int(args.width if args.width is not None else defaults.width)
    requested_height = int(args.height if args.height is not None else defaults.height)
    width = _normalize_dimension(requested_width)
    height = _normalize_dimension(requested_height)

    audio_info = probe_audio_info(Path(audio_path))
    image_paths = list_frame_images(
        Path(frames_dir),
        image_load_cap=defaults.image_load_cap,
        skip_first_images=defaults.skip_first_images,
        select_every_nth=defaults.select_every_nth,
    )
    segments = plan_segments(audio_info.duration_seconds, segment_seconds, fps, image_paths, image_selection_seed)

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir is not None else _default_output_dir(Path(audio_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = _build_runtime_config(args, output_dir)
    debug_logger = RunDebugLogger(
        enabled=runtime.debug,
        log_path=runtime.debug_log_path,
        echo=bool(args.debug),
    )
    debug_logger.event(
        "run_start",
        workflow_path=defaults.workflow_path,
        audio_path=audio_path,
        conditioning_audio=conditioning_audio,
        frames_dir=frames_dir,
        output_dir=output_dir,
        segment_seconds=segment_seconds,
        fps=fps,
        requested_width=requested_width,
        requested_height=requested_height,
        width=width,
        height=height,
        use_text_to_video=use_text_to_video,
        prompt_encoder_device=runtime.prompt_encoder_device,
        performance_profile=runtime.performance_profile,
        streaming_prefetch_count=runtime.streaming_prefetch_count,
        prompt_streaming_prefetch_count=runtime.prompt_streaming_prefetch_count,
        max_batch_size=runtime.max_batch_size,
        compile_transformer=runtime.compile_transformer,
    )
    if runtime.run_commands or runtime.emit_run_script:
        runtime = LTX23RuntimeConfig(
            pipeline_module=runtime.pipeline_module,
            python_executable=runtime.python_executable,
            ltx_repo_root=runtime.ltx_repo_root,
            checkpoint_path=Path(_required_path(runtime.checkpoint_path, "LTX2_CHECKPOINT_PATH")),
            distilled_lora_path=Path(_required_path(runtime.distilled_lora_path, "LTX2_DISTILLED_LORA_PATH")),
            distilled_lora_strength=runtime.distilled_lora_strength,
            spatial_upsampler_path=Path(_required_path(runtime.spatial_upsampler_path, "LTX2_SPATIAL_UPSAMPLER_PATH")),
            gemma_root=Path(_required_gemma_root(runtime.gemma_root, "LTX2_GEMMA_ROOT")),
            output_dir=runtime.output_dir,
            num_inference_steps=runtime.num_inference_steps,
            image_strength=runtime.image_strength,
            image_crf=runtime.image_crf,
            quantization=runtime.quantization,
            enhance_prompt=runtime.enhance_prompt,
            video_cfg_guidance_scale=runtime.video_cfg_guidance_scale,
            prompt_encoder_device=runtime.prompt_encoder_device,
            performance_profile=runtime.performance_profile,
            streaming_prefetch_count=runtime.streaming_prefetch_count,
            prompt_streaming_prefetch_count=runtime.prompt_streaming_prefetch_count,
            max_batch_size=runtime.max_batch_size,
            compile_transformer=runtime.compile_transformer,
            debug=runtime.debug,
            debug_log_path=runtime.debug_log_path,
            run_commands=runtime.run_commands,
            emit_run_script=runtime.emit_run_script,
            overwrite=runtime.overwrite,
            extra_ltx_args=runtime.extra_ltx_args,
        )
    prepared_audio_paths = (
        _prepare_conditioning_audio_chunks(
            Path(conditioning_audio),
            segments,
            fps=fps,
            output_dir=output_dir,
            overwrite=runtime.overwrite,
        )
        if runtime.run_commands or runtime.emit_run_script
        else None
    )
    debug_logger.event(
        "segments_planned",
        segment_count=len(segments),
        prepared_conditioning_audio=prepared_audio_paths is not None,
        conditioning_chunk_count=0 if prepared_audio_paths is None else len(prepared_audio_paths),
    )
    segment_commands = build_segment_commands(
        defaults=defaults,
        runtime=runtime,
        source_audio=Path(audio_path),
        conditioning_audio=Path(conditioning_audio),
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        fps=fps,
        use_text_to_video=use_text_to_video,
        ltx_seed_base=ltx_seed_base,
        segments=segments,
        prepared_audio_paths=prepared_audio_paths,
    )

    for command in segment_commands:
        Path(command.output_path).parent.mkdir(parents=True, exist_ok=True)

    rendered_segments: list[Path] = []
    video_only_path: Path | None = None
    final_video_path: Path | None = None
    if runtime.run_commands:
        runtime_load_started = time.perf_counter()
        ltx_runtime = _load_ltx_inference_runtime(runtime.ltx_repo_root)
        debug_logger.event(
            "runtime_loaded",
            seconds=round(time.perf_counter() - runtime_load_started, 3),
            repo_root=runtime.ltx_repo_root,
            pipeline_module=runtime.pipeline_module,
        )
        rendered_segments = _run_segments_in_process(
            ltx_runtime=ltx_runtime,
            segment_commands=segment_commands,
            runtime_config=runtime,
            debug_logger=debug_logger,
        )
        if rendered_segments and not args.skip_final_concat:
            concat_started = time.perf_counter()
            video_only_path = _concat_video_streams(
                rendered_segments,
                output_dir / f"{defaults.output_prefix}_video_only.mp4",
                overwrite=runtime.overwrite,
            )
            debug_logger.event(
                "video_concat_done",
                seconds=round(time.perf_counter() - concat_started, 3),
                video_only_path=video_only_path,
                rendered_segments=rendered_segments,
            )
            mux_started = time.perf_counter()
            final_video_path = _mux_original_audio(
                video_only_path,
                Path(audio_path),
                output_dir / f"{defaults.output_prefix}.mp4",
                overwrite=runtime.overwrite,
            )
            debug_logger.event(
                "final_mux_done",
                seconds=round(time.perf_counter() - mux_started, 3),
                final_video_path=final_video_path,
            )

    run_script_path: Path | None = None
    if runtime.emit_run_script:
        run_script_path = _emit_run_script(
            runtime,
            source_audio=Path(audio_path),
            output_prefix=defaults.output_prefix,
            segment_commands=segment_commands,
            include_final_concat=not args.skip_final_concat,
        )
        debug_logger.event("run_script_emitted", run_script_path=run_script_path)

    manifest = build_base_manifest(
        workflow_path=defaults.workflow_path,
        audio_info=audio_info,
        frames_dir=Path(frames_dir),
        fps=fps,
        segment_seconds=segment_seconds,
        seed=image_selection_seed,
        image_load_cap=defaults.image_load_cap,
        skip_first_images=defaults.skip_first_images,
        select_every_nth=defaults.select_every_nth,
        segments=list(segments),
    )
    manifest["source_audio"] = str(Path(audio_path).resolve())
    manifest["conditioning_audio"] = str(Path(conditioning_audio).resolve())
    manifest["prompt"] = prompt
    manifest["negative_prompt"] = negative_prompt
    manifest["use_text_to_video"] = use_text_to_video
    manifest["text_to_video"] = use_text_to_video
    manifest["use_only_vocals"] = defaults.use_only_vocals
    manifest["requested_width"] = requested_width
    manifest["requested_height"] = requested_height
    manifest["width"] = width
    manifest["height"] = height
    manifest["ltx_seed_base"] = ltx_seed_base
    manifest["pipeline_module"] = runtime.pipeline_module
    manifest["prompt_encoder_device"] = runtime.prompt_encoder_device
    manifest["performance_profile"] = runtime.performance_profile
    manifest["streaming_prefetch_count"] = runtime.streaming_prefetch_count
    manifest["prompt_streaming_prefetch_count"] = runtime.prompt_streaming_prefetch_count
    manifest["max_batch_size"] = runtime.max_batch_size
    manifest["compile_transformer"] = runtime.compile_transformer
    manifest["ltx_repo_root"] = str(runtime.ltx_repo_root) if runtime.ltx_repo_root is not None else None
    manifest["run_script"] = str(run_script_path) if run_script_path is not None else None
    manifest["command_preview"] = [segment.command for segment in segment_commands]
    manifest["segment_commands"] = [asdict(segment) for segment in segment_commands]
    manifest["rendered_segments"] = [str(path) for path in rendered_segments]
    manifest["video_only_output"] = str(video_only_path) if video_only_path is not None else None
    manifest["final_video"] = str(final_video_path) if final_video_path is not None else None
    manifest["debug_log"] = str(runtime.debug_log_path) if runtime.debug_log_path is not None else None
    manifest["notes"] = [
        "Backend target: official LTX-2 a2vid two-stage pipeline in-process.",
        "Segment commands use full audio plus --audio-start-time/--audio-max-duration.",
        "Final mux always restores the original source audio after segment video concatenation.",
        "If use_only_vocals is enabled in the workflow, pass --conditioning-audio with a prepared stem.",
        f"Prompt encoder device override: {runtime.prompt_encoder_device}.",
        f"Performance profile: {runtime.performance_profile}.",
        f"Prompt streaming prefetch count: {runtime.prompt_streaming_prefetch_count}.",
        "Use --debug to emit step-by-step progress plus a JSONL debug log.",
        "Resolution is normalized down to multiples of 64 for the official two-stage backend.",
    ]
    manifest_path = write_manifest(manifest, output_dir / args.manifest_name)
    debug_logger.event("manifest_written", manifest_path=manifest_path, final_video=final_video_path)

    print(f"Workflow defaults: {defaults.workflow_path}")
    print(f"Audio: {audio_info.path}")
    print(f"Conditioning audio: {Path(conditioning_audio).resolve()}")
    print(f"Frames: {Path(frames_dir).resolve()}")
    print(f"Segments planned: {len(segment_commands)}")
    print(
        "Runtime tuning: "
        f"profile={runtime.performance_profile}, "
        f"streaming_prefetch_count={runtime.streaming_prefetch_count}, "
        f"prompt_streaming_prefetch_count={runtime.prompt_streaming_prefetch_count}, "
        f"max_batch_size={runtime.max_batch_size}, "
        f"compile_transformer={runtime.compile_transformer}"
    )
    print(f"Manifest: {manifest_path}")
    if runtime.debug_log_path is not None:
        print(f"Debug log: {runtime.debug_log_path}")
    if run_script_path is not None:
        print(f"Run script: {run_script_path}")
    if final_video_path is not None:
        print(f"Final video: {final_video_path}")
    elif runtime.run_commands:
        print("Final video: skipped")
    else:
        print("Render execution: skipped")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(argv)


__all__ = [
    "DEFAULT_MANIFEST_NAME",
    "DEFAULT_PIPELINE_MODULE",
    "DEFAULT_WORKFLOW",
    "DEFAULT_NEGATIVE_PROMPT",
    "LTX23RuntimeConfig",
    "LTX23WorkflowDefaults",
    "Ltx23Assets",
    "SegmentCommand",
    "SegmentRenderRequest",
    "build_segment_command",
    "build_segment_commands",
    "load_ltx23_workflow_defaults",
    "main",
    "parse_args",
    "plan_segments",
    "run",
]


if __name__ == "__main__":
    raise SystemExit(main())
