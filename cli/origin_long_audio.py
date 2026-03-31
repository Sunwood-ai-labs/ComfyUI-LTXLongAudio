from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_MANIFEST_NAME = "origin_long_audio_manifest.json"
DEFAULT_WORKFLOW = Path("samples/workflows/LTX_2.3_Image_or_Text_&_Audio_2_Video_App_Origin.json")

WORKFLOW_NODE_IDS = {
    "frames_dir": 167,
    "fps": 285,
    "segment_seconds": 291,
    "audio": 372,
    "seed": 399,
    "prompt": 352,
    "text_to_video": 290,
    "width": 292,
    "height": 293,
    "video_combine": 140,
}


@dataclass(frozen=True)
class WorkflowDefaults:
    workflow_path: Path
    audio: str | None
    frames_dir: str | None
    segment_seconds: int
    fps: float
    seed: int
    prompt: str
    text_to_video: bool
    width: int
    height: int
    output_prefix: str


@dataclass(frozen=True)
class ResolvedInputs:
    workflow_path: Path
    audio_path: Path
    frames_dir: Path
    segment_seconds: int
    fps: float
    seed: int


@dataclass(frozen=True)
class SegmentPlan:
    index: int
    start_time: float
    nominal_seconds: float
    exact_seconds: float
    frames: int
    is_last_segment: bool
    image_index: int
    image_path: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the long-audio loop from "
            "LTX_2.3_Image_or_Text_&_Audio_2_Video_App_Origin.json without ComfyUI."
        )
    )
    parser.add_argument("--workflow", type=Path, default=DEFAULT_WORKFLOW, help="Workflow JSON used for defaults.")
    parser.add_argument("--audio", type=Path, default=None, help="Override source audio path.")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Override source image folder.")
    parser.add_argument("--segment-seconds", type=int, default=None, help="Override segment length in seconds.")
    parser.add_argument("--fps", type=float, default=None, help="Override frame rate used for segment quantization.")
    parser.add_argument("--seed", type=int, default=None, help="Override deterministic image-selection seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cli_output") / "origin_long_audio",
        help="Directory for manifest output and optional extracted audio.",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help=f"Manifest filename. Defaults to {DEFAULT_MANIFEST_NAME}.",
    )
    parser.add_argument(
        "--extract-audio",
        action="store_true",
        help="Extract one WAV file per planned segment with ffmpeg.",
    )
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable for --extract-audio.")
    parser.add_argument("--ffprobe", default="ffprobe", help="ffprobe executable for duration probing.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting the output directory.")
    return parser.parse_args(argv)


def _coerce_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def load_workflow_defaults(workflow_path: Path) -> WorkflowDefaults:
    resolved = workflow_path.expanduser().resolve()
    data = json.loads(resolved.read_text(encoding="utf-8"))
    node_by_id = {int(node["id"]): node for node in data.get("nodes", [])}

    def widget_value(node_id: int, index: int, default: object) -> object:
        node = node_by_id.get(node_id)
        if node is None:
            return default
        values = node.get("widgets_values") or []
        if index >= len(values):
            return default
        return values[index]

    return WorkflowDefaults(
        workflow_path=resolved,
        audio=_coerce_text(widget_value(WORKFLOW_NODE_IDS["audio"], 0, None)),
        frames_dir=_coerce_text(widget_value(WORKFLOW_NODE_IDS["frames_dir"], 0, None)),
        segment_seconds=int(widget_value(WORKFLOW_NODE_IDS["segment_seconds"], 0, 20)),
        fps=float(widget_value(WORKFLOW_NODE_IDS["fps"], 0, 24.0)),
        seed=int(widget_value(WORKFLOW_NODE_IDS["seed"], 0, 1234)),
        prompt=str(widget_value(WORKFLOW_NODE_IDS["prompt"], 0, "") or ""),
        text_to_video=bool(widget_value(WORKFLOW_NODE_IDS["text_to_video"], 0, False)),
        width=int(widget_value(WORKFLOW_NODE_IDS["width"], 0, 832)),
        height=int(widget_value(WORKFLOW_NODE_IDS["height"], 0, 480)),
        output_prefix=str(widget_value(WORKFLOW_NODE_IDS["video_combine"], 2, "LTX-2.3-longaudio-randomimg") or "LTX-2.3-longaudio-randomimg"),
    )


def find_repo_root(start: Path) -> Path:
    resolved = start.expanduser().resolve()
    candidates = [resolved.parent, *resolved.parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return resolved.parent


def resolve_input_path(
    value: str | Path | None,
    *,
    label: str,
    workflow_path: Path,
    repo_root: Path,
) -> Path:
    if value is None:
        raise ValueError(f"{label} is not set in the workflow; pass --{label.replace('_', '-')}.")

    raw = Path(value).expanduser()
    if raw.is_absolute():
        if raw.exists():
            return raw.resolve()
        raise FileNotFoundError(f"{label} does not exist: {raw}")

    search_roots = [
        Path.cwd(),
        workflow_path.parent,
        repo_root,
        repo_root / "samples" / "input",
    ]
    for root in search_roots:
        candidate = (root / raw).resolve()
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve {label} '{value}'. "
        f"Searched: {', '.join(str(root) for root in search_roots)}"
    )


def resolve_inputs(args: argparse.Namespace) -> ResolvedInputs:
    defaults = load_workflow_defaults(args.workflow)
    repo_root = find_repo_root(defaults.workflow_path)

    audio_value = args.audio if args.audio is not None else defaults.audio
    frames_value = args.frames_dir if args.frames_dir is not None else defaults.frames_dir

    return ResolvedInputs(
        workflow_path=defaults.workflow_path,
        audio_path=resolve_input_path(audio_value, label="audio", workflow_path=defaults.workflow_path, repo_root=repo_root),
        frames_dir=resolve_input_path(
            frames_value,
            label="frames_dir",
            workflow_path=defaults.workflow_path,
            repo_root=repo_root,
        ),
        segment_seconds=max(int(args.segment_seconds if args.segment_seconds is not None else defaults.segment_seconds), 1),
        fps=max(float(args.fps if args.fps is not None else defaults.fps), 1.0),
        seed=int(args.seed if args.seed is not None else defaults.seed),
    )


def list_image_files(frames_dir: Path) -> list[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"frames_dir does not exist: {frames_dir}")
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"frames_dir is not a directory: {frames_dir}")

    images = sorted(path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No supported images found in {frames_dir}")
    return images


def probe_audio_duration(audio_path: Path, *, ffprobe: str = "ffprobe") -> float:
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout or "{}")
        duration = float(payload["format"]["duration"])
        if duration <= 0:
            raise ValueError(f"Audio duration must be positive: {audio_path}")
        return duration
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError):
        if audio_path.suffix.lower() == ".wav":
            with wave.open(str(audio_path), "rb") as reader:
                frame_rate = max(reader.getframerate(), 1)
                return reader.getnframes() / float(frame_rate)
        raise


def plan_segments(total_duration: float, segment_seconds: int, fps: float) -> list[dict[str, float | int | bool]]:
    total_duration = max(float(total_duration), 0.0)
    segment_seconds = max(int(segment_seconds), 1)
    fps = max(float(fps), 1.0)
    segment_count = int(math.ceil(total_duration / segment_seconds)) if total_duration > 0 else 0

    plans: list[dict[str, float | int | bool]] = []
    for index in range(segment_count):
        start_time = float(index * segment_seconds)
        remaining = max(total_duration - start_time, 0.0)
        current_seconds = min(float(segment_seconds), remaining)
        is_last_segment = remaining < float(segment_seconds)
        frame_blocks = math.floor((current_seconds * fps) / 8.0) if current_seconds > 0 else 0
        frames = 1 + max(frame_blocks, 0) * 8
        exact_seconds = float((frames - 1) / fps) if frames > 1 else 0.0
        plans.append(
            {
                "index": index,
                "start_time": start_time,
                "nominal_seconds": float(current_seconds),
                "exact_seconds": exact_seconds,
                "frames": int(frames),
                "is_last_segment": bool(is_last_segment),
            }
        )
    return plans


def build_segment_plan(
    total_duration: float,
    image_paths: Sequence[Path],
    *,
    segment_seconds: int,
    fps: float,
    seed: int,
) -> list[SegmentPlan]:
    plans = plan_segments(total_duration, segment_seconds, fps)
    if not image_paths:
        raise ValueError("At least one image is required to build the segment plan.")

    segment_plans: list[SegmentPlan] = []
    for plan in plans:
        index = int(plan["index"])
        rng = random.Random(int(seed) + (index * 9973))
        image_index = rng.randrange(len(image_paths))
        segment_plans.append(
            SegmentPlan(
                index=index,
                start_time=float(plan["start_time"]),
                nominal_seconds=float(plan["nominal_seconds"]),
                exact_seconds=float(plan["exact_seconds"]),
                frames=int(plan["frames"]),
                is_last_segment=bool(plan["is_last_segment"]),
                image_index=image_index,
                image_path=str(image_paths[image_index].resolve()),
            )
        )
    return segment_plans


def ensure_output_dir(output_dir: Path, *, overwrite: bool) -> Path:
    resolved = output_dir.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a directory: {resolved}")
        if any(resolved.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {resolved}. "
                "Pass --overwrite to reuse it."
            )
    else:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def extract_audio_segments(
    audio_path: Path,
    plans: Sequence[SegmentPlan],
    *,
    output_dir: Path,
    ffmpeg: str = "ffmpeg",
    overwrite: bool = False,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    for plan in plans:
        chunk_path = output_dir / f"segment_{plan.index + 1:03d}.wav"
        if chunk_path.exists() and not overwrite:
            raise FileExistsError(f"Chunk file already exists: {chunk_path}. Pass --overwrite to replace it.")

        duration = plan.exact_seconds if plan.exact_seconds > 0 else plan.nominal_seconds
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-i",
            str(audio_path),
            "-ss",
            f"{plan.start_time:.6f}",
            "-t",
            f"{duration:.6f}",
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(chunk_path),
        ]
        subprocess.run(command, check=True)
        chunk_paths.append(chunk_path.resolve())
    return chunk_paths


def build_manifest(
    inputs: ResolvedInputs,
    *,
    total_duration: float,
    plans: Sequence[SegmentPlan],
    extracted_audio: Sequence[Path] | None = None,
) -> dict[str, object]:
    extracted_map = {path.stem: path for path in extracted_audio or []}
    segments: list[dict[str, object]] = []
    for plan in plans:
        audio_key = f"segment_{plan.index + 1:03d}"
        segment_entry = asdict(plan)
        segment_entry["audio_path"] = str(extracted_map[audio_key]) if audio_key in extracted_map else None
        segments.append(segment_entry)

    return {
        "workflow_path": str(inputs.workflow_path),
        "audio_path": str(inputs.audio_path),
        "frames_dir": str(inputs.frames_dir),
        "segment_seconds": inputs.segment_seconds,
        "fps": inputs.fps,
        "seed": inputs.seed,
        "audio_duration": total_duration,
        "segment_count": len(plans),
        "segments": segments,
    }


def write_manifest(manifest: dict[str, object], output_path: Path) -> Path:
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    inputs = resolve_inputs(args)
    output_dir = ensure_output_dir(args.output_dir, overwrite=args.overwrite)

    image_paths = list_image_files(inputs.frames_dir)
    total_duration = probe_audio_duration(inputs.audio_path, ffprobe=args.ffprobe)
    plans = build_segment_plan(
        total_duration,
        image_paths,
        segment_seconds=inputs.segment_seconds,
        fps=inputs.fps,
        seed=inputs.seed,
    )

    extracted_audio: list[Path] | None = None
    if args.extract_audio:
        extracted_audio = extract_audio_segments(
            inputs.audio_path,
            plans,
            output_dir=output_dir / "audio_segments",
            ffmpeg=args.ffmpeg,
            overwrite=args.overwrite,
        )

    manifest = build_manifest(inputs, total_duration=total_duration, plans=plans, extracted_audio=extracted_audio)
    manifest_path = write_manifest(manifest, output_dir / args.manifest_name)

    print(f"Planned {len(plans)} segment(s) from {inputs.audio_path}")
    print(f"Frames directory: {inputs.frames_dir}")
    print(f"Manifest: {manifest_path}")
    if extracted_audio:
        print(f"Audio segments: {len(extracted_audio)} file(s) under {output_dir / 'audio_segments'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
