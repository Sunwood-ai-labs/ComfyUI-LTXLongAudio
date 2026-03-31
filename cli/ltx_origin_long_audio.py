from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "samples"
    / "workflows"
    / "LTX_2.3_Image_or_Text_&_Audio_2_Video_App_Origin.json"
)
DEFAULT_SEGMENT_SECONDS = 20
DEFAULT_FPS = 24.0
DEFAULT_RANDOM_SEED = 1234
DEFAULT_OUTPUT_PREFIX = "ltx-origin-cli"


@dataclass(frozen=True)
class WorkflowDefaults:
    workflow_path: Path
    frames_directory: str | None
    image_load_cap: int
    skip_first_images: int
    select_every_nth: int
    source_audio: str | None
    segment_seconds: int
    fps: float
    random_seed: int
    prompt: str
    use_text_to_video: bool
    use_only_vocals: bool
    output_prefix: str


@dataclass(frozen=True)
class SegmentPlan:
    index: int
    start_time: float
    nominal_seconds: float
    exact_seconds: float
    frame_count: int
    is_last_segment: bool


@dataclass(frozen=True)
class SegmentArtifact:
    index: int
    image_path: Path
    audio_path: Path | None
    video_path: Path | None
    start_time: float
    exact_seconds: float
    frame_count: int


def _read_workflow(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_node(nodes: list[dict[str, Any]], *, title: str) -> dict[str, Any]:
    for node in nodes:
        if node.get("title") == title:
            return node
    raise KeyError(f"Workflow node with title '{title}' was not found.")


def _widget_value(node: dict[str, Any], index: int, default: Any) -> Any:
    values = node.get("widgets_values") or []
    if index >= len(values):
        return default
    value = values[index]
    return default if value is None else value


def parse_origin_workflow_defaults(workflow_path: Path = DEFAULT_WORKFLOW) -> WorkflowDefaults:
    resolved = workflow_path.expanduser().resolve()
    data = _read_workflow(resolved)
    nodes = data.get("nodes", [])

    image_node = _find_node(nodes, title="Segment Image Folder")
    segment_node = _find_node(nodes, title="SEGMENT SECONDS")
    fps_node = _find_node(nodes, title="FPS")
    audio_node = _find_node(nodes, title="Source Song Upload")
    seed_node = _find_node(nodes, title="RANDOM IMAGE SEED")
    prompt_node = _find_node(nodes, title="PROMPT")
    text_to_video_node = _find_node(nodes, title="Text To Video (no image ref)")
    vocals_node = _find_node(nodes, title="USE ONLY VOCALS")
    video_node = _find_node(nodes, title="Video Combine (20s Long Audio)")

    frames_directory = _widget_value(image_node, 0, "") or None
    source_audio = _widget_value(audio_node, 0, "") or None
    prompt = _widget_value(prompt_node, 0, "")
    output_prefix = _widget_value(video_node, 2, DEFAULT_OUTPUT_PREFIX) or DEFAULT_OUTPUT_PREFIX

    return WorkflowDefaults(
        workflow_path=resolved,
        frames_directory=frames_directory,
        image_load_cap=int(_widget_value(image_node, 1, 0)),
        skip_first_images=int(_widget_value(image_node, 2, 0)),
        select_every_nth=int(_widget_value(image_node, 3, 1)),
        source_audio=source_audio,
        segment_seconds=int(_widget_value(segment_node, 0, DEFAULT_SEGMENT_SECONDS)),
        fps=float(_widget_value(fps_node, 0, DEFAULT_FPS)),
        random_seed=int(_widget_value(seed_node, 0, DEFAULT_RANDOM_SEED)),
        prompt=str(prompt),
        use_text_to_video=bool(_widget_value(text_to_video_node, 0, False)),
        use_only_vocals=bool(_widget_value(vocals_node, 0, False)),
        output_prefix=str(output_prefix),
    )


def _resolve_repo_relative_path(value: str | None, *, repo_root: Path) -> Path | None:
    if not value:
        return None

    raw = Path(value).expanduser()
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend(
            [
                repo_root / raw,
                repo_root / "samples" / "input" / raw,
                repo_root / "input" / raw,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def list_frame_images(
    frames_dir: Path,
    *,
    image_load_cap: int = 0,
    skip_first_images: int = 0,
    select_every_nth: int = 1,
) -> list[Path]:
    resolved = frames_dir.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Frames directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Frames path is not a directory: {resolved}")

    image_paths = sorted(
        path
        for path in resolved.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No image files were found in: {resolved}")

    start_index = max(int(skip_first_images), 0)
    nth = max(int(select_every_nth), 1)
    selected = image_paths[start_index::nth]
    if int(image_load_cap) > 0:
        selected = selected[: int(image_load_cap)]
    if not selected:
        raise ValueError("Frame selection resulted in zero images.")
    return selected


def probe_audio_duration(audio_path: Path, *, ffprobe_exe: str = "ffprobe") -> float:
    resolved = audio_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file does not exist: {resolved}")

    if resolved.suffix.lower() == ".wav":
        with wave.open(str(resolved), "rb") as reader:
            frame_rate = max(reader.getframerate(), 1)
            return float(reader.getnframes() / frame_rate)

    command = [
        ffprobe_exe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(resolved),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    duration = payload.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError(f"ffprobe did not return a duration for {resolved}")
    return float(duration)


def plan_segment(audio_duration: float, segment_seconds: int, index: int, fps: float) -> SegmentPlan:
    total_duration = max(float(audio_duration), 0.0)
    safe_segment_seconds = max(int(segment_seconds), 1)
    safe_index = max(int(index), 0)
    safe_fps = max(float(fps), 1.0)

    start_time = float(safe_index * safe_segment_seconds)
    remaining = max(total_duration - start_time, 0.0)
    current_seconds = min(float(safe_segment_seconds), remaining)
    is_last_segment = remaining < float(safe_segment_seconds)
    frame_blocks = math.floor((current_seconds * safe_fps) / 8.0) if current_seconds > 0 else 0
    frame_count = 1 + max(frame_blocks, 0) * 8
    exact_seconds = float((frame_count - 1) / safe_fps) if frame_count > 1 else 0.0

    return SegmentPlan(
        index=safe_index,
        start_time=start_time,
        nominal_seconds=float(current_seconds),
        exact_seconds=exact_seconds,
        frame_count=int(frame_count),
        is_last_segment=bool(is_last_segment),
    )


def build_segment_plans(audio_duration: float, segment_seconds: int, fps: float) -> list[SegmentPlan]:
    safe_duration = max(float(audio_duration), 0.0)
    safe_segment_seconds = max(int(segment_seconds), 1)
    segment_count = int(math.ceil(safe_duration / safe_segment_seconds)) if safe_duration > 0 else 0
    plans = [plan_segment(safe_duration, safe_segment_seconds, index, fps) for index in range(segment_count)]
    for plan in plans:
        if plan.exact_seconds <= 0:
            raise ValueError(
                "Segment duration collapsed to zero after frame quantization. "
                "Reduce fps or use a longer audio segment."
            )
    return plans


def pick_segment_image_index(image_count: int, segment_index: int, seed: int) -> int:
    safe_image_count = max(int(image_count), 1)
    safe_segment_index = max(int(segment_index), 0)
    rng = random.Random(int(seed) + (safe_segment_index * 9973))
    return rng.randrange(safe_image_count)


def select_segment_images(image_paths: list[Path], segment_plans: list[SegmentPlan], seed: int) -> list[Path]:
    return [image_paths[pick_segment_image_index(len(image_paths), plan.index, seed)] for plan in segment_plans]


def default_output_dir(audio_path: Path) -> Path:
    return audio_path.expanduser().resolve().with_name(f"{audio_path.stem}_origin_cli")


def _run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        binary = Path(command[0]).name
        raise FileNotFoundError(f"Required executable was not found on PATH: {binary}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Command failed with exit code {exc.returncode}: {' '.join(command)}\n{exc.stderr.strip()}"
        ) from exc


def _prepare_output_dir(path: Path, *, overwrite: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a directory: {resolved}")
        if any(resolved.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {resolved}. "
                "Pass --overwrite to reuse it."
            )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def extract_audio_chunks(
    audio_path: Path,
    segment_plans: list[SegmentPlan],
    *,
    output_dir: Path,
    ffmpeg_exe: str = "ffmpeg",
    overwrite: bool = False,
) -> list[Path]:
    audio_dir = output_dir / "audio_chunks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths: list[Path] = []
    for plan in segment_plans:
        chunk_path = audio_dir / f"segment_{plan.index + 1:03d}.wav"
        if chunk_path.exists() and not overwrite:
            raise FileExistsError(f"Chunk already exists: {chunk_path}")
        command = [
            ffmpeg_exe,
            "-y" if overwrite else "-n",
            "-i",
            str(audio_path),
            "-ss",
            f"{plan.start_time:.6f}",
            "-t",
            f"{plan.exact_seconds:.6f}",
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(chunk_path),
        ]
        _run_command(command)
        chunk_paths.append(chunk_path)
    return chunk_paths


def render_segment_videos(
    segment_plans: list[SegmentPlan],
    image_paths: list[Path],
    audio_paths: list[Path],
    *,
    fps: float,
    output_dir: Path,
    ffmpeg_exe: str = "ffmpeg",
    overwrite: bool = False,
) -> list[Path]:
    video_dir = output_dir / "video_segments"
    video_dir.mkdir(parents=True, exist_ok=True)
    segment_videos: list[Path] = []
    for plan, image_path, audio_path in zip(segment_plans, image_paths, audio_paths, strict=True):
        segment_path = video_dir / f"segment_{plan.index + 1:03d}.mp4"
        if segment_path.exists() and not overwrite:
            raise FileExistsError(f"Segment video already exists: {segment_path}")
        command = [
            ffmpeg_exe,
            "-y" if overwrite else "-n",
            "-loop",
            "1",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(image_path),
            "-i",
            str(audio_path),
            "-t",
            f"{plan.exact_seconds:.6f}",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(segment_path),
        ]
        _run_command(command)
        segment_videos.append(segment_path)
    return segment_videos


def concatenate_segment_videos(
    segment_videos: list[Path],
    *,
    output_path: Path,
    ffmpeg_exe: str = "ffmpeg",
    overwrite: bool = False,
) -> Path:
    if not segment_videos:
        raise ValueError("At least one segment video is required for concatenation.")

    concat_file = output_path.parent / "video_segments.txt"
    concat_lines = [f"file '{path.as_posix()}'" for path in segment_videos]
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    command = [
        ffmpeg_exe,
        "-y" if overwrite else "-n",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]
    _run_command(command)
    return output_path


def write_manifest(
    output_dir: Path,
    *,
    workflow_defaults: WorkflowDefaults,
    audio_path: Path,
    frame_images: list[Path],
    segment_plans: list[SegmentPlan],
    selected_images: list[Path],
    audio_chunks: list[Path] | None,
    segment_videos: list[Path] | None,
    final_video: Path | None,
) -> Path:
    manifest_path = output_dir / "manifest.json"
    artifacts: list[dict[str, Any]] = []
    for index, plan in enumerate(segment_plans):
        artifacts.append(
            {
                "index": plan.index,
                "start_time": plan.start_time,
                "nominal_seconds": plan.nominal_seconds,
                "exact_seconds": plan.exact_seconds,
                "frame_count": plan.frame_count,
                "is_last_segment": plan.is_last_segment,
                "selected_image": str(selected_images[index]),
                "audio_chunk": str(audio_chunks[index]) if audio_chunks is not None else None,
                "segment_video": str(segment_videos[index]) if segment_videos is not None else None,
            }
        )

    manifest = {
        "workflow_defaults": {
            **asdict(workflow_defaults),
            "workflow_path": str(workflow_defaults.workflow_path),
        },
        "audio_path": str(audio_path),
        "frame_image_count": len(frame_images),
        "final_video": str(final_video) if final_video is not None else None,
        "segments": artifacts,
        "excluded_workflow_stages": [
            "Prompt enhancement and text/image conditioning nodes",
            "Model loading and latent sampler nodes",
            "Optional vocals separation switch",
            "ComfyUI loop-control and App-mode UI nodes",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the long-audio loop from the Origin workflow as a simple Python CLI.",
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=DEFAULT_WORKFLOW,
        help="Path to the Origin workflow JSON used for defaults.",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        default=None,
        help="Source audio file. If omitted, the workflow default is used when it resolves on disk.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory containing still images. If omitted, the workflow default is used when it resolves on disk.",
    )
    parser.add_argument("--segment-seconds", type=int, default=None, help="Override segment length in seconds.")
    parser.add_argument("--fps", type=float, default=None, help="Override output frame rate.")
    parser.add_argument("--seed", type=int, default=None, help="Override deterministic image selection seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <audio-stem>_origin_cli next to the source audio.",
    )
    parser.add_argument("--ffmpeg-exe", default="ffmpeg", help="ffmpeg executable path or name.")
    parser.add_argument("--ffprobe-exe", default="ffprobe", help="ffprobe executable path or name.")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only write the manifest and skip audio extraction and video rendering.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Write the manifest and audio chunks, but skip per-segment and final video rendering.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory and replacing existing outputs.",
    )
    return parser.parse_args(argv)


def _resolve_required_audio(args_audio: Path | None, defaults: WorkflowDefaults, *, repo_root: Path) -> Path:
    if args_audio is not None:
        resolved = args_audio.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Audio file does not exist: {resolved}")
        return resolved

    from_workflow = _resolve_repo_relative_path(defaults.source_audio, repo_root=repo_root)
    if from_workflow is not None:
        return from_workflow
    raise FileNotFoundError(
        "No audio file was resolved. Pass --audio explicitly because the workflow default does not exist locally."
    )


def _resolve_required_frames_dir(args_frames_dir: Path | None, defaults: WorkflowDefaults, *, repo_root: Path) -> Path:
    if args_frames_dir is not None:
        resolved = args_frames_dir.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Frames directory does not exist: {resolved}")
        return resolved

    from_workflow = _resolve_repo_relative_path(defaults.frames_directory, repo_root=repo_root)
    if from_workflow is not None:
        return from_workflow
    raise FileNotFoundError(
        "No frames directory was resolved. Pass --frames-dir explicitly because the workflow default is blank."
    )


def run(args: argparse.Namespace) -> dict[str, Path | None]:
    defaults = parse_origin_workflow_defaults(args.workflow)
    repo_root = defaults.workflow_path.parents[1]

    audio_path = _resolve_required_audio(args.audio, defaults, repo_root=repo_root)
    frames_dir = _resolve_required_frames_dir(args.frames_dir, defaults, repo_root=repo_root)

    segment_seconds = int(args.segment_seconds if args.segment_seconds is not None else defaults.segment_seconds)
    fps = float(args.fps if args.fps is not None else defaults.fps)
    seed = int(args.seed if args.seed is not None else defaults.random_seed)

    frame_images = list_frame_images(
        frames_dir,
        image_load_cap=defaults.image_load_cap,
        skip_first_images=defaults.skip_first_images,
        select_every_nth=defaults.select_every_nth,
    )
    audio_duration = probe_audio_duration(audio_path, ffprobe_exe=args.ffprobe_exe)
    segment_plans = build_segment_plans(audio_duration, segment_seconds, fps)
    selected_images = select_segment_images(frame_images, segment_plans, seed)

    output_dir = _prepare_output_dir(args.output_dir or default_output_dir(audio_path), overwrite=args.overwrite)

    audio_chunks: list[Path] | None = None
    segment_videos: list[Path] | None = None
    final_video: Path | None = None

    if not args.plan_only:
        audio_chunks = extract_audio_chunks(
            audio_path,
            segment_plans,
            output_dir=output_dir,
            ffmpeg_exe=args.ffmpeg_exe,
            overwrite=args.overwrite,
        )

        if not args.skip_video:
            segment_videos = render_segment_videos(
                segment_plans,
                selected_images,
                audio_chunks,
                fps=fps,
                output_dir=output_dir,
                ffmpeg_exe=args.ffmpeg_exe,
                overwrite=args.overwrite,
            )
            final_video = concatenate_segment_videos(
                segment_videos,
                output_path=output_dir / f"{defaults.output_prefix}.mp4",
                ffmpeg_exe=args.ffmpeg_exe,
                overwrite=args.overwrite,
            )

    manifest_path = write_manifest(
        output_dir,
        workflow_defaults=defaults,
        audio_path=audio_path,
        frame_images=frame_images,
        segment_plans=segment_plans,
        selected_images=selected_images,
        audio_chunks=audio_chunks,
        segment_videos=segment_videos,
        final_video=final_video,
    )

    return {
        "output_dir": output_dir,
        "manifest_path": manifest_path,
        "final_video": final_video,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(f"Output directory: {result['output_dir']}")
    print(f"Manifest: {result['manifest_path']}")
    if result["final_video"] is not None:
        print(f"Final video: {result['final_video']}")
    else:
        print("Final video: skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
