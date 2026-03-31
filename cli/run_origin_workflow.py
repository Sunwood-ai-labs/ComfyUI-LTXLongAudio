from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from cli.long_audio_core import (
    build_manifest,
    export_audio_segments,
    list_frame_images,
    plan_segments,
    probe_audio_info,
    write_manifest,
)


DEFAULT_WORKFLOW_PATH = Path("samples/workflows/LTX_2.3_Image_or_Text_&_Audio_2_Video_App_Origin.json")
DEFAULT_SAMPLE_AUDIO = Path("samples/input/ltx-demo-tone.wav")
DEFAULT_SAMPLE_FRAMES = Path("samples/input/frames_pool")


@dataclass(frozen=True)
class WorkflowDefaults:
    workflow_path: Path
    source_audio: str | None
    frames_dir: str | None
    segment_seconds: int
    fps: float
    seed: int
    image_load_cap: int
    skip_first_images: int
    select_every_nth: int


def _node_widget_value(node: dict, index: int, fallback):
    values = node.get("widgets_values") or []
    return values[index] if len(values) > index else fallback


def load_origin_workflow_defaults(workflow_path: Path) -> WorkflowDefaults:
    import json

    resolved = workflow_path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))

    source_audio = None
    frames_dir = None
    segment_seconds = 20
    fps = 24.0
    seed = 1234
    image_load_cap = 0
    skip_first_images = 0
    select_every_nth = 1

    for node in payload.get("nodes", []):
        title = node.get("title")
        node_type = node.get("type")
        if node_type == "LoadAudio" and title == "Source Song Upload":
            source_audio = _node_widget_value(node, 0, None)
        elif node_type == "LTXLoadImages" and title == "Segment Image Folder":
            frames_dir = _node_widget_value(node, 0, None) or None
            image_load_cap = int(_node_widget_value(node, 1, 0) or 0)
            skip_first_images = int(_node_widget_value(node, 2, 0) or 0)
            select_every_nth = int(_node_widget_value(node, 3, 1) or 1)
        elif title == "SEGMENT SECONDS":
            segment_seconds = int(_node_widget_value(node, 0, 20) or 20)
        elif title == "FPS":
            fps = float(_node_widget_value(node, 0, 24.0) or 24.0)
        elif title == "RANDOM IMAGE SEED":
            seed = int(_node_widget_value(node, 0, 1234) or 1234)

    return WorkflowDefaults(
        workflow_path=resolved,
        source_audio=source_audio,
        frames_dir=frames_dir,
        segment_seconds=segment_seconds,
        fps=fps,
        seed=seed,
        image_load_cap=image_load_cap,
        skip_first_images=skip_first_images,
        select_every_nth=select_every_nth,
    )


def _resolve_candidate_path(base: Path, candidate: str | None) -> Path | None:
    if not candidate:
        return None
    path = Path(candidate)
    if path.is_absolute() and path.exists():
        return path.resolve()
    direct = (base / candidate).resolve()
    if direct.exists():
        return direct
    sample_input = (base.parent / "input" / candidate).resolve()
    if sample_input.exists():
        return sample_input
    return None


def _default_output_dir(audio_path: Path) -> Path:
    return audio_path.with_name(f"{audio_path.stem}_origin_cli")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the long-audio planning portion of the Origin workflow without ComfyUI. "
            "This CLI does not perform LTX model inference or MP4 generation."
        )
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        default=DEFAULT_WORKFLOW_PATH,
        help="Path to the Origin workflow JSON used only for default values.",
    )
    parser.add_argument("--audio", type=Path, default=None, help="Source audio path. Overrides the workflow widget value.")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Directory containing still images. Overrides the workflow widget value.",
    )
    parser.add_argument("--segment-seconds", type=int, default=None, help="Segment length in seconds.")
    parser.add_argument("--fps", type=float, default=None, help="Frames-per-second used for segment quantization.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic frame selection.")
    parser.add_argument("--image-load-cap", type=int, default=None, help="Optional cap on frame count.")
    parser.add_argument("--skip-first-images", type=int, default=None, help="Number of leading images to skip.")
    parser.add_argument("--select-every-nth", type=int, default=None, help="Only keep every Nth image.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for manifest and optional audio chunks.")
    parser.add_argument(
        "--manifest-name",
        default="origin_long_audio_manifest.json",
        help="Output manifest filename. Defaults to origin_long_audio_manifest.json.",
    )
    parser.add_argument(
        "--export-wav-chunks",
        action="store_true",
        help="Export one WAV file per planned segment using ffmpeg or WAV fallback.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow reusing a non-empty output directory.")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    defaults = load_origin_workflow_defaults(args.workflow)
    workflow_root = defaults.workflow_path.parent

    audio_path = (
        args.audio
        or _resolve_candidate_path(workflow_root, defaults.source_audio)
        or _resolve_candidate_path(workflow_root, str(DEFAULT_SAMPLE_AUDIO))
    )
    if audio_path is None:
        raise FileNotFoundError("No audio file could be resolved. Pass --audio explicitly.")

    frames_dir = (
        args.frames_dir
        or _resolve_candidate_path(workflow_root, defaults.frames_dir)
        or _resolve_candidate_path(workflow_root, str(DEFAULT_SAMPLE_FRAMES))
    )
    if frames_dir is None:
        raise FileNotFoundError("No frames directory could be resolved. Pass --frames-dir explicitly.")

    segment_seconds = args.segment_seconds if args.segment_seconds is not None else defaults.segment_seconds
    fps = args.fps if args.fps is not None else defaults.fps
    seed = args.seed if args.seed is not None else defaults.seed
    image_load_cap = args.image_load_cap if args.image_load_cap is not None else defaults.image_load_cap
    skip_first_images = args.skip_first_images if args.skip_first_images is not None else defaults.skip_first_images
    select_every_nth = args.select_every_nth if args.select_every_nth is not None else defaults.select_every_nth

    audio_info = probe_audio_info(audio_path)
    image_paths = list_frame_images(
        frames_dir,
        image_load_cap=image_load_cap,
        skip_first_images=skip_first_images,
        select_every_nth=select_every_nth,
    )
    segments = plan_segments(audio_info.duration_seconds, segment_seconds, fps, image_paths, seed)

    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir is not None else _default_output_dir(audio_path))
    manifest_segments = segments
    if args.export_wav_chunks:
        manifest_segments = export_audio_segments(
            audio_info.path,
            segments,
            output_dir / "audio_segments",
            overwrite=args.overwrite,
        )

    manifest = build_manifest(
        workflow_path=defaults.workflow_path,
        audio_info=audio_info,
        frames_dir=frames_dir.resolve(),
        fps=fps,
        segment_seconds=segment_seconds,
        seed=seed,
        image_load_cap=image_load_cap,
        skip_first_images=skip_first_images,
        select_every_nth=select_every_nth,
        segments=manifest_segments,
    )
    manifest_path = write_manifest(manifest, output_dir / args.manifest_name)

    print(f"Workflow defaults: {defaults.workflow_path}")
    print(f"Audio: {audio_info.path}")
    print(f"Frames: {frames_dir.resolve()}")
    print(f"Segments planned: {len(manifest_segments)}")
    print(f"Manifest: {manifest_path}")
    if args.export_wav_chunks:
        print(f"Audio chunks: {output_dir / 'audio_segments'}")
    return 0


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())

