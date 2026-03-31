from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_SUFFIX_WIDTH = 3


@dataclass(frozen=True)
class AudioInfo:
    path: Path
    duration_seconds: float
    sample_rate: int | None = None
    channels: int | None = None
    probe: str = "unknown"


@dataclass(frozen=True)
class SegmentSpec:
    index: int
    start_time: float
    current_seconds: float
    exact_seconds: float
    frames: int
    is_last_segment: bool
    selected_image_index: int
    selected_image_path: str
    audio_chunk_path: str | None = None


def _ffprobe_executable() -> str | None:
    return shutil.which("ffprobe")


def _ffmpeg_executable() -> str | None:
    return shutil.which("ffmpeg")


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a directory: {output_dir}")
        if any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Pass --overwrite to reuse it."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def probe_audio_info(audio_path: Path) -> AudioInfo:
    resolved = audio_path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file does not exist: {resolved}")

    ffprobe = _ffprobe_executable()
    if ffprobe is not None:
        command = [
            ffprobe,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-print_format",
            "json",
            str(resolved),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        payload = json.loads(completed.stdout or "{}")
        format_info = payload.get("format", {})
        streams = payload.get("streams", [])
        audio_stream = next((stream for stream in streams if stream.get("codec_type") == "audio"), {})
        raw_duration = audio_stream.get("duration") or format_info.get("duration")
        duration_seconds = float(raw_duration) if raw_duration is not None else 0.0
        sample_rate = int(audio_stream["sample_rate"]) if audio_stream.get("sample_rate") else None
        channels = int(audio_stream["channels"]) if audio_stream.get("channels") else None
        return AudioInfo(
            path=resolved,
            duration_seconds=max(duration_seconds, 0.0),
            sample_rate=sample_rate,
            channels=channels,
            probe="ffprobe",
        )

    if resolved.suffix.lower() != ".wav":
        raise RuntimeError("ffprobe is required for non-WAV input files.")

    with wave.open(str(resolved), "rb") as reader:
        sample_rate = reader.getframerate()
        total_frames = reader.getnframes()
        channels = reader.getnchannels()
    duration_seconds = total_frames / float(max(sample_rate, 1))
    return AudioInfo(
        path=resolved,
        duration_seconds=max(duration_seconds, 0.0),
        sample_rate=sample_rate,
        channels=channels,
        probe="wave",
    )


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

    candidates = [
        path
        for path in sorted(resolved.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    candidates = candidates[int(skip_first_images) :: max(int(select_every_nth), 1)]
    if int(image_load_cap) > 0:
        candidates = candidates[: int(image_load_cap)]
    if not candidates:
        raise FileNotFoundError(f"No images found in '{resolved}'.")
    return candidates


def compute_segment_count(audio_duration: float, segment_seconds: int) -> int:
    total_duration = max(float(audio_duration), 0.0)
    safe_segment_seconds = max(int(segment_seconds), 1)
    return int(math.ceil(total_duration / safe_segment_seconds)) if total_duration > 0 else 0


def plan_segment(audio_duration: float, segment_seconds: int, index: int, fps: float) -> dict[str, Any]:
    total_duration = max(float(audio_duration), 0.0)
    safe_segment_seconds = max(int(segment_seconds), 1)
    safe_index = max(int(index), 0)
    safe_fps = max(float(fps), 1.0)
    segment_count = compute_segment_count(total_duration, safe_segment_seconds)
    start_time = float(safe_index * safe_segment_seconds)
    remaining = max(total_duration - start_time, 0.0)
    current_seconds = min(float(safe_segment_seconds), remaining)
    is_last_segment = remaining < float(safe_segment_seconds)
    frame_blocks = math.floor((current_seconds * safe_fps) / 8.0) if current_seconds > 0 else 0
    frames = 1 + max(frame_blocks, 0) * 8
    exact_seconds = float((frames - 1) / safe_fps) if frames > 1 else 0.0
    return {
        "index": safe_index,
        "start_time": start_time,
        "current_seconds": float(current_seconds),
        "exact_seconds": exact_seconds,
        "frames": int(frames),
        "is_last_segment": bool(is_last_segment),
        "segment_count": segment_count,
    }


def pick_segment_image_index(image_count: int, segment_index: int, seed: int) -> int:
    safe_image_count = max(int(image_count), 1)
    safe_segment_index = max(int(segment_index), 0)
    rng = random.Random(int(seed) + (safe_segment_index * 9973))
    return rng.randrange(safe_image_count)


def plan_segments(
    audio_duration: float,
    segment_seconds: int,
    fps: float,
    image_paths: list[Path],
    seed: int,
) -> list[SegmentSpec]:
    segment_count = compute_segment_count(audio_duration, segment_seconds)
    segments: list[SegmentSpec] = []
    for segment_index in range(segment_count):
        planned = plan_segment(audio_duration, segment_seconds, segment_index, fps)
        image_index = pick_segment_image_index(len(image_paths), segment_index, seed)
        segments.append(
            SegmentSpec(
                index=segment_index,
                start_time=float(planned["start_time"]),
                current_seconds=float(planned["current_seconds"]),
                exact_seconds=float(planned["exact_seconds"]),
                frames=int(planned["frames"]),
                is_last_segment=bool(planned["is_last_segment"]),
                selected_image_index=image_index,
                selected_image_path=str(image_paths[image_index]),
            )
        )
    return segments


def _wave_export_segment(audio_path: Path, segment: SegmentSpec, output_path: Path) -> None:
    with wave.open(str(audio_path), "rb") as reader:
        sample_rate = reader.getframerate()
        start_frame = max(0, int(round(segment.start_time * sample_rate)))
        duration_seconds = segment.exact_seconds if segment.exact_seconds > 0 else segment.current_seconds
        frame_count = max(0, int(round(duration_seconds * sample_rate)))
        reader.setpos(min(start_frame, reader.getnframes()))
        chunk_bytes = reader.readframes(frame_count)

        with wave.open(str(output_path), "wb") as writer:
            writer.setnchannels(reader.getnchannels())
            writer.setsampwidth(reader.getsampwidth())
            writer.setframerate(sample_rate)
            writer.setcomptype(reader.getcomptype(), reader.getcompname())
            writer.writeframes(chunk_bytes)


def export_audio_segments(
    audio_path: Path,
    segments: list[SegmentSpec],
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> list[SegmentSpec]:
    _prepare_output_dir(output_dir, overwrite=overwrite)
    ffmpeg = _ffmpeg_executable()
    exported: list[SegmentSpec] = []

    for segment in segments:
        output_path = output_dir / f"segment_{segment.index + 1:0{DEFAULT_SUFFIX_WIDTH}d}.wav"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Chunk file already exists: {output_path}. Pass --overwrite to replace it.")

        if ffmpeg is not None:
            duration_seconds = segment.exact_seconds if segment.exact_seconds > 0 else segment.current_seconds
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{segment.start_time:.6f}",
                "-i",
                str(audio_path),
                "-t",
                f"{duration_seconds:.6f}",
                "-vn",
                "-acodec",
                "pcm_s16le",
                str(output_path),
            ]
            subprocess.run(command, check=True, capture_output=True)
        elif audio_path.suffix.lower() == ".wav":
            _wave_export_segment(audio_path, segment, output_path)
        else:
            raise RuntimeError("ffmpeg is required to export audio segments for non-WAV inputs.")

        exported.append(
            SegmentSpec(
                index=segment.index,
                start_time=segment.start_time,
                current_seconds=segment.current_seconds,
                exact_seconds=segment.exact_seconds,
                frames=segment.frames,
                is_last_segment=segment.is_last_segment,
                selected_image_index=segment.selected_image_index,
                selected_image_path=segment.selected_image_path,
                audio_chunk_path=str(output_path),
            )
        )
    return exported


def build_manifest(
    *,
    workflow_path: Path,
    audio_info: AudioInfo,
    frames_dir: Path,
    fps: float,
    segment_seconds: int,
    seed: int,
    image_load_cap: int,
    skip_first_images: int,
    select_every_nth: int,
    segments: list[SegmentSpec],
) -> dict[str, Any]:
    return {
        "workflow_path": str(workflow_path),
        "source_audio": str(audio_info.path),
        "frames_dir": str(frames_dir),
        "audio_probe": audio_info.probe,
        "audio_duration_seconds": audio_info.duration_seconds,
        "fps": float(fps),
        "segment_seconds": int(segment_seconds),
        "seed": int(seed),
        "image_load_cap": int(image_load_cap),
        "skip_first_images": int(skip_first_images),
        "select_every_nth": int(select_every_nth),
        "segment_count": len(segments),
        "segments": [asdict(segment) for segment in segments],
    }


def write_manifest(manifest: dict[str, Any], manifest_path: Path) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest_path

