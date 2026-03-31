from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path
from typing import NamedTuple


DEFAULT_CHUNK_SECONDS = 20.0
DEFAULT_SUFFIX_WIDTH = 3


class ChunkWriteResult(NamedTuple):
    path: Path
    frame_count: int
    duration_seconds: float


def default_output_dir(input_wav: Path) -> Path:
    return input_wav.with_name(f"{input_wav.stem}_chunks")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a WAV file into fixed-length chunks.")
    parser.add_argument("input_wav", type=Path, help="Path to the source WAV file.")
    parser.add_argument(
        "--chunk-seconds",
        "--seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help=f"Chunk length in seconds. Defaults to {DEFAULT_CHUNK_SECONDS:g}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for chunk files. Defaults to <input-stem>_chunks next to the source WAV.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename prefix. Defaults to the source WAV stem.",
    )
    parser.add_argument(
        "--suffix-width",
        type=int,
        default=DEFAULT_SUFFIX_WIDTH,
        help=f"Zero-padding width for chunk indexes. Defaults to {DEFAULT_SUFFIX_WIDTH}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting chunk files and writing into a non-empty output directory.",
    )
    return parser.parse_args(argv)


def chunk_frame_count(frame_rate: int, chunk_seconds: float) -> int:
    return max(1, int(math.ceil(frame_rate * chunk_seconds)))


def output_chunk_path(output_dir: Path, prefix: str, index: int, suffix_width: int) -> Path:
    return output_dir / f"{prefix}_{index:0{suffix_width}d}.wav"


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


def _validate_args(
    input_wav: Path,
    *,
    chunk_seconds: float,
    suffix_width: int,
) -> Path:
    resolved_input = input_wav.expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Input WAV does not exist: {resolved_input}")
    if resolved_input.suffix.lower() != ".wav":
        raise ValueError(f"Input file must be a .wav file: {resolved_input}")
    if chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be greater than 0.")
    if suffix_width <= 0:
        raise ValueError("--suffix-width must be greater than 0.")
    return resolved_input


def split_wav_file(
    input_wav: Path,
    *,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    output_dir: Path | None = None,
    prefix: str | None = None,
    suffix_width: int = DEFAULT_SUFFIX_WIDTH,
    overwrite: bool = False,
) -> list[ChunkWriteResult]:
    resolved_input = _validate_args(
        input_wav,
        chunk_seconds=chunk_seconds,
        suffix_width=suffix_width,
    )
    target_output_dir = (output_dir.expanduser().resolve() if output_dir is not None else default_output_dir(resolved_input))
    target_prefix = prefix or resolved_input.stem

    with wave.open(str(resolved_input), "rb") as reader:
        total_frames = reader.getnframes()
        if total_frames == 0:
            return []

        frames_per_chunk = chunk_frame_count(reader.getframerate(), chunk_seconds)
        _prepare_output_dir(target_output_dir, overwrite=overwrite)

        results: list[ChunkWriteResult] = []
        remaining_frames = total_frames
        chunk_index = 1
        while remaining_frames > 0:
            frames_to_read = min(frames_per_chunk, remaining_frames)
            chunk_bytes = reader.readframes(frames_to_read)
            chunk_path = output_chunk_path(target_output_dir, target_prefix, chunk_index, suffix_width)
            if chunk_path.exists() and not overwrite:
                raise FileExistsError(f"Chunk file already exists: {chunk_path}. Pass --overwrite to replace it.")

            with wave.open(str(chunk_path), "wb") as writer:
                writer.setnchannels(reader.getnchannels())
                writer.setsampwidth(reader.getsampwidth())
                writer.setframerate(reader.getframerate())
                writer.setcomptype(reader.getcomptype(), reader.getcompname())
                writer.writeframes(chunk_bytes)

            results.append(
                ChunkWriteResult(
                    path=chunk_path,
                    frame_count=frames_to_read,
                    duration_seconds=frames_to_read / float(reader.getframerate()),
                )
            )
            remaining_frames -= frames_to_read
            chunk_index += 1

    return results


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = split_wav_file(
        args.input_wav,
        chunk_seconds=args.chunk_seconds,
        output_dir=args.output_dir,
        prefix=args.prefix,
        suffix_width=args.suffix_width,
        overwrite=args.overwrite,
    )

    if not results:
        print("No audio frames found; no chunk files were created.")
        return 0

    print(f"Wrote {len(results)} chunk(s) to {results[0].path.parent}")
    for result in results:
        print(f"{result.path.name}: {result.duration_seconds:.3f}s ({result.frame_count} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
