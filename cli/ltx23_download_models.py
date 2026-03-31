from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_REPO_ID = "Lightricks/LTX-2.3"
DEFAULT_GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"
DEFAULT_ASSETS_ROOT = Path("models") / "ltx23_official"
DEFAULT_MANIFEST_NAME = "ltx23_assets_manifest.json"
DEFAULT_RESERVE_GB = 20


@dataclass(frozen=True)
class AssetSpec:
    key: str
    repo_id: str
    local_subdir: str
    description: str
    filename: str | None = None
    is_snapshot: bool = False


@dataclass(frozen=True)
class DownloadedAsset:
    key: str
    repo_id: str
    local_path: str
    description: str
    filename: str | None = None
    is_snapshot: bool = False
    size_bytes: int | None = None


@dataclass(frozen=True)
class HFClients:
    api: Any
    hf_hub_download: Any
    snapshot_download: Any


def build_official_asset_specs(
    *,
    model_repo_id: str = DEFAULT_MODEL_REPO_ID,
    gemma_repo_id: str = DEFAULT_GEMMA_REPO_ID,
    include_gemma: bool = True,
) -> list[AssetSpec]:
    specs = [
        AssetSpec(
            key="checkpoint",
            repo_id=model_repo_id,
            filename="ltx-2.3-22b-dev.safetensors",
            local_subdir="checkpoints",
            description="Official LTX-2.3 development checkpoint for two-stage pipelines.",
        ),
        AssetSpec(
            key="distilled_lora",
            repo_id=model_repo_id,
            filename="ltx-2.3-22b-distilled-lora-384.safetensors",
            local_subdir="loras",
            description="Official distilled LoRA required by the two-stage upsampling pass.",
        ),
        AssetSpec(
            key="spatial_upsampler",
            repo_id=model_repo_id,
            filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            local_subdir="upscalers",
            description="Official spatial upscaler for the second stage.",
        ),
    ]
    if include_gemma:
        specs.append(
            AssetSpec(
                key="gemma_root",
                repo_id=gemma_repo_id,
                local_subdir="gemma",
                description="Gemma text encoder snapshot required by the official pipelines.",
                is_snapshot=True,
            )
        )
    return specs


def _load_hf_clients() -> HFClients:
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for model downloads. Install it with `uv add huggingface_hub` "
            "or `uv pip install huggingface_hub`."
        ) from exc
    return HFClients(
        api=HfApi(),
        hf_hub_download=hf_hub_download,
        snapshot_download=snapshot_download,
    )


def _resolve_hf_token(cli_token: str | None) -> str | None:
    return cli_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _collect_repo_sizes(api: Any, repo_id: str, *, token: str | None) -> dict[str, int]:
    info = api.model_info(repo_id=repo_id, files_metadata=True, token=token)
    sizes: dict[str, int] = {}
    for sibling in getattr(info, "siblings", []) or []:
        name = getattr(sibling, "rfilename", None)
        size = getattr(sibling, "size", None)
        if name and size is not None:
            sizes[str(name)] = int(size)
    return sizes


def estimate_required_bytes(specs: list[AssetSpec], *, clients: HFClients, token: str | None) -> int:
    total = 0
    per_repo_cache: dict[str, dict[str, int]] = {}
    for spec in specs:
        if spec.repo_id not in per_repo_cache:
            per_repo_cache[spec.repo_id] = _collect_repo_sizes(clients.api, spec.repo_id, token=token)
        sizes = per_repo_cache[spec.repo_id]
        if spec.is_snapshot:
            total += sum(sizes.values())
        elif spec.filename and spec.filename in sizes:
            total += sizes[spec.filename]
    return total


def ensure_disk_space(target_root: Path, *, required_bytes: int, reserve_bytes: int) -> None:
    probe_path = target_root.expanduser().resolve()
    while not probe_path.exists():
        parent = probe_path.parent
        if parent == probe_path:
            break
        probe_path = parent
    usage = shutil.disk_usage(probe_path)
    if usage.free < (required_bytes + reserve_bytes):
        required_gb = (required_bytes + reserve_bytes) / float(1024**3)
        free_gb = usage.free / float(1024**3)
        raise RuntimeError(
            f"Not enough free disk space under {probe_path}. "
            f"Need about {required_gb:.1f} GiB including reserve, only {free_gb:.1f} GiB free."
        )


def download_assets(
    specs: list[AssetSpec],
    *,
    assets_root: Path,
    clients: HFClients,
    token: str | None,
) -> list[DownloadedAsset]:
    downloaded: list[DownloadedAsset] = []
    for spec in specs:
        local_dir = (assets_root / spec.local_subdir).expanduser().resolve()
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            if spec.is_snapshot:
                local_path = Path(
                    clients.snapshot_download(
                        repo_id=spec.repo_id,
                        local_dir=str(local_dir),
                        token=token,
                    )
                ).resolve()
                size_bytes = None
            else:
                assert spec.filename is not None
                local_path = Path(
                    clients.hf_hub_download(
                        repo_id=spec.repo_id,
                        filename=spec.filename,
                        local_dir=str(local_dir),
                        token=token,
                    )
                ).resolve()
                size_bytes = local_path.stat().st_size if local_path.exists() else None
        except Exception as exc:
            message = str(exc)
            if "gated repo" in message.lower() or exc.__class__.__name__ == "GatedRepoError":
                raise RuntimeError(
                    f"Failed to download '{spec.key}' from '{spec.repo_id}'. "
                    "This repository is gated. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN, "
                    "or rerun with --skip-gemma if you only want the public assets first."
                ) from exc
            raise
        downloaded.append(
            DownloadedAsset(
                key=spec.key,
                repo_id=spec.repo_id,
                local_path=str(local_path),
                description=spec.description,
                filename=spec.filename,
                is_snapshot=spec.is_snapshot,
                size_bytes=size_bytes,
            )
        )
    return downloaded


def build_exports(downloaded: list[DownloadedAsset]) -> dict[str, str]:
    by_key = {asset.key: asset for asset in downloaded}
    exports: dict[str, str] = {}
    if "checkpoint" in by_key:
        exports["LTX2_CHECKPOINT_PATH"] = by_key["checkpoint"].local_path
    if "distilled_lora" in by_key:
        exports["LTX2_DISTILLED_LORA_PATH"] = by_key["distilled_lora"].local_path
    if "spatial_upsampler" in by_key:
        exports["LTX2_SPATIAL_UPSAMPLER_PATH"] = by_key["spatial_upsampler"].local_path
    if "gemma_root" in by_key:
        exports["LTX2_GEMMA_ROOT"] = by_key["gemma_root"].local_path
    return exports


def write_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official model assets required by the LTX 2.3 GPU runner.")
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--model-repo-id", default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--gemma-repo-id", default=DEFAULT_GEMMA_REPO_ID)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--skip-gemma", action="store_true")
    parser.add_argument("--reserve-gb", type=float, default=float(DEFAULT_RESERVE_GB))
    parser.add_argument("--skip-disk-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    assets_root = args.assets_root.expanduser().resolve()
    token = _resolve_hf_token(args.hf_token)
    specs = build_official_asset_specs(
        model_repo_id=args.model_repo_id,
        gemma_repo_id=args.gemma_repo_id,
        include_gemma=not args.skip_gemma,
    )
    clients = _load_hf_clients()
    estimated_bytes = estimate_required_bytes(specs, clients=clients, token=token)
    reserve_bytes = int(max(args.reserve_gb, 0.0) * (1024**3))
    if not args.skip_disk_check:
        ensure_disk_space(assets_root, required_bytes=estimated_bytes, reserve_bytes=reserve_bytes)

    downloaded: list[DownloadedAsset] = []
    if not args.dry_run:
        downloaded = download_assets(specs, assets_root=assets_root, clients=clients, token=token)

    exports = build_exports(downloaded)
    manifest = {
        "assets_root": str(assets_root),
        "model_repo_id": args.model_repo_id,
        "gemma_repo_id": None if args.skip_gemma else args.gemma_repo_id,
        "estimated_required_bytes": estimated_bytes,
        "reserve_bytes": reserve_bytes,
        "dry_run": bool(args.dry_run),
        "downloads": [asdict(asset) for asset in downloaded],
        "exports": exports,
        "notes": [
            "Reference notebook also downloaded ComfyUI-specific GGUF and VAE extras.",
            "This downloader only fetches the official assets required by cli/ltx23_gpu_ready.py.",
            "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN if your environment requires authenticated access.",
        ],
    }
    manifest_path = write_manifest(manifest, assets_root / args.manifest_name)

    print(f"Assets root: {assets_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Estimated bytes: {estimated_bytes}")
    if args.dry_run:
        print("Download: skipped (--dry-run)")
    else:
        print(f"Downloaded assets: {len(downloaded)}")
    if exports:
        print("Export these environment variables:")
        for key, value in exports.items():
            print(f"export {key}=\"{value}\"")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
