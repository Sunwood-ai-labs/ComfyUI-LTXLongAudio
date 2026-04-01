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
DEFAULT_GGUF_REPO_ID = "unsloth/LTX-2.3-GGUF"
DEFAULT_COMFY_TEXT_ENCODER_REPO_ID = "Comfy-Org/ltx-2"
DEFAULT_MMPROJ_REPO_ID = "unsloth/gemma-3-12b-it-qat-GGUF"
DEFAULT_MELBAND_REPO_ID = "Kijai/MelBandRoFormer_comfy"
DEFAULT_TAE_REPO_ID = "Kijai/LTX2.3_comfy"
ASSET_PROFILE_OFFICIAL = "official"
ASSET_PROFILE_NOTEBOOK_COMFY = "notebook-comfy"
DEFAULT_ASSET_PROFILE = ASSET_PROFILE_OFFICIAL
DEFAULT_NOTEBOOK_DISTILLED_LORA_STRENGTH = 0.6
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


def build_notebook_comfy_asset_specs(
    *,
    model_repo_id: str = DEFAULT_MODEL_REPO_ID,
    gguf_repo_id: str = DEFAULT_GGUF_REPO_ID,
    comfy_text_encoder_repo_id: str = DEFAULT_COMFY_TEXT_ENCODER_REPO_ID,
    mmproj_repo_id: str = DEFAULT_MMPROJ_REPO_ID,
    melband_repo_id: str = DEFAULT_MELBAND_REPO_ID,
    tae_repo_id: str = DEFAULT_TAE_REPO_ID,
) -> list[AssetSpec]:
    return [
        AssetSpec(
            key="unet_gguf",
            repo_id=gguf_repo_id,
            filename="ltx-2.3-22b-dev-Q4_K_M.gguf",
            local_subdir="unet",
            description="Notebook UNet GGUF checkpoint for the low-VRAM Comfy stack.",
        ),
        AssetSpec(
            key="gemma_fp8_scaled",
            repo_id=comfy_text_encoder_repo_id,
            filename="split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors",
            local_subdir="text_encoders",
            description="Notebook Gemma fp8 text encoder for the Comfy stack.",
        ),
        AssetSpec(
            key="mmproj_gguf",
            repo_id=mmproj_repo_id,
            filename="mmproj-BF16.gguf",
            local_subdir="text_encoders",
            description="Notebook GGUF mmproj text-encoder companion asset.",
        ),
        AssetSpec(
            key="embeddings_connectors",
            repo_id=gguf_repo_id,
            filename="text_encoders/ltx-2.3-22b-dev_embeddings_connectors.safetensors",
            local_subdir="text_encoders",
            description="Notebook LTX embeddings connectors file.",
        ),
        AssetSpec(
            key="video_vae",
            repo_id=gguf_repo_id,
            filename="vae/ltx-2.3-22b-dev_video_vae.safetensors",
            local_subdir="vae",
            description="Notebook video VAE.",
        ),
        AssetSpec(
            key="audio_vae",
            repo_id=gguf_repo_id,
            filename="vae/ltx-2.3-22b-dev_audio_vae.safetensors",
            local_subdir="vae",
            description="Notebook audio VAE.",
        ),
        AssetSpec(
            key="spatial_upsampler",
            repo_id=model_repo_id,
            filename="ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            local_subdir="latent_upscale_models",
            description="Notebook spatial upscaler used by the Origin workflow.",
        ),
        AssetSpec(
            key="distilled_lora",
            repo_id=model_repo_id,
            filename="ltx-2.3-22b-distilled-lora-384.safetensors",
            local_subdir="loras",
            description="Notebook distilled LoRA used by the Origin workflow.",
        ),
        AssetSpec(
            key="melband_model",
            repo_id=melband_repo_id,
            filename="MelBandRoformer_fp16.safetensors",
            local_subdir="diffusion_models",
            description="Notebook MelBandRoFormer model for vocal-conditioned audio features.",
        ),
        AssetSpec(
            key="tae_vae",
            repo_id=tae_repo_id,
            filename="vae/taeltx2_3.safetensors",
            local_subdir="vae",
            description="Notebook TAE VAE helper used by the saved workflow.",
        ),
    ]


def build_asset_specs(
    *,
    asset_profile: str,
    model_repo_id: str = DEFAULT_MODEL_REPO_ID,
    gemma_repo_id: str = DEFAULT_GEMMA_REPO_ID,
    gguf_repo_id: str = DEFAULT_GGUF_REPO_ID,
    comfy_text_encoder_repo_id: str = DEFAULT_COMFY_TEXT_ENCODER_REPO_ID,
    mmproj_repo_id: str = DEFAULT_MMPROJ_REPO_ID,
    melband_repo_id: str = DEFAULT_MELBAND_REPO_ID,
    tae_repo_id: str = DEFAULT_TAE_REPO_ID,
    include_gemma: bool = True,
) -> list[AssetSpec]:
    if asset_profile == ASSET_PROFILE_OFFICIAL:
        return build_official_asset_specs(
            model_repo_id=model_repo_id,
            gemma_repo_id=gemma_repo_id,
            include_gemma=include_gemma,
        )
    if asset_profile == ASSET_PROFILE_NOTEBOOK_COMFY:
        return build_notebook_comfy_asset_specs(
            model_repo_id=model_repo_id,
            gguf_repo_id=gguf_repo_id,
            comfy_text_encoder_repo_id=comfy_text_encoder_repo_id,
            mmproj_repo_id=mmproj_repo_id,
            melband_repo_id=melband_repo_id,
            tae_repo_id=tae_repo_id,
        )
    raise ValueError(f"Unknown asset profile: {asset_profile}")


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
                cached_path = Path(
                    clients.hf_hub_download(
                        repo_id=spec.repo_id,
                        filename=spec.filename,
                        token=token,
                    )
                ).resolve()
                local_path = (local_dir / Path(spec.filename).name).resolve()
                if cached_path != local_path:
                    shutil.copy2(cached_path, local_path)
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
    if "unet_gguf" in by_key:
        exports["LTX23_NOTEBOOK_UNET_GGUF_PATH"] = by_key["unet_gguf"].local_path
    if "gemma_fp8_scaled" in by_key:
        exports["LTX23_NOTEBOOK_GEMMA_FP8_PATH"] = by_key["gemma_fp8_scaled"].local_path
    if "mmproj_gguf" in by_key:
        exports["LTX23_NOTEBOOK_MMPROJ_GGUF_PATH"] = by_key["mmproj_gguf"].local_path
    if "embeddings_connectors" in by_key:
        exports["LTX23_NOTEBOOK_EMBEDDINGS_CONNECTORS_PATH"] = by_key["embeddings_connectors"].local_path
    if "video_vae" in by_key:
        exports["LTX23_NOTEBOOK_VIDEO_VAE_PATH"] = by_key["video_vae"].local_path
    if "audio_vae" in by_key:
        exports["LTX23_NOTEBOOK_AUDIO_VAE_PATH"] = by_key["audio_vae"].local_path
    if "spatial_upsampler" in by_key:
        exports["LTX23_NOTEBOOK_SPATIAL_UPSAMPLER_PATH"] = by_key["spatial_upsampler"].local_path
    if "distilled_lora" in by_key:
        exports["LTX23_NOTEBOOK_DISTILLED_LORA_PATH"] = by_key["distilled_lora"].local_path
        exports["LTX23_NOTEBOOK_DISTILLED_LORA_STRENGTH"] = str(DEFAULT_NOTEBOOK_DISTILLED_LORA_STRENGTH)
    if "melband_model" in by_key:
        exports["LTX23_NOTEBOOK_MELBAND_PATH"] = by_key["melband_model"].local_path
    if "tae_vae" in by_key:
        exports["LTX23_NOTEBOOK_TAE_VAE_PATH"] = by_key["tae_vae"].local_path
    return exports


def write_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the official model assets required by the LTX 2.3 GPU runner.")
    parser.add_argument(
        "--asset-profile",
        choices=(ASSET_PROFILE_OFFICIAL, ASSET_PROFILE_NOTEBOOK_COMFY),
        default=DEFAULT_ASSET_PROFILE,
        help="official downloads the current ltx_pipelines assets; notebook-comfy downloads the notebook/Origin workflow asset set.",
    )
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--model-repo-id", default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--gemma-repo-id", default=DEFAULT_GEMMA_REPO_ID)
    parser.add_argument("--gguf-repo-id", default=DEFAULT_GGUF_REPO_ID)
    parser.add_argument("--comfy-text-encoder-repo-id", default=DEFAULT_COMFY_TEXT_ENCODER_REPO_ID)
    parser.add_argument("--mmproj-repo-id", default=DEFAULT_MMPROJ_REPO_ID)
    parser.add_argument("--melband-repo-id", default=DEFAULT_MELBAND_REPO_ID)
    parser.add_argument("--tae-repo-id", default=DEFAULT_TAE_REPO_ID)
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
    specs = build_asset_specs(
        asset_profile=args.asset_profile,
        model_repo_id=args.model_repo_id,
        gemma_repo_id=args.gemma_repo_id,
        gguf_repo_id=args.gguf_repo_id,
        comfy_text_encoder_repo_id=args.comfy_text_encoder_repo_id,
        mmproj_repo_id=args.mmproj_repo_id,
        melband_repo_id=args.melband_repo_id,
        tae_repo_id=args.tae_repo_id,
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
        "asset_profile": args.asset_profile,
        "assets_root": str(assets_root),
        "model_repo_id": args.model_repo_id,
        "gemma_repo_id": None if args.skip_gemma else args.gemma_repo_id,
        "estimated_required_bytes": estimated_bytes,
        "reserve_bytes": reserve_bytes,
        "dry_run": bool(args.dry_run),
        "downloads": [asdict(asset) for asset in downloaded],
        "exports": exports,
        "notes": (
            [
                "Asset profile: official.",
                "This downloader fetches the current ltx_pipelines runtime assets used by cli/ltx23_gpu_ready.py.",
                "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN if your environment requires authenticated access.",
            ]
            if args.asset_profile == ASSET_PROFILE_OFFICIAL
            else [
                "Asset profile: notebook-comfy.",
                "This downloader follows the notebook and Origin workflow asset closure, including GGUF, split VAEs, MelBand, and TAE helper weights.",
                "The saved Origin workflow uses ltx-2.3-spatial-upscaler-x2-1.0.safetensors from latent_upscale_models and distilled LoRA strength 0.6.",
                "These assets do not imply runtime parity by themselves; the current GPU runner still targets the official ltx_pipelines backend.",
            ]
        ),
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
