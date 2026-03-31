from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cli.ltx23_download_models as downloader


class _FakeApi:
    def model_info(self, repo_id: str, files_metadata: bool, token: str | None):  # noqa: ARG002
        if repo_id == "Lightricks/LTX-2.3":
            siblings = [
                SimpleNamespace(rfilename="ltx-2.3-22b-dev.safetensors", size=11),
                SimpleNamespace(rfilename="ltx-2.3-22b-distilled-lora-384.safetensors", size=7),
                SimpleNamespace(rfilename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors", size=5),
            ]
        else:
            siblings = [
                SimpleNamespace(rfilename="config.json", size=3),
                SimpleNamespace(rfilename="weights.safetensors", size=13),
            ]
        return SimpleNamespace(siblings=siblings)


def _fake_hf_download(*, repo_id: str, filename: str, local_dir: str, token: str | None):  # noqa: ARG001
    path = Path(local_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(repo_id, encoding="utf-8")
    return str(path)


def _fake_snapshot_download(*, repo_id: str, local_dir: str, token: str | None):  # noqa: ARG001
    target = Path(local_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(repo_id, encoding="utf-8")
    return str(target)


def test_run_downloads_assets_and_writes_manifest(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        downloader,
        "_load_hf_clients",
        lambda: downloader.HFClients(
            api=_FakeApi(),
            hf_hub_download=_fake_hf_download,
            snapshot_download=_fake_snapshot_download,
        ),
    )

    exit_code = downloader.run(["--assets-root", str(tmp_path), "--reserve-gb", "0"])

    manifest = json.loads((tmp_path / "ltx23_assets_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(manifest["downloads"]) == 4
    assert manifest["exports"]["LTX2_CHECKPOINT_PATH"].endswith("ltx-2.3-22b-dev.safetensors")
    assert manifest["exports"]["LTX2_GEMMA_ROOT"] == str((tmp_path / "gemma").resolve())


def test_dry_run_skips_downloads_but_keeps_estimate(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        downloader,
        "_load_hf_clients",
        lambda: downloader.HFClients(
            api=_FakeApi(),
            hf_hub_download=_fake_hf_download,
            snapshot_download=_fake_snapshot_download,
        ),
    )

    exit_code = downloader.run(["--assets-root", str(tmp_path), "--reserve-gb", "0", "--dry-run"])

    manifest = json.loads((tmp_path / "ltx23_assets_manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert manifest["dry_run"] is True
    assert manifest["downloads"] == []
    assert manifest["estimated_required_bytes"] == 39


def test_disk_check_uses_nearest_existing_parent(tmp_path: Path):
    target = tmp_path / "nested" / "models" / "ltx23"

    downloader.ensure_disk_space(target, required_bytes=1, reserve_bytes=0)
