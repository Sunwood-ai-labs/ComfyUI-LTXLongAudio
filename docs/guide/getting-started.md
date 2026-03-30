# Getting Started

## Runtime paths

This repository supports two common flows:

1. Clone it into `ComfyUI/custom_nodes` from Google Colab or a local ComfyUI checkout.
2. Use the repository root itself as a QA workspace for tests, layout checks, and docs builds.

## Install into ComfyUI

```bash
cd /content/ComfyUI/custom_nodes
git clone https://github.com/Sunwood-ai-labs/ComfyUI-LTXLongAudio.git
uv pip install -r ComfyUI-LTXLongAudio/requirements.txt
```

Then restart ComfyUI so the native `LTX*` node schemas refresh cleanly.

## Local repository QA

The repository keeps developer checks under `uv`:

```bash
uv run pytest

uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups \
  --require-app-mode

uv run python scripts/run_comfyui_api_smoke.py \
  --workflow samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --comfy-root /path/to/ComfyUI
```

## Runtime expectations

- `ffmpeg` must be available before you run final video preview or the API smoke script.
- Audio loading uses `torchaudio`, which is why the runtime install stays in `requirements.txt`.
- The prompt API smoke runner auto-detects surrounding paths when this repo is cloned under `ComfyUI/custom_nodes`. Otherwise pass at least `--comfy-root`.

## Recommended first pass

1. Open the bundled smoke workflow.
2. Confirm that `samples/input/frames_pool` appears in the folder list.
3. Upload an audio file through `LoadAudio`.
4. Keep the default 20-second chunk size for the first run.
5. Verify that `LTXVideoCombine` returns a preview payload in the output panel.
