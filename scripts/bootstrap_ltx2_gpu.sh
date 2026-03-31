#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${1:-$HOME/vendor/LTX-2}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required" >&2
  exit 1
fi

mkdir -p "$(dirname "$TARGET_DIR")"

if [ ! -d "$TARGET_DIR/.git" ]; then
  git clone --depth 1 https://github.com/Lightricks/LTX-2.git "$TARGET_DIR"
else
  git -C "$TARGET_DIR" fetch --depth 1 origin main
  git -C "$TARGET_DIR" reset --hard FETCH_HEAD
fi

cd "$TARGET_DIR"
uv sync --frozen

cat <<EOF
LTX-2 bootstrap complete.
Repo: $TARGET_DIR
Python: $TARGET_DIR/.venv/bin/python

Recommended smoke:
  $TARGET_DIR/.venv/bin/python -m ltx_pipelines.a2vid_two_stage --help

Recommended next step from the ComfyUI-LTXLongAudio repo:
  uv run python cli/ltx23_download_models.py --assets-root "$HOME/models/ltx23_official"
  export LD_LIBRARY_PATH="/usr/lib64-nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"

Environment variables accepted by cli/ltx23_gpu_runner.py:
  export LTX2_PYTHON="$TARGET_DIR/.venv/bin/python"
  export LTX2_REPO_ROOT="$TARGET_DIR"
  export LTX2_CHECKPOINT_PATH="/path/to/ltx2_checkpoint.safetensors"
  export LTX2_DISTILLED_LORA_PATH="/path/to/distilled_lora.safetensors"
  export LTX2_SPATIAL_UPSAMPLER_PATH="/path/to/spatial_upsampler.safetensors"
  export LTX2_GEMMA_ROOT="/path/to/gemma"
EOF
