# ComfyUI-LTXLongAudio

Custom nodes for long-audio LTX workflows in ComfyUI.

This repository is intended for setups where custom nodes are published on GitHub and cloned from Google Colab into `ComfyUI/custom_nodes`.

## Included Nodes

- `LTX Long Audio Segment Info`
  Computes per-segment timing and frame information for loop-based long-audio workflows.
- `LTX Random Image Index`
  Returns a deterministic random image index for a segment index and seed.
- `LTX*` workflow helper nodes
  Provide long-audio math, loop, image-loading, audio-loading, audio-concatenation, and video-combine helpers.
- Compatibility shims for the workflow
  This repo now exports the node ids used by `Easy Use`, `VideoHelperSuite`, and `KJNodes` that are referenced by the bundled LTX workflow, so Google Colab only needs this repo for those parts.

## Install

```bash
cd /content/ComfyUI/custom_nodes
git clone https://github.com/Sunwood-ai-labs/ComfyUI-LTXLongAudio.git
pip install -r ComfyUI-LTXLongAudio/requirements.txt
```

Then restart ComfyUI.

## Google Colab

This repository is designed for the common Colab flow where custom nodes are installed with `git clone` into `ComfyUI/custom_nodes`.

The current workflow expects:

- one song file in the ComfyUI input root
- one frame-image folder under the ComfyUI input root
- `ffmpeg` available in the runtime

## Notes

- Segment frame counts are quantized in blocks of 8 frames to stay friendly with LTX-style workflows.
- `LTX2 Sampling Preview Override` is intentionally a lightweight compatibility shim. It keeps the workflow loadable without pulling in extra preview JS.
- Audio loading uses `torchaudio`.

## License

GPL-3.0-or-later
