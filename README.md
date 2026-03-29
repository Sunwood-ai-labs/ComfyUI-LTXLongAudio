# ComfyUI-LTXLongAudio

Minimal custom nodes for long-audio LTX workflows in ComfyUI.

This repository is intended for setups where custom nodes are published on GitHub and cloned from Google Colab into `ComfyUI/custom_nodes`.

## Included Nodes

- `LTX Long Audio Segment Info`
  Computes per-segment timing and frame information for loop-based long-audio workflows.
- `LTX Random Image Index`
  Returns a deterministic random image index for a segment index and seed.

## Install

```bash
cd /content/ComfyUI/custom_nodes
git clone https://github.com/Sunwood-ai-labs/ComfyUI-LTXLongAudio.git
```

Then restart ComfyUI.

## Current Status

This is the initial public scaffold. The first two nodes are meant to support:

- long audio segmentation
- deterministic random image selection
- future LTX audio-to-video workflow cleanup

## Notes

- Segment frame counts are quantized in blocks of 8 frames to stay friendly with LTX-style workflows.
- Very small tail segments may round down to zero effective duration depending on FPS.

## License

MIT
