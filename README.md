# ComfyUI-LTXLongAudio

Native `LTX*` custom nodes for long-audio LTX workflows in ComfyUI.

This repository is intended for setups where custom nodes are published on GitHub and cloned from Google Colab into `ComfyUI/custom_nodes`.

## Included Nodes

- `LTXLongAudioSegmentInfo`
  Computes per-segment timing and frame information for loop-based long-audio workflows.
- `LTXRandomImageIndex`
  Returns a deterministic random image index for a segment index and seed.
- `LTXLoadAudioUpload` and `LTXLoadImages`
  Load the source song and a reference-frame directory directly from the ComfyUI input folder.
- `LTXSimpleMath`, `LTXCompare`, `LTXIfElse`, `LTXIndexAnything`, `LTXBatchAnything`
  Workflow helpers used to drive segment loops and image selection.
- `LTXWhileLoopStart`, `LTXWhileLoopEnd`, `LTXForLoopStart`, `LTXForLoopEnd`
  Native loop-control nodes for long-audio workflows.
- `LTXAudioConcatenate` and `LTXVideoCombine`
  Merge per-segment audio and mux the final video without relying on external node packs.
- `LTXIntConstant`, `LTXSimpleCalculator`, `LTXVAELoader`, `LTXImageResize`, `LTXChunkFeedForward`, `LTXSamplingPreviewOverride`, `LTXNormalizedAttentionGuidance`
  Native replacements for the extra workflow utilities used by the bundled LTX graph.

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

## Samples

- Sample workflow: `samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json`
- Sample input placeholders: `samples/input/`
- Layout checker: `scripts/check_workflow_layout.py`

Validate grouped workflow layouts before publishing:

```bash
uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups
```

## Notes

- Segment frame counts are quantized in blocks of 8 frames to stay friendly with LTX-style workflows.
- Audio loading uses `torchaudio`.
- Final video muxing uses `ffmpeg`, so the Colab runtime should install it before launching ComfyUI.

## License

GPL-3.0-or-later
