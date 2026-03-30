# ComfyUI-LTXLongAudio

Native `LTX*` custom nodes for long-audio LTX workflows in ComfyUI.

This repository is intended for setups where custom nodes are published on GitHub and cloned from Google Colab into `ComfyUI/custom_nodes`.

## Included Nodes

- `LTXLongAudioSegmentInfo`
  Computes per-segment timing and frame information for loop-based long-audio workflows.
- `LTXRandomImageIndex`
  Returns a deterministic random image index for a segment index and seed.
- `LTXLoadAudioUpload`, `LTXLoadImageUpload`, `LTXBatchUploadedFrames`, and `LTXLoadImages`
  Load an uploaded song, upload individual reference frames, batch uploaded frames, or load a reference-frame directory directly from the ComfyUI input folder.
- `LTXAudioSlice`
  Cuts a segment from an already loaded audio clip, so workflows only need one audio upload control.
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

The bundled smoke workflow is folder-plus-audio:

- upload one song with the built-in `LoadAudio` control
- choose one frame folder from the ComfyUI input list
- turn the selected still image into a dummy still-video batch
- preview the generated mp4 directly in ComfyUI
- keep `ffmpeg` available in the runtime

## Samples

- Sample workflow: `samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json`
- Sample input placeholders: `samples/input/`
- Layout checker: `scripts/check_workflow_layout.py`

The bundled smoke workflow uses concrete sample defaults: `samples/input/frames_pool` for the frame folder and `HOWL AT THE HAIRPIN2.wav` for the audio input.

The sample uses ComfyUI's built-in `LoadAudio` node for the upload widget, plus `LTXAudioDuration` and `LTXAudioSlice` for long-audio helpers. This keeps the Desktop and App mode upload UI reliable while still testing the custom long-audio logic.

The smoke script stages those sample assets into the ComfyUI input directory and validates the workflow as-is by default. Use `--auto-fill-missing` only if you intentionally want fallback behavior for blank widgets.

If App mode still shows stale controls after updating this repository, fully restart the ComfyUI backend or Desktop app. A hot reload can keep old custom-node input schemas alive.

Validate grouped workflow layouts before publishing. Group overlaps, title-band collisions, node-node overlaps, App mode metadata, and common runtime contract issues are checked:

```bash
uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups \
  --require-app-mode

# Run the exact smoke workflow through a real ComfyUI /prompt API session.
uv run python scripts/run_comfyui_api_smoke.py \
  --workflow D:/Prj/ComfyUI_LTX2_3_TI2V/LTXLongAudio_CustomNodes_SmokeTest.json
```

The bundled sample workflow includes App mode metadata via `extra.linearData` and `extra.linearMode`, matching the current ComfyUI frontend builder behavior.

## Notes

- Segment frame counts are quantized in blocks of 8 frames to stay friendly with LTX-style workflows.
- Audio loading uses `torchaudio`.
- Final video muxing uses `ffmpeg`, so the Colab runtime should install it before launching ComfyUI.

## License

GPL-3.0-or-later
