---
layout: home

hero:
  name: ComfyUI-LTXLongAudio
  text: Long-audio chunks, native loops, and still-video previews
  tagline: Keep the full ComfyUI surface native from upload widgets to final MP4 output.
  image:
    src: /logo.svg
    alt: ComfyUI-LTXLongAudio logo
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: Sample Workflow
      link: /guide/usage
    - theme: alt
      text: GitHub
      link: https://github.com/Sunwood-ai-labs/ComfyUI-LTXLongAudio

features:
  - title: Native LTX workflow surface
    details: Bundle loop control, chunk planning, media assembly, and preview output without relying on extra node packs.
  - title: App mode-ready smoke path
    details: Ship one focused workflow with four exposed controls, sample folder defaults, and a deterministic frame-per-chunk preview path.
  - title: Publish with QA
    details: Validate layout contracts, App mode metadata, and prompt API execution with repository-native scripts before you share a graph.
---

## What ships

- Native input nodes for audio upload, image upload, batched frames, and folder selection.
- Long-audio helpers for segment timing, deterministic frame selection, sliced audio, and final audio concatenation.
- Native loop-control nodes and utility replacements used by the bundled LTX workflow.
- A smoke workflow under `samples/workflows/` plus lightweight sample image folders in `samples/input/`.

## What to expect from the sample

The bundled workflow keeps the UI surface intentionally small:

| App mode input | Purpose |
| --- | --- |
| `Frames Folder` | Picks the folder that feeds per-chunk still frames. |
| `Source Audio Upload` | Lets you upload the long-form song directly in ComfyUI. |
| `Segment Seconds` | Controls chunk length, defaulting to 20 seconds. |
| `Random Seed` | Keeps frame selection deterministic across reruns. |

The workflow currently analyzes to `3` groups and `9` nodes, with one final output node selected for App mode preview.

## Verification entry points

- `uv run pytest`
- `uv run python scripts/check_workflow_layout.py samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json --require-all-nodes-in-groups --require-app-mode`
- `uv run python scripts/run_comfyui_api_smoke.py --workflow samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json --comfy-root /path/to/ComfyUI`

::: tip
`run_comfyui_api_smoke.py` keeps `ltx-demo-tone.wav` as the tracked default. If you also keep a longer local `HOWL AT THE HAIRPIN2.wav`, the script prefers it and stages it under the tracked widget filename.
:::
