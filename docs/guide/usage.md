# Usage

## Smoke workflow defaults

The bundled workflow lives at `samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json`.

Current checked expectations:

| Item | Value |
| --- | --- |
| Frame folder default | `samples/input/frames_pool` |
| Audio widget default | `HOWL AT THE HAIRPIN2.wav` |
| App mode inputs | `Frames Folder`, `Source Audio Upload`, `Segment Seconds`, `Random Seed` |
| App mode output node | `LTXVideoCombine` |

## Sample assets

Tracked sample assets stay intentionally light:

- `samples/input/frames_pool` ships quarter-resolution stills at `688x384`.
- `samples/input/demo_frames` ships tiny debug stills at `192x108`.
- `samples/input/ltx-demo-tone.wav` is the tracked fallback audio for scripted smoke runs.

The recommended long-form sample filename remains `HOWL AT THE HAIRPIN2.wav`. If that longer file is missing, the API smoke runner automatically stages the fallback tone file under the expected upload name.

## Input discovery behavior

The upload and folder nodes intentionally search both the ComfyUI input directory and the repository sample directory:

- audio dropdowns can expose files from the runtime input directory and `samples/input`
- image dropdowns do the same for still images
- folder dropdowns include both ComfyUI input folders and `samples/input/*` directories
- upload-oriented COMBO widgets keep a blank first option so the App mode surface can start cleanly

## Publishing checks

Use the static checker before you publish a workflow update:

```bash
uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups \
  --require-app-mode
```

Then use the live smoke script when a real ComfyUI install is available:

```bash
uv run python scripts/run_comfyui_api_smoke.py \
  --workflow samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json
```
