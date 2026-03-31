# Samples

`workflows/LTXLongAudio_CustomNodes_SmokeTest.json` is a lightweight ComfyUI workflow for validating the custom nodes in this repository without loading the full LTX generation stack.

`workflows/LTX_2.3_Image_or_Text_&_Audio_2_Video_App_Origin.json` is the heavier verified long-audio Origin workflow that keeps the full LTX 2.3 generation path and swaps in the native `LTX*` long-audio loop helpers from this repository.

Expected runtime inputs:

- The bundled workflow now opens with concrete sample defaults.
- Default frame folder: `samples/input/frames_pool`
- Default audio widget name: `ltx-demo-tone.wav`
- You can still replace either value in App mode or the node graph.

Bundled smoke-test assets:

- tracked fallback audio: `samples/input/ltx-demo-tone.wav`
- image folder: `samples/input/frames_pool`

Local one-off files can still be dropped into `samples/input` or your ComfyUI `input` directory. Folder and audio dropdowns list both locations.

The workflow exercises:

- `LoadAudio`
- `LTXAudioDuration`
- `LTXLoadImages`
- `LTXLongAudioSegmentInfo`
- `LTXBuildChunkedStillVideo`
- `LTXIntConstant`
- `LTXVideoCombine`

Before publishing a workflow update, you can also lint the layout. The checker verifies group bounds, title-band collisions, node-node overlaps, App mode metadata, and common runtime contract issues such as missing required links or linked combo widgets:

```bash
uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups \
  --require-app-mode
```

The smoke workflow is App mode ready. Its exposed controls are:

- `Source Audio Upload` -> `audio`
- `Frames Folder` -> `directory`
- `Segment Seconds` -> `value`
- `Random Seed` -> `value`
- output -> `Video Combine + Preview (20s Chunks)`

The generated result is a dummy long-audio still-video: the custom node splits the song into 20-second chunks, picks a deterministic random frame for each chunk from the selected folder, runs a dummy segment renderer for each chunk internally, concatenates all chunk audio and frame batches, and previews the final mp4 directly from the output node.

The Origin workflow is intended for the real LTX stack rather than the lightweight dummy smoke path. It keeps:

- image-plus-prompt conditioning when `Use Text to Video` is `OFF`
- `Prompt Enhancer` bypassed in the saved workflow state
- folder-based image selection and source-audio slicing inside the long-audio loop

If you keep an optional longer `HOWL AT THE HAIRPIN2.wav` sample locally, `scripts/run_comfyui_api_smoke.py` prefers it and stages it under the tracked upload name for prompt API validation.

If App mode still shows older controls after you update the custom node, fully restart the ComfyUI backend or Desktop app before reopening the workflow.

You can also validate the exact workflow end-to-end through ComfyUI's `/prompt` API:

```bash
uv run python scripts/run_comfyui_api_smoke.py \
  --workflow samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --comfy-root /path/to/ComfyUI
```
