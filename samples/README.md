# Samples

`workflows/LTXLongAudio_CustomNodes_SmokeTest.json` is a lightweight ComfyUI workflow for validating the custom nodes in this repository without loading the full LTX generation stack.

Expected runtime inputs:

- The bundled workflow now opens with blank controls instead of hardcoded file paths.
- Pick one frame folder from your ComfyUI `input` directory and upload one source audio file in App mode.

Bundled smoke-test assets:

- audio: `samples/input/HOWL AT THE HAIRPIN2.wav`
- image folder: `samples/input/frames_pool`

Local one-off files can still be dropped into `samples/input` or your ComfyUI `input` directory. Folder and audio dropdowns list both locations, and blank is allowed until you decide what to use.

The workflow exercises:

- `LoadAudio`
- `LTXAudioDuration`
- `LTXLoadImages`
- `LTXLongAudioSegmentInfo`
- `LTXRandomImageIndex`
- `LTXAudioSlice`
- `LTXIntConstant`
- `LTXSimpleMath`
- `LTXSimpleCalculator`
- `LTXCompare`
- `LTXIfElse`
- `LTXIndexAnything`
- `LTXBatchAnything`
- `LTXAudioConcatenate`
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
- output -> `Video Combine (Smoke Test)`

If App mode still shows older controls after you update the custom node, fully restart the ComfyUI backend or Desktop app before reopening the workflow.

You can also validate the exact workflow end-to-end through ComfyUI's `/prompt` API:

```bash
uv run python scripts/run_comfyui_api_smoke.py \
  --workflow D:/Prj/ComfyUI_LTX2_3_TI2V/LTXLongAudio_CustomNodes_SmokeTest.json
```
