# Samples

`workflows/LTXLongAudio_CustomNodes_SmokeTest.json` is a lightweight ComfyUI workflow for validating the custom nodes in this repository without loading the full LTX generation stack.

Expected runtime inputs:

- The bundled workflow now defaults to assets inside `samples/input`.
- You can keep using the bundled sample files, or switch the controls to entries from your ComfyUI `input` directory.

Bundled smoke-test assets:

- audio: `samples/input/ltx-demo-tone.wav`
- image folder: `samples/input/demo_frames`

Local one-off files can still be dropped into `samples/input` or your ComfyUI `input` directory. The node dropdowns will list both locations.

The workflow exercises:

- `LTXLoadImages`
- `LTXLoadAudioUpload`
- `LTXLongAudioSegmentInfo`
- `LTXRandomImageIndex`
- `LTXIntConstant`
- `LTXSimpleMath`
- `LTXSimpleCalculator`
- `LTXCompare`
- `LTXIfElse`
- `LTXIndexAnything`
- `LTXBatchAnything`
- `LTXAudioConcatenate`
- `LTXVideoCombine`

Before publishing a workflow update, you can also lint the layout. The checker verifies group bounds, title-band collisions, node-node overlaps, and App mode metadata:

```bash
uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups \
  --require-app-mode
```

The smoke workflow is App mode ready. Its exposed controls are:

- `Frames Folder` -> `directory`
- `Source Audio` -> `audio`
- `Segment Audio` -> `audio`
- `Segment Seconds` -> `value`
- `Random Seed` -> `value`
- output -> `Video Combine (Smoke Test)`
