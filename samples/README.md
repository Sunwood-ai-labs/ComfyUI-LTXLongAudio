# Samples

`workflows/LTXLongAudio_CustomNodes_SmokeTest.json` is a lightweight ComfyUI workflow for validating the custom nodes in this repository without loading the full LTX generation stack.

Expected runtime inputs:

- Put one short audio file in your ComfyUI `input` directory.
- Put at least two same-size images in `input/frames_pool`.

Suggested smoke-test asset names:

- audio: `tone.wav`
- image folder: `frames_pool`

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

Before publishing a workflow update, you can also lint the layout:

```bash
uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups
```
