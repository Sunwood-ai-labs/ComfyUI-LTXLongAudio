# Troubleshooting

## App mode still shows old controls

Fully restart the ComfyUI backend or desktop app after updating custom nodes. A hot reload can keep stale schema definitions alive.

## The preview looks too short

The bundled graph is meant to render the full source audio length, not only the first chunk. If the preview looks unexpectedly short, stale backend state is the first thing to check.

## The audio file is missing from the dropdown

Remember how discovery works:

- repository sample files appear with the `samples/input/...` prefix for folder-style entries
- uploaded audio still relies on what is visible to ComfyUI's input system
- the smoke script can stage fallback audio for prompt API validation, but the interactive UI still expects either an uploaded audio file or a matching runtime input asset

## `ffmpeg` is missing

Final muxing depends on `ffmpeg`. Make it visible in the runtime before launching ComfyUI, or set the explicit path for the API smoke script when needed.

## What to run before publishing

```bash
uv run pytest

uv run python scripts/check_workflow_layout.py \
  samples/workflows/LTXLongAudio_CustomNodes_SmokeTest.json \
  --require-all-nodes-in-groups \
  --require-app-mode
```

Use the prompt API smoke runner only when you have a real ComfyUI environment available.
