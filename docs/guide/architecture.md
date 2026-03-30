# Architecture

## Long-audio pipeline

The repository keeps the long-audio path explicit:

1. `LTXAudioDuration` and `LTXLongAudioSegmentInfo` plan segments from the real source audio length.
2. `LTXBuildChunkedStillVideo` splits the song into chunk-sized windows.
3. Each chunk chooses one deterministic frame from the selected folder.
4. Internal dummy segment rendering produces a still-video slice for each chunk.
5. Image batches and audio segments are concatenated back into one previewable result.
6. `LTXVideoCombine` muxes the final MP4 preview payload.

Segment frame counts are quantized in blocks of 8 frames to stay compatible with LTX-style downstream assumptions.

## Node families

### Inputs and staging

- `LTXLoadAudioUpload`
- `LTXLoadImageUpload`
- `LTXLoadImages`
- `LTXBatchUploadedFrames`
- `LTXRepeatImageBatch`

### Chunk planning and media assembly

- `LTXAudioDuration`
- `LTXLongAudioSegmentInfo`
- `LTXRandomImageIndex`
- `LTXAudioSlice`
- `LTXBuildChunkedStillVideo`
- `LTXDummyRenderSegment`
- `LTXAppendImageBatch`
- `LTXAppendAudio`
- `LTXEnsureImageBatch`
- `LTXEnsureAudio`
- `LTXAudioConcatenate`
- `LTXVideoCombine`

### Loop control and workflow helpers

- `LTXWhileLoopStart`, `LTXWhileLoopEnd`
- `LTXForLoopStart`, `LTXForLoopEnd`
- `LTXIfElse`
- `LTXCompare`
- `LTXSimpleMath`
- `LTXSimpleCalculator`
- `LTXIntConstant`
- `LTXIndexAnything`
- `LTXBatchAnything`
- `LTXSeedList`
- `LTXShowAnything`

### LTX utility replacements

- `LTXVAELoader`
- `LTXImageResize`
- `LTXChunkFeedForward`
- `LTXSamplingPreviewOverride`
- `LTXNormalizedAttentionGuidance`

## Repository layout

```text
docs/                         VitePress docs and shared SVG identity
samples/input/                lightweight sample frames and fallback audio
samples/workflows/            smoke workflow JSON
scripts/check_workflow_layout.py
scripts/run_comfyui_api_smoke.py
tests/                        import and workflow regression tests
nodes.py                      custom node implementations and mappings
```
