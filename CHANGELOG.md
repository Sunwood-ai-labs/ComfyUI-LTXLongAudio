# Changelog

## Unreleased

- Added a bilingual repository surface with English and Japanese top-level guides
- Added VitePress docs, shared SVG identity assets, and GitHub Pages deployment workflow
- Fixed the API smoke runner default workflow path and aligned local verification around `uv`

## 0.3.0 - 2026-03-30

- Switched the public workflow surface to native `LTX*` node types instead of legacy compatibility names
- Added native loop-control mappings for `LTXWhileLoop*` and `LTXForLoop*`
- Updated docs and tests for the native long-audio workflow release

## 0.2.0 - 2026-03-30

- Added compatibility shims for the workflow nodes previously taken from `Easy Use`, `VideoHelperSuite`, and `KJNodes`
- Added long-audio helpers for audio loading, frame-folder loading, loop math, audio concatenation, and final video muxing
- Updated docs, tests, runtime requirements, and repository license for the compatibility-shim release

## 0.1.0 - 2026-03-30

- Initial public scaffold for `ComfyUI-LTXLongAudio`
- Added `LTX Long Audio Segment Info`
- Added `LTX Random Image Index`
