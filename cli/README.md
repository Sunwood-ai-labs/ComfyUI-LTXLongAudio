# CLI

This folder contains a ComfyUI-free first pass of the long-audio loop from
`samples/workflows/LTX_2.3_Image_or_Text_&_Audio_2_Video_App_Origin.json`.

Current scope:

- read workflow defaults from the Origin JSON
- resolve source audio and frame-folder inputs
- build frame-quantized long-audio segment plans
- select one deterministic image per segment
- optionally extract per-segment WAV files with `origin_long_audio.py`
- prepare GPU-ready LTX-2.3 segment commands for the official `ltx_pipelines` runtime
- emit a runnable `run_segments.sh` handoff script for GPU instances
- optionally run official LTX-2.3 inference in-process from one Python runner and restore the original full-length audio in the final mux
- optionally split prompt encoding onto CPU while keeping the diffusion stages on GPU for low-VRAM runs
- write a manifest that records selected images, segment timings, and command previews

Deliberately excluded from this first CLI milestone:

- automatic vocals separation
- checkpoint download or model installation
- ComfyUI graph execution
- ComfyUI loop-control nodes

Main entrypoints:

- `origin_long_audio.py`: simple long-audio planning + optional WAV extraction
- `ltx23_download_models.py`: download the official assets required by the GPU runner
- `ltx23_gpu_ready.py`: emit or run official LTX-2.3 segment inference commands
- `prepare_smoke_assets.py`: clip `HOWL AT THE HAIRPIN2.wav` and resize `momiji_studio` for lightweight smoke runs
- `ltx_origin_long_audio.py`: experimental still-image MP4 prototype
- `python -m cli.audio_segmentation`: visualize a WAV file, estimate natural split points, and optionally export chunk WAV files

The CLI stays inside Python packages. It does not import or execute ComfyUI at runtime. The notebook and Origin workflow are treated as reference material for defaults, model filenames, and asset closure only.

Run the simple planner with `uv`:

```bash
uv run python cli/origin_long_audio.py \
  --audio /path/to/song.mp3 \
  --frames-dir /path/to/frames \
  --overwrite
```

Prepare GPU-ready LTX-2.3 commands:

```bash
uv run python cli/ltx23_download_models.py \
  --assets-root /workspace/models/ltx23_official

uv run python cli/ltx23_gpu_ready.py \
  --audio /path/to/song.mp3 \
  --frames-dir /path/to/frames \
  --checkpoint-path /workspace/models/ltx23_official/checkpoints/ltx-2.3-22b-dev.safetensors \
  --distilled-lora-path /workspace/models/ltx23_official/loras/ltx-2.3-22b-distilled-lora-384.safetensors \
  --spatial-upsampler-path /workspace/models/ltx23_official/upscalers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /workspace/models/ltx23_official/gemma \
  --output-dir /workspace/ltx23-run \
  --conditioning-audio /path/to/optional-vocals.wav \
  --emit-run-script \
  --overwrite
```

If you want the notebook / Origin workflow asset closure instead of the official `ltx_pipelines` asset set, use the dedicated profile:

```bash
uv run python cli/ltx23_download_models.py \
  --asset-profile notebook-comfy \
  --assets-root /workspace/models/ltx23_notebook
```

That profile downloads the notebook-aligned files:

- `ltx-2.3-22b-dev-Q4_K_M.gguf`
- `gemma_3_12B_it_fp8_scaled.safetensors`
- `mmproj-BF16.gguf`
- `ltx-2.3-22b-dev_embeddings_connectors.safetensors`
- `ltx-2.3-22b-dev_video_vae.safetensors`
- `ltx-2.3-22b-dev_audio_vae.safetensors`
- `ltx-2.3-spatial-upscaler-x2-1.0.safetensors`
- `ltx-2.3-22b-distilled-lora-384.safetensors`
- `MelBandRoformer_fp16.safetensors`
- `taeltx2_3.safetensors`

When `--run` is enabled, `ltx23_gpu_ready.py` now executes the official LTX pipeline in-process instead of spawning one Python subprocess per segment. The current interpreter must be able to import the official `ltx_pipelines` packages. On GPU boxes that usually means running this CLI from the LTX-2 environment and pointing `--ltx-repo-root` at the cloned official repository.

When VRAM is tight, add `--prompt-encoder-device cpu` to keep the Gemma prompt encoder off the GPU. The encoded prompt tensors are copied back to the pipeline device before diffusion starts, so the main denoising path still runs on GPU.

When a run is slow or stalls, add `--debug`. The runner will emit step-by-step progress to stdout and write `ltx23_debug.jsonl` under the output directory. That log includes:

- runtime import timing
- pipeline build timing and device placement
- phase-level timing for `prompt_encoder`, `audio_conditioner`, `image_conditioner`, `stage_1`, `upsampler`, `stage_2`, and `video_decoder`
- per-segment start / pipeline / encode timing
- CUDA memory snapshots before and after each segment
- final concat / mux timing

The manifest now also includes a `timings` section with the aggregated stage durations, and the runner prints the same summary to stdout after each run. That makes it easy to compare `prompt_encoder`, `stage_1`, `stage_2`, `video_decoder`, segment pipeline time, encode time, concat time, and final mux time without manually parsing the JSONL log.

Performance tuning is now first-class instead of hidden behind `--extra-ltx-arg`:

- `--performance-profile throughput`: disables layer streaming unless you explicitly set it and defaults `--max-batch-size` to `4`
- `--performance-profile low-vram`: enables `--streaming-prefetch-count 1` unless you explicitly override it and also defaults `--max-batch-size` to `4`
- `--streaming-prefetch-count N`: explicit official layer-streaming control
- `--prompt-streaming-prefetch-count N`: in-process only override for the prompt encoder, useful when Gemma must stream but diffusion should stay GPU-resident
- `--max-batch-size N`: explicit guidance batching control
- `--compile-transformer`: forwards to the official `--compile` flag

The in-process runner now also forwards the resolved `max_batch_size` into Stage 2. The official CLI path only exposes it on the top-level pipeline call, but the upstream `A2VidPipelineTwoStage` implementation applies it to Stage 1 and forgets it for Stage 2. This runner patches that gap so the same batching hint reaches both denoising stages.

Why this matters on L4: `--streaming-prefetch-count 1` pushes the official PromptEncoder and DiffusionStage into CPU-built layer streaming mode. That saves VRAM, but it can leave GPU utilization very low. For throughput-focused runs on a 24 GB card, start with `--performance-profile throughput --quantization fp8-cast`. If Gemma then OOMs, add `--prompt-streaming-prefetch-count 1` before falling back to full low-VRAM mode.

Example on a GPU instance with the official repo bootstrapped under `/workspace/LTX-2`:

```bash
cd /workspace/ComfyUI-LTXLongAudio
/workspace/LTX-2/.venv/bin/python -m cli.ltx23_gpu_ready \
  --audio /path/to/song.mp3 \
  --frames-dir /path/to/frames \
  --checkpoint-path /workspace/models/ltx23_official/checkpoints/ltx-2.3-22b-dev.safetensors \
  --distilled-lora-path /workspace/models/ltx23_official/loras/ltx-2.3-22b-distilled-lora-384.safetensors \
  --spatial-upsampler-path /workspace/models/ltx23_official/upscalers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /workspace/models/ltx23_official/gemma \
  --ltx-repo-root /workspace/LTX-2 \
  --output-dir /workspace/ltx23-run \
  --prompt-encoder-device cpu \
  --performance-profile low-vram \
  --run \
  --overwrite
```

Prepare the repo's bundled smoke assets from the larger sample WAV + studio frames:

```bash
uv run python cli/prepare_smoke_assets.py \
  --audio "samples/input/HOWL AT THE HAIRPIN2.wav" \
  --frames-dir samples/input/momiji_studio \
  --output-dir cli_output/smoke_assets/howl20_momiji \
  --clip-seconds 20 \
  --resize-width 384 \
  --resize-height 216 \
  --overwrite
```

The prepared output will contain:

- `audio/HOWL AT THE HAIRPIN2_20s.wav`
- `frames/*.png` resized to `384x216`
- `smoke_assets_manifest.json`

Validated lightweight GPU smoke run:

```bash
uv run python cli/ltx23_gpu_ready.py \
  --audio cli_output/smoke_assets/howl20_momiji/audio/HOWL\ AT\ THE\ HAIRPIN2_20s.wav \
  --frames-dir cli_output/smoke_assets/howl20_momiji/frames \
  --prompt "a" \
  --negative-prompt "" \
  --checkpoint-path /workspace/models/ltx23_official/checkpoints/ltx-2.3-22b-dev.safetensors \
  --distilled-lora-path /workspace/models/ltx23_official/loras/ltx-2.3-22b-distilled-lora-384.safetensors \
  --spatial-upsampler-path /workspace/models/ltx23_official/upscalers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /workspace/models/ltx23_official/gemma \
  --ltx-python /workspace/LTX-2/.venv/bin/python \
  --ltx-repo-root /workspace/LTX-2 \
  --output-dir /workspace/ltx23_howl20_smoke \
  --segment-seconds 20 \
  --fps 1 \
  --width 128 \
  --height 64 \
  --num-inference-steps 1 \
  --quantization fp8-cast \
  --video-cfg-guidance-scale 1.0 \
  --performance-profile throughput \
  --debug \
  --run \
  --overwrite
```

This is the smallest image-conditioned recipe currently verified to finish on the L4 validation instance. It produced:

- one rendered segment
- `ltx23_gpu_ready_manifest.json`
- `LTX-2.3-longaudio-randomimg.mp4`

Notes:

- `ltx23_gpu_ready.py` normalizes width and height down to multiples of 64 for the official two-stage backend.
- `--run` keeps the official LTX pipeline inside one Python process. `--emit-run-script` remains available as a legacy handoff/debug path and still writes one command per segment.
- `--emit-run-script` and `--run` both require real model asset paths; prepare-only without those flags can still emit a preview manifest.
- When `--conditioning-audio` is provided, segment commands condition on that file while the final mux still restores the original `--audio`.
- When `--run` or `--emit-run-script` is used, segment conditioning audio is prepared per chunk as stereo WAV under `conditioning_audio/`.
- `ltx23_download_models.py` now supports two asset profiles: `official` for the current `ltx_pipelines` runner, and `notebook-comfy` for the notebook / Origin workflow asset closure.
- `ltx23_gpu_ready.py` still targets the official Python `ltx_pipelines` backend today. Downloading the notebook profile does not mean the runner starts importing ComfyUI.
- The Gemma snapshot used by the official pipeline is gated on Hugging Face, so `HF_TOKEN` may be required for a full download.
- `ltx23_download_models.py --skip-gemma` lets you fetch the public assets first and add Gemma later.

If you want the still-video prototype only, use `ltx_origin_long_audio.py --plan-only`.

Natural segmentation helper:

```bash
uv run python -m cli.audio_segmentation \
  /path/to/song.wav \
  --max-segment-seconds 15 \
  --split \
  --overwrite
```
