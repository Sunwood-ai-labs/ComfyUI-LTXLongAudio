# LTX-2.3 GPU Instance Preparation

This document covers the non-ComfyUI path for running the long-audio Origin
workflow on a GPU instance.

Current scope:

- use this repository to plan long-audio segments and deterministic frame picks
- emit one official `ltx_pipelines.a2vid_two_stage` command per segment
- optionally run those commands directly when the official LTX-2 environment is installed

This does not claim full ComfyUI parity. It targets the LTX-2.3 inference stage.

## Official runtime

The official runtime and checkpoints come from Lightricks:

- GitHub: <https://github.com/Lightricks/LTX-2>
- Model card: <https://huggingface.co/Lightricks/LTX-2.3>

The model card states:

- Python `>= 3.12`
- CUDA `> 12.7`
- PyTorch `~= 2.7`

Recommended setup on the GPU instance:

```bash
bash scripts/bootstrap_ltx2_gpu.sh /workspace/LTX-2
source /workspace/LTX-2/.venv/bin/activate
export LD_LIBRARY_PATH=/usr/lib64-nvidia:/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.8/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

Required model assets from the `Lightricks/LTX-2.3` model card:

- full checkpoint
- distilled LoRA
- spatial upscaler
- Gemma text encoder root

This repo can now download those official assets for you:

```bash
uv run python cli/ltx23_download_models.py \
  --assets-root /workspace/models/ltx23_official
```

If you want the notebook / Origin workflow asset closure instead, use the notebook-aligned profile:

```bash
uv run python cli/ltx23_download_models.py \
  --asset-profile notebook-comfy \
  --assets-root /workspace/models/ltx23_notebook
```

That profile stages the GGUF UNet, split VAEs, Gemma fp8 text encoder, connectors, MelBand, and TAE helper weights. It also follows the saved Origin workflow defaults for `latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors` and distilled LoRA strength `0.6`.

If your environment needs authenticated Hugging Face access, set `HF_TOKEN`
or pass `--hf-token`.
The default Gemma repo used by the official pipeline is gated, so anonymous
downloads can fetch the public LTX assets but will stop at Gemma unless you
authenticate.
If you want to fetch the public assets first and add Gemma later, use
`--skip-gemma`.

If `nvidia-smi` or `torch.cuda.is_available()` cannot see the GPU even though
`/dev/nvidia*` exists, export `LD_LIBRARY_PATH` as shown above before running
the official pipeline.

The command examples below still target the official `ltx_pipelines` backend and therefore use `/workspace/models/ltx23_official/...`, not `/workspace/models/ltx23_notebook/...`.

## Preparing segment commands

Run from this repository:

```bash
uv run python cli/ltx23_gpu_ready.py \
  --workflow samples/workflows/LTX_2.3_Image_or_Text_&_Audio_2_Video_App_Origin.json \
  --audio samples/input/ltx-demo-tone.wav \
  --frames-dir samples/input/frames_pool \
  --checkpoint-path /workspace/models/ltx23_official/checkpoints/ltx-2.3-22b-dev.safetensors \
  --distilled-lora-path /workspace/models/ltx23_official/loras/ltx-2.3-22b-distilled-lora-384.safetensors \
  --spatial-upsampler-path /workspace/models/ltx23_official/upscalers/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --gemma-root /workspace/models/ltx23_official/gemma \
  --ltx-repo-root /workspace/LTX-2 \
  --output-dir /workspace/ltx23-run \
  --emit-run-script \
  --overwrite
```

This writes:

- `ltx23_gpu_ready_manifest.json`
- `run_segments.sh`

Behavior notes:

- `--emit-run-script` validates that checkpoint, distilled LoRA, upsampler, and Gemma paths exist.
- Width and height are normalized down to multiples of 64 before segment commands are written.
- Generated segment commands use the full conditioning audio plus `--audio-start-time` and `--audio-max-duration`.
- Final concatenation restores the original source audio after stitching the rendered video segments.

To run the same job directly on the GPU instance instead of only writing the script, add:

```bash
--run
```

## Validated L4 smoke recipe

The following path was validated end-to-end on the GPU handoff instance:

1. Prepare a smaller dataset from the bundled large WAV and studio stills:

```bash
uv run python cli/prepare_smoke_assets.py \
  --audio "samples/input/HOWL AT THE HAIRPIN2.wav" \
  --frames-dir samples/input/momiji_studio \
  --output-dir /workspace/ltx23_assets_howl20_momiji \
  --clip-seconds 20 \
  --resize-width 384 \
  --resize-height 216 \
  --overwrite
```

2. Run the smallest image-conditioned job that completed on the L4:

```bash
uv run python cli/ltx23_gpu_ready.py \
  --audio /workspace/ltx23_assets_howl20_momiji/audio/HOWL\ AT\ THE\ HAIRPIN2_20s.wav \
  --frames-dir /workspace/ltx23_assets_howl20_momiji/frames \
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
  --extra-ltx-arg=--streaming-prefetch-count \
  --extra-ltx-arg=1 \
  --run \
  --overwrite
```

Observed output on the validation instance:

- final video: `/workspace/ltx23_howl20_smoke/LTX-2.3-longaudio-randomimg.mp4`
- manifest: `/workspace/ltx23_howl20_smoke/ltx23_gpu_ready_manifest.json`
- one selected resized frame from `momiji_studio`
- one prepared stereo conditioning chunk under `/workspace/ltx23_howl20_smoke/conditioning_audio/`

This recipe intentionally uses `fps=1`, `128x64`, `num-inference-steps=1`, and a minimal prompt because larger settings were not reliable on that `NVIDIA L4 24GB` instance.

## Optional vocals-only conditioning

The ComfyUI workflow can condition on vocals-only audio while preserving the
original audio for final mux. The adapter does not separate vocals itself.

If you already have a vocals stem, pass it as:

```bash
--conditioning-audio /path/to/vocals.wav
```

The generated segment commands will use that file for LTX conditioning, while
the final mux still uses the original `--audio` input.
