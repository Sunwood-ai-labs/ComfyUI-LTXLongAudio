from __future__ import annotations

import importlib
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import Any, Callable, TypeVar

NOTEBOOK_REFERENCED_BACKEND = "notebook-referenced-python"

T = TypeVar("T")


def _runtime_modules() -> SimpleNamespace:
    return SimpleNamespace(
        torch=importlib.import_module("torch"),
        blocks=importlib.import_module("ltx_pipelines.utils.blocks"),
        loader=importlib.import_module("ltx_core.loader"),
        registry=importlib.import_module("ltx_core.loader.registry"),
        video_vae=importlib.import_module("ltx_core.model.video_vae"),
        audio_vae=importlib.import_module("ltx_core.model.audio_vae"),
        upsampler=importlib.import_module("ltx_core.model.upsampler"),
        text_gemma=importlib.import_module("ltx_core.text_encoders.gemma"),
        gpu_model_module=importlib.import_module("ltx_pipelines.utils.gpu_model"),
        helpers=importlib.import_module("ltx_pipelines.utils.helpers"),
        pipeline_module=importlib.import_module("ltx_pipelines.a2vid_two_stage"),
    )


class MetadataBackedStateDictLoader:
    def __init__(self, metadata_path: str) -> None:
        self._metadata_path = metadata_path
        mods = _runtime_modules()
        self._metadata_loader = mods.loader.SafetensorsModelStateDictLoader()
        self._weight_loader = mods.loader.SafetensorsStateDictLoader()

    def metadata(self, path: str) -> dict[str, Any]:
        return self._metadata_loader.metadata(self._metadata_path)

    def load(self, path: str | list[str], sd_ops: Any = None, device: Any = None) -> Any:
        return self._weight_loader.load(path, sd_ops=sd_ops, device=device)


class NotebookReferencedPromptEncoder:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        gemma_root: str,
        gemma_text_encoder_path: str,
        embeddings_connectors_path: str,
        dtype: Any,
        device: Any,
        registry: Any = None,
    ) -> None:
        mods = _runtime_modules()
        self._mods = mods
        self._dtype = dtype
        self._device = device
        active_registry = registry or mods.registry.DummyRegistry()
        module_ops = mods.text_gemma.module_ops_from_gemma_root(gemma_root)

        self._text_encoder_builder = mods.loader.SingleGPUModelBuilder(
            model_path=gemma_text_encoder_path,
            model_class_configurator=mods.text_gemma.GemmaTextEncoderConfigurator,
            model_sd_ops=mods.text_gemma.GEMMA_LLM_KEY_OPS,
            module_ops=(mods.text_gemma.GEMMA_MODEL_OPS, *module_ops),
            registry=active_registry,
        )
        self._embeddings_processor_builder = mods.loader.SingleGPUModelBuilder(
            model_path=embeddings_connectors_path,
            model_class_configurator=mods.text_gemma.EmbeddingsProcessorConfigurator,
            model_sd_ops=mods.text_gemma.EMBEDDINGS_PROCESSOR_KEY_OPS,
            model_loader=MetadataBackedStateDictLoader(checkpoint_path),
            registry=active_registry,
        )

    def _text_encoder_ctx(self, streaming_prefetch_count: int | None) -> AbstractContextManager[Any]:
        if streaming_prefetch_count is not None:
            return self._mods.blocks._streaming_model(
                self._text_encoder_builder.build(device=self._mods.torch.device("cpu"), dtype=self._dtype).eval(),
                layers_attr="model.model.language_model.layers",
                target_device=self._device,
                prefetch_count=streaming_prefetch_count,
            )
        return self._mods.gpu_model_module.gpu_model(
            self._text_encoder_builder.build(device=self._device, dtype=self._dtype).eval()
        )

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
        streaming_prefetch_count: int | None = None,
    ) -> list[Any]:
        with self._text_encoder_ctx(streaming_prefetch_count) as text_encoder:
            if enhance_first_prompt:
                prompts = list(prompts)
                prompts[0] = self._mods.helpers.generate_enhanced_prompt(
                    text_encoder,
                    prompts[0],
                    enhance_prompt_image,
                    seed=enhance_prompt_seed,
                )
            raw_outputs = [text_encoder.encode(prompt) for prompt in prompts]

        with self._mods.gpu_model_module.gpu_model(
            self._embeddings_processor_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        ) as embeddings_processor:
            return [embeddings_processor.process_hidden_states(hidden_states, mask) for hidden_states, mask in raw_outputs]


class NotebookReferencedImageConditioner:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        video_vae_path: str,
        dtype: Any,
        device: Any,
        registry: Any = None,
    ) -> None:
        mods = _runtime_modules()
        self._mods = mods
        self._dtype = dtype
        self._device = device
        self._encoder_builder = mods.loader.SingleGPUModelBuilder(
            model_path=video_vae_path,
            model_class_configurator=mods.video_vae.VideoEncoderConfigurator,
            model_sd_ops=mods.video_vae.VAE_ENCODER_COMFY_KEYS_FILTER,
            model_loader=MetadataBackedStateDictLoader(checkpoint_path),
            registry=registry or mods.registry.DummyRegistry(),
        )

    def __call__(self, fn: Callable[[Any], T]) -> T:
        with self._mods.gpu_model_module.gpu_model(
            self._encoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        ) as encoder:
            return fn(encoder)


class NotebookReferencedAudioConditioner:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        audio_vae_path: str,
        dtype: Any,
        device: Any,
        registry: Any = None,
    ) -> None:
        mods = _runtime_modules()
        self._mods = mods
        self._dtype = dtype
        self._device = device
        self._encoder_builder = mods.loader.SingleGPUModelBuilder(
            model_path=audio_vae_path,
            model_class_configurator=mods.audio_vae.AudioEncoderConfigurator,
            model_sd_ops=mods.audio_vae.AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            model_loader=MetadataBackedStateDictLoader(checkpoint_path),
            registry=registry or mods.registry.DummyRegistry(),
        )

    def __call__(self, fn: Callable[[Any], T]) -> T:
        with self._mods.gpu_model_module.gpu_model(
            self._encoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        ) as encoder:
            return fn(encoder)


class NotebookReferencedVideoUpsampler:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        video_vae_path: str,
        upsampler_path: str,
        dtype: Any,
        device: Any,
        registry: Any = None,
    ) -> None:
        mods = _runtime_modules()
        self._mods = mods
        self._dtype = dtype
        self._device = device
        active_registry = registry or mods.registry.DummyRegistry()
        self._encoder_builder = mods.loader.SingleGPUModelBuilder(
            model_path=video_vae_path,
            model_class_configurator=mods.video_vae.VideoEncoderConfigurator,
            model_sd_ops=mods.video_vae.VAE_ENCODER_COMFY_KEYS_FILTER,
            model_loader=MetadataBackedStateDictLoader(checkpoint_path),
            registry=active_registry,
        )
        self._upsampler_builder = mods.loader.SingleGPUModelBuilder(
            model_path=upsampler_path,
            model_class_configurator=mods.upsampler.LatentUpsamplerConfigurator,
            registry=active_registry,
        )

    def __call__(self, latent: Any) -> Any:
        with (
            self._mods.gpu_model_module.gpu_model(
                self._encoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
            ) as encoder,
            self._mods.gpu_model_module.gpu_model(
                self._upsampler_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
            ) as upsampler,
        ):
            return self._mods.upsampler.upsample_video(latent=latent, video_encoder=encoder, upsampler=upsampler)


class NotebookReferencedVideoDecoder:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        video_vae_path: str,
        dtype: Any,
        device: Any,
        registry: Any = None,
    ) -> None:
        mods = _runtime_modules()
        self._mods = mods
        self._dtype = dtype
        self._device = device
        self._decoder_builder = mods.loader.SingleGPUModelBuilder(
            model_path=video_vae_path,
            model_class_configurator=mods.video_vae.VideoDecoderConfigurator,
            model_sd_ops=mods.video_vae.VAE_DECODER_COMFY_KEYS_FILTER,
            model_loader=MetadataBackedStateDictLoader(checkpoint_path),
            registry=registry or mods.registry.DummyRegistry(),
        )

    def __call__(self, latent: Any, tiling_config: Any = None, generator: Any = None) -> Any:
        decoder = self._decoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        return self._mods.blocks._cleanup_iter(decoder.decode_video(latent, tiling_config, generator), decoder)


class NotebookReferencedA2VidPipelineTwoStage:
    def __init__(
        self,
        *,
        checkpoint_path: str,
        distilled_lora: tuple[Any, ...] | list[Any],
        spatial_upsampler_path: str,
        gemma_root: str,
        gemma_text_encoder_path: str,
        embeddings_connectors_path: str,
        video_vae_path: str,
        audio_vae_path: str,
        loras: tuple[Any, ...] | list[Any],
        device: Any = None,
        quantization: Any = None,
        registry: Any = None,
        torch_compile: bool = False,
    ) -> None:
        mods = _runtime_modules()
        self.device = device or mods.pipeline_module.get_device()
        self.dtype = mods.torch.bfloat16
        active_registry = registry or mods.registry.DummyRegistry()

        self.prompt_encoder = NotebookReferencedPromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            gemma_text_encoder_path=gemma_text_encoder_path,
            embeddings_connectors_path=embeddings_connectors_path,
            dtype=self.dtype,
            device=self.device,
            registry=active_registry,
        )
        self.image_conditioner = NotebookReferencedImageConditioner(
            checkpoint_path=checkpoint_path,
            video_vae_path=video_vae_path,
            dtype=self.dtype,
            device=self.device,
            registry=active_registry,
        )
        self.audio_conditioner = NotebookReferencedAudioConditioner(
            checkpoint_path=checkpoint_path,
            audio_vae_path=audio_vae_path,
            dtype=self.dtype,
            device=self.device,
            registry=active_registry,
        )
        self.stage_1 = mods.blocks.DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=active_registry,
            torch_compile=torch_compile,
        )
        self.stage_2 = mods.blocks.DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=(*tuple(loras), *tuple(distilled_lora)),
            quantization=quantization,
            registry=active_registry,
            torch_compile=torch_compile,
        )
        self.upsampler = NotebookReferencedVideoUpsampler(
            checkpoint_path=checkpoint_path,
            video_vae_path=video_vae_path,
            upsampler_path=spatial_upsampler_path,
            dtype=self.dtype,
            device=self.device,
            registry=active_registry,
        )
        self.video_decoder = NotebookReferencedVideoDecoder(
            checkpoint_path=checkpoint_path,
            video_vae_path=video_vae_path,
            dtype=self.dtype,
            device=self.device,
            registry=active_registry,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _runtime_modules().pipeline_module.A2VidPipelineTwoStage.__call__(self, *args, **kwargs)


def build_notebook_referenced_pipeline(**kwargs: Any) -> NotebookReferencedA2VidPipelineTwoStage:
    return NotebookReferencedA2VidPipelineTwoStage(**kwargs)


__all__ = [
    "NOTEBOOK_REFERENCED_BACKEND",
    "NotebookReferencedA2VidPipelineTwoStage",
    "NotebookReferencedPromptEncoder",
    "build_notebook_referenced_pipeline",
]
