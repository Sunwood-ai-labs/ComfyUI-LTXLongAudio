import importlib.util
from pathlib import Path

import pytest


def _load_nodes_module():
    module_path = Path(__file__).resolve().parents[1] / "nodes.py"
    spec = importlib.util.spec_from_file_location("comfyui_ltx_long_audio.nodes", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_node_module_exports():
    module = _load_nodes_module()

    assert "LTXLongAudioSegmentInfo" in module.NODE_CLASS_MAPPINGS
    assert "LTXRandomImageIndex" in module.NODE_CLASS_MAPPINGS
    assert "LTXLoadImageUpload" in module.NODE_CLASS_MAPPINGS
    assert "LTXBatchUploadedFrames" in module.NODE_CLASS_MAPPINGS
    assert "LTXRepeatImageBatch" in module.NODE_CLASS_MAPPINGS
    assert "LTXAppendImageBatch" in module.NODE_CLASS_MAPPINGS
    assert "LTXEnsureImageBatch" in module.NODE_CLASS_MAPPINGS
    assert "LTXForLoopStart" in module.NODE_CLASS_MAPPINGS
    assert "LTXForLoopEnd" in module.NODE_CLASS_MAPPINGS
    assert "LTXAudioSlice" in module.NODE_CLASS_MAPPINGS
    assert "LTXAudioDuration" in module.NODE_CLASS_MAPPINGS
    assert "LTXDummyRenderSegment" in module.NODE_CLASS_MAPPINGS
    assert "LTXSplitAudioIntoChunks" in module.NODE_CLASS_MAPPINGS
    assert "LTXRandomSelectChunkImages" in module.NODE_CLASS_MAPPINGS
    assert "LTXDummyRenderChunkSequence" in module.NODE_CLASS_MAPPINGS
    assert "LTXConcatenateDummySegments" in module.NODE_CLASS_MAPPINGS
    assert "LTXBuildChunkedStillVideo" in module.NODE_CLASS_MAPPINGS
    assert "LTXAppendAudio" in module.NODE_CLASS_MAPPINGS
    assert "LTXEnsureAudio" in module.NODE_CLASS_MAPPINGS
    assert "LTXVideoCombine" in module.NODE_CLASS_MAPPINGS
    assert "LTXSimpleCalculator" in module.NODE_CLASS_MAPPINGS
    for legacy in ("easy forLoopStart", "VHS_VideoCombine", "SimpleCalculatorKJ"):
        assert legacy not in module.NODE_CLASS_MAPPINGS


def test_pure_node_behaviors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_nodes_module()

    segment = module.LTXLongAudioSegmentInfo()
    start_time, segment_seconds, frames, exact_seconds, is_last, segment_count = segment.segment_info(26.0, 10, 2, 24.0)
    assert start_time == 20.0
    assert segment_seconds == 6.0
    assert frames == 145
    assert exact_seconds == 6.0
    assert is_last is True
    assert segment_count == 3

    simple_math = module.NODE_CLASS_MAPPINGS["LTXSimpleMath"]()
    assert simple_math.execute("ceil(a / b)", a=21, b=10) == (3, 3.0, True)

    compare = module.NODE_CLASS_MAPPINGS["LTXCompare"]()
    assert compare.compare(5, 10, "a < b") == (True,)

    formula = module.NODE_CLASS_MAPPINGS["LTXSimpleCalculator"]()
    assert formula.calculate("a + b", a=2, b=3) == (5.0, 5, True)

    if module.torch is not None:
        frame_batch = module.NODE_CLASS_MAPPINGS["LTXBatchUploadedFrames"]()
        image_a = module.torch.zeros((1, 4, 4, 3), dtype=module.torch.float32)
        image_b = module.torch.ones((1, 4, 4, 3), dtype=module.torch.float32)
        batched_images, frame_count = frame_batch.batch_images(image_a, image_2=image_b)
        assert batched_images.shape == (2, 4, 4, 3)
        assert frame_count == 2

        repeat_batch = module.NODE_CLASS_MAPPINGS["LTXRepeatImageBatch"]()
        repeated_images, repeated_count = repeat_batch.repeat_image(image_a, 3)
        assert repeated_images.shape == (3, 4, 4, 3)
        assert repeated_count == 3

        append_images = module.NODE_CLASS_MAPPINGS["LTXAppendImageBatch"]()
        merged_images, merged_count = append_images.append_images(image_b, previous_images=image_a)
        assert merged_images.shape == (2, 4, 4, 3)
        assert merged_count == 2

        monkeypatch.setattr(module, "FRAME_STORE_SPILL_THRESHOLD_BYTES", 1)
        spilled_images, spilled_count = append_images.append_images(image_b, previous_images=image_a)
        assert spilled_count == 2
        assert spilled_images[module.FRAME_STORE_MAGIC] is True
        assert len(spilled_images["segments"]) == 2
        assert all(Path(segment["dir"]).exists() for segment in spilled_images["segments"])

        ensure_images = module.NODE_CLASS_MAPPINGS["LTXEnsureImageBatch"]()
        ensured_images, ensured_count = ensure_images.ensure_images(merged_images)
        assert ensured_images.shape == (2, 4, 4, 3)
        assert ensured_count == 2

        audio_slice = module.NODE_CLASS_MAPPINGS["LTXAudioSlice"]()
        sample_rate = 8
        waveform = module.torch.arange(0, 32, dtype=module.torch.float32).reshape(1, 1, 32)
        sliced_audio, sliced_duration = audio_slice.slice_audio(
            {"waveform": waveform, "sample_rate": sample_rate},
            start_time=1.0,
            duration=1.5,
        )
        assert sliced_audio["waveform"].shape[-1] == 12
        assert sliced_duration == 1.5

        audio_duration = module.NODE_CLASS_MAPPINGS["LTXAudioDuration"]()
        assert audio_duration.get_duration({"waveform": waveform, "sample_rate": sample_rate}) == (4.0,)

        dummy_segment = module.NODE_CLASS_MAPPINGS["LTXDummyRenderSegment"]()
        segment_images, segment_audio, segment_fps, segment_frame_count, segment_duration = dummy_segment.render(
            image_a,
            {"waveform": waveform[..., :16], "sample_rate": sample_rate},
            fps=2.0,
        )
        assert segment_images.shape == (5, 4, 4, 3)
        assert segment_audio["waveform"].shape[-1] == 16
        assert segment_fps == 2.0
        assert segment_frame_count == 5
        assert segment_duration == 2.0

        split_chunks = module.NODE_CLASS_MAPPINGS["LTXSplitAudioIntoChunks"]()
        audio_chunks, audio_chunk_count, total_duration = split_chunks.split(
            {"waveform": waveform, "sample_rate": sample_rate},
            segment_seconds=2,
        )
        assert audio_chunk_count == 2
        assert total_duration == 4.0
        assert len(audio_chunks["segments"]) == 2
        assert audio_chunks["segments"][0]["waveform"].shape[-1] == 16

        random_chunk_images = module.NODE_CLASS_MAPPINGS["LTXRandomSelectChunkImages"]()
        selected_images, selected_count = random_chunk_images.select(batched_images, audio_chunks, seed=1234)
        assert selected_images.shape == (2, 4, 4, 3)
        assert selected_count == 2

        render_chunk_sequence = module.NODE_CLASS_MAPPINGS["LTXDummyRenderChunkSequence"]()
        dummy_segments, sequence_fps, sequence_count = render_chunk_sequence.render(selected_images, audio_chunks, fps=2.0)
        assert sequence_fps == 2.0
        assert sequence_count == 2
        assert len(dummy_segments["segments"]) == 2

        concatenate_dummy_segments = module.NODE_CLASS_MAPPINGS["LTXConcatenateDummySegments"]()
        concatenated_images, concatenated_audio, concatenated_fps, concatenated_count = concatenate_dummy_segments.concatenate(dummy_segments)
        assert concatenated_images.shape == (10, 4, 4, 3)
        assert concatenated_audio["waveform"].shape[-1] == 32
        assert concatenated_fps == 2.0
        assert concatenated_count == 2

        chunk_builder = module.NODE_CLASS_MAPPINGS["LTXBuildChunkedStillVideo"]()
        built_images, built_audio, built_fps, built_segment_count = chunk_builder.build(
            batched_images,
            {"waveform": waveform, "sample_rate": sample_rate},
            segment_seconds=2,
            seed=1234,
            fps=2.0,
        )
        assert built_images.shape == (10, 4, 4, 3)
        assert built_audio["waveform"].shape[-1] == 32
        assert built_fps == 2.0
        assert built_segment_count == 2

        append_audio = module.NODE_CLASS_MAPPINGS["LTXAppendAudio"]()
        merged_audio, merged_duration = append_audio.append_audio(
            {"waveform": waveform[..., 16:], "sample_rate": sample_rate},
            previous_audio={"waveform": waveform[..., :16], "sample_rate": sample_rate},
        )
        assert merged_audio["waveform"].shape[-1] == 32
        assert merged_duration == 4.0

        ensure_audio = module.NODE_CLASS_MAPPINGS["LTXEnsureAudio"]()
        ensured_audio, ensured_audio_duration = ensure_audio.ensure_audio(merged_audio)
        assert ensured_audio["waveform"].shape[-1] == 32
        assert ensured_audio_duration == 4.0

        monkeypatch.setattr(module, "_output_directory", lambda save_output: str(tmp_path))
        monkeypatch.setattr(module, "_ffmpeg_executable", lambda: "ffmpeg")
        ffmpeg_commands = []

        def _fake_run(command, check, capture_output):
            ffmpeg_commands.append(command)
            return None

        monkeypatch.setattr(module.subprocess, "run", _fake_run)
        video_combine = module.NODE_CLASS_MAPPINGS["LTXVideoCombine"]()
        preview_result = video_combine.combine_video(
            repeated_images,
            frame_rate=2.0,
            filename_prefix="ltx-preview-test",
            save_output=False,
            trim_to_audio=False,
        )
        assert "ui" in preview_result
        assert "images" in preview_result["ui"]
        assert preview_result["ui"]["animated"][0] is True
        assert ffmpeg_commands

        spilled_preview = video_combine.combine_video(
            spilled_images,
            frame_rate=2.0,
            filename_prefix="ltx-preview-store-test",
            save_output=False,
            trim_to_audio=False,
        )
        assert "ui" in spilled_preview
        assert spilled_preview["ui"]["animated"][0] is True
        assert all(not Path(segment["dir"]).exists() for segment in spilled_images["segments"])


def test_sample_inputs_are_listed_and_resolved(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_nodes_module()

    repo_root = tmp_path / "repo"
    sample_input = repo_root / "samples" / "input"
    sample_input.mkdir(parents=True)
    (sample_input / "demo.wav").write_bytes(b"demo")
    (sample_input / "demo.png").write_bytes(b"demo-image")
    (sample_input / "frames_pool").mkdir()

    comfy_input = tmp_path / "comfy_input"
    comfy_input.mkdir()
    (comfy_input / "root.wav").write_bytes(b"root")
    (comfy_input / "root.png").write_bytes(b"root-image")
    (comfy_input / "3d").mkdir()

    monkeypatch.setattr(module, "_repository_root", lambda: repo_root)
    monkeypatch.setattr(module, "_sample_input_directory", lambda: sample_input)
    monkeypatch.setattr(module, "_input_directory", lambda: str(comfy_input))
    monkeypatch.setattr(module, "folder_paths", None)

    assert module._list_input_audio_files() == ["root.wav", "samples/input/demo.wav"]
    assert module._list_input_image_files() == ["root.png", "samples/input/demo.png"]
    assert module._list_input_subdirectories() == ["3d", "samples/input/frames_pool"]
    assert module._resolve_input_path("root.wav") == str(comfy_input / "root.wav")
    assert module._resolve_input_path("samples/input/demo.wav") == str(sample_input / "demo.wav")
    assert module._resolve_input_path("samples/input/frames_pool") == str(sample_input / "frames_pool")


def test_upload_inputs_include_blank_and_upload_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_nodes_module()

    comfy_input = tmp_path / "comfy_input"
    comfy_input.mkdir()
    (comfy_input / "root.wav").write_bytes(b"root")
    (comfy_input / "root.png").write_bytes(b"root-image")

    monkeypatch.setattr(module, "_input_directory", lambda: str(comfy_input))
    monkeypatch.setattr(module, "_sample_input_directory", lambda: tmp_path / "missing_samples")
    monkeypatch.setattr(module, "folder_paths", None)

    audio_spec = module.NODE_CLASS_MAPPINGS["LTXLoadAudioUpload"].INPUT_TYPES()["required"]["audio"]
    image_spec = module.NODE_CLASS_MAPPINGS["LTXLoadImageUpload"].INPUT_TYPES()["required"]["image"]
    directory_spec = module.NODE_CLASS_MAPPINGS["LTXLoadImages"].INPUT_TYPES()["required"]["directory"]

    assert audio_spec[0] == "COMBO"
    assert audio_spec[1]["audio_upload"] is True
    assert audio_spec[1]["options"][0] == ""
    assert "root.wav" in audio_spec[1]["options"]

    assert image_spec[0] == "COMBO"
    assert image_spec[1]["image_upload"] is True
    assert image_spec[1]["options"][0] == ""
    assert "root.png" in image_spec[1]["options"]

    assert directory_spec[0] == "COMBO"
    assert directory_spec[1]["options"][0] == ""


def test_ffmpeg_executable_uses_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module = _load_nodes_module()

    ffmpeg_path = tmp_path / "ffmpeg-custom.exe"
    ffmpeg_path.write_bytes(b"")

    monkeypatch.setenv("LTX_FFMPEG_EXE", str(ffmpeg_path))
    monkeypatch.setattr(module.shutil, "which", lambda name: None)

    assert module._ffmpeg_executable() == str(ffmpeg_path)
