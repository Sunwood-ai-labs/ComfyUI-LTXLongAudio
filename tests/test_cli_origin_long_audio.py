import importlib.util
import sys
from pathlib import Path


def _load_cli_module():
    module_path = Path(__file__).resolve().parents[1] / "cli" / "ltx_origin_long_audio.py"
    spec = importlib.util.spec_from_file_location("comfyui_ltx_long_audio.cli.origin", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_origin_workflow_defaults_uses_sample_json():
    module = _load_cli_module()

    defaults = module.parse_origin_workflow_defaults()

    assert defaults.workflow_path == module.DEFAULT_WORKFLOW.resolve()
    assert defaults.segment_seconds == 20
    assert defaults.fps == 24.0
    assert defaults.random_seed == 1234
    assert defaults.frames_directory is None
    assert defaults.source_audio == "Grateful to be here.mp3"
    assert defaults.use_text_to_video is False
    assert defaults.use_only_vocals is True


def test_build_segment_plans_matches_existing_segment_math():
    module = _load_cli_module()

    plans = module.build_segment_plans(26.0, 10, 24.0)

    assert len(plans) == 3
    assert plans[0].start_time == 0.0
    assert plans[0].exact_seconds == 10.0
    assert plans[1].start_time == 10.0
    assert plans[1].exact_seconds == 10.0
    assert plans[2].start_time == 20.0
    assert plans[2].nominal_seconds == 6.0
    assert plans[2].frame_count == 145
    assert plans[2].exact_seconds == 6.0
    assert plans[2].is_last_segment is True


def test_select_segment_images_is_deterministic(tmp_path: Path):
    module = _load_cli_module()
    image_paths = []
    for name in ("a.png", "b.png", "c.png"):
        path = tmp_path / name
        path.write_bytes(b"img")
        image_paths.append(path)

    plans = module.build_segment_plans(21.0, 10, 24.0)

    first = module.select_segment_images(image_paths, plans, 1234)
    second = module.select_segment_images(image_paths, plans, 1234)
    third = module.select_segment_images(image_paths, plans, 1235)

    assert first == second
    assert first != third


def test_list_frame_images_applies_skip_and_stride(tmp_path: Path):
    module = _load_cli_module()

    for index in range(5):
        (tmp_path / f"{index:03d}.png").write_bytes(b"img")

    selected = module.list_frame_images(tmp_path, skip_first_images=1, select_every_nth=2)

    assert [path.name for path in selected] == ["001.png", "003.png"]
