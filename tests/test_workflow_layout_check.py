import importlib.util
from pathlib import Path


def _load_layout_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "check_workflow_layout.py"
    spec = importlib.util.spec_from_file_location("comfyui_ltx_long_audio.layout_check", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sample_workflow_layout_passes():
    module = _load_layout_module()
    workflow_path = Path(__file__).resolve().parents[1] / "samples" / "workflows" / "LTXLongAudio_CustomNodes_SmokeTest.json"

    report = module.analyze_workflow(
        workflow_path,
        title_padding=80.0,
        inner_padding=12.0,
        check_node_overlap=False,
        require_all_nodes_in_groups=True,
    )

    assert report["issues"] == []
    assert report["group_count"] == 3
    assert report["node_count"] == 17
    assert report["node_group_matches"]["Video Combine (Smoke Test)"] == ["Output"]


def test_layout_checker_reports_overlaps(tmp_path):
    module = _load_layout_module()
    workflow_path = tmp_path / "broken_workflow.json"
    workflow_path.write_text(
        """
{
  "nodes": [
    {
      "id": 1,
      "type": "LTXIntConstant",
      "title": "Header Collision",
      "pos": [30, 30],
      "size": [120, 80]
    }
  ],
  "groups": [
    {
      "id": 1,
      "title": "Group A",
      "bounding": [0, 0, 220, 220]
    },
    {
      "id": 2,
      "title": "Group B",
      "bounding": [120, 20, 220, 220]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    report = module.analyze_workflow(workflow_path, require_all_nodes_in_groups=True)

    assert any("group overlap" in issue for issue in report["issues"])
    assert any("group header overlap" in issue for issue in report["issues"])
