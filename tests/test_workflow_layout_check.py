import importlib.util
import json
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
        require_all_nodes_in_groups=True,
        require_app_mode=True,
    )

    assert report["issues"] == []
    assert report["group_count"] == 3
    assert report["node_count"] == 17
    assert report["node_group_matches"]["Frames Folder"] == ["Inputs"]
    assert report["node_group_matches"]["Source Audio Duration"] == ["Inputs"]
    assert report["node_group_matches"]["Still Frame To Video Batch"] == ["Validation Logic"]
    assert report["node_group_matches"]["Video Combine (Smoke Test)"] == ["Output"]
    assert report["app_mode"]["enabled"] is True
    assert report["app_mode"]["selected_inputs"] == [
        [1, "directory"],
        [2, "audio"],
        [4, "value"],
        [5, "value"],
    ]
    assert report["app_mode"]["selected_outputs"] == [17]


def test_sample_workflow_defaults_are_populated():
    workflow_path = Path(__file__).resolve().parents[1] / "samples" / "workflows" / "LTXLongAudio_CustomNodes_SmokeTest.json"
    data = json.loads(workflow_path.read_text(encoding="utf-8"))

    frames_node = next(node for node in data["nodes"] if node["id"] == 1)
    audio_node = next(node for node in data["nodes"] if node["id"] == 2)

    assert frames_node["type"] == "LTXLoadImages"
    assert frames_node["widgets_values"][0] == "samples/input/frames_pool"
    assert audio_node["type"] == "LoadAudio"
    assert audio_node["widgets_values"][0] == "HOWL AT THE HAIRPIN2.wav"


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
    },
    {
      "id": 2,
      "type": "LTXIntConstant",
      "title": "Node Collision",
      "pos": [70, 60],
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
    assert any("node overlap" in issue for issue in report["issues"])


def test_layout_checker_reports_invalid_app_mode(tmp_path):
    module = _load_layout_module()
    workflow_path = tmp_path / "broken_app_workflow.json"
    workflow_path.write_text(
        """
{
  "nodes": [
    {
      "id": 1,
      "type": "LTXIntConstant",
      "title": "Segment Seconds",
      "pos": [0, 100],
      "size": [220, 86],
      "inputs": [
        {
          "name": "value",
          "type": "INT",
          "widget": {
            "name": "value"
          },
          "link": null
        }
      ]
    },
    {
      "id": 2,
      "type": "LTXBatchAnything",
      "title": "Not An Output Node",
      "pos": [320, 100],
      "size": [220, 86],
      "inputs": []
    }
  ],
  "groups": [
    {
      "id": 1,
      "title": "Group A",
      "bounding": [-20, 0, 620, 260]
    }
  ],
  "extra": {
    "linearMode": true,
    "linearData": {
      "inputs": [[999, "value"], [1, "missing_widget"]],
      "outputs": [2]
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    report = module.analyze_workflow(
        workflow_path,
        require_all_nodes_in_groups=True,
        require_app_mode=True,
    )

    assert any("app mode missing input node" in issue for issue in report["issues"])
    assert any("app mode missing widget" in issue for issue in report["issues"])
    assert any("app mode invalid output node" in issue for issue in report["issues"])


def test_layout_checker_reports_runtime_contract_errors(tmp_path):
    module = _load_layout_module()
    workflow_path = tmp_path / "broken_runtime_workflow.json"
    workflow_path.write_text(
        """
{
  "last_link_id": 1,
  "nodes": [
    {
      "id": 1,
      "type": "LTXIntConstant",
      "title": "Segment Index",
      "pos": [0, 0],
      "size": [220, 86],
      "inputs": [
        {
          "name": "value",
          "type": "INT",
          "widget": {
            "name": "value"
          },
          "link": null
        }
      ],
      "outputs": [
        {
          "name": "value",
          "type": "INT",
          "links": [1]
        }
      ],
      "widgets_values": [0]
    },
    {
      "id": 2,
      "type": "LTXAudioConcatenate",
      "title": "Audio Concatenate",
      "pos": [320, 0],
      "size": [220, 86],
      "inputs": [
        {
          "name": "audio1",
          "type": "AUDIO",
          "link": null
        },
        {
          "name": "audio2",
          "type": "AUDIO",
          "link": null
        },
        {
          "name": "direction",
          "type": "COMBO",
          "link": 1
        }
      ],
      "widgets_values": ["sideways"]
    },
    {
      "id": 3,
      "type": "LTXCompare",
      "title": "Compare",
      "pos": [640, 0],
      "size": [220, 86],
      "inputs": [
        {
          "name": "a",
          "type": "*",
          "link": 1
        },
        {
          "name": "b",
          "type": "*",
          "link": null
        },
        {
          "name": "comparison",
          "type": "COMBO",
          "link": null
        }
      ],
      "widgets_values": ["a > b"]
    }
  ],
  "links": [
    [1, 1, 0, 2, 2, "INT"]
  ],
  "groups": [
    {
      "id": 1,
      "title": "Group A",
      "bounding": [-20, -20, 980, 220]
    }
  ]
}
""".strip(),
        encoding="utf-8",
    )

    report = module.analyze_workflow(workflow_path, require_all_nodes_in_groups=True)

    assert any("invalid combo value" in issue for issue in report["issues"])
    assert any("linked combo input" in issue for issue in report["issues"])
    assert any("missing required input" in issue for issue in report["issues"])
