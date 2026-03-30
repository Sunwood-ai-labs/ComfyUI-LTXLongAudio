import argparse
import importlib.util
import json
import sys
from pathlib import Path

INVALID_APP_MODE_NODE_TYPES = {"Note", "MarkdownNote"}


def rect_from_xywh(values):
    x, y, width, height = values
    return float(x), float(y), float(x + width), float(y + height)


def rects_overlap(left, right):
    return left[0] < right[2] and left[2] > right[0] and left[1] < right[3] and left[3] > right[1]


def rect_inside(inner, outer, *, left_padding, top_padding, right_padding, bottom_padding):
    return (
        inner[0] >= outer[0] + left_padding
        and inner[1] >= outer[1] + top_padding
        and inner[2] <= outer[2] - right_padding
        and inner[3] <= outer[3] - bottom_padding
    )


def group_header_rect(group_rect, title_padding):
    return group_rect[0], group_rect[1], group_rect[2], min(group_rect[1] + float(title_padding), group_rect[3])


def describe_titles(items):
    return ", ".join(f"'{item}'" for item in items)


def load_local_node_registry():
    module_path = Path(__file__).resolve().parents[1] / "nodes.py"
    if not module_path.exists():
        return {}

    spec = importlib.util.spec_from_file_location("comfyui_ltx_long_audio.nodes", module_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        return {}
    spec.loader.exec_module(module)
    return getattr(module, "NODE_CLASS_MAPPINGS", {})


def analyze_app_mode(data, nodes, *, node_registry=None, require_app_mode=False):
    issues = []
    extra = data.get("extra") or {}
    linear_mode = extra.get("linearMode")
    linear_data = extra.get("linearData")

    has_app_mode_metadata = "linearMode" in extra or "linearData" in extra
    if require_app_mode and not has_app_mode_metadata:
        issues.append("app mode missing: expected extra.linearMode and extra.linearData")
        return {"enabled": False, "issues": issues, "selected_inputs": [], "selected_outputs": []}

    if not has_app_mode_metadata:
        return {"enabled": False, "issues": issues, "selected_inputs": [], "selected_outputs": []}

    if "linearMode" in extra and not isinstance(linear_mode, bool):
        issues.append("app mode invalid: extra.linearMode must be a boolean")
    if linear_data is None or not isinstance(linear_data, dict):
        issues.append("app mode invalid: extra.linearData must be an object")
        return {"enabled": bool(linear_mode) if isinstance(linear_mode, bool) else False, "issues": issues, "selected_inputs": [], "selected_outputs": []}

    inputs = linear_data.get("inputs", [])
    outputs = linear_data.get("outputs", [])
    if not isinstance(inputs, list):
        issues.append("app mode invalid: extra.linearData.inputs must be a list")
        inputs = []
    if not isinstance(outputs, list):
        issues.append("app mode invalid: extra.linearData.outputs must be a list")
        outputs = []

    node_by_id = {node["id"]: node for node in nodes}
    seen_inputs = set()
    seen_outputs = set()

    for entry in inputs:
        if not isinstance(entry, list) or len(entry) != 2:
            issues.append(f"app mode invalid input entry: expected [nodeId, widgetName], got {entry!r}")
            continue
        node_id, widget_name = entry
        node = node_by_id.get(node_id)
        if node is None:
            issues.append(f"app mode missing input node: node {node_id!r} does not exist")
            continue
        if node.get("type") in INVALID_APP_MODE_NODE_TYPES:
            issues.append(f"app mode invalid input node: '{node.get('title') or node.get('type')}' cannot be used as an input")
        available_widget_names = {
            input_def.get("name")
            for input_def in node.get("inputs", [])
            if isinstance(input_def, dict) and input_def.get("name")
        }
        if widget_name not in available_widget_names:
            issues.append(
                f"app mode missing widget: '{node.get('title') or node.get('type')}' does not expose widget '{widget_name}'"
            )
        if (node_id, widget_name) in seen_inputs:
            issues.append(f"app mode duplicate input: node {node_id!r} widget '{widget_name}' is listed more than once")
        seen_inputs.add((node_id, widget_name))

    resolved_registry = node_registry if node_registry is not None else {}
    for node_id in outputs:
        node = node_by_id.get(node_id)
        if node is None:
            issues.append(f"app mode missing output node: node {node_id!r} does not exist")
            continue
        if node.get("type") in INVALID_APP_MODE_NODE_TYPES:
            issues.append(f"app mode invalid output node: '{node.get('title') or node.get('type')}' cannot be used as an output")
        node_class = resolved_registry.get(node.get("type"))
        if node_class is not None and not bool(getattr(node_class, "OUTPUT_NODE", False)):
            issues.append(
                f"app mode invalid output node: '{node.get('title') or node.get('type')}' is not marked as OUTPUT_NODE"
            )
        if node_id in seen_outputs:
            issues.append(f"app mode duplicate output: node {node_id!r} is listed more than once")
        seen_outputs.add(node_id)

    if linear_mode is True and not outputs:
        issues.append("app mode invalid: extra.linearMode=true requires at least one selected output")

    return {
        "enabled": bool(linear_mode) if isinstance(linear_mode, bool) else False,
        "issues": issues,
        "selected_inputs": inputs,
        "selected_outputs": outputs,
    }


def analyze_workflow(
    path,
    *,
    title_padding=80.0,
    inner_padding=12.0,
    check_node_overlap=True,
    require_all_nodes_in_groups=False,
    require_app_mode=False,
):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    groups = []
    for group in data.get("groups", []):
        groups.append(
            {
                "id": group.get("id"),
                "title": group.get("title") or f"group-{group.get('id', '?')}",
                "rect": rect_from_xywh(group["bounding"]),
            }
        )

    nodes = []
    for node in data.get("nodes", []):
        rect = rect_from_xywh([node["pos"][0], node["pos"][1], node["size"][0], node["size"][1]])
        nodes.append(
            {
                "id": node.get("id"),
                "title": node.get("title") or node.get("type") or f"node-{node.get('id', '?')}",
                "type": node.get("type", ""),
                "rect": rect,
                "inputs": node.get("inputs", []),
                "outputs": node.get("outputs", []),
            }
        )

    issues = []
    node_registry = load_local_node_registry()

    for index, left in enumerate(groups):
        for right in groups[index + 1:]:
            if rects_overlap(left["rect"], right["rect"]):
                issues.append(f"group overlap: '{left['title']}' overlaps '{right['title']}'")

    node_group_matches = {}
    for node in nodes:
        safe_groups = []
        touched_groups = []
        header_groups = []
        for group in groups:
            if rects_overlap(node["rect"], group["rect"]):
                touched_groups.append(group["title"])
            if rects_overlap(node["rect"], group_header_rect(group["rect"], title_padding)):
                header_groups.append(group["title"])
            if rect_inside(
                node["rect"],
                group["rect"],
                left_padding=float(inner_padding),
                top_padding=max(float(inner_padding), float(title_padding)),
                right_padding=float(inner_padding),
                bottom_padding=float(inner_padding),
            ):
                safe_groups.append(group["title"])

        node_group_matches[node["title"]] = safe_groups

        if len(safe_groups) > 1:
            issues.append(f"node in multiple groups: '{node['title']}' is inside {describe_titles(safe_groups)}")
            continue

        if safe_groups:
            continue

        if header_groups:
            issues.append(f"group header overlap: '{node['title']}' overlaps the title area of {describe_titles(header_groups)}")
            continue

        if touched_groups:
            issues.append(f"group frame overlap: '{node['title']}' touches {describe_titles(touched_groups)} but is not safely inside")
            continue

        if require_all_nodes_in_groups and groups:
            issues.append(f"ungrouped node: '{node['title']}' is not inside any group")

    if check_node_overlap:
        for index, left in enumerate(nodes):
            for right in nodes[index + 1:]:
                if rects_overlap(left["rect"], right["rect"]):
                    issues.append(f"node overlap: '{left['title']}' overlaps '{right['title']}'")

    app_mode_report = analyze_app_mode(
        data,
        nodes,
        node_registry=node_registry,
        require_app_mode=require_app_mode,
    )
    issues.extend(app_mode_report["issues"])

    return {
        "path": str(Path(path).resolve()),
        "node_count": len(nodes),
        "group_count": len(groups),
        "issues": issues,
        "node_group_matches": node_group_matches,
        "app_mode": app_mode_report,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Detect overlapping groups and nodes in a ComfyUI workflow JSON.")
    parser.add_argument("workflow", nargs="+", help="Path to one or more ComfyUI workflow JSON files.")
    parser.add_argument("--title-padding", type=float, default=80.0, help="Reserved top area inside each group title band.")
    parser.add_argument("--inner-padding", type=float, default=12.0, help="Safe padding from each group border.")
    overlap_group = parser.add_mutually_exclusive_group()
    overlap_group.add_argument(
        "--check-node-overlap",
        dest="check_node_overlap",
        action="store_true",
        help="Detect node-node overlaps. Enabled by default.",
    )
    overlap_group.add_argument(
        "--no-check-node-overlap",
        dest="check_node_overlap",
        action="store_false",
        help="Skip node-node overlap detection.",
    )
    parser.set_defaults(check_node_overlap=True)
    parser.add_argument(
        "--require-all-nodes-in-groups",
        action="store_true",
        help="Treat nodes outside every group as an error when the workflow uses groups.",
    )
    parser.add_argument(
        "--require-app-mode",
        action="store_true",
        help="Fail when the workflow does not include valid App mode metadata.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    exit_code = 0
    for workflow in args.workflow:
        report = analyze_workflow(
            workflow,
            title_padding=args.title_padding,
            inner_padding=args.inner_padding,
            check_node_overlap=args.check_node_overlap,
            require_all_nodes_in_groups=args.require_all_nodes_in_groups,
            require_app_mode=args.require_app_mode,
        )
        if report["issues"]:
            exit_code = 1
            print(f"FAIL {report['path']}")
            for issue in report["issues"]:
                print(f"  - {issue}")
        else:
            app_mode_suffix = ""
            if report["app_mode"]["enabled"] or args.require_app_mode:
                app_mode_suffix = (
                    f", app_inputs={len(report['app_mode']['selected_inputs'])},"
                    f" app_outputs={len(report['app_mode']['selected_outputs'])}"
                )
            print(
                f"PASS {report['path']} (groups={report['group_count']}, nodes={report['node_count']}{app_mode_suffix})"
            )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
