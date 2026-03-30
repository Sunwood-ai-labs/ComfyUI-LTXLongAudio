import argparse
import json
import sys
from pathlib import Path


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


def analyze_workflow(path, *, title_padding=80.0, inner_padding=12.0, check_node_overlap=False, require_all_nodes_in_groups=False):
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
            }
        )

    issues = []

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

    return {
        "path": str(Path(path).resolve()),
        "node_count": len(nodes),
        "group_count": len(groups),
        "issues": issues,
        "node_group_matches": node_group_matches,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Detect overlapping groups and nodes in a ComfyUI workflow JSON.")
    parser.add_argument("workflow", nargs="+", help="Path to one or more ComfyUI workflow JSON files.")
    parser.add_argument("--title-padding", type=float, default=80.0, help="Reserved top area inside each group title band.")
    parser.add_argument("--inner-padding", type=float, default=12.0, help="Safe padding from each group border.")
    parser.add_argument("--check-node-overlap", action="store_true", help="Also detect node-node overlaps.")
    parser.add_argument(
        "--require-all-nodes-in-groups",
        action="store_true",
        help="Treat nodes outside every group as an error when the workflow uses groups.",
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
        )
        if report["issues"]:
            exit_code = 1
            print(f"FAIL {report['path']}")
            for issue in report["issues"]:
                print(f"  - {issue}")
        else:
            print(f"PASS {report['path']} (groups={report['group_count']}, nodes={report['node_count']})")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
