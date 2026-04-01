from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_ROOT = REPO_ROOT / "cli"


def _collect_import_targets(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            targets.append(node.module or "")
    return targets


def test_cli_modules_do_not_import_comfy_runtime() -> None:
    cli_files = sorted(CLI_ROOT.glob("*.py"))
    assert cli_files, "expected CLI modules to exist"

    offenders: dict[str, list[str]] = {}
    for path in cli_files:
        bad_targets = [target for target in _collect_import_targets(path) if "comfy" in target.lower()]
        if bad_targets:
            offenders[str(path.relative_to(REPO_ROOT))] = bad_targets

    assert offenders == {}
