#!/usr/bin/env python3
"""Regenerate root-level layout fixtures from source definitions.

The generated .canvas files are disposable outputs. If a human edits them in
Obsidian, rerun this script to restore the canonical fixtures from
fixture-definitions.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    fixture_dir = Path(__file__).resolve().parent
    repo_root = fixture_dir.parent

    definitions_path = fixture_dir / "fixture-definitions.json"
    blank_canvas_path = repo_root / "examples" / "blank.canvas"
    canvas_tool_path = repo_root / "canvas-tool.py"

    if not definitions_path.is_file():
        raise SystemExit(f"Missing definitions file: {definitions_path}")
    if not blank_canvas_path.is_file():
        raise SystemExit(f"Missing blank canvas template: {blank_canvas_path}")
    if not canvas_tool_path.is_file():
        raise SystemExit(f"Missing canvas tool: {canvas_tool_path}")

    with definitions_path.open("r", encoding="utf-8") as handle:
        definitions = json.load(handle)

    if not isinstance(definitions, dict) or not definitions:
        raise SystemExit("fixture-definitions.json must contain a non-empty object")

    generated = []
    for filename, spec in definitions.items():
        if not filename.endswith(".canvas"):
            raise SystemExit(f"Fixture key must end with .canvas: {filename}")
        if not isinstance(spec, dict):
            raise SystemExit(f"Fixture spec must be an object: {filename}")

        payload = {
            "groups": spec.get("groups", []),
            "tasks": spec.get("tasks", []),
        }

        target_path = fixture_dir / filename
        shutil.copyfile(blank_canvas_path, target_path)

        subprocess.run(
            [sys.executable, str(canvas_tool_path), str(target_path), "batch"],
            input=json.dumps(payload).encode("utf-8"),
            cwd=repo_root,
            check=True,
        )
        generated.append(target_path.name)

    print("Regenerated fixtures:")
    for name in generated:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
