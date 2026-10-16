"""Create a portable integrity inventory, excluding runtime environments and itself."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache", "releases"}


def entries(root: Path):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in EXCLUDED_PARTS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if (
            relative.as_posix()
            in {"artifacts/manifest.json", "artifacts/runs/robosuite-install.log"}
            or path.name == ".DS_Store"
        ):
            continue
        yield relative.as_posix(), path


def build(root: Path):
    inventory = [
        {
            "path": name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in entries(root)
    ]
    result = {
        "schema_version": 1,
        "files": inventory,
        "note": "Local integrity evidence, not a digital signature or external attestation. Virtual environments, caches, releases and the active robosuite installation log are excluded.",
    }
    (root / "artifacts/manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Inventoried {len(inventory)} files ({sum(item['size'] for item in inventory):,} bytes)."
    )


if __name__ == "__main__":
    build(ROOT)
