"""Verify every recorded delivery file, rejecting paths escaping the project root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def verify(root: Path):
    root = root.resolve()
    manifest = json.loads((root / "artifacts/manifest.json").read_text())
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Path escapes project root: {entry['path']}")
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise ValueError(f"Missing/size mismatch: {entry['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Hash mismatch: {entry['path']}")
    print(f"Verified all {len(manifest['files'])} manifest entries.")


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
