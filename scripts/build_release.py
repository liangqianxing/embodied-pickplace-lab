"""Build a portable snapshot locally; publishing is a separate explicit action."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from build_manifest import entries

ROOT = Path(__file__).resolve().parents[1]


def build(root: Path):
    target = root / "releases/pickplace-lab-research-v1.zip"
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = []
    prefix = "pickplace-lab-research-v1/"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in entries(root):
            if name.startswith("artifacts/runs/"):
                continue
            if name == "artifacts/relocation-verification.json":
                continue
            content = path.read_bytes()
            if path.suffix in (
                ".md",
                ".html",
                ".json",
                ".py",
                ".txt",
                ".sh",
                ".toml",
                ".js",
                ".css",
            ):
                # Only replace the explicit local checkout path, never arbitrary file content.
                content = content.replace(str(root).encode(), b"/path/to/embodied-pickplace-lab")
            archive.writestr(prefix + name, content)
            manifest.append(
                {"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            )
        archive.writestr(
            prefix + "artifacts/manifest.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "files": manifest,
                    "note": "Portable MIT project snapshot; see LICENSE and THIRD_PARTY_NOTICES.md. Local paths sanitized; environments and run logs omitted. No independent-replication or external-impact claim.",
                },
                indent=2,
            )
            + "\n",
        )
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC verification failed")
        for item in manifest:
            content = archive.read(prefix + item["path"])
            if hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise RuntimeError(f"ZIP hash mismatch: {item['path']}")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n")
    print(
        f"Verified portable release: {target} ({target.stat().st_size:,} bytes, {len(manifest)} files)"
    )
    return target


if __name__ == "__main__":
    build(ROOT)
