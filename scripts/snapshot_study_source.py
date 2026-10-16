"""Preserve exactly the code referenced by a study protocol for future reproduction."""

import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1]
study = root / "artifacts/study/v1"
protocol = json.loads((study / "protocol.json").read_text())
for name, expected in protocol["source_sha256"].items():
    source = root / name
    actual = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Source changed before snapshot: {name}")
    target = study / "source_snapshot" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Existing snapshot differs: {name}")
    shutil.copyfile(source, target)
print(f"Preserved {len(protocol['source_sha256'])} frozen source/config files.")
