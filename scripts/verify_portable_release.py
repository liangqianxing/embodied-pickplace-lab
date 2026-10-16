"""Check the portable ZIP's internal hashes and science provenance without extracting."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path):
    prefix = "pickplace-lab-research-v1/"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP paths")
        for name in names:
            parsed = PurePosixPath(name)
            if parsed.is_absolute() or ".." in parsed.parts or not name.startswith(prefix):
                raise ValueError(f"Unsafe archive path: {name}")
        manifest = json.loads(archive.read(prefix + "artifacts/manifest.json"))
        for record in manifest["files"]:
            data = archive.read(prefix + record["path"])
            if len(data) != record["size"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"Manifest mismatch: {record['path']}")
        for protocol_path, source_key in (
            ("artifacts/study/v1/protocol.json", "source_sha256"),
            ("benchmarks/latency_guard/artifacts/v1/frozen.json", "source_sha256"),
            ("benchmarks/robosuite/artifacts/study-v1_1/frozen.json", "source_sha256"),
        ):
            if prefix + protocol_path not in names:
                continue
            protocol = json.loads(archive.read(prefix + protocol_path))
            parent = str(PurePosixPath(protocol_path).parent)
            for name, expected in protocol[source_key].items():
                snapshot = archive.read(prefix + parent + "/source_snapshot/" + name)
                if hashlib.sha256(snapshot).hexdigest() != expected:
                    raise ValueError(f"Portable frozen source differs: {name}")
        study_prefix = prefix + "artifacts/study/v1/"
        protocol = json.loads(archive.read(study_prefix + "protocol.json"))
        dataset = archive.read(prefix + "artifacts/data/expert.npz")
        if hashlib.sha256(dataset).hexdigest() != protocol["dataset_sha256"]:
            raise ValueError("Portable dataset differs")
        config = protocol["configuration"]
        for model in config["models"]:
            for seed in config["training_seeds"]:
                name = f"checkpoints/{model['name']}-{seed}"
                metadata = json.loads(archive.read(study_prefix + name + ".json"))
                checkpoint = archive.read(study_prefix + name + ".pt")
                if hashlib.sha256(checkpoint).hexdigest() != metadata["checkpoint_sha256"]:
                    raise ValueError(f"Portable checkpoint differs: {name}")
        panda_prefix = prefix + "benchmarks/robosuite/artifacts/study-v1_1/"
        if panda_prefix + "completed.json" in names:
            panda_protocol = json.loads(archive.read(panda_prefix + "frozen.json"))
            panda_metadata = json.loads(archive.read(panda_prefix + "data/expert.json"))
            panda_data = archive.read(panda_prefix + "data/expert.npz")
            if hashlib.sha256(panda_data).hexdigest() != panda_metadata["sha256"]:
                raise ValueError("Portable Panda dataset differs")
            for seed in panda_protocol["config"]["training_seeds"]:
                stem = panda_prefix + f"checkpoints/bc-{seed}"
                metadata = json.loads(archive.read(stem + ".json"))
                actual = hashlib.sha256(archive.read(stem + ".pt")).hexdigest()
                if actual != metadata["checkpoint_sha256"]:
                    raise ValueError(f"Portable Panda checkpoint differs: {seed}")
    print(
        f"Portable ZIP verified: {len(manifest['files'])} files, frozen source snapshots, dataset and 25 research weights."
    )


if __name__ == "__main__":
    verify(ROOT / "releases/pickplace-lab-research-v1.zip")
