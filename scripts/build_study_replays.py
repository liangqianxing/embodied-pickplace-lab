"""Re-execute first successes/failures and assert equality with frozen trial outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from pickplace.data import file_hash
from pickplace.study import episode, write_json
from pickplace.study_analysis import load_cube
from pickplace.study_policy import StudyPolicy


def build(output: Path):
    protocol, _, _, _, _ = load_cube(output)
    config = protocol["configuration"]
    torch.set_num_threads(1)
    inventory = []
    for seed in config["training_seeds"]:
        for spec in config["models"]:
            model = StudyPolicy.load(output / f"checkpoints/{spec['name']}-{seed}.pt")
            for condition in config["conditions"]:
                group = f"{spec['name']}-{seed}-{condition['name']}"
                data = json.loads((output / f"trials/{group}.json").read_text())
                for success in (True, False):
                    first = next((r for r in data["trials"] if r["success"] == success), None)
                    if first is None:
                        continue
                    replay = episode(model, condition, first["seed"], record=True)
                    for key, value in first.items():
                        if replay[key] != value:
                            raise ValueError(
                                f"Rerun differs at {group} scene {first['seed']} key {key}"
                            )
                    target = output / f"replays/{group}-{first['seed']}.json"
                    write_json(
                        target,
                        {
                            "model": spec["name"],
                            "training_seed": seed,
                            "condition_settings": condition,
                            **replay,
                        },
                    )
                    inventory.append(
                        {
                            "file": str(target.relative_to(output)),
                            "sha256": file_hash(target),
                            "model": spec["name"],
                            "training_seed": seed,
                            "condition": condition["name"],
                            "seed": first["seed"],
                            "success": success,
                            "frames": len(replay["frames"]),
                        }
                    )
    write_json(output / "replay_index.json", inventory)
    print(f"Verified and saved {len(inventory)} deterministic first-success/first-failure replays.")


if __name__ == "__main__":
    build(Path(__file__).resolve().parents[1] / "artifacts/study/v1")
