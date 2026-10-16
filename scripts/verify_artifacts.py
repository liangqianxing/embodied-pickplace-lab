"""Validate actual records and hashes, not just file existence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pickplace.data import file_hash, load_data
from pickplace.evaluate import wilson_interval


def verify(root: Path):
    dataset = root / "artifacts/data/expert.npz"
    meta = json.loads(dataset.with_suffix(".json").read_text())
    data = load_data(dataset)
    assert file_hash(dataset) == meta["sha256"], "Dataset hash changed"
    assert len(data["actions"]) == meta["transitions"]
    train_ids = set(data["episode_ids"][~data["validation"]].tolist())
    val_ids = set(data["episode_ids"][data["validation"]].tolist())
    assert len(train_ids) == meta["train_episodes"]
    assert len(val_ids) == meta["validation_episodes"]
    assert not train_ids & val_ids
    for name in ("bc32", "bc256", "dagger"):
        checkpoint = root / f"artifacts/checkpoints/{name}.pt"
        model_meta = json.loads(checkpoint.with_suffix(".json").read_text())
        assert file_hash(checkpoint) == model_meta["checkpoint_sha256"]
        assert model_meta["dataset_sha256"] == meta["sha256"]
        assert model_meta["best_val_loss"] == min(row["val_loss"] for row in model_meta["history"])
    evaluation = json.loads((root / "artifacts/results/evaluation.json").read_text())
    replay_records = json.loads((root / "artifacts/results/replays.json").read_text())
    report = json.loads((root / "artifacts/report.json").read_text())
    assert report["evaluation"] == evaluation
    test_ids = set()
    rows = {}
    total = 0
    for row in evaluation["results"]:
        trials = row["trials"]
        assert row["episodes"] == len(trials) == evaluation["episodes_per_condition"]
        ids = [trial["seed"] for trial in trials]
        start = evaluation[f"seed_start_{row['distribution']}"]
        assert ids == list(range(start, start + len(trials)))
        successes = sum(trial["success"] for trial in trials)
        assert successes == row["successes"]
        assert row["success_rate"] == successes / len(trials)
        np.testing.assert_allclose(
            row["confidence_interval"], wilson_interval(successes, len(trials))
        )
        assert sum(row["failures"].values()) + successes == len(trials)
        assert all(bool(trial["failure"] is None) == trial["success"] for trial in trials)
        test_ids.update(ids)
        rows[(row["policy"], row["distribution"])] = row
        total += len(trials)
    assert not test_ids & (train_ids | val_ids)
    for replay in replay_records:
        row = rows[(replay["policy"], replay["distribution"])]
        first = next(trial for trial in row["trials"] if trial["success"] == replay["success"])
        assert replay["seed"] == first["seed"], "Replay selection is not first-in-order"
        assert len(replay["frames"]) == replay["steps"] + 1
        assert replay["steps"] == first["steps"]
        assert replay["success"] == first["success"]
        assert all(len(frame["qpos"]) == 12 for frame in replay["frames"])
        if replay["success"]:
            assert any(frame["cube"][2] > 0.10 for frame in replay["frames"])
    media = json.loads((root / "artifacts/media/demo.json").read_text())
    assert any(
        replay["id"] == media["replay_id"] and replay["success"] == media["success"]
        for replay in replay_records
    )
    print(
        f"Verified {len(train_ids)} training + {len(val_ids)} validation episodes, "
        f"3 checkpoints, {total} evaluation trials and {len(replay_records)} replay records."
    )


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
