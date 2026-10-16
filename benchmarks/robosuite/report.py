"""Audit complete Panda study and re-execute selected outcomes before reporting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from evaluate import episode
from learning import LiftPolicy, digest, save_json
from study import OUT, summarize

HERE = Path(__file__).resolve().parent


def verify():
    torch.set_num_threads(1)
    frozen = json.loads((OUT / "frozen.json").read_text())
    config = frozen["config"]
    frozen_hash = digest(OUT / "frozen.json")
    for name, expected in frozen["source_sha256"].items():
        if digest(OUT / "source_snapshot" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    dataset = OUT / "data/expert.npz"
    metadata = json.loads(dataset.with_suffix(".json").read_text())
    if metadata["sha256"] != digest(dataset):
        raise ValueError("Dataset hash mismatch")
    with np.load(dataset, allow_pickle=False) as archive:
        train_ids = set(archive["episode_ids"][~archive["validation"]].tolist())
        validation_ids = set(archive["episode_ids"][archive["validation"]].tolist())
        if (
            not np.isfinite(archive["observations"]).all()
            or not np.isfinite(archive["actions"]).all()
        ):
            raise ValueError("Nonfinite dataset")
    test_ids = set(
        range(
            config["test_scenario_start"], config["test_scenario_start"] + config["test_episodes"]
        )
    )
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise ValueError("Episode split overlap")
    if train_ids != set(range(92000, 92128)) or validation_ids != set(range(93000, 93016)):
        raise ValueError("Unexpected training/validation episodes")
    if len(metadata["episodes"]) != 144 or not all(r["held_8_steps"] for r in metadata["episodes"]):
        raise ValueError("Unsuccessful or incomplete demonstrations")
    for seed in config["training_seeds"]:
        path = OUT / f"checkpoints/bc-{seed}.pt"
        record = json.loads(path.with_suffix(".json").read_text())
        if record["checkpoint_sha256"] != digest(path) or record["dataset_sha256"] != digest(
            dataset
        ):
            raise ValueError("Checkpoint provenance mismatch")
        if record["best_val_loss"] != min(r["val_loss"] for r in record["history"]):
            raise ValueError("Checkpoint not selected by best validation loss")
    report = summarize(config, frozen_hash)
    replays = []
    for seed in config["training_seeds"]:
        model = LiftPolicy.load(OUT / f"checkpoints/bc-{seed}.pt")
        for condition in config["conditions"]:
            group = json.loads((OUT / f"trials/bc-{seed}-{condition['name']}.json").read_text())
            for success in (True, False):
                first = next((r for r in group["trials"] if r["held_8_steps"] == success), None)
                if first is None:
                    continue
                replay = episode(
                    model, first["seed"], condition["delay_steps"], condition["filter"], record=True
                )
                for key, value in first.items():
                    if replay[key] != value:
                        raise ValueError(
                            f"Replay differs: seed={seed}, scene={first['seed']}, {key}"
                        )
                path = OUT / f"replays/bc-{seed}-{condition['name']}-{first['seed']}.json"
                save_json(path, {"training_seed": seed, "condition": condition, **replay})
                replays.append(
                    {
                        "path": str(path.relative_to(OUT)),
                        "sha256": digest(path),
                        "training_seed": seed,
                        "condition": condition["name"],
                        "scenario_seed": first["seed"],
                        "held_8_steps": success,
                        "official_success": first["official_success"],
                    }
                )
    save_json(OUT / "replay_index.json", replays)
    lines = [
        "# Panda Lift: Local Standard-Task Replication",
        "",
        f"Frozen: {frozen['frozen_at']}",
        "",
        f"**{report['total_learned_trials']:,} learned-policy trials + {report['baseline_trials']} baseline trials.**",
        "",
        "Five optimizer seeds, one shared 128-episode training dataset, 16 held-out validation episodes, 50 new scene seeds paired across all execution conditions.",
        "",
        "| Condition | Official success mean +/- SD | Eight-step hold mean +/- SD | Mean held-episode steps |",
        "|---|---:|---:|---:|",
    ]
    for row in report["rows"]:
        steps = f"{row['mean_held_steps']:.1f}" if row["mean_held_steps"] is not None else "N/A"
        lines.append(
            f"| {row['condition']} | {row['official_mean']:.1%} +/- {row['official_sd']:.1%} | {row['hold_mean']:.1%} +/- {row['hold_sd']:.1%} | {steps} |"
        )
    low, high = report["primary_bootstrap_95"]
    lines += [
        "",
        "## Primary comparison",
        "",
        f"EMA minus raw at 50 ms, official success: {report['primary_mean_difference'] * 100:+.1f} pp; exploratory paired two-axis bootstrap [{low * 100:+.1f}, {high * 100:+.1f}] pp.",
        f"Supplementary eight-step hold difference: {report['hold_difference'] * 100:+.1f} pp.",
        "",
        "## Baselines",
        "",
    ]
    for baseline in report["baselines"]:
        lines.append(
            f"- {baseline['name']}: official {baseline['official_successes']}/{baseline['episodes']}; hold {baseline['hold_successes']}/{baseline['episodes']}."
        )
    lines += [
        "",
        "## Evidence and limitations",
        "",
        f"- {len(replays)} automatically selected first-success/first-failure hold traces re-executed with exact scalar outcome agreement.",
        "- Official success means the object crosses the upstream Lift height threshold at an observed control boundary. Hold is a separate stronger diagnostic, not the upstream benchmark definition.",
        "- Horizon is 200 controls, termination after eight successful observations, and learned rotational deltas are zero. This is not a leaderboard-equivalent protocol.",
        "- States and joint proprioception are observed; no images, language, hardware or real-time safety are evaluated.",
        "- Optimization/shuffling seeds vary while data stay fixed; uncertainty is not the same as the custom study that also resamples demonstration subsets.",
        "- Identical outcomes can make empirical bootstrap intervals degenerate. A 100% sample rate or [0,0] contrast interval is not proof of universal success or exact population equality.",
        "- The external library provides the task. This is our local replication, not independent third-party verification or external adoption.",
        "- A first attempt stopped on metadata serialization before completed data/model/test artifacts; its frozen directory is retained. Current protocol v1.1 fixes serialization without changing experimental settings.",
        "",
        "## Environment",
        "",
        f"`{frozen['versions']}`",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(
        f"Verified {report['total_learned_trials'] + report['baseline_trials']} trials, five models, disjoint datasets and {len(replays)} exact reruns.",
        flush=True,
    )
    return report


if __name__ == "__main__":
    verify()
