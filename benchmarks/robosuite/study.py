"""Frozen, resumable Panda Lift study using separate, pinned package environment."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from evaluate import episode
from learning import LiftPolicy, collect, digest, save_json, train

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "artifacts/study-v1_1"


def freeze():
    config = json.loads((HERE / "protocol.json").read_text())
    names = [
        "smoke.py",
        "development.py",
        "learning.py",
        "evaluate.py",
        "study.py",
        "protocol.json",
        "requirements-lock.txt",
    ]
    source = {name: digest(HERE / name) for name in names}
    versions = {
        name: importlib.metadata.version(name)
        for name in ["robosuite", "mujoco", "numpy", "torch", "numba", "scipy"]
    }
    state = {
        "config": config,
        "source_sha256": source,
        "versions": versions,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if versions["robosuite"] != "1.5.2" or versions["mujoco"] != "3.3.7":
        raise RuntimeError("Use the verified robosuite/MuJoCo version pins")
    path = OUT / "frozen.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key, value in state.items():
            if previous[key] != value:
                raise RuntimeError(f"Frozen study mismatch: {key}")
    else:
        save_json(path, {**state, "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()})
        for name in names:
            target = OUT / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((HERE / name).read_bytes())
    return config, digest(path)


def bootstrap(values, repetitions=5000, seed=20261015):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        rows = rng.integers(0, values.shape[0], values.shape[0])
        columns = rng.integers(0, values.shape[1], values.shape[1])
        samples.append(float(values[np.ix_(rows, columns)].mean()))
    return np.quantile(samples, [0.025, 0.975]).tolist()


def summarize(config, frozen_hash):
    cube = np.empty(
        (len(config["training_seeds"]), len(config["conditions"]), config["test_episodes"])
    )
    hold = np.empty_like(cube)
    rows = []
    expected = list(
        range(
            config["test_scenario_start"], config["test_scenario_start"] + config["test_episodes"]
        )
    )
    for c, condition in enumerate(config["conditions"]):
        lengths = []
        for s, seed in enumerate(config["training_seeds"]):
            path = OUT / f"trials/bc-{seed}-{condition['name']}.json"
            group = json.loads(path.read_text())
            if (
                group["frozen_sha256"] != frozen_hash
                or group["condition"] != condition
                or group["training_seed"] != seed
                or [r["seed"] for r in group["trials"]] != expected
            ):
                raise ValueError(f"Incomplete/mismatched trials: {path}")
            cube[s, c] = [r["official_success"] for r in group["trials"]]
            hold[s, c] = [r["held_8_steps"] for r in group["trials"]]
            lengths.extend(r["steps"] for r in group["trials"] if r["held_8_steps"])
        rates = cube[:, c, :].mean(1)
        holds = hold[:, c, :].mean(1)
        rows.append(
            {
                "condition": condition["name"],
                "official_mean": float(rates.mean()),
                "official_sd": float(rates.std(ddof=1)),
                "official_seed_rates": rates.tolist(),
                "hold_mean": float(holds.mean()),
                "hold_sd": float(holds.std(ddof=1)),
                "hold_seed_rates": holds.tolist(),
                "official_bootstrap_95": bootstrap(cube[:, c, :]),
                "mean_held_steps": float(np.mean(lengths)) if lengths else None,
            }
        )
    names = [c["name"] for c in config["conditions"]]
    difference = cube[:, names.index("delay50_ema"), :] - cube[:, names.index("delay50"), :]
    hold_difference = hold[:, names.index("delay50_ema"), :] - hold[:, names.index("delay50"), :]
    baseline = []
    for name in config["baselines"]:
        group = json.loads((OUT / f"trials/{name}.json").read_text())
        if (
            group["frozen_sha256"] != frozen_hash
            or [r["seed"] for r in group["trials"]] != expected
        ):
            raise ValueError(f"Incomplete baseline {name}")
        baseline.append(
            {
                "name": name,
                "episodes": len(expected),
                "official_successes": sum(r["official_success"] for r in group["trials"]),
                "hold_successes": sum(r["held_8_steps"] for r in group["trials"]),
            }
        )
    report = {
        "frozen_sha256": frozen_hash,
        "total_learned_trials": int(cube.size),
        "baseline_trials": len(baseline) * len(expected),
        "rows": rows,
        "baselines": baseline,
        "primary_mean_difference": float(difference.mean()),
        "primary_bootstrap_95": bootstrap(difference),
        "hold_difference": float(hold_difference.mean()),
        "hold_bootstrap_95": bootstrap(hold_difference),
        "caution": "Five optimization seeds, fixed demonstration data. Exploratory interval, no multiplicity correction. Standard-task local replication, not independent external validation.",
    }
    save_json(OUT / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    torch.set_num_threads(1)
    config, frozen_hash = freeze()
    start = time.perf_counter()
    dataset = OUT / "data/expert.npz"
    if dataset.exists():
        metadata = json.loads(dataset.with_suffix(".json").read_text())
        if digest(dataset) != metadata["sha256"]:
            raise ValueError("Dataset hash mismatch")
    else:
        collect(
            dataset.parent, config["expert_train_episodes"], config["expert_validation_episodes"]
        )
    for seed in config["training_seeds"]:
        path = OUT / f"checkpoints/bc-{seed}.pt"
        if path.exists() and path.with_suffix(".json").exists():
            metadata = json.loads(path.with_suffix(".json").read_text())
            if metadata["checkpoint_sha256"] != digest(path) or metadata[
                "dataset_sha256"
            ] != digest(dataset):
                raise ValueError(f"Checkpoint mismatch: {path}")
        else:
            train(dataset, path, seed, config["epochs"])
    expected = list(
        range(
            config["test_scenario_start"], config["test_scenario_start"] + config["test_episodes"]
        )
    )
    # Complete all validation-selected training before any deployment test.
    for seed in config["training_seeds"]:
        checkpoint = OUT / f"checkpoints/bc-{seed}.pt"
        model = LiftPolicy.load(checkpoint)
        for condition in config["conditions"]:
            path = OUT / f"trials/bc-{seed}-{condition['name']}.json"
            records = []
            if path.exists():
                previous = json.loads(path.read_text())
                if previous["frozen_sha256"] != frozen_hash or previous[
                    "checkpoint_sha256"
                ] != digest(checkpoint):
                    raise ValueError(f"Trial provenance mismatch: {path}")
                records = previous["trials"]
                if [r["seed"] for r in records] != expected[: len(records)]:
                    raise ValueError(f"Trial sequence mismatch: {path}")
            for scene in expected[len(records) :]:
                records.append(episode(model, scene, condition["delay_steps"], condition["filter"]))
                save_json(
                    path,
                    {
                        "training_seed": seed,
                        "condition": condition,
                        "frozen_sha256": frozen_hash,
                        "checkpoint_sha256": digest(checkpoint),
                        "trials": records,
                        "complete": len(records) == len(expected),
                    },
                )
            print(
                f"EVAL seed={seed} {condition['name']} official={sum(r['official_success'] for r in records)}/{len(records)} hold={sum(r['held_8_steps'] for r in records)}/{len(records)}",
                flush=True,
            )
    for name in config["baselines"]:
        path = OUT / f"trials/{name}.json"
        records = []
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["frozen_sha256"] != frozen_hash:
                raise ValueError(f"Baseline provenance mismatch: {path}")
            records = previous["trials"]
            if [r["seed"] for r in records] != expected[: len(records)]:
                raise ValueError(f"Baseline sequence mismatch: {path}")
        for scene in expected[len(records) :]:
            records.append(episode(name, scene))
            save_json(
                path,
                {
                    "baseline": name,
                    "frozen_sha256": frozen_hash,
                    "trials": records,
                    "complete": len(records) == len(expected),
                },
            )
        print(
            f"BASELINE {name}: {sum(r['official_success'] for r in records)}/{len(records)}",
            flush=True,
        )
    summarize(config, frozen_hash)
    save_json(
        OUT / "completed.json",
        {
            "frozen_sha256": frozen_hash,
            "seconds_this_invocation": time.perf_counter() - start,
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        },
    )


if __name__ == "__main__":
    main()
