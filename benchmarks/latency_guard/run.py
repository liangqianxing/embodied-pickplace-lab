"""Frozen-controller latency intervention on newly reserved evaluation scenarios."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from pickplace.data import file_hash
from pickplace.robustness import DeploymentEnv
from pickplace.study import write_json
from pickplace.study_analysis import two_axis_bootstrap
from pickplace.study_policy import StudyPolicy

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


class ExecutionFilter:
    def __init__(self, mode: str):
        if mode not in ("identity", "gain_half", "ema_half"):
            raise ValueError("Unknown filter mode")
        self.mode = mode
        self.previous = np.zeros(3, dtype=np.float32)

    def apply(self, action):
        result = np.asarray(action, dtype=np.float32).copy()
        if result.shape != (4,) or not np.isfinite(result).all():
            raise ValueError("Expected finite four-dimensional action")
        if self.mode == "gain_half":
            result[:3] *= 0.5
        elif self.mode == "ema_half":
            result[:3] = 0.5 * result[:3] + 0.5 * self.previous
        self.previous = result[:3].copy()
        return result


def rollout(model, condition, seed, variant, record=False):
    env = DeploymentEnv(condition)
    command_filter = ExecutionFilter(variant)
    obs, _ = env.reset(seed)
    frames = [env.env.snapshot()] if record else []
    try:
        for _ in range(env.env.max_steps):
            raw = model.act(obs)
            command = command_filter.apply(raw)
            obs, _, terminated, truncated, info = env.step(command)
            if record:
                frames.append(
                    {
                        **env.env.snapshot(),
                        "raw_action": raw.tolist(),
                        "filtered_action": command.tolist(),
                        "applied_action": info["applied_action"],
                    }
                )
            if terminated or truncated:
                break
        result = {
            "seed": seed,
            "success": info["success"],
            "steps": info["steps"],
            "goal_distance": info["goal_distance"],
            "lifted": info["lifted"],
        }
        if record:
            result["frames"] = frames
        return result
    finally:
        env.close()


def freeze(output):
    config_path = HERE / "protocol.json"
    config = json.loads(config_path.read_text())
    names = [
        "src/pickplace/env.py",
        "src/pickplace/assets/scene.xml",
        "src/pickplace/policy.py",
        "src/pickplace/study_policy.py",
        "src/pickplace/robustness.py",
        "src/pickplace/study_analysis.py",
        "src/pickplace/study.py",
        "benchmarks/latency_guard/run.py",
        "benchmarks/latency_guard/protocol.json",
    ]
    source = {name: file_hash(ROOT / name) for name in names}
    checkpoints = {
        str(seed): file_hash(
            ROOT / config["base_checkpoint_directory"] / f"{config['base_model']}-{seed}.pt"
        )
        for seed in config["training_seeds"]
    }
    state = {
        "config": config,
        "source_sha256": source,
        "checkpoint_sha256": checkpoints,
        "base_protocol_sha256": file_hash(ROOT / "artifacts/study/v1/protocol.json"),
    }
    path = output / "frozen.json"
    if path.exists():
        saved = json.loads(path.read_text())
        for key, value in state.items():
            if saved[key] != value:
                raise RuntimeError(f"Frozen follow-up mismatch: {key}")
    else:
        write_json(
            path, {**state, "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
        )
        for name in names:
            target = output / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
    return config, file_hash(path)


def run(output):
    config, frozen_hash = freeze(output)
    torch.set_num_threads(1)
    start = time.perf_counter()
    for seed in config["training_seeds"]:
        model = StudyPolicy.load(
            ROOT / config["base_checkpoint_directory"] / f"{config['base_model']}-{seed}.pt"
        )
        for variant in config["variants"]:
            for condition in config["conditions"]:
                path = output / f"trials/{variant}-{seed}-{condition['name']}.json"
                expected_seeds = list(
                    range(config["scenario_start"], config["scenario_start"] + config["episodes"])
                )
                if path.exists():
                    prior = json.loads(path.read_text())
                    if (
                        prior["frozen_sha256"] != frozen_hash
                        or [r["seed"] for r in prior["trials"]] != expected_seeds
                    ):
                        raise RuntimeError(f"Trial provenance mismatch: {path}")
                    continue
                trials = [rollout(model, condition, scene, variant) for scene in expected_seeds]
                write_json(
                    path,
                    {
                        "variant": variant,
                        "training_seed": seed,
                        "condition": condition,
                        "frozen_sha256": frozen_hash,
                        "trials": trials,
                    },
                )
                print(
                    f"{variant:10s} seed={seed} {condition['name']:12s} "
                    f"{sum(r['success'] for r in trials)}/{len(trials)}",
                    flush=True,
                )
    write_json(
        output / "completed.json",
        {
            "seconds_this_invocation": time.perf_counter() - start,
            "frozen_sha256": frozen_hash,
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        },
    )
    summarize(output)


def summarize(output):
    frozen = json.loads((output / "frozen.json").read_text())
    config = frozen["config"]
    shape = (
        len(config["variants"]),
        len(config["training_seeds"]),
        len(config["conditions"]),
        config["episodes"],
    )
    cube = np.empty(shape)
    rows = []
    for v, variant in enumerate(config["variants"]):
        for c, condition in enumerate(config["conditions"]):
            success_lengths = []
            all_lengths = []
            for s, seed in enumerate(config["training_seeds"]):
                path = output / f"trials/{variant}-{seed}-{condition['name']}.json"
                group = json.loads(path.read_text())
                if group["frozen_sha256"] != file_hash(output / "frozen.json"):
                    raise ValueError(f"Hash mismatch: {path}")
                trials = group["trials"]
                if [r["seed"] for r in trials] != list(
                    range(config["scenario_start"], config["scenario_start"] + config["episodes"])
                ):
                    raise ValueError(f"Scenario mismatch: {path}")
                cube[v, s, c] = [float(r["success"]) for r in trials]
                success_lengths.extend(r["steps"] for r in trials if r["success"])
                all_lengths.extend(r["steps"] for r in trials)
            rates = cube[v, :, c, :].mean(1)
            rows.append(
                {
                    "variant": variant,
                    "condition": condition["name"],
                    "mean": float(rates.mean()),
                    "sd": float(rates.std(ddof=1)),
                    "seed_rates": rates.tolist(),
                    "mean_successful_steps": float(np.mean(success_lengths))
                    if success_lengths
                    else None,
                    "mean_all_steps": float(np.mean(all_lengths)),
                    "bootstrap_95": two_axis_bootstrap(cube[v, :, c, :]),
                }
            )
    diff = (
        cube[config["variants"].index("ema_half"), :, 1, :]
        - cube[config["variants"].index("identity"), :, 1, :]
    )
    clean_diff = (
        cube[config["variants"].index("ema_half"), :, 0, :]
        - cube[config["variants"].index("identity"), :, 0, :]
    )
    result = {
        "frozen_sha256": file_hash(output / "frozen.json"),
        "total_trials": int(cube.size),
        "rows": rows,
        "primary_mean_difference": float(diff.mean()),
        "primary_bootstrap_95": two_axis_bootstrap(diff),
        "clean_mean_difference": float(clean_diff.mean()),
        "clean_bootstrap_95": two_axis_bootstrap(clean_diff),
        "caution": "Five training/data-subset seeds; exploratory interval. Post-v1 model-family selection. No algorithm novelty or hardware claim.",
    }
    write_json(output / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "artifacts/v1")
    args = parser.parse_args()
    run(args.output.resolve())
