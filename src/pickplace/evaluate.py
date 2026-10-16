"""Paired held-out trials, Wilson intervals, and real simulator replay traces."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .env import PickPlaceEnv
from .expert import ScriptedExpert
from .policy import Policy


def wilson_interval(successes: int, total: int) -> list[float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial counts")
    p, z = successes / total, 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def rollout(
    policy: Policy | str, seed: int, distribution: str = "id", *, record: bool = False
) -> dict:
    env = PickPlaceEnv()
    obs, _ = env.reset(seed=seed, options={"distribution": distribution})
    expert = ScriptedExpert() if isinstance(policy, str) and policy == "expert" else None
    rng = np.random.default_rng(seed + 1_000_000)
    frames = [env.snapshot()] if record else []
    if isinstance(policy, str) and policy not in ("expert", "random"):
        raise ValueError("Unknown policy")
    try:
        for _ in range(env.max_steps):
            if expert is not None:
                action = expert.act(env)
            elif isinstance(policy, str):
                action = rng.uniform(-1, 1, 4).astype(np.float32)
            else:
                action = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if record:
                frames.append(env.snapshot())
            if terminated or truncated:
                break
        failure = None
        if not info["success"]:
            if not info["lifted"]:
                failure = "no_lift"
            elif info["goal_distance"] >= 0.045:
                failure = "misplaced_or_dropped"
            else:
                failure = "release_or_settle"
        result = {
            "seed": seed,
            "distribution": distribution,
            **info,
            "simulated_seconds": env.steps * 0.05,
            "failure": failure,
        }
        if record:
            result["frames"] = frames
        return result
    finally:
        env.close()


def evaluate(
    policies: dict[str, Policy | str], output: Path, *, episodes: int = 100, seed_start: int = 20000
) -> dict:
    if not 1 <= episodes <= 1000:
        raise ValueError("episodes must be in [1, 1000]")
    torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    results, replays = [], []
    for name, policy in policies.items():
        for distribution, start in [("id", seed_start), ("ood", seed_start + 10000)]:
            trials = [
                rollout(policy, seed, distribution) for seed in range(start, start + episodes)
            ]
            successes = sum(trial["success"] for trial in trials)
            completed = [trial["steps"] for trial in trials if trial["success"]]
            reasons = {
                key: sum(trial["failure"] == key for trial in trials)
                for key in ("no_lift", "misplaced_or_dropped", "release_or_settle")
            }
            row = {
                "policy": name,
                "distribution": distribution,
                "episodes": episodes,
                "successes": successes,
                "success_rate": successes / episodes,
                "confidence_interval": wilson_interval(successes, episodes),
                "mean_steps_success": float(np.mean(completed)) if completed else None,
                "mean_goal_distance": float(np.mean([t["goal_distance"] for t in trials])),
                "failures": reasons,
                "trials": trials,
            }
            results.append(row)
            # First success and first failure are selected deterministically, not hand-picked.
            for success in (True, False):
                example = next((trial for trial in trials if trial["success"] == success), None)
                if example is not None:
                    replay = rollout(policy, example["seed"], distribution, record=True)
                    replay.update(
                        {"policy": name, "id": f"{name}-{distribution}-{example['seed']}"}
                    )
                    replays.append(replay)
            print(
                f"{name:12s} {distribution}: {successes}/{episodes} ({100 * successes / episodes:.1f}%)",
                flush=True,
            )
    report = {
        "schema_version": 1,
        "episodes_per_condition": episodes,
        "seed_start_id": seed_start,
        "seed_start_ood": seed_start + 10000,
        "confidence_method": "Wilson 95% (episode uncertainty, not training-seed uncertainty)",
        "results": results,
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "replays.json").write_text(json.dumps(replays, separators=(",", ":")) + "\n")
    return report
