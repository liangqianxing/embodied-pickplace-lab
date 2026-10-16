"""Seeded expert trajectories with explicit episode-level train/validation splits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .env import PickPlaceEnv
from .expert import ScriptedExpert


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(path: Path, episodes: int = 256, validation_episodes: int = 32) -> dict:
    if not 1 <= episodes <= 1000 or not 1 <= validation_episodes <= 1000:
        raise ValueError("Episode counts must be between 1 and 1000 (reserved seed ranges)")
    path.parent.mkdir(parents=True, exist_ok=True)
    observations, actions, episode_ids, phases, validation = [], [], [], [], []
    episode_stats = []
    env = PickPlaceEnv()
    try:
        for is_val, count, start_seed in [(False, episodes, 0), (True, validation_episodes, 10000)]:
            for seed in range(start_seed, start_seed + count):
                obs, _ = env.reset(seed=seed)
                expert = ScriptedExpert()
                for _ in range(env.max_steps):
                    action = expert.act(env)
                    observations.append(obs)
                    actions.append(action)
                    episode_ids.append(seed)
                    phases.append(expert.phase)
                    validation.append(is_val)
                    obs, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                episode_stats.append({"seed": seed, "validation": is_val, **info})
                if not info["success"]:
                    raise RuntimeError(f"Expert failed at seed {seed}; refusing silent filtering")
    finally:
        env.close()
    np.savez_compressed(
        path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        episode_ids=np.asarray(episode_ids),
        phases=np.asarray(phases),
        validation=np.asarray(validation, dtype=bool),
    )
    meta = {
        "schema_version": 1,
        "train_episodes": episodes,
        "validation_episodes": validation_episodes,
        "transitions": len(actions),
        "sha256": file_hash(path),
        "episodes": episode_stats,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Collected {len(actions)} transitions from {len(episode_stats)} episodes", flush=True)
    return meta


def load_data(path: Path, train_episodes: int | None = None) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if train_episodes is not None:
        ids = np.unique(data["episode_ids"][~data["validation"]])
        if not 1 <= train_episodes <= len(ids):
            raise ValueError(f"train_episodes must be in [1, {len(ids)}]")
        keep = data["validation"] | np.isin(data["episode_ids"], ids[:train_episodes])
        data = {key: value[keep] for key, value in data.items()}
    if set(data["episode_ids"][data["validation"]]) & set(data["episode_ids"][~data["validation"]]):
        raise ValueError("Train/validation episode leakage")
    return data
