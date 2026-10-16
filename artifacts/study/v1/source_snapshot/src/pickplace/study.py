"""Frozen, resumable factorial study; no automatic hyperparameter search."""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import platform
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
from torch.nn import functional as F

from .data import file_hash, load_data
from .robustness import DeploymentEnv, augment_observations
from .study_policy import StudyPolicy


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes(root: Path) -> dict:
    paths = sorted((root / "src/pickplace").rglob("*.py"))
    paths += sorted((root / "src/pickplace/assets").rglob("*.xml"))
    paths += [root / "configs/research_v1.json", root / "docs/RESEARCH_PROTOCOL.md"]
    return {str(path.relative_to(root)): file_hash(path) for path in paths}


def select_data(dataset: Path, count: int, seed: int):
    data = load_data(dataset)
    ids = np.unique(data["episode_ids"][~data["validation"]])
    if not 1 <= count <= len(ids):
        raise ValueError("Invalid episode count")
    selected = np.sort(np.random.default_rng(seed + 100_000).choice(ids, count, replace=False))
    keep = data["validation"] | np.isin(data["episode_ids"], selected)
    return {key: value[keep] for key, value in data.items()}, selected.tolist()


def fit_variant(
    dataset: Path, output: Path, spec: dict, config: dict, seed: int, protocol_hash: str
):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    shuffle = torch.Generator().manual_seed(seed + 200_000)
    noise = torch.Generator().manual_seed(seed + 300_000)
    data, selected = select_data(dataset, spec["episodes"], seed)
    mask = data["validation"]
    x, y = (torch.from_numpy(data[key][~mask]) for key in ("observations", "actions"))
    xv, yv = (torch.from_numpy(data[key][mask]) for key in ("observations", "actions"))
    model = StudyPolicy(relative=spec["relative"])
    features = model.features(x)
    model.mean.copy_(features.mean(0))
    model.scale.copy_(features.std(0).clamp(min=0.02))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=0.0001
    )
    history, best_loss, best_epoch, best_state = [], float("inf"), None, None
    started = time.perf_counter()

    def objective(inputs, targets):
        motion, grip = model(inputs)
        return F.mse_loss(motion, targets[:, :3]) + 0.15 * F.binary_cross_entropy_with_logits(
            grip, (targets[:, 3] + 1) * 0.5
        )

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total = 0.0
        for batch in torch.randperm(len(x), generator=shuffle).split(config["batch_size"]):
            inputs = x[batch]
            if spec["augment"]:
                inputs = augment_observations(inputs, noise, **config["augmentation"])
            optimizer.zero_grad(set_to_none=True)
            loss = objective(inputs, y[batch])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(objective(xv, yv))
        history.append({"epoch": epoch, "train_loss": total / len(x), "val_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "width": model.width,
            "relative": model.relative,
            "state_dict": best_state,
        },
        output,
    )
    metadata = {
        "spec": spec,
        "seed": seed,
        "protocol_sha256": protocol_hash,
        "dataset_sha256": file_hash(dataset),
        "selected_episodes": selected,
        "validation_episode_ids": np.unique(data["episode_ids"][mask]).tolist(),
        "training_transitions": len(x),
        "validation_transitions": len(xv),
        "epochs": config["epochs"],
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "history": history,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.perf_counter() - started,
        "checkpoint_sha256": file_hash(output),
    }
    write_json(output.with_suffix(".json"), metadata)
    print(
        f"TRAIN {spec['name']} seed={seed} val={best_loss:.5f} ({metadata['seconds']:.1f}s)",
        flush=True,
    )
    return metadata


def episode(policy: StudyPolicy, condition: dict, seed: int, *, record: bool = False):
    env = DeploymentEnv(condition)
    observation, _ = env.reset(seed)
    frames = [env.env.snapshot()] if record else []
    max_height = float(env.env.cube[2])
    try:
        for _ in range(env.env.max_steps):
            action = policy.act(observation)
            observation, _, terminated, truncated, info = env.step(action)
            max_height = max(max_height, float(env.env.cube[2]))
            if record:
                frames.append(
                    {
                        **env.env.snapshot(),
                        "requested_action": action.tolist(),
                        "applied_action": info["applied_action"],
                    }
                )
            if terminated or truncated:
                break
        failure = (
            None
            if info["success"]
            else (
                "no_lift"
                if not info["lifted"]
                else (
                    "misplaced_or_dropped"
                    if info["goal_distance"] >= 0.045
                    else "release_or_settle"
                )
            )
        )
        result = {
            "seed": seed,
            "condition": condition["name"],
            "success": info["success"],
            "lifted": info["lifted"],
            "steps": info["steps"],
            "goal_distance": info["goal_distance"],
            "max_cube_height": max_height,
            "failure": failure,
        }
        if record:
            result["frames"] = frames
        return result
    finally:
        env.close()


def freeze(root: Path, output: Path, config_path: Path):
    config = json.loads(config_path.read_text())
    protocol = output / "protocol.json"
    current = {
        "configuration": config,
        "source_sha256": source_hashes(root),
        "dataset_sha256": file_hash(root / "artifacts/data/expert.npz"),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "mujoco", "gymnasium")
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if protocol.exists():
        existing = json.loads(protocol.read_text())
        for key, value in current.items():
            if existing.get(key) != value:
                raise RuntimeError(f"Frozen protocol mismatch ({key}); use a new output version")
    else:
        write_json(
            protocol, {**current, "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}
        )
    return config, file_hash(protocol)


def run(root: Path, output: Path, config_path: Path):
    config, protocol_hash = freeze(root, output, config_path)
    dataset = root / "artifacts/data/expert.npz"
    torch.set_num_threads(1)
    started = time.perf_counter()
    for seed in config["training_seeds"]:
        for spec in config["models"]:
            path = output / f"checkpoints/{spec['name']}-{seed}.pt"
            if path.exists() and path.with_suffix(".json").exists():
                meta = json.loads(path.with_suffix(".json").read_text())
                if meta["protocol_sha256"] != protocol_hash or meta[
                    "checkpoint_sha256"
                ] != file_hash(path):
                    raise RuntimeError(f"Checkpoint provenance mismatch: {path}")
            else:
                fit_variant(dataset, path, spec, config, seed, protocol_hash)
    # All model selection is complete before the first deployment-test episode.
    for seed in config["training_seeds"]:
        for spec in config["models"]:
            model = StudyPolicy.load(output / f"checkpoints/{spec['name']}-{seed}.pt")
            for condition in config["conditions"]:
                path = output / f"trials/{spec['name']}-{seed}-{condition['name']}.json"
                if path.exists():
                    previous = json.loads(path.read_text())
                    expected_seeds = list(
                        range(
                            config["test_seed_start"],
                            config["test_seed_start"] + config["test_episodes"],
                        )
                    )
                    if (
                        previous["protocol_sha256"] != protocol_hash
                        or [r["seed"] for r in previous["trials"]] != expected_seeds
                    ):
                        raise RuntimeError(f"Trial provenance mismatch: {path}")
                    continue
                trials = [
                    episode(model, condition, scenario)
                    for scenario in range(
                        config["test_seed_start"],
                        config["test_seed_start"] + config["test_episodes"],
                    )
                ]
                write_json(
                    path,
                    {
                        "model": spec["name"],
                        "training_seed": seed,
                        "condition": condition,
                        "protocol_sha256": protocol_hash,
                        "trials": trials,
                    },
                )
                print(
                    f"TEST {spec['name']:18s} seed={seed} {condition['name']:12s} "
                    f"{sum(r['success'] for r in trials)}/{len(trials)}",
                    flush=True,
                )
    write_json(
        output / "completed.json",
        {
            "protocol_sha256": protocol_hash,
            "completed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "wall_seconds_this_invocation": time.perf_counter() - started,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/study/v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/research_v1.json"))
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.config.resolve())


if __name__ == "__main__":
    main()
