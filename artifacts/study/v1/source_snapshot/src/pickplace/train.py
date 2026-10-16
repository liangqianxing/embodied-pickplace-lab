"""Deterministic CPU training with validation-loss checkpoint selection."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import file_hash, load_data
from .env import PickPlaceEnv
from .expert import ScriptedExpert
from .policy import Policy


def fit(
    data: dict,
    output: Path,
    *,
    epochs: int = 100,
    seed: int = 42,
    model: Policy | None = None,
    metadata: dict | None = None,
) -> tuple[Policy, dict]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = model or Policy()
    mask = data["validation"]
    x = torch.from_numpy(data["observations"][~mask])
    y = torch.from_numpy(data["actions"][~mask])
    xv = torch.from_numpy(data["observations"][mask])
    yv = torch.from_numpy(data["actions"][mask])
    if not len(x) or not len(xv):
        raise ValueError("Both training and validation data are required")
    model.mean.copy_(x.mean(0))
    model.scale.copy_(x.std(0).clamp(min=0.02))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    history, best_loss, best_epoch, best_state = [], float("inf"), 0, None
    start = time.perf_counter()

    def loss_fn(xb, yb):
        movement, gripper = model(xb)
        return F.mse_loss(movement, yb[:, :3]) + 0.15 * F.binary_cross_entropy_with_logits(
            gripper, (yb[:, 3] + 1) * 0.5
        )

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(x))
        total = 0.0
        for indices in permutation.split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(x[indices], y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += loss.item() * len(indices)
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            val_loss = loss_fn(xv, yv).item()
        history.append({"epoch": epoch, "train_loss": total / len(x), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"{output.stem}: epoch {epoch}/{epochs} train={total / len(x):.5f} val={val_loss:.5f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "width": model.width, "state_dict": best_state}, output)
    report = {
        "seed": seed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "training_transitions": len(x),
        "validation_transitions": len(xv),
        "seconds": time.perf_counter() - start,
        "checkpoint_sha256": file_hash(output),
        "history": history,
        **(metadata or {}),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return model, report


def train(
    dataset: Path, output: Path, *, episodes: int = 256, epochs: int = 100, seed: int = 42
) -> dict:
    data = load_data(dataset, episodes)
    _, report = fit(
        data,
        output,
        epochs=epochs,
        seed=seed,
        metadata={
            "method": "bc",
            "expert_episodes": episodes,
            "dataset_sha256": file_hash(dataset),
        },
    )
    return report


def dagger(
    dataset: Path,
    initial: Path,
    output: Path,
    *,
    rounds: int = 3,
    episodes_per_round: int = 64,
    epochs: int = 60,
    seed: int = 42,
) -> dict:
    if not 1 <= rounds <= 3 or not 1 <= episodes_per_round <= 1000:
        raise ValueError("Use 1-3 rounds and 1-1000 episodes per round (reserved seed ranges)")
    data = load_data(dataset)
    model = Policy.load(initial)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    summaries = []
    env = PickPlaceEnv()
    try:
        for round_index in range(rounds):
            observations, actions, ids, phases = [], [], [], []
            beta = 0.5 ** (round_index + 1)
            successes = 0
            for episode in range(episodes_per_round):
                episode_seed = 2000 + round_index * 1000 + episode
                obs, _ = env.reset(seed=episode_seed)
                oracle = ScriptedExpert()
                for _ in range(env.max_steps):
                    label = oracle.act(env)
                    observations.append(obs)
                    actions.append(label)
                    ids.append(episode_seed)
                    phases.append(oracle.phase)
                    executed = label if rng.random() < beta else model.act(obs)
                    obs, _, terminated, truncated, info = env.step(executed)
                    if terminated or truncated:
                        break
                successes += int(info["success"])
            extra = {
                "observations": np.asarray(observations, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.float32),
                "episode_ids": np.asarray(ids),
                "phases": np.asarray(phases),
                "validation": np.zeros(len(ids), dtype=bool),
            }
            data = {key: np.concatenate([value, extra[key]]) for key, value in data.items()}
            print(
                f"DAgger round {round_index + 1}: mixed-policy success {successes}/{episodes_per_round}, {len(ids)} labels",
                flush=True,
            )
            model, report = fit(
                data,
                output,
                epochs=epochs,
                seed=seed + round_index,
                model=model,
                metadata={
                    "method": "dagger",
                    "round": round_index + 1,
                    "expert_episodes": len(
                        np.unique(data["episode_ids"][data["episode_ids"] < 1000])
                    ),
                    "dataset_sha256": file_hash(dataset),
                    "beta": beta,
                },
            )
            summaries.append(
                {
                    "round": round_index + 1,
                    "beta": beta,
                    "labels": len(ids),
                    "mixed_policy_successes": successes,
                    "episodes": episodes_per_round,
                    "training": report,
                }
            )
    finally:
        env.close()
    output.with_name(output.stem + "_dagger.json").write_text(
        json.dumps(summaries, indent=2) + "\n"
    )
    np.savez_compressed(output.with_name(output.stem + "_aggregated.npz"), **data)
    return report
