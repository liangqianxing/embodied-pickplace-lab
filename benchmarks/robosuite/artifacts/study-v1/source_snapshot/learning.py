"""State-based Panda Lift cloning; isolated from the custom-task experiment package."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from development import LiftExpert, field
from smoke import make_env
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def vector(observation, previous_action):
    eef = field(observation, "eef_pos")
    cube = np.asarray(observation["cube_pos"])
    return np.concatenate(
        [
            eef,
            cube,
            cube - eef,
            observation["cube_quat"],
            field(observation, "eef_quat"),
            field(observation, "gripper_qpos"),
            field(observation, "gripper_qvel"),
            field(observation, "joint_pos_cos"),
            field(observation, "joint_pos_sin"),
            field(observation, "joint_vel"),
            previous_action,
        ]
    ).astype(np.float32)


class LiftPolicy(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.register_buffer("mean", torch.zeros(input_size))
        self.register_buffer("scale", torch.ones(input_size))
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, observations):
        output = self.network((observations - self.mean) / self.scale)
        return 0.5 * torch.tanh(output[:, :3]), output[:, 3]

    @torch.inference_mode()
    def act(self, observation):
        motion, grip = self(torch.from_numpy(observation).unsqueeze(0))
        action = np.zeros(7, dtype=np.float32)
        action[:3] = motion[0].numpy()
        action[-1] = 1.0 if grip[0].item() >= 0 else -1.0
        return action

    @classmethod
    def load(cls, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(checkpoint["input_size"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model


def collect(output: Path, episodes=128, validation_episodes=16):
    if not 1 <= episodes <= 500 or not 1 <= validation_episodes <= 500:
        raise ValueError("Episode counts must be in [1, 500]")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    observations, actions, ids, val_mask = [], [], [], []
    for validation, count, start in ((False, episodes, 92000), (True, validation_episodes, 93000)):
        for seed in range(start, start + count):
            env = make_env(seed)
            oracle = LiftExpert()
            raw = env.reset()
            previous = np.zeros(7, dtype=np.float32)
            stable = 0
            local_obs, local_actions = [], []
            try:
                for step in range(200):
                    obs = vector(raw, previous)
                    action = oracle.act(raw)
                    local_obs.append(obs)
                    local_actions.append(action)
                    raw, _, done, _ = env.step(action)
                    previous = action.copy()
                    success = bool(env._check_success())
                    stable = stable + 1 if success else 0
                    if stable >= 8 or done:
                        break
                record = {
                    "seed": seed,
                    "validation": validation,
                    "held_8_steps": stable >= 8,
                    "steps": step + 1,
                    "cube_size": env.cube.size.tolist(),
                }
                records.append(record)
                if stable < 8:
                    save_json(
                        output / "collection_failed.json", {"failed": record, "attempted": records}
                    )
                    raise RuntimeError(f"Expert failed at {seed}; no silent trajectory filtering")
                observations.extend(local_obs)
                actions.extend(local_actions)
                ids.extend([seed] * len(local_actions))
                val_mask.extend([validation] * len(local_actions))
                if len(records) % 16 == 0:
                    print(f"COLLECT {len(records)}/{episodes + validation_episodes}", flush=True)
            finally:
                env.close()
    path = output / "expert.npz"
    np.savez_compressed(
        path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        episode_ids=np.asarray(ids),
        validation=np.asarray(val_mask, dtype=bool),
    )
    save_json(
        output / "expert.json",
        {
            "episodes": records,
            "sha256": digest(path),
            "transitions": len(actions),
            "input_size": len(observations[0]),
            "train_episodes": episodes,
            "validation_episodes": validation_episodes,
        },
    )
    return path


def train(dataset: Path, output: Path, seed=17, epochs=150):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    with np.load(dataset, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    mask = data["validation"]
    train_ids = set(data["episode_ids"][~mask].tolist())
    val_ids = set(data["episode_ids"][mask].tolist())
    if train_ids & val_ids:
        raise ValueError("Train/validation episode leakage")
    x, xv = (torch.from_numpy(data["observations"][m]) for m in (~mask, mask))
    y, yv = (torch.from_numpy(data["actions"][m]) for m in (~mask, mask))
    model = LiftPolicy(x.shape[1])
    model.mean.copy_(x.mean(0))
    model.scale.copy_(x.std(0).clamp(min=0.02))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0001)
    best, best_epoch, best_state = float("inf"), 0, None
    history = []
    started = time.perf_counter()

    def objective(inputs, targets):
        motion, grip = model(inputs)
        return F.mse_loss(motion, targets[:, :3]) + 0.15 * F.binary_cross_entropy_with_logits(
            grip, (targets[:, -1] + 1) * 0.5
        )

    for epoch in range(1, epochs + 1):
        total = 0.0
        model.train()
        for indices in torch.randperm(len(x)).split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = objective(x[indices], y[indices])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += loss.item() * len(indices)
        scheduler.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(objective(xv, yv))
        history.append({"epoch": epoch, "train_loss": total / len(x), "val_loss": val_loss})
        if val_loss < best:
            best, best_epoch, best_state = val_loss, epoch, copy.deepcopy(model.state_dict())
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_size": model.input_size, "state_dict": best_state}, output)
    save_json(
        output.with_suffix(".json"),
        {
            "seed": seed,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best,
            "history": history,
            "dataset_sha256": digest(dataset),
            "checkpoint_sha256": digest(output),
            "train_ids": sorted(train_ids),
            "validation_ids": sorted(val_ids),
            "seconds": time.perf_counter() - started,
        },
    )
    print(f"TRAIN seed={seed}, val={best:.5f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--validation-episodes", type=int, default=16)
    p.add_argument("--output", type=Path, default=HERE / "artifacts/data")
    p = sub.add_parser("train")
    p.add_argument("--data", type=Path, default=HERE / "artifacts/data/expert.npz")
    p.add_argument("--output", type=Path, default=HERE / "artifacts/checkpoints/bc-17.pt")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.output, args.episodes, args.validation_episodes)
    else:
        train(args.data, args.output, args.seed, args.epochs)


if __name__ == "__main__":
    main()
