"""Official task success plus separately labeled eight-step hold diagnostic."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from development import LiftExpert, field
from learning import LiftPolicy, save_json, vector
from smoke import make_env

HERE = Path(__file__).resolve().parent


def episode(model, seed: int, delay_steps: int = 0, filter_mode: str = "identity", record=False):
    if delay_steps not in (0, 1, 2):
        raise ValueError("Only 0, 1 and 2 step delays are defined")
    if filter_mode not in ("identity", "ema_half"):
        raise ValueError("Unknown execution filter")
    env = make_env(seed)
    raw = env.reset()
    previous = np.zeros(7, dtype=np.float32)
    filtered_xyz = np.zeros(3, dtype=np.float32)
    idle = np.zeros(7, dtype=np.float32)
    idle[-1] = -1
    queue = deque(idle.copy() for _ in range(delay_steps))
    expert = LiftExpert() if isinstance(model, str) and model == "expert" else None
    rng = np.random.default_rng(seed + 1_000_000)
    ever = False
    stable = 0
    first_success = None
    peak = float(raw["cube_pos"][2])
    frames = []
    try:
        for step in range(200):
            if expert is not None:
                requested = expert.act(raw)
            elif isinstance(model, str) and model == "random":
                requested = rng.uniform(-1, 1, 7).astype(np.float32)
            elif isinstance(model, str):
                raise ValueError("Unknown baseline")
            else:
                requested = model.act(vector(raw, previous))
            filtered = requested.copy()
            if filter_mode == "ema_half":
                filtered[:3] = 0.5 * requested[:3] + 0.5 * filtered_xyz
            filtered_xyz = filtered[:3].copy()
            queue.append(filtered)
            executed = queue.popleft()
            raw, _, done, _ = env.step(executed)
            previous = executed.copy()
            success = bool(env._check_success())
            ever |= success
            if success and first_success is None:
                first_success = step + 1
            stable = stable + 1 if success else 0
            peak = max(peak, float(raw["cube_pos"][2]))
            if record:
                frames.append(
                    {
                        "step": step + 1,
                        "qpos": env.sim.data.qpos.tolist(),
                        "cube": raw["cube_pos"].tolist(),
                        "eef": field(raw, "eef_pos").tolist(),
                        "requested_action": requested.tolist(),
                        "filtered_action": filtered.tolist(),
                        "applied_action": executed.tolist(),
                        "official_success": success,
                    }
                )
            if stable >= 8 or done:
                break
        result = {
            "seed": seed,
            "delay_steps": delay_steps,
            "filter": filter_mode,
            "official_success": ever,
            "held_8_steps": stable >= 8,
            "steps": step + 1,
            "first_success_step": first_success,
            "max_cube_height": peak,
            "cube_size": env.cube.size.tolist(),
        }
        if record:
            result["frames"] = frames
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline", choices=["expert", "random"])
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--delay-steps", type=int, default=0)
    parser.add_argument("--filter", choices=["identity", "ema_half"], default="identity")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.checkpoint is None) == (args.baseline is None):
        parser.error("Choose exactly one checkpoint or baseline")
    torch.set_num_threads(1)
    model = LiftPolicy.load(args.checkpoint) if args.checkpoint else args.baseline
    trials = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        trial = episode(model, seed, args.delay_steps, args.filter)
        trials.append(trial)
        save_json(
            args.output,
            {
                "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "baseline": args.baseline,
                "requested_episodes": args.episodes,
                "status": "complete" if len(trials) == args.episodes else "partial",
                "trials": trials,
            },
        )
        print(
            f"seed={seed} official={trial['official_success']} hold={trial['held_8_steps']} steps={trial['steps']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
