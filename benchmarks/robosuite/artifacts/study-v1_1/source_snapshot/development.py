"""Development-only scripted control audit for Panda Lift; no learning/test claims."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from smoke import make_env


def field(observation, suffix):
    matches = [name for name in observation if name.startswith("robot0") and name.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one robot field ending {suffix}, found {matches}")
    return np.asarray(observation[matches[0]])


class LiftExpert:
    """Approach, descend, close, lift; no access to hidden learner state."""

    def __init__(self):
        self.phase = "approach"
        self.phase_steps = 0

    def act(self, observation):
        eef = field(observation, "eef_pos")
        cube = np.asarray(observation["cube_pos"])
        if self.phase == "approach":
            target = cube + [0, 0, 0.10]
            if np.linalg.norm(eef - target) < 0.012:
                self.phase = "descend"
                self.phase_steps = 0
        if self.phase == "descend":
            target = cube + [0, 0, 0.005]
            if np.linalg.norm(eef - target) < 0.008:
                self.phase = "close"
                self.phase_steps = 0
        if self.phase == "close":
            target = cube + [0, 0, 0.005]
            if self.phase_steps >= 12:
                self.phase = "lift"
                self.phase_steps = 0
        if self.phase == "lift":
            target = np.r_[cube[:2], 1.05]
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip((target - eef) / 0.05, -0.5, 0.5)
        action[6] = 1 if self.phase in ("close", "lift") else -1
        self.phase_steps += 1
        return action


def run(seed):
    env = make_env(seed)
    expert = LiftExpert()
    observations = env.reset()
    phase_changes, previous_phase = [], None
    frames = []
    success, stable = False, 0
    started = time.perf_counter()
    try:
        for step in range(200):
            action = expert.act(observations)
            if expert.phase != previous_phase:
                phase_changes.append({"step": step, "phase": expert.phase})
                previous_phase = expert.phase
            observations, reward, done, _ = env.step(action)
            current_success = bool(env._check_success())
            success |= current_success
            stable = stable + 1 if current_success else 0
            frames.append(
                {
                    "step": step + 1,
                    "cube": observations["cube_pos"].tolist(),
                    "eef": field(observations, "eef_pos").tolist(),
                    "action": action.tolist(),
                    "reward": float(reward),
                    "official_success": current_success,
                }
            )
            if stable >= 8 or done:
                break
        return {
            "seed": seed,
            "ever_official_success": success,
            "held_8_steps": stable >= 8,
            "steps": len(frames),
            "phase_changes": phase_changes,
            "wall_seconds": time.perf_counter() - started,
            "frames": frames,
        }
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=90010)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts/development.json",
    )
    args = parser.parse_args()
    results = []
    for seed in range(args.seed_start, args.seed_start + args.episodes):
        result = run(seed)
        results.append(result)
        print(
            f"development seed {seed}: success={result['ever_official_success']} "
            f"hold={result['held_8_steps']} steps={result['steps']} "
            f"phases={result['phase_changes']}",
            flush=True,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "status": "development-only; not held-out learning evaluation",
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
