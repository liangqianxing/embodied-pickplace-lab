"""Controlled observation, latency, and dynamics shifts without changing task success."""

from __future__ import annotations

from collections import deque

import mujoco
import numpy as np
import torch

from .env import PickPlaceEnv

ABSOLUTE_XY = (0, 1, 3, 4, 6, 7)


def noisy_observation(
    observation: np.ndarray,
    rng: np.random.Generator,
    grip_std: float = 0,
    cube_std: float = 0,
    goal_std: float = 0,
) -> np.ndarray:
    result = observation.copy()
    for start, std in ((0, grip_std), (3, cube_std), (6, goal_std)):
        if std < 0:
            raise ValueError("Noise standard deviations must be nonnegative")
        # Draw even at zero std: every condition has a fixed innovation schedule.
        result[start : start + 3] += rng.normal(0, std, 3).astype(np.float32)
    if grip_std == cube_std == goal_std == 0:
        return result
    result[9:12] = result[3:6] - result[0:3]
    result[12:15] = result[6:9] - result[3:6]
    return result


def augment_observations(
    observations: torch.Tensor,
    generator: torch.Generator,
    grip_std: float,
    cube_std: float,
    goal_std: float,
) -> torch.Tensor:
    result = observations.clone()
    noise = torch.randn((len(result), 9), generator=generator, dtype=result.dtype)
    scales = result.new_tensor([grip_std] * 3 + [cube_std] * 3 + [goal_std] * 3)
    result[:, :9] += noise * scales
    result[:, 9:12] = result[:, 3:6] - result[:, 0:3]
    result[:, 12:15] = result[:, 6:9] - result[:, 3:6]
    return result


class DeploymentEnv:
    """Explicit deployment adapter; exposes only measured observations to learners."""

    def __init__(self, condition: dict):
        self.condition = dict(condition)
        self.env = PickPlaceEnv()
        mass_scale = float(condition.get("mass_scale", 1))
        friction_scale = float(condition.get("friction_scale", 1))
        delay = condition.get("delay_steps", 0)
        if not 0 < mass_scale <= 10 or not 0 < friction_scale <= 10:
            raise ValueError("Dynamics scale factors must be in (0, 10]")
        if not isinstance(delay, int) or not 0 <= delay <= 10:
            raise ValueError("delay_steps must be an integer in [0, 10]")
        body = self.env.model.body("cube").id
        self.env.model.body_mass[body] *= mass_scale
        self.env.model.body_inertia[body] *= mass_scale
        self.env.model.geom_friction[:, 0] *= friction_scale
        mujoco.mj_setConst(self.env.model, self.env.data)
        self.delay_steps = delay
        self.queue: deque[np.ndarray] = deque()
        self.rng = np.random.default_rng(0)

    def _measure(self, observation):
        return noisy_observation(
            observation,
            self.rng,
            **{key: self.condition.get(key, 0.0) for key in ("grip_std", "cube_std", "goal_std")},
        )

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed + 2_000_000)
        self.queue = deque(
            np.array([0, 0, 0, 1], dtype=np.float32) for _ in range(self.delay_steps)
        )
        observation, info = self.env.reset(
            seed=seed, options={"distribution": self.condition.get("distribution", "id")}
        )
        return self._measure(observation), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must be finite with shape (4,)")
        self.queue.append(action.copy())
        executed = self.queue.popleft()
        obs, reward, terminated, truncated, info = self.env.step(executed)
        return (
            self._measure(obs),
            reward,
            terminated,
            truncated,
            {**info, "applied_action": executed.tolist()},
        )

    def close(self):
        self.env.close()
