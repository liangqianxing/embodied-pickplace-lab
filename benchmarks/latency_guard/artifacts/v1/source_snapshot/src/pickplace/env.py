"""MuJoCo Cartesian gripper task; the cube is never attached or teleported."""

from __future__ import annotations

from importlib.resources import files
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class PickPlaceEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["rgb_array"], "render_fps": 20}
    action_scale = 0.025
    observation_size = 26

    def __init__(self, *, max_steps: int = 220, render_mode: str | None = None):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Only rgb_array rendering is supported")
        self.model = mujoco.MjModel.from_xml_path(str(files("pickplace") / "assets/scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (26,), dtype=np.float32)
        self._renderer = None
        self._cube_body = self.model.body("cube").id
        self._cube_geom = self.model.geom("cube_geom").id
        self._pad_geoms = {self.model.geom("left_pad").id, self.model.geom("right_pad").id}
        self.goal = np.zeros(3)
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.steps = 0
        self.lifted = False
        self.stable_steps = 0
        self._done = False

    @property
    def grip(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    @property
    def cube(self) -> np.ndarray:
        return self.data.xpos[self._cube_body].copy()

    @property
    def opening(self) -> float:
        return float(self.data.qpos[3:5].sum())

    def contacts(self) -> int:
        pads = set()
        for contact in self.data.contact:
            pair = {int(contact.geom1), int(contact.geom2)}
            if self._cube_geom in pair:
                pads.update(pair & self._pad_geoms)
        return len(pads)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        distribution = options.get("distribution", "id")
        if distribution not in ("id", "ood"):
            raise ValueError("distribution must be id or ood")
        mujoco.mj_resetData(self.model, self.data)
        bounds = np.array([0.16, 0.12] if distribution == "id" else [0.23, 0.18])
        obj = self.np_random.uniform(-bounds, bounds)
        # OOD starts outside the training rectangle, rather than overlapping it.
        if distribution == "ood":
            axis = int(self.np_random.integers(2))
            obj[axis] = self.np_random.choice([-1, 1]) * self.np_random.uniform(
                [0.18, 0.14][axis], bounds[axis]
            )
        for _ in range(1000):
            target = self.np_random.uniform(-bounds, bounds)
            if np.linalg.norm(target - obj) >= 0.16:
                break
        else:
            raise RuntimeError("Could not sample a separated object and goal")
        self.goal = np.r_[target, 0.025]
        self.model.site("goal").pos[:] = [*target, 0.001]
        self.data.qpos[:5] = [0, -0.18, 0.28, 0.055, 0.055]
        self.data.qpos[5:12] = [*obj, 0.025, 1, 0, 0, 0]
        self.data.ctrl[:] = self.data.qpos[:5]
        self.previous_action[:] = 0
        self.steps = self.stable_steps = 0
        self.lifted = self._done = False
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)
        return self._observation(), self._info()

    def _observation(self) -> np.ndarray:
        grip, cube = self.grip, self.cube
        return np.concatenate(
            [
                grip,
                cube,
                self.goal,
                cube - grip,
                self.goal - cube,
                [self.opening],
                self.data.qvel[:3],
                self.data.qvel[5:8],
                self.previous_action,
            ]
        ).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "success": self.stable_steps >= 8,
            "lifted": self.lifted,
            "goal_distance": float(np.linalg.norm(self.cube[:2] - self.goal[:2])),
            "contacts": self.contacts(),
            "steps": self.steps,
        }

    def step(self, action):
        if self._done:
            raise RuntimeError("Episode is done; call reset before step")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("action must be a finite vector of shape (4,)")
        action = np.clip(action, -1, 1)
        target = np.clip(
            self.grip + self.action_scale * action[:3], [-0.27, -0.23, 0.103], [0.27, 0.23, 0.40]
        )
        opening = 0.008 + (float(action[3]) + 1) * 0.5 * (0.055 - 0.008)
        self.data.ctrl[:] = [*target, opening, opening]
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self.previous_action = action.copy()
        cube = self.cube
        self.lifted = bool(self.lifted or cube[2] > 0.10)
        distance = np.linalg.norm(cube[:2] - self.goal[:2])
        placed = (
            self.lifted
            and distance < 0.045
            and abs(cube[2] - 0.025) < 0.012
            and np.linalg.norm(self.data.qvel[5:8]) < 0.08
            and self.grip[2] > 0.18
            and self.opening > 0.075
        )
        self.stable_steps = self.stable_steps + 1 if placed else 0
        success = self.stable_steps >= 8
        failure = abs(cube[0]) > 0.37 or abs(cube[1]) > 0.31 or cube[2] < -0.02
        terminated = bool(success or failure)
        truncated = bool(self.steps >= self.max_steps and not terminated)
        self._done = terminated or truncated
        reward = float(success) - 0.1 * float(distance)
        return self._observation(), reward, terminated, truncated, self._info()

    def snapshot(self) -> dict:
        return {
            "grip": self.grip.tolist(),
            "cube": self.cube.tolist(),
            "goal": self.goal.tolist(),
            "opening": self.opening,
            "qpos": self.data.qpos.tolist(),
            "contacts": self.contacts(),
            "lifted": self.lifted,
            "time": round(self.steps * 0.05, 3),
        }

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("Initialize with render_mode='rgb_array'")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="overview")
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
