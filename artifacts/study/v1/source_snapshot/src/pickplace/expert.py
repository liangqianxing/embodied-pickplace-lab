"""Finite-state geometric expert used only to label demonstrations."""

import numpy as np

from .env import PickPlaceEnv


class ScriptedExpert:
    phases = ("approach", "descend", "grasp", "lift", "carry", "lower", "release", "retreat")

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = 0

    def act(self, env: PickPlaceEnv) -> np.ndarray:
        grip, cube, goal = env.grip, env.cube, env.goal
        p = self.phase
        if p == 0 and np.linalg.norm(grip[:2] - cube[:2]) < 0.008 and abs(grip[2] - 0.20) < 0.01:
            p = 1
        if p == 1 and grip[2] < 0.109:
            p = 2
        if p == 2 and env.contacts() == 2 and env.opening < 0.055:
            p = 3
        if p == 3 and grip[2] > 0.245:
            p = 4
        if p == 4 and np.linalg.norm(grip[:2] - goal[:2]) < 0.008:
            p = 5
        if p == 5 and grip[2] < 0.110:
            p = 6
        if p == 6 and env.opening > 0.095:
            p = 7
        self.phase = p
        if p == 0:
            target = np.r_[cube[:2], 0.20]
        elif p in (1, 2):
            target = np.r_[cube[:2], 0.103]
        elif p == 3:
            target = np.r_[cube[:2], 0.26]
        elif p == 4:
            target = np.r_[goal[:2], 0.26]
        elif p in (5, 6):
            target = np.r_[goal[:2], 0.103]
        else:
            target = np.r_[goal[:2], 0.26]
        action = np.r_[
            np.clip((target - grip) / env.action_scale, -1, 1), -1.0 if p in (2, 3, 4, 5) else 1.0
        ]
        return action.astype(np.float32)
