"""Executable integration gate for the official Panda Lift environment, not a result claim."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config


def make_env(seed: int):
    controller = load_composite_controller_config(robot="Panda")
    return suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller,
        use_camera_obs=False,
        use_object_obs=True,
        has_renderer=False,
        has_offscreen_renderer=False,
        control_freq=20,
        horizon=200,
        ignore_done=False,
        hard_reset=False,
        seed=seed,
    )


def inspect(seed: int):
    env = make_env(seed)
    try:
        observation = env.reset()
        initial = {key: np.asarray(value).tolist() for key, value in observation.items()}
        low, high = env.action_spec
        action = np.zeros_like(low)
        action[-1] = -1.0
        for _ in range(5):
            observation, reward, _done, _info = env.step(action)
            assert all(np.isfinite(np.asarray(value)).all() for value in observation.values())
            assert np.isfinite(reward)
        return {
            "initial_observation": initial,
            "observation_shapes": {
                key: list(np.asarray(value).shape) for key, value in observation.items()
            },
            "action_low": low.tolist(),
            "action_high": high.tolist(),
            "success_after_five_idle_steps": bool(env._check_success()),
            "controller": env.robots[0].composite_controller_config,
        }
    finally:
        env.close()


def main():
    a = inspect(90000)
    b = inspect(90000)
    for key in a["initial_observation"]:
        np.testing.assert_array_equal(a["initial_observation"][key], b["initial_observation"][key])
    result = {
        "status": "integration smoke passed; no trained-policy benchmark yet",
        "versions": {
            name: importlib.metadata.version(name) for name in ("robosuite", "mujoco", "numpy")
        },
        "seed": 90000,
        "repeatable_reset": True,
        **a,
    }
    output = Path(__file__).resolve().parent / "artifacts/smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(output)
    print(json.dumps(result["observation_shapes"], indent=2))


if __name__ == "__main__":
    main()
