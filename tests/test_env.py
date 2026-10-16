import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pickplace.env import PickPlaceEnv
from pickplace.expert import ScriptedExpert


def test_gymnasium_contract():
    env = PickPlaceEnv()
    check_env(env, skip_render_check=True)
    env.close()


def test_seeded_reset_and_step_are_deterministic():
    a, b = PickPlaceEnv(), PickPlaceEnv()
    oa, ia = a.reset(seed=123)
    ob, ib = b.reset(seed=123)
    np.testing.assert_array_equal(oa, ob)
    assert ia == ib
    rng = np.random.default_rng(8)
    for _ in range(20):
        action = rng.uniform(-1, 1, 4)
        ra, rb = a.step(action), b.step(action)
        np.testing.assert_array_equal(ra[0], rb[0])
        assert ra[1:] == rb[1:]
    a.close()
    b.close()


@pytest.mark.parametrize("action", [[0, 0], [0, 0, np.nan, 1], [np.inf, 0, 0, 0]])
def test_invalid_actions_rejected(action):
    env = PickPlaceEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)
    env.close()


def test_clipping_and_time_limit():
    env = PickPlaceEnv(max_steps=2)
    env.reset(seed=1)
    obs, _, terminated, truncated, _ = env.step([100, -100, 100, 100])
    assert env.observation_space.contains(obs)
    assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step([0, 0, 0, 1])
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step([0, 0, 0, 0])
    env.close()


@pytest.mark.parametrize("distribution", ["id", "ood"])
@pytest.mark.parametrize("seed", [3, 29, 70])
def test_contact_based_expert_completes(distribution, seed):
    env = PickPlaceEnv()
    env.reset(seed=seed, options={"distribution": distribution})
    expert = ScriptedExpert()
    bilateral_contact = False
    for _ in range(env.max_steps):
        _, _, done, truncated, info = env.step(expert.act(env))
        bilateral_contact |= info["contacts"] == 2
        if done or truncated:
            break
    assert info["success"] and info["lifted"]
    assert bilateral_contact
    assert env.model.neq == 0  # No equality/weld constraints attach the cube.
    assert expert.phase == 7
    env.close()


def test_ood_cube_starts_outside_training_rectangle():
    env = PickPlaceEnv()
    for seed in range(30):
        env.reset(seed=seed, options={"distribution": "ood"})
        assert abs(env.cube[0]) > 0.16 or abs(env.cube[1]) > 0.12
    env.close()


def test_placement_without_lifting_does_not_count():
    env = PickPlaceEnv(max_steps=15)
    env.reset(seed=4)
    env.goal[:2] = env.cube[:2]
    for _ in range(15):
        _, _, _, _, info = env.step([0, 0, 0, 1])
        assert not info["success"]
    env.close()


def test_invalid_distribution_and_render_mode():
    with pytest.raises(ValueError):
        PickPlaceEnv(render_mode="human")
    env = PickPlaceEnv()
    with pytest.raises(ValueError):
        env.reset(options={"distribution": "invalid"})
    env.close()
