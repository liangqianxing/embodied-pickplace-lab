from pathlib import Path

import numpy as np
import pytest
import torch

from pickplace.env import PickPlaceEnv
from pickplace.robustness import DeploymentEnv, augment_observations, noisy_observation
from pickplace.study import fit_variant, select_data
from pickplace.study_analysis import two_axis_bootstrap
from pickplace.study_policy import StudyPolicy


def test_clean_adapter_matches_base_environment():
    a, b = PickPlaceEnv(), DeploymentEnv({"name": "id"})
    try:
        x, _ = a.reset(seed=71001)
        y, _ = b.reset(71001)
        np.testing.assert_array_equal(x, y)
        for action in ([0.3, 0.2, -0.4, 1], [0, 0, 0, -1]):
            x = a.step(action)[0]
            y = b.step(action)[0]
            np.testing.assert_array_equal(x, y)
    finally:
        a.close()
        b.close()


def test_consistent_noise_channels_and_no_input_mutation():
    env = PickPlaceEnv()
    obs, _ = env.reset(seed=71002)
    original = obs.copy()
    first = noisy_observation(obs, np.random.default_rng(8), 0.001, 0.003, 0.001)
    second = noisy_observation(obs, np.random.default_rng(8), 0.001, 0.003, 0.001)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(obs, original)
    np.testing.assert_allclose(first[9:12], first[3:6] - first[:3])
    np.testing.assert_allclose(first[12:15], first[6:9] - first[3:6])
    np.testing.assert_array_equal(first[15:], obs[15:])
    env.close()


def test_training_noise_consistency():
    x = torch.zeros(20, 26)
    result = augment_observations(x, torch.Generator().manual_seed(3), 0.001, 0.003, 0.001)
    torch.testing.assert_close(result[:, 9:12], result[:, 3:6] - result[:, :3])
    torch.testing.assert_close(result[:, 12:15], result[:, 6:9] - result[:, 3:6])
    assert x.count_nonzero() == 0
    assert result[:, :9].count_nonzero() > 0


def test_one_step_latency_and_reset():
    env = DeploymentEnv({"name": "delay", "delay_steps": 1})
    try:
        env.reset(71003)
        a = np.array([0.4, 0.1, -0.1, -1], dtype=np.float32)
        first = env.step(a)
        assert first[4]["applied_action"] == [0, 0, 0, 1]
        second = env.step([0, 0, 0, 1])
        np.testing.assert_array_equal(second[0][-4:], a)
        env.reset(71003)
        assert env.step(a)[4]["applied_action"] == [0, 0, 0, 1]
    finally:
        env.close()


def test_dynamics_change_mass_inertia_and_sliding_friction():
    base = DeploymentEnv({"name": "id"})
    changed = DeploymentEnv({"name": "dynamics", "mass_scale": 2, "friction_scale": 0.55})
    try:
        body = base.env.model.body("cube").id
        assert changed.env.model.body_mass[body] == pytest.approx(
            2 * base.env.model.body_mass[body]
        )
        np.testing.assert_allclose(
            changed.env.model.body_inertia[body], 2 * base.env.model.body_inertia[body]
        )
        np.testing.assert_allclose(
            changed.env.model.geom_friction[:, 0], 0.55 * base.env.model.geom_friction[:, 0]
        )
        np.testing.assert_array_equal(
            changed.env.model.geom_friction[:, 1:], base.env.model.geom_friction[:, 1:]
        )
    finally:
        base.close()
        changed.close()


def test_relative_policy_is_planar_translation_invariant_and_serializes(tmp_path):
    torch.manual_seed(8)
    model = StudyPolicy(relative=True)
    x = torch.randn(15, 26)
    translated = x.clone()
    for start in (0, 3, 6):
        translated[:, start : start + 2] += torch.tensor([0.2, -0.1])
    for a, b in zip(model(x), model(translated), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    path = tmp_path / "model.pt"
    torch.save({"relative": True, "width": model.width, "state_dict": model.state_dict()}, path)
    loaded = StudyPolicy.load(path)
    for a, b in zip(model(x), loaded(x), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_paired_bootstrap_constant_effect_and_reproducibility():
    ones = np.ones((5, 100))
    assert two_axis_bootstrap(ones, repetitions=100) == [1, 1]
    zeros = np.zeros((5, 100))
    assert two_axis_bootstrap(zeros, repetitions=100) == [0, 0]
    values = np.random.default_rng(5).normal(size=(5, 20))
    assert two_axis_bootstrap(values, repetitions=100) == two_axis_bootstrap(
        values, repetitions=100
    )
    with pytest.raises(ValueError):
        two_axis_bootstrap(np.zeros(3))


def test_sampled_episode_subset_and_short_train(tmp_path):
    from pickplace.data import collect

    data_path = tmp_path / "expert.npz"
    collect(data_path, 8, 3)
    data, chosen = select_data(data_path, 4, 17)
    assert len(chosen) == 4 and set(chosen) <= set(range(8))
    assert set(data["episode_ids"][data["validation"]]) == {10000, 10001, 10002}
    config = {
        "epochs": 3,
        "batch_size": 256,
        "augmentation": {"grip_std": 0.001, "cube_std": 0.003, "goal_std": 0.001},
    }
    spec = {"name": "test", "episodes": 4, "relative": True, "augment": True}
    report = fit_variant(data_path, tmp_path / "test.pt", spec, config, 17, "test-protocol")
    assert report["selected_episodes"] == chosen
    assert report["history"][-1]["train_loss"] < report["history"][0]["train_loss"]
    assert Path(tmp_path / "test.pt").is_file()
