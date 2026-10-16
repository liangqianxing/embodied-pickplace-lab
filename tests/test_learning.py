import json

import numpy as np
import pytest
import torch

from pickplace.data import collect, file_hash, load_data
from pickplace.evaluate import rollout, wilson_interval
from pickplace.policy import Policy
from pickplace.train import fit


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "expert.npz"
    collect(path, episodes=8, validation_episodes=3)
    return path


def test_episode_split_and_subset(dataset):
    data = load_data(dataset, train_episodes=4)
    val = data["validation"]
    assert len(np.unique(data["episode_ids"][~val])) == 4
    assert len(np.unique(data["episode_ids"][val])) == 3
    assert not set(data["episode_ids"][val]) & set(data["episode_ids"][~val])
    assert data["observations"].shape[1] == 26
    assert data["actions"].shape[1] == 4
    assert np.isfinite(data["observations"]).all()
    meta = json.loads(dataset.with_suffix(".json").read_text())
    assert meta["sha256"] == file_hash(dataset)
    with pytest.raises(ValueError):
        load_data(dataset, train_episodes=99)


def test_training_loss_checkpoint_and_determinism(dataset, tmp_path):
    data = load_data(dataset)
    model, report = fit(data, tmp_path / "first.pt", epochs=12, seed=7)
    loaded = Policy.load(tmp_path / "first.pt")
    assert report["history"][-1]["train_loss"] < report["history"][0]["train_loss"]
    action = model.act(data["observations"][0])
    np.testing.assert_array_equal(action, loaded.act(data["observations"][0]))
    assert action.shape == (4,) and np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 1
    assert report["best_val_loss"] == min(row["val_loss"] for row in report["history"])
    second, _ = fit(data, tmp_path / "second.pt", epochs=12, seed=7)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[key], rtol=0, atol=0)


def test_wilson_interval():
    low, high = wilson_interval(100, 100)
    assert 0.96 < low < 0.97 and high == pytest.approx(1)
    low, high = wilson_interval(0, 100)
    assert low == pytest.approx(0, abs=1e-10) and 0.03 < high < 0.04
    with pytest.raises(ValueError):
        wilson_interval(1, 0)


def test_replay_contains_actual_states_and_seed():
    a = rollout("expert", 15200, record=True)
    b = rollout("expert", 15200, record=True)
    assert a == b
    assert a["success"]
    assert len(a["frames"]) == a["steps"] + 1
    assert len(a["frames"][0]["qpos"]) == 12
    assert any(frame["contacts"] == 2 for frame in a["frames"])
    assert json.loads(json.dumps(a))["seed"] == 15200


def test_random_baseline_and_failure_diagnostic():
    result = rollout("random", 15201)
    assert not result["success"]
    assert result["failure"] in ("no_lift", "misplaced_or_dropped", "release_or_settle")
