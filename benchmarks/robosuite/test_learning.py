"""Isolated integration tests; run with this benchmark's virtual environment."""

import numpy as np
import torch
from learning import LiftPolicy, vector
from smoke import make_env


def test_observation_vector_and_forward_model():
    torch.set_num_threads(1)
    env = make_env(91000)
    try:
        observation = env.reset()
        x = vector(observation, np.zeros(7, dtype=np.float32))
        assert x.ndim == 1 and np.isfinite(x).all()
        assert x.shape == (49,)
        model = LiftPolicy(len(x))
        action = model.act(x)
        assert action.shape == (7,)
        assert np.isfinite(action).all()
        assert np.max(np.abs(action[:3])) <= 0.5
        assert (action[3:6] == 0).all()
        assert action[-1] in (-1, 1)
        low, high = env.action_spec
        assert (action >= low).all() and (action <= high).all()
        result = env.step(action)
        assert np.isfinite(vector(result[0], action)).all()
    finally:
        env.close()


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(17)
    model = LiftPolicy(49)
    path = tmp_path / "policy.pt"
    torch.save({"input_size": 49, "state_dict": model.state_dict()}, path)
    loaded = LiftPolicy.load(path)
    x = np.random.default_rng(17).normal(size=49).astype(np.float32)
    np.testing.assert_array_equal(model.act(x), loaded.act(x))


def test_collection_metadata_and_evaluation_serialization(tmp_path):
    import json

    from evaluate import episode
    from learning import collect

    path = collect(tmp_path / "data", episodes=1, validation_episodes=1)
    with np.load(path, allow_pickle=False) as data:
        assert data["observations"].shape[1] == 49
        assert data["actions"].shape[1] == 7
        assert set(data["episode_ids"].tolist()) == {92000, 93000}
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert all(len(record["cube_size"]) == 3 for record in metadata["episodes"])
    assert all(record["held_8_steps"] for record in metadata["episodes"])
    trial = episode("expert", 91001)
    assert trial["official_success"] and trial["held_8_steps"]
    assert len(json.loads(json.dumps(trial))["cube_size"]) == 3
