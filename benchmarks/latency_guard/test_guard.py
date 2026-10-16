import numpy as np
import pytest
from run import ExecutionFilter, rollout


def test_identity_copy():
    action = np.array([0.3, -0.2, 0.8, -1], dtype=np.float32)
    output = ExecutionFilter("identity").apply(action)
    np.testing.assert_array_equal(output, action)
    output[0] = 0
    assert action[0] != 0


def test_gain_does_not_modify_gripper():
    result = ExecutionFilter("gain_half").apply([1, -1, 0.2, -1])
    np.testing.assert_allclose(result, [0.5, -0.5, 0.1, -1])


def test_ema_has_known_response_and_preserves_gripper():
    f = ExecutionFilter("ema_half")
    np.testing.assert_allclose(f.apply([1, 0, 0, 1]), [0.5, 0, 0, 1])
    np.testing.assert_allclose(f.apply([-1, 0, 0, -1]), [-0.25, 0, 0, -1])
    np.testing.assert_allclose(f.apply([1, 0, 0, 1]), [0.375, 0, 0, 1])
    fresh = ExecutionFilter("ema_half")
    np.testing.assert_allclose(fresh.apply([1, 0, 0, -1]), [0.5, 0, 0, -1])


def test_invalid_mode_and_action():
    with pytest.raises(ValueError):
        ExecutionFilter("unknown")
    with pytest.raises(ValueError):
        ExecutionFilter("ema_half").apply([1, 2])


def test_identity_rollout_exactly_matches_frozen_evaluator():
    from pathlib import Path

    import torch

    from pickplace.study import episode
    from pickplace.study_policy import StudyPolicy

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    model = StudyPolicy.load(root / "artifacts/study/v1/checkpoints/raw256-17.pt")
    condition = {"name": "delay_50ms", "distribution": "id", "delay_steps": 1}
    a = episode(model, condition, 75000)
    b = rollout(model, condition, 75000, "identity")
    for key in b:
        assert b[key] == a[key]
