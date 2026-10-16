"""Real Panda evaluation semantics and deterministic command-delay checks."""

from itertools import pairwise

import numpy as np
from evaluate import episode


def test_same_seed_expert_rollout_is_repeatable():
    first = episode("expert", 91010, record=True)
    second = episode("expert", 91010, record=True)
    assert first == second
    assert first["official_success"] and first["held_8_steps"]
    assert len(first["frames"]) == first["steps"]
    assert first["first_success_step"] <= first["steps"] - 7
    assert all(frame["official_success"] for frame in first["frames"][-8:])


def test_delayed_action_is_exactly_previous_filtered_request():
    trial = episode("expert", 91011, delay_steps=1, filter_mode="ema_half", record=True)
    frames = trial["frames"]
    np.testing.assert_array_equal(frames[0]["applied_action"], [0, 0, 0, 0, 0, 0, -1])
    for previous, current in pairwise(frames):
        np.testing.assert_array_equal(current["applied_action"], previous["filtered_action"])
    for frame in frames:
        assert frame["requested_action"][-1] == frame["filtered_action"][-1]
    previous_xyz = np.zeros(3, dtype=np.float32)
    for frame in frames:
        expected = (
            0.5 * np.asarray(frame["requested_action"][:3], dtype=np.float32) + 0.5 * previous_xyz
        )
        np.testing.assert_array_equal(expected, frame["filtered_action"][:3])
        previous_xyz = expected
