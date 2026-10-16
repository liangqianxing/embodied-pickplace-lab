# Draft: asymmetric NumPy/MuJoCo enum equality breaks joint-type membership checks

**Not submitted. No upstream acknowledgement or accepted contribution exists.**

## Suggested title

Panda initialization fails at `get_joint_qpos_addr` with MuJoCo 3.16.0 / NumPy 1.26.4 due to enum membership comparison

## Environment

- macOS Apple Silicon
- Python 3.11.15
- robosuite 1.5.2
- MuJoCo 3.16.0
- NumPy 1.26.4 (resolved via the pinned Mink dependency)

## Minimal binding reproduction

```python
import mujoco
import numpy as np

value = np.int32(3)
hinge = mujoco.mjtJoint.mjJNT_HINGE
print(value == hinge)  # True
print(hinge == value)  # False
print(value in (hinge, mujoco.mjtJoint.mjJNT_SLIDE))  # False
print(int(value) in (int(hinge), int(mujoco.mjtJoint.mjJNT_SLIDE)))  # True
```

## Environment-level reproduction

```python
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=load_composite_controller_config(robot="Panda"),
    use_camera_obs=False,
    has_renderer=False,
    has_offscreen_renderer=False,
    seed=90000,
)
```

Observed failure is the hinge/slide membership assertion inside `robosuite/utils/binding_utils.py:get_joint_qpos_addr`. The first Panda joint has numeric type 3 (hinge), but tuple membership returns false because the enum-to-NumPy scalar comparison differs from the reverse comparison.

## Expected behavior

A numerically matching hinge joint should not be rejected during reference setup.

## Verified local workaround

Pinning MuJoCo to 3.3.7 in the same isolated environment makes the equality directions agree and lets Panda Lift initialize and execute. We did not patch the installed package to bypass its assertion.

## Candidate defensive fix, not yet validated upstream

Converting joint types and enum constants to integer values before membership checks avoids dependence on scalar/enum equality direction. Other comparisons and bindings should be audited before treating this as full compatibility. A version upper bound could instead document the currently supported range if upstream prefers.

## Scope and caveats

This local reproduction does not establish whether the root cause belongs in MuJoCo bindings, NumPy compatibility or robosuite's defensive comparisons. Check upstream issue history and latest supported versions before submission; the report intentionally avoids claiming novelty or ownership of an already-known issue.

Before posting, remove any personal paths from traceback excerpts. Attach the compact reproduction and working/failing version matrices, not the entire local project or virtual environment.
