# Compatibility finding: NumPy scalar vs MuJoCo enum membership

Status: local reproducible finding; not yet submitted to an upstream project. No claim of an accepted bug report or contribution.

The unconstrained dependency installation resolved to robosuite 1.5.2, MuJoCo 3.16.0 and NumPy 1.26.4. Creating Panda Lift failed in `robosuite.utils.binding_utils.MjModel.get_joint_qpos_addr` at the assertion testing membership in `(mjtJoint.mjJNT_HINGE, mjtJoint.mjJNT_SLIDE)`.

The actual robot joint is a hinge (`numpy.int32(3)`). A minimal reproduction without creating an environment shows:

```python
joint_type = np.int32(3)
hinge = mujoco.mjtJoint.mjJNT_HINGE
joint_type == hinge  # True
hinge == joint_type  # False in the observed environment
joint_type in (hinge, slide_enum)  # False in the observed environment
int(joint_type) == int(hinge)  # True
```

This explains why the tuple-membership assertion rejects a numeric hinge in that environment. It does not by itself identify which upstream package should change or prove compatibility across versions.

Reproducer: `reproduce_enum_compat.py`. Raw observation: `artifacts/runs/robosuite-enum-reproducer.json` in the project root. Initial dependency snapshot: `requirements-initial-incompatible.txt`.

A candidate defensive upstream change would compare integer representations for joint-type membership. That has not been submitted, and changing a single assertion does not prove all other bindings are compatible. The local integration instead tests an isolated MuJoCo version pin; the original experiment environment remains unchanged.

Before external reporting, check current upstream issues, reproduce in a minimal clean environment, include exact versions and a compact traceback, and ask the user for permission to post under their identity.

## Verified local resolution

The isolated pin `mujoco==3.3.7` makes both comparison directions and tuple membership agree. The actual Panda Lift smoke passed, including same-seed reset equality, and the default scripted grasp/lift controller succeeded and held the task for eight steps on all 10 development seeds (90010–90019). `requirements-lock.txt` records the working environment; the original project remains on MuJoCo 3.16.0 / NumPy 2.4.6.

Warnings about missing `robosuite_models`, private macros and optional GR1 Mink whole-body IK remain; this experiment uses built-in Panda with OSC_POSE, not those optional features.

The first formal protocol (study-v1) then stopped on metadata serialization because `env.cube.size` is a Python list, not an ndarray. No training or test data were completed. The code now explicitly converts with `np.asarray`, the regression is covered by a real collect/evaluate integration test, and study-v1_1 is a new frozen artifact directory. Failed protocol/logs are retained rather than silently replaced.
