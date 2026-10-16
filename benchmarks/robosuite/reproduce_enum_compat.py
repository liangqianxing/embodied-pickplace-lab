"""Minimal, non-mutating reproduction of MuJoCo enum / NumPy membership mismatch."""

import json

import mujoco
import numpy as np

joint_type = np.int32(3)
hinge = mujoco.mjtJoint.mjJNT_HINGE
print(
    json.dumps(
        {
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "numpy_eq_enum": bool(joint_type == hinge),
            "enum_eq_numpy": bool(hinge == joint_type),
            "tuple_membership": joint_type in (hinge, mujoco.mjtJoint.mjJNT_SLIDE),
            "integer_membership": int(joint_type)
            in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)),
        },
        indent=2,
    )
)
