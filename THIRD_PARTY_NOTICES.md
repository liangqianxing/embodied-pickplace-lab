# Third-party software and assets

The root MIT license covers original project material. It does not replace the
licenses of dependencies, robot assets, or upstream material.

This repository does not vendor installed Python packages, MuJoCo binaries, or
robosuite robot mesh/texture libraries. Install them from the upstream packages
using the two separate environment specifications. Generated demonstrations,
trained project policies, result records, screenshots, and simulation renders are
included; the screenshots and renders are not physical-robot recordings.

## Primary dependencies

The following identifiers come from the installed distributions used for the
recorded experiments, not from an assertion that all transitive dependencies
have a single license. See each distribution for its complete notices.

| Dependency | Recorded license information | Role |
|---|---|---|
| MuJoCo | Apache-2.0, plus bundled third-party notices | Simulation and rendering |
| Gymnasium | MIT | Custom environment interface |
| PyTorch | Composite notices including BSD, Apache, MIT, and other bundled licenses | Training and inference |
| NumPy | Composite notices including BSD-3-Clause and bundled licenses | Numeric arrays |
| Pillow | MIT-CMU | Images and animations |
| robosuite 1.5.2 | MIT, with upstream MuJoCo notice | Panda Lift task and rendered robot assets |

The robosuite copyright and permission notice is preserved in
`licenses/robosuite-MIT.txt`. Panda images and animations were rendered from a
locally installed robosuite task; its assets remain subject to upstream terms.

## Upstream projects

- MuJoCo: https://github.com/google-deepmind/mujoco
- Gymnasium: https://github.com/Farama-Foundation/Gymnasium
- PyTorch: https://github.com/pytorch/pytorch
- NumPy: https://github.com/numpy/numpy
- Pillow: https://github.com/python-pillow/Pillow
- robosuite: https://github.com/ARISE-Initiative/robosuite/tree/v1.5.2

Behavior cloning, DAgger, and exponential moving-average filtering are existing
methods. This project does not claim their invention, affiliation with the
upstream authors, or upstream endorsement. The compatibility issue draft has
not been submitted or accepted.
