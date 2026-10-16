# Panda Lift: Local Standard-Task Replication

**Completed:** five trained state-based BC policies, 1,500 held-out learned-policy trials, 100 baseline trials, 55 exactly reproduced selected trajectories, five integration tests, and actual MuJoCo rendering.

This is a seven-joint Panda arm on robosuite's official Lift task, not a renamed version of the custom Cartesian gripper. It is a **local standard-task replication**, not an independent third-party replication, leaderboard result or hardware experiment.

## Results

Five optimizer/shuffle seeds share 128 training demonstrations and 16 validation demonstrations. Each execution condition uses the same 50 new scenario seeds, including the environment's seeded initial joint noise, cube size, pose and yaw.

| Condition | Official success mean | Eight-step hold mean |
|---|---:|---:|
| Clean, original | 100.0% | 100.0% |
| Clean, EMA(0.5) | 77.2% | 77.2% |
| 50 ms delay, original | 67.2% | 66.4% |
| 50 ms delay, EMA(0.5) | 60.8% | 60.0% |
| 100 ms delay, original | 70.8% | 70.0% |
| 100 ms delay, EMA(0.5) | 77.6% | 75.2% |

Prespecified main comparison, EMA minus original at 50 ms: **-6.4 percentage points**, exploratory paired two-axis bootstrap 95% interval **[-23.2, +10.4] pp**. This does not support transferring the custom task's 50 ms improvement. EMA also hurts clean success substantially. Do not turn the positive 100 ms secondary comparison into the main result after seeing the data.

Expert and random clean baselines achieve 50/50 and 0/50, respectively, for both official and hold metrics. Full seed-level rates, SDs, intervals and trials are in `artifacts/study-v1_1/`.

The upstream success threshold is object-center height > table height + 0.04 m. We sample this at control boundaries and report whether it was ever reached within our 200-control horizon. Eight consecutive successful boundaries (0.4 s) are a separate hold diagnostic. We stop after hold success or timeout; learned rotational delta actions stay zero. These differences prevent claiming standard leaderboard equivalence.

## Reproduction

Verified on macOS Apple Silicon / Python 3.11.15 with robosuite 1.5.2, MuJoCo 3.3.7 and NumPy 1.26.4. Use the isolated environment; never install this lock over the original NumPy 2 experiment environment.

```bash
# From the project root. Existing local environment is already prepared.
bash benchmarks/robosuite/run_all.sh

# Individual stages
benchmarks/robosuite/.venv/bin/python benchmarks/robosuite/smoke.py
benchmarks/robosuite/.venv/bin/python -m pytest -q benchmarks/robosuite/test_learning.py benchmarks/robosuite/test_evaluation.py
benchmarks/robosuite/.venv/bin/python benchmarks/robosuite/study.py
benchmarks/robosuite/.venv/bin/python benchmarks/robosuite/report.py
benchmarks/robosuite/.venv/bin/python benchmarks/robosuite/render.py
```

For a new macOS environment, create `benchmarks/robosuite/.venv` with Python 3.11 and install `benchmarks/robosuite/requirements-lock.txt`. PyObjC packages are macOS-specific; the lock is not claimed to install directly on Linux. Exact source/dependency provenance is checked before a study resume.

## Implementation

- `smoke.py`: actual Panda environment creation, finite observations and deterministic reset verification.
- `development.py`: approach → descend → close → lift finite-state demonstration controller, 10/10 development scenes held successfully.
- `learning.py`: 49-dimensional state/proprioception features, collection, validation-selected three-layer MLP. No image, expert-phase or timer input.
- `evaluate.py`: requested → filtered → queued → actually applied command semantics; official and hold outcomes stay separate.
- `study.py`: freeze-before-training protocol, all training before deployment testing, atomic per-episode persistence and provenance-checked resume.
- `report.py`: data/model hashes, split disjointness, complete-case summaries, and exact outcome re-execution of each group's first held success/failure.
- `render.py`: uses saved MuJoCo qpos with the seeded scene and actual visual meshes; no synthesized robot motion or learned camera input.

## Preserved failures and compatibility

The initially unconstrained dependency installation selected MuJoCo 3.16.0. Its enum-to-NumPy comparison rejected a valid hinge in a tuple membership check. A minimal reproduction and verified 3.3.7 workaround are in `COMPATIBILITY.md`; an unsubmitted upstream issue draft is retained. It is **not** an accepted community contribution.

The first frozen formal attempt (`artifacts/study-v1/`) aborted in the first collection episode's metadata serialization (`cube.size` is a list). No completed dataset, model or deployment tests existed. The corrected v1.1 uses a separate frozen directory, identical experiment settings and an added regression test. Failed logs are retained.

Optional warnings for private macros, extra robot models and GR1 Mink whole-body IK remain; the tested setup uses built-in Panda and OSC_POSE, not those features.

## Boundaries

One task, state input, fixed expert data and five optimization seeds. No vision/VLA, independently replicated results, real robot, safety guarantee or external adoption. Empirical bootstrap intervals may degenerate at identical outcomes and do not prove population success equals 100%.

Upstream source inspected at tag v1.5.2: `robosuite/environments/manipulation/lift.py`, `robosuite/controllers/composite/composite_controller_factory.py`, and `robosuite/utils/binding_utils.py`. Repository: `https://github.com/ARISE-Initiative/robosuite`.
