# Fixed-weight latency intervention

**Completed:** 4,500 new closed-loop trials on five frozen BC256 models. This is a prospective follow-up to an observed failure, not an invented new algorithm or a hardware result.

| Execution wrapper | Clean | 50 ms delay | 100 ms delay |
|---|---:|---:|---:|
| Original | 100.0% | 56.0% | 52.4% |
| Half XYZ gain | 36.6% | 84.0% | 88.6% |
| EMA, coefficient 0.5 | 99.6% | 86.6% | 40.6% |

The prespecified main EMA effect at 50 ms is **+30.6 percentage points**, exploratory paired two-axis bootstrap 95% interval **[+17.0, +42.6] pp**. Clean performance changes by -0.4 pp. EMA does **not** fix 100 ms latency, and halving gain badly damages clean performance for some models.

Same frozen weights, paired scenes, fixed control horizon and original success criteria. The scalar coefficient was fixed before new tests, not selected from a sweep. Only translational actions are filtered; gripper action is unchanged. Five seeds describe both original training-subset and optimizer variation. Selected model-family bias from the preceding study remains.

```bash
# From project root; use the original project virtual environment.
.venv/bin/pytest -q benchmarks/latency_guard/test_guard.py
.venv/bin/python benchmarks/latency_guard/run.py
.venv/bin/python benchmarks/latency_guard/report.py
.venv/bin/python benchmarks/latency_guard/render_pair.py
```

Full metadata: `artifacts/v1/frozen.json`; trial data: `artifacts/v1/trials/`; report: `artifacts/v1/REPORT.md`. The report script checks frozen source/checkpoint hashes and exactly reruns three automatically selected pairs: a repair at 50 ms, a clean regression, and a 100 ms regression.

Actual MuJoCo pair rendering: `artifacts/v1/paired-delay-50ms.gif`. After one side completes, its final physics state is held on screen; the clock explicitly shows the final time. This is replay visualization, not continuous successful control after episode termination.

This is an **execution filter**, not a certified safety guard. It provides no collision, force, joint-limit, or hardware safety guarantees. Do not deploy directly on hardware.
