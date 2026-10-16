# Fixed-Weight Latency Follow-up

Protocol frozen: 2026-10-15T15:54:26.225916+08:00

**4,500 newly reserved closed-loop test trials; five frozen BC256 training runs.**

This prospective follow-up was motivated by the earlier study's observed failures; it is not an original preregistered hypothesis or an external benchmark.

| Execution | Condition | Mean success | Across-run SD | Successful steps | All-trial steps |
|---|---|---:|---:|---:|---:|
| identity | clean | 100.0% | 0.0% | 59.4 | 59.4 |
| identity | delay_50ms | 56.0% | 11.9% | 74.2 | 138.3 |
| identity | delay_100ms | 52.4% | 29.9% | 114.8 | 164.8 |
| gain_half | clean | 36.6% | 47.5% | 88.9 | 172.0 |
| gain_half | delay_50ms | 84.0% | 10.2% | 93.8 | 114.0 |
| gain_half | delay_100ms | 88.6% | 6.7% | 111.1 | 123.5 |
| ema_half | clean | 99.6% | 0.9% | 68.1 | 68.7 |
| ema_half | delay_50ms | 86.6% | 14.7% | 82.6 | 101.0 |
| ema_half | delay_100ms | 40.6% | 24.5% | 111.7 | 176.0 |

## Prespecified primary contrast

EMA(0.5) minus identity at 50ms: **+30.6 percentage points**; exploratory paired bootstrap interval [+17.0, +42.6] pp.
Clean difference: -0.4 pp; interval [-1.8, +0.0] pp.

## What changed

- Same frozen neural-network weights and same scene seeds across execution wrappers.
- identity sends the raw action; gain_half halves XYZ only; ema_half computes 0.5 * requested XYZ + 0.5 * previous filtered XYZ.
- Gripper commands are unchanged. Filter state resets to zero at each episode. Observation previous-action channels describe actually executed commands.
- Every wrapper uses the original 220-step horizon and success criteria. No coefficient sweep or retraining.

## Honest interpretation

- EMA helps this 50ms latency setting with a small clean-success decrease, but NOT the 100ms setting; do not claim universal delay robustness.
- Reduced XYZ gain is not an equivalent safe fallback: it damages clean performance substantially for some training runs.
- Successful-episode length is conditional on success and not a matched speed comparison. All-trial length includes failures/timeouts.
- Dynamics and feedback distributions change under filtering. Results demonstrate a wrapper intervention, not proof of one exclusive failure mechanism.
- Only five training/data-subset seeds. Model family was selected after study v1; remaining comparisons are exploratory and not multiplicity-corrected.
- EMA filtering is a standard baseline, not a novel algorithm. There is no hardware or external-adoption claim.

## Paired replay evidence

- paired-delay_50ms.json: first identity failure repaired by EMA; training 17, scene 80002.
- paired-clean.json: first clean success regressed by EMA; training 97, scene 80015.
- paired-delay_100ms.json: first 100ms success regressed by EMA; training 17, scene 80007.
