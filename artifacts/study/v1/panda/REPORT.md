# Panda Lift: Local Standard-Task Replication

Frozen: 2026-10-15T22:07:51.740234+08:00

**1,500 learned-policy trials + 100 baseline trials.**

Five optimizer seeds, one shared 128-episode training dataset, 16 held-out validation episodes, 50 new scene seeds paired across all execution conditions.

| Condition | Official success mean +/- SD | Eight-step hold mean +/- SD | Mean held-episode steps |
|---|---:|---:|---:|
| clean | 100.0% +/- 0.0% | 100.0% +/- 0.0% | 53.7 |
| clean_ema | 77.2% +/- 13.8% | 77.2% +/- 13.8% | 71.0 |
| delay50 | 67.2% +/- 12.0% | 66.4% +/- 11.1% | 83.7 |
| delay50_ema | 60.8% +/- 9.1% | 60.0% +/- 8.6% | 80.1 |
| delay100 | 70.8% +/- 6.4% | 70.0% +/- 7.9% | 79.5 |
| delay100_ema | 77.6% +/- 6.4% | 75.2% +/- 6.4% | 76.7 |

## Primary comparison

EMA minus raw at 50 ms, official success: -6.4 pp; exploratory paired two-axis bootstrap [-23.2, +10.4] pp.
Supplementary eight-step hold difference: -6.4 pp.

## Baselines

- expert: official 50/50; hold 50/50.
- random: official 0/50; hold 0/50.

## Evidence and limitations

- 55 automatically selected first-success/first-failure hold traces re-executed with exact scalar outcome agreement.
- Official success means the object crosses the upstream Lift height threshold at an observed control boundary. Hold is a separate stronger diagnostic, not the upstream benchmark definition.
- Horizon is 200 controls, termination after eight successful observations, and learned rotational deltas are zero. This is not a leaderboard-equivalent protocol.
- States and joint proprioception are observed; no images, language, hardware or real-time safety are evaluated.
- Optimization/shuffling seeds vary while data stay fixed; uncertainty is not the same as the custom study that also resamples demonstration subsets.
- Identical outcomes can make empirical bootstrap intervals degenerate. A 100% sample rate or [0,0] contrast interval is not proof of universal success or exact population equality.
- The external library provides the task. This is our local replication, not independent third-party verification or external adoption.
- A first attempt stopped on metadata serialization before completed data/model/test artifacts; its frozen directory is retained. Current protocol v1.1 fixes serialization without changing experimental settings.

## Environment

`{'robosuite': '1.5.2', 'mujoco': '3.3.7', 'numpy': '1.26.4', 'torch': '2.14.1', 'numba': '0.68.0', 'scipy': '1.17.1'}`
