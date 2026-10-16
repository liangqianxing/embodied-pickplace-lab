# Coordinate / Observation Noise Study v1

Protocol frozen: 2026-10-15T15:16:54.099384+08:00

**15,000 closed-loop trials; 5 training/data-subset seeds.**

Success rate: mean +/- sample SD across training runs; each run uses 100 scenarios by default.

| Model | id | position | sensor_3mm | delay_50ms | dynamics | combined | Shifted mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw32 | 94.8% +/- 8.5% | 68.2% +/- 11.2% | 97.2% +/- 3.1% | 87.0% +/- 8.7% | 92.8% +/- 11.3% | 60.4% +/- 13.1% | 81.1% |
| relative32 | 85.6% +/- 19.0% | 73.6% +/- 16.6% | 95.0% +/- 2.7% | 80.6% +/- 14.2% | 77.8% +/- 18.0% | 53.4% +/- 20.4% | 76.1% |
| noise32 | 96.6% +/- 4.6% | 62.6% +/- 10.6% | 97.6% +/- 2.6% | 86.8% +/- 11.7% | 96.2% +/- 5.4% | 56.8% +/- 15.2% | 80.0% |
| relative_noise32 | 96.2% +/- 1.9% | 77.6% +/- 4.6% | 97.0% +/- 2.0% | 69.0% +/- 34.7% | 82.0% +/- 15.7% | 41.4% +/- 24.8% | 73.4% |
| raw256 | 100.0% +/- 0.0% | 99.0% +/- 1.2% | 99.8% +/- 0.4% | 49.6% +/- 17.8% | 100.0% +/- 0.0% | 71.0% +/- 12.5% | 83.9% |

## Prespecified primary comparison

relative_noise32 minus raw32: **-7.72 percentage points**.
Exploratory paired two-axis bootstrap 95% interval: [-12.60, -2.68] percentage points.

This is not an independent external replication or evidence of real-world deployment success.
Five seeds are too few for precise tail inference; per-seed outcomes are retained in summary.json.
The 32-episode subsets and optimizer seeds vary together. raw256 is a larger-data reference, not an equal-budget competitor.
Training augmentation matches the prescribed sensor-noise level; do not call this unknown-sensor generalization.
The remaining condition comparisons and factorial interactions are exploratory and not multiplicity-corrected.

## Artifacts

- protocol.json: frozen configuration, source hashes, dependency versions.
- checkpoints/: weights, selected episode IDs, validation curves and hashes.
- trials/: every scenario outcome grouped by model, training seed and condition.
- trials.csv: all per-episode records in one machine-readable file.
- summary.json: seed-level rates, intervals and prespecified primary contrast.
