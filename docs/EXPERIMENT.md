# Experiment Report

Generated: 2026-10-15T14:36:35.215718+08:00

## Held-out evaluation

Identical initial-condition seeds across policies; no test episodes in training.

| Policy | Distribution | Success | Wilson 95% CI | Mean successful steps |
|---|---|---:|---|---:|
| random | id | 0/100 | 0.0% - 3.7% | N/A |
| random | ood | 0/100 | 0.0% - 3.7% | N/A |
| expert | id | 100/100 | 96.3% - 100.0% | 59.5 |
| expert | ood | 100/100 | 96.3% - 100.0% | 65.9 |
| bc32 | id | 96/100 | 90.2% - 98.4% | 60.6 |
| bc32 | ood | 66/100 | 56.3% - 74.5% | 67.9 |
| bc256 | id | 100/100 | 96.3% - 100.0% | 58.6 |
| bc256 | ood | 100/100 | 96.3% - 100.0% | 65.8 |
| dagger | id | 98/100 | 93.0% - 99.4% | 60.2 |
| dagger | ood | 81/100 | 72.2% - 87.5% | 70.0 |

## Failure counts

No lift / misplaced or dropped / release or settle:

- random (id): {'no_lift': 100, 'misplaced_or_dropped': 0, 'release_or_settle': 0}
- random (ood): {'no_lift': 100, 'misplaced_or_dropped': 0, 'release_or_settle': 0}
- expert (id): {'no_lift': 0, 'misplaced_or_dropped': 0, 'release_or_settle': 0}
- expert (ood): {'no_lift': 0, 'misplaced_or_dropped': 0, 'release_or_settle': 0}
- bc32 (id): {'no_lift': 3, 'misplaced_or_dropped': 1, 'release_or_settle': 0}
- bc32 (ood): {'no_lift': 18, 'misplaced_or_dropped': 16, 'release_or_settle': 0}
- bc256 (id): {'no_lift': 0, 'misplaced_or_dropped': 0, 'release_or_settle': 0}
- bc256 (ood): {'no_lift': 0, 'misplaced_or_dropped': 0, 'release_or_settle': 0}
- dagger (id): {'no_lift': 0, 'misplaced_or_dropped': 1, 'release_or_settle': 1}
- dagger (ood): {'no_lift': 8, 'misplaced_or_dropped': 5, 'release_or_settle': 6}

## Protocol and limitations

- Training uses 256 expert episodes, with a nested 32-episode data ablation. Validation: 32 separate episodes.
- DAgger uses additional expert labels on mixed learner/expert trajectories; this is NOT an equal-label-budget comparison.
- DAgger validation loss measures the original expert validation distribution, not off-policy recovery quality.
- Training seeds: 0-255; validation: 10000-10031; diagnostic development rollouts: 11000-11019.
- DAgger seeds: 2000-2063, 3000-3063, 4000-4063. Test: 20000-20099 and 30000-30099 for the default 100-trial run.
- OOD: cube begins outside the training rectangle; position support changes, not visual appearance, mass, or friction.
- Wilson intervals describe finite evaluation episodes, not variance across independent training seeds.
- Failure labels are final-state diagnostics, not proven causal explanations.
- The expert has a finite-state controller; the learned policy only gets 26 observation values, including its previous action.
- No images, natural-language instructions, real hardware, real-robot validation, or sim-to-real claims.
- Exact bitwise reproducibility across MuJoCo/PyTorch versions or different platforms is not guaranteed.

## Provenance

- Dataset SHA-256: `87fa71f5ab36fa656bafc85576d3eabb266e98872935b5fb3714b724a8acda8d`
- Versions: `{'mujoco': '3.16.0', 'gymnasium': '1.4.0', 'torch': '2.14.1', 'numpy': '2.4.6'}`
