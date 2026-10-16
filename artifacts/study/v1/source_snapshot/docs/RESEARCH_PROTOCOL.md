# Research Extension: Frozen Protocol v1

This study is specified before looking at its held-out results. It is an **internal, timestamped protocol**, not a third-party preregistration.

## Question and falsifiable hypotheses

Can low-data behavior cloning become more robust by (a) removing absolute planar coordinates and (b) training on geometrically consistent noisy observations?

- H1: jointly using both improves average shifted-condition success over raw BC32.
- H2: removing absolute coordinates specifically helps shifted starting positions.
- H3: noisy-observation training helps sensor noise; it need not help latency or altered dynamics.
- A null or negative result must remain in the report. No requirement that the proposed variant win.

## Factorial design

Use four 32-episode models forming a 2 × 2 factorial ablation: raw, relative only, noise only, relative + noise. Add raw BC256 as a larger-demonstration reference, not an equal-data competitor.

Each model is trained with five seeds: 17, 29, 43, 71, 97. For each seed, all four low-data models use the same randomly sampled 32 training episodes, initialization seed, optimizer and training budget. Across seeds the subset and optimization both change; report this mixed source of variation explicitly. All models share the original 32 held-out validation episodes. Test seeds are newly reserved at 50000–50099.

Representation: zero the six absolute X/Y coordinates (gripper, object, goal). Retain relative vectors, heights, velocities, gripper opening and prior action. The policy is invariant to joint planar translation of the three positions, but the environment is **not** translation invariant at workspace boundaries. Stored parameter shapes are matched; the relative representation has fewer effectively active input weights.

Noise: independently perturb measured gripper/object/goal positions with Gaussian standard deviations 1/3/1 mm, then recompute relative vectors from the perturbed positions. Do not perturb absolute and relative channels independently. Keep the expert action label unchanged. This is robustness to known measurement-noise levels, not a claim of transfer to unseen sensors.

Architecture and objective match the original 3 × 128 MLP; train 150 epochs, minibatch 256, AdamW with cosine decay, select by minimum clean validation loss. No deployment-test success is used for selection.

## Deployment shifts

Six conditions: clean ID, new starting positions, positional observation noise, one control-step command latency (50 ms), heavier object with globally reduced sliding friction, and their combination. Exact values are in `configs/research_v1.json`.

Delay executes one queued action, initialized to zero XYZ motion / open fingers. Previous-action observations describe the action actually applied, not the pending request. There is no hidden-policy fallback or expert correction at evaluation.

Dynamic perturbations change mass and inertia together (2×), and every geom's sliding friction (0.55×). Torsional/rolling friction stays fixed. Position actuators and task success criteria stay unchanged.

## Analysis

Primary endpoint: equal-weighted mean success across the five non-clean conditions. Primary comparison: relative+noise32 minus raw32. Report all six conditions and all five seed-level rates, not only this scalar.

Compute a paired two-axis bootstrap: resample training-seed and test-scenario indices with replacement, sharing indices between compared methods. This accounts for repeated scenarios across models and seeds; percentile intervals from only five seeds remain unstable and must be described as exploratory. For the primary endpoint, use the same resampled scenario indices across conditions. Do not treat 500 correlated trials as 500 independent training runs.

Preserve per-episode outcomes, validation curves, model and source hashes, configuration, dataset subsets, and first successful/failed traces. No post-test tuning in the v1 run. Any later change needs a new protocol/version and clearly labeled exploratory/confirmatory status.

## Publication boundaries

This extends a custom educational simulator into a research-style study. It does **not** yet establish benchmark competitiveness, visual robotics performance, hardware transfer, a novel method, public adoption, citations, or employment impact. Those remain separate deliverables.
