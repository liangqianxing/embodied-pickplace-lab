# Reproducibility and Provenance

## Original pipeline

Run `bash scripts/run_all.sh` in the project root. It reruns data collection, both BC data budgets, DAgger, paired evaluation, reporting, rendering and tests. It overwrites the original pipeline's named outputs.

Run `python scripts/verify_artifacts.py` with the project virtual environment to independently recompute dataset/checkpoint hashes, split disjointness, reported success counts, confidence intervals, and deterministic replay-selection rules from the stored records. This verifies record consistency; it does not independently re-execute every original episode.

## Frozen research v1

Run `bash scripts/run_research.sh`. The protocol is written before training. All model selection finishes before the first deployment-test trial. Completed checkpoint and trial groups resume from disk only when provenance matches. A changed source, dependency version, dataset or protocol raises an error instead of silently mixing experiments.

`artifacts/study/v1/source_snapshot/` preserves the code/configuration described by the protocol. Future development may change the working tree; preserve the snapshot. Reproduce from a separate checkout/directory populated with these snapshot sources, the matching original dataset, project packaging files, and the recorded dependency versions. Do not overwrite the active workspace to restore it.

To recompute only statistics, use:

```bash
.venv/bin/python -m pickplace.study_analysis --output artifacts/study/v1
```

`load_cube` requires every expected model/seed/condition trial group, checks expected scenario ordering and protocol/checkpoint hashes, and rejects incomplete-case analysis. Source snapshots establish local traceability, not trusted external attestation.

## Integrity inventory

After all jobs and documentation updates finish:

```bash
.venv/bin/python scripts/build_manifest.py
.venv/bin/python scripts/verify_manifest.py
```

The manifest excludes virtual environments, caches, version-control internals and release archives. Rebuild it only after intentional changes. A matching hash inventory does not prove scientific correctness; tests, raw trials and algorithm review are separate requirements.

## Limits

- CPU algorithms are seeded and tested for repeatability on this machine. Cross-version/platform bitwise equivalence is not promised.
- Original pipeline and research study use different training protocols; do not compare their rows as if only one factor changed.
- Reference trial seeds from the original pipeline must not be reused to claim unseen deployment validation after inspecting them.
- Public impact, real-hardware reliability and external benchmark competitiveness are not established by local checks.

After the frozen study has completed, `python scripts/build_study_replays.py` re-executes each group's first successful and failed trial, checks every scalar result exactly against the original record, and preserves the physics states and requested/applied commands. This provides direct rerun evidence for those selected trials, not for all 15,000 trials.

## Prospective latency follow-up

The separate protocol in `benchmarks/latency_guard/protocol.json` was fixed after discovering v1 latency failures but before its new scene tests. It evaluates frozen BC256 models, not newly trained checkpoints. `run.py` preserves its own source/configuration hashes and copies source snapshots. `report.py` regenerates all summary statistics and exactly reruns automatically selected repair/regression pairs.

This is a second study on new scenes, not an amendment that replaces negative v1 results. It must be presented with its 100ms failure boundary and small clean-success regression.

## Portable release snapshot

`python scripts/build_release.py` creates a ZIP without runtime environments or command logs and substitutes the explicit local checkout path in text artifacts. `python scripts/verify_portable_release.py` then verifies internal hashes and that source snapshots, dataset and research weights still match the scientific provenance records after sanitization. Do not assume a successful ZIP build proves installability on another computer.

## Relocated artifact check

`python scripts/verify_relocated_release.py` validates and extracts the portable ZIP to a temporary directory, starts Python with `-S` (no site initialization/editable install hooks), and resolves project imports only from relocated sources plus the explicitly selected original dependency directory. It runs source-resolution, manifest, original artifact, study artifact and both test-suite checks there. If the two prepared environments live in another checkout, pass `--dependency-root /path/to/prepared-checkout`; this still reuses dependencies rather than testing a fresh installation.

This catches accidental dependence on the original source/data paths. It **reuses installed dependencies** and is not a fresh installation, cross-platform result or independent third-party replication. The sidecar `artifacts/relocation-verification.json` records the exact tested ZIP hash and is omitted from the ZIP to avoid a self-referential archive hash. Rebuild the ZIP before rerunning this check; then regenerate the local manifest.
