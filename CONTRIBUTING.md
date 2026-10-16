# Reproduction and contributions

Both successful and failed reproductions are useful. Open a reproduction issue
with the commit/tag, operating system and CPU architecture, Python version,
environment specification, exact command, and observed result. Remove tokens,
personal filesystem paths, private hostnames, and unrelated data from logs.

Start with record checks rather than retraining:

```bash
python3 scripts/verify_manifest.py
.venv/bin/python scripts/verify_artifacts.py
.venv/bin/python scripts/verify_study.py
.venv/bin/python -m pytest -q tests benchmarks/latency_guard/test_guard.py
```

The first command needs only Python's standard library. The others require the
main environment described in README.md. For Panda, follow the isolated setup in
`benchmarks/robosuite/README.md` and run:

```bash
benchmarks/robosuite/.venv/bin/python -m pytest -q \
  benchmarks/robosuite/test_learning.py benchmarks/robosuite/test_evaluation.py
```

Do not silently overwrite frozen protocols or snapshots. Run new experiments in
a separate output directory, report every prespecified condition, and separate
new results from the published records. Use paired scene seeds for paired
comparisons. A record check or a green integrity workflow is not an independent
replication of all 22,100 episodes.

Keep changes small and testable. Contributions are provided under the project's
MIT license, subject to any explicitly identified upstream material and notices.
