#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON="${PANDA_PYTHON:-benchmarks/robosuite/.venv/bin/python}"
"$PYTHON" -m pip check
"$PYTHON" benchmarks/robosuite/smoke.py
"$PYTHON" -m pytest -q benchmarks/robosuite/test_learning.py benchmarks/robosuite/test_evaluation.py
"$PYTHON" benchmarks/robosuite/study.py
"$PYTHON" benchmarks/robosuite/report.py
"$PYTHON" benchmarks/robosuite/render.py
