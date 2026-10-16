#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -m pytest -q
"$PYTHON" -m pickplace.study
"$PYTHON" -m pickplace.study_analysis
"$PYTHON" scripts/snapshot_study_source.py
"$PYTHON" scripts/verify_artifacts.py
