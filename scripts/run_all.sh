#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -m pickplace.cli collect
"$PYTHON" -m pickplace.cli train --episodes 32 --output artifacts/checkpoints/bc32.pt
"$PYTHON" -m pickplace.cli train
"$PYTHON" -m pickplace.cli dagger
"$PYTHON" -m pickplace.cli evaluate
"$PYTHON" -m pickplace.cli report
"$PYTHON" -m pickplace.cli render
"$PYTHON" -m pytest -q
