#!/usr/bin/env bash
# Regenerate the HTML API documentation (docs/) and the pytest report (tests/test_report.txt).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
pdoc -o docs occupancy
python -m pytest -v | tee tests/test_report.txt
