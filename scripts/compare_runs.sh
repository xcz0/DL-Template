#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/compare_runs.sh
#   scripts/compare_runs.sh --glob "outputs/*/*" --write reports/summary.json --write-csv reports/summary.csv

exec uv run python src/tools/compare_runs.py "$@"
