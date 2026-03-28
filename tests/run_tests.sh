#!/usr/bin/env bash
set -euo pipefail

if ! python -c "import xdist" >/dev/null 2>&1; then
  echo "pytest-xdist is required to run this script. Install it with: python -m pip install pytest-xdist" >&2
  exit 1
fi

python -m pytest -n auto "$@"
