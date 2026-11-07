#!/usr/bin/env bash
# Runs the full test-suite: Python (pytest: ops, OpInfo forward + gradients,
# runtime API) and, unless SKIP_CTESTS=1, the C++ gtest suite.
set -euo pipefail
cd "$(dirname "$0")"
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

python -m pytest tests -q "$@"

if [ "${SKIP_CTESTS:-0}" != "1" ]; then
  ./build-ctests.sh > /dev/null
  ./run-ctests.sh
fi
