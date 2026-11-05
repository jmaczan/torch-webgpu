#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
ctest --test-dir build/ctests --output-on-failure -j "${CTEST_JOBS:-4}" "$@"
