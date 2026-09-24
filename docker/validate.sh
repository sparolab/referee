#!/bin/bash
# Runs every check the repository ships.
set -uo pipefail

CORE=${REFEREE_ROOT:-/referee}/cpp/referee
BIN=${REFEREE_CORE_BUILD:-/opt/referee/core-build}

failures=0
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

step "C++ descriptors vs the published implementation"
"$BIN/validate_descriptor" "$CORE/examples" "$CORE/tools/reference/published_descriptors.txt"
[ $? -eq 0 ] || failures=$((failures+1))

step "Python package vs the same reference"
cd "${REFEREE_ROOT:-/referee}/python" && python3 -m pytest tests/ -q
[ $? -eq 0 ] || failures=$((failures+1))

printf '\n'
if [ "$failures" -eq 0 ]; then echo "all checks passed"; else echo "$failures check(s) failed"; fi
exit $((failures == 0 ? 0 : 1))
