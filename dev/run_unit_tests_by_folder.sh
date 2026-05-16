#!/usr/bin/env bash
# Run unit tests folder-by-folder to isolate leaks/hangs.
# Each folder gets pytest --timeout=2 -n auto, with a bash-level 30s kill.

set -uo pipefail

TESTS_DIR="tests/unit"
FAILED=()
PASSED=()
TIMED_OUT=()

# Collect top-level entries (directories + standalone .py files)
entries=()
for entry in "$TESTS_DIR"/*/; do
    [ -d "$entry" ] && [[ "$(basename "$entry")" != __pycache__ ]] && entries+=("${entry%/}")
done
for entry in "$TESTS_DIR"/test_*.py; do
    [ -f "$entry" ] && entries+=("$entry")
done

for entry in "${entries[@]}"; do
    name=$(basename "$entry")
    echo "========================================"
    echo "Running: $entry"
    echo "========================================"
    timeout 30 uv run pytest "$entry" -n auto --timeout=2 -x 2>&1
    rc=$?
    if [ $rc -eq 124 ]; then
        echo "TIMED OUT (bash 30s): $entry"
        TIMED_OUT+=("$name")
    elif [ $rc -ne 0 ]; then
        echo "FAILED: $entry (exit $rc)"
        FAILED+=("$name")
    else
        PASSED+=("$name")
    fi
    echo ""
done

echo "========================================"
echo "SUMMARY"
echo "========================================"
echo "Passed:    ${#PASSED[@]}"
echo "Failed:    ${#FAILED[@]}"
echo "Timed out: ${#TIMED_OUT[@]}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
fi
if [ ${#TIMED_OUT[@]} -gt 0 ]; then
    echo ""
    echo "Timed out:"
    for t in "${TIMED_OUT[@]}"; do echo "  - $t"; done
fi

[ ${#FAILED[@]} -eq 0 ] && [ ${#TIMED_OUT[@]} -eq 0 ]
