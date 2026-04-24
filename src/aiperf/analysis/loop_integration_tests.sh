#!/usr/bin/env bash
# Run integration tests one file at a time with pytest-xdist -n auto.
# Stops on the first failing file so it can be fixed, then rerun to continue.
#
# Usage:
#   scripts/loop_integration_tests.sh                # start fresh run
#   scripts/loop_integration_tests.sh --resume       # skip files already marked PASS
#   scripts/loop_integration_tests.sh --from <file>  # start from a specific file
#   scripts/loop_integration_tests.sh --reset        # clear state and start over
#
# State:
#   .loop_integration_state/passed.txt  - one path per line, files that passed
#   .loop_integration_state/last.log    - full output of most recent run
#   .loop_integration_state/<file>.log  - per-file log

set -u
set -o pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STATE_DIR=".loop_integration_state"
PASSED_FILE="$STATE_DIR/passed.txt"
LAST_LOG="$STATE_DIR/last.log"
mkdir -p "$STATE_DIR"
touch "$PASSED_FILE"

RESUME=0
FROM=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume) RESUME=1; shift ;;
        --from) FROM="$2"; shift 2 ;;
        --reset) rm -f "$PASSED_FILE"; touch "$PASSED_FILE"; echo "state reset"; shift ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

mapfile -t ALL_FILES < <(ls tests/integration/test_*.py | sort)

# Activate venv if available (same pattern the Makefile uses).
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1

started=0
for f in "${ALL_FILES[@]}"; do
    if [[ -n "$FROM" && $started -eq 0 ]]; then
        [[ "$f" == *"$FROM"* ]] && started=1 || continue
    fi
    if [[ $RESUME -eq 1 ]] && grep -Fxq "$f" "$PASSED_FILE"; then
        echo "[skip] $f (already passed)"
        continue
    fi

    log="$STATE_DIR/$(basename "$f").log"
    echo
    echo "========================================================================"
    echo "[run ] $f"
    echo "========================================================================"

    # Per-file pytest invocation. Mirrors the make test-integration flags.
    pytest "$f" \
        -m 'integration and not stress and not performance' \
        -n auto \
        --tb=short \
        --no-looptime \
        2>&1 | tee "$log" | tee "$LAST_LOG"
    rc=${PIPESTATUS[0]}

    if [[ $rc -eq 0 ]]; then
        echo "[pass] $f"
        grep -Fxq "$f" "$PASSED_FILE" || echo "$f" >> "$PASSED_FILE"
    elif [[ $rc -eq 5 ]]; then
        # pytest exit 5 = no tests collected (e.g. all deselected by markers)
        echo "[skip] $f (no tests collected)"
        grep -Fxq "$f" "$PASSED_FILE" || echo "$f" >> "$PASSED_FILE"
    else
        echo "[FAIL] $f  (exit $rc)  log=$log"
        exit "$rc"
    fi
done

echo
echo "all integration test files passed"
