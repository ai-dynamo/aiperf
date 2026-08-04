#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Proof that `global` at workers>1 draws the single issuer's conversation
# multiset. Runs INSIDE the c4-standard-144 pod.
#
#   BIN=/work/target/release/aiperf TAG=postB rig-corpus-proof.sh
#
# WHY A SEPARATE CONFIG FROM rig-dispatch-ab.sh: that config is
# `entries == requests == 200000`, i.e. exactly one pass over the corpus with
# NO recycling. Partitioned and full-corpus sampling agree in that degenerate
# case, so the throughput config cannot show this defect at all. Divergence
# needs recycling (requests > entries) with a corpus size that does not divide
# the worker count -- 1000 conversations over 16 workers gives shards of 63 and
# 62, which wrap at different points.
#
# Expected per-conversation draw counts at entries=1000, requests=20000:
#   single issuer (workers=1) : exactly 20 for every conversation
#   partitioned (pre-change)  : spread (a 63-conversation shard draws 1250/63,
#                               a 62-conversation shard draws 1250/62)
#   position-addressed (post) : exactly 20 -- identical to the single issuer
set -euo pipefail

PORT=${PORT:-8152}
BIN=${BIN:-/work/target/release/aiperf}
MOCK=${MOCK:-/work/target/release/aiperf-mock-server}
TAG=${TAG:-run}
OUT=${OUT:-/work/goal/corpus-proof}
ENTRIES=${ENTRIES:-1000}
REQUESTS=${REQUESTS:-20000}
WORKERS=${WORKERS:-16}

export HF_HOME=/work/hf50 HF_HUB_OFFLINE=1 NO_PROXY=127.0.0.1,localhost
mkdir -p "$OUT"

mock_up() { curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }
start_mock() {
  if mock_up; then return 0; fi
  setsid nohup "$MOCK" --port $PORT --fast --random-seed 42 \
    > "$OUT/mock.log" 2>&1 < /dev/null & disown
  for _ in $(seq 1 30); do mock_up && return 0; sleep 1; done
  echo "FATAL: mock did not come up" >&2; return 1
}

write_cfg() { # $1 = workers, $2 = dispatch, $3 = path
  cat > "$3" <<YAML
schemaVersion: "2.0"
benchmark:
  model: meta-llama/Llama-3.1-8B-Instruct
  endpoint:
    url: http://127.0.0.1:$PORT
    type: chat
    streaming: true
  dataset:
    type: synthetic
    entries: $ENTRIES
    prompts:
      isl: 550
      osl: 1
  phases:
    type: concurrency
    concurrency: 512
    requests: $REQUESTS
runtime:
  workers: $1
  dispatch: $2
YAML
}

# Per-conversation draw counts, as "<count> <conversation_id>" sorted by id.
histogram() {
  python3 - "$1" <<'PY'
import json, sys, collections
counts = collections.Counter()
with open(sys.argv[1]) as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        if str(meta.get("benchmark_phase", "")).lower() == "warmup":
            continue
        cid = meta.get("conversation_id")
        if cid is None:
            print("FATAL: record without conversation_id", file=sys.stderr)
            raise SystemExit(1)
        counts[cid] += 1
for cid in sorted(counts):
    print(f"{counts[cid]} {cid}")
PY
}

run_one() { # $1 = label, $2 = workers, $3 = dispatch
  local art="$OUT/art-$TAG-$1"
  rm -rf "$art"
  start_mock || exit 1
  write_cfg "$2" "$3" "$OUT/cfg-$TAG-$1.yaml"
  # `|| rc=$?` rather than a bare `$?` capture: the profile run's non-zero exit
  # is inspected below, and this keeps that intentional under `set -e` without
  # depending on the caller invoking `run_one` in a condition context.
  local rc=0
  timeout 1800 "$BIN" profile --config "$OUT/cfg-$TAG-$1.yaml" --random-seed 42 \
    --export-level records --artifact-dir "$art" > "$OUT/run-$TAG-$1.log" 2>&1 || rc=$?
  if [ $rc -ne 0 ]; then
    echo "FATAL: $1 exited $rc; see $OUT/run-$TAG-$1.log" >&2
    tail -20 "$OUT/run-$TAG-$1.log" >&2
    return 1
  fi
  histogram "$art/profile_export.jsonl" > "$OUT/hist-$TAG-$1.txt" || return 1
  local total distinct lo hi
  total=$(awk '{s+=$1} END{print s}' "$OUT/hist-$TAG-$1.txt")
  distinct=$(wc -l < "$OUT/hist-$TAG-$1.txt")
  lo=$(awk 'NR==1{m=$1} $1<m{m=$1} END{print m}' "$OUT/hist-$TAG-$1.txt")
  hi=$(awk 'NR==1{m=$1} $1>m{m=$1} END{print m}' "$OUT/hist-$TAG-$1.txt")
  printf '%-28s records=%-7s conversations=%-6s per-conversation min=%-4s max=%s\n' \
    "$1" "$total" "$distinct" "$lo" "$hi"
}

echo "=== corpus proof [$TAG] entries=$ENTRIES requests=$REQUESTS workers=$WORKERS ==="
run_one "single-issuer" 1 global || exit 1
run_one "global-w$WORKERS" "$WORKERS" global || exit 1

echo
if diff -q "$OUT/hist-$TAG-single-issuer.txt" "$OUT/hist-$TAG-global-w$WORKERS.txt" >/dev/null; then
  echo "RESULT [$TAG]: MATCH - global w$WORKERS draws the single issuer's conversation multiset"
else
  echo "RESULT [$TAG]: DIVERGENT - global w$WORKERS differs from the single issuer"
  echo "  conversations whose draw count differs:"
  # `diff` exits 1 here BY DEFINITION (this is the divergent branch), and
  # `pipefail` propagates that through the pipe, so both reports need `|| true`
  # to survive `set -e` -- without it the script would abort at the exact moment
  # it is supposed to explain the divergence.
  diff "$OUT/hist-$TAG-single-issuer.txt" "$OUT/hist-$TAG-global-w$WORKERS.txt" \
    | grep -c '^[<>]' | xargs printf '    %s differing lines\n' || true
  diff "$OUT/hist-$TAG-single-issuer.txt" "$OUT/hist-$TAG-global-w$WORKERS.txt" | head -10 || true
fi
