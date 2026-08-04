#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Throughput under a NON-EVEN RECYCLE. Runs INSIDE the c4-standard-144 pod.
#
#   BIN=/work/target/release/aiperf TAG=postB RUNS=6 rig-thr-recycle.sh
#
# WHY THIS EXISTS, separate from rig-dispatch-ab.sh:
#
# rig-dispatch-ab.sh runs entries == requests == 200000 -- one pass, NO recycling.
# That config has two problems as a cost measurement for position-addressed
# sampling:
#
#   1. It never exercises the recycle path at all, which is the only place
#      partitioned and full-corpus drawing actually differ.
#   2. It is the WORST CASE for metadata residency. Pre-change each worker held
#      only its 1/W residue (200000/16 = 12500 conversations); post-change every
#      worker holds all 200000. `session()` does a `metadata_by_id` hash lookup
#      PER DRAW, so that config inflates the per-request map from 12.5k to 200k
#      entries -- a cache-locality effect that has nothing to do with recycling.
#
# A small prime corpus with heavy recycling separates the two: the map stays
# tiny (ENTRIES entries either way, so residency is near-identical pre/post),
# while every draw past the first pass goes through the recycle path. ENTRIES is
# prime and does not divide WORKERS, so shards are uneven and their cycles fall
# out of step -- the condition under which the pre-change draw sequence is
# actually wrong.
set -uo pipefail

PORT=${PORT:-8153}
BIN=${BIN:-/work/target/release/aiperf}
MOCK=${MOCK:-/work/target/release/aiperf-mock-server}
TAG=${TAG:-run}
RUNS=${RUNS:-6}
OUT=${OUT:-/work/goal/thr-recycle}
ENTRIES=${ENTRIES:-997}
REQUESTS=${REQUESTS:-200000}
WORKERS=${WORKERS:-16}
TSV=${TSV:-$OUT/thr-recycle-$TAG.tsv}

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

cfg="$OUT/cfg-$TAG.yaml"
cat > "$cfg" <<YAML
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
  workers: $WORKERS
  dispatch: global
YAML

[ -f "$TSV" ] || printf 'tag\trun\trc\trps\tisl\tisl_std\tosl\tcount\terr_pct\tbench_s\n' > "$TSV"

echo "=== throughput under non-even recycle [$TAG] entries=$ENTRIES requests=$REQUESTS workers=$WORKERS ==="
echo "    recycles ~$((REQUESTS / ENTRIES))x; $ENTRIES % $WORKERS = $((ENTRIES % WORKERS)) (uneven shards)"

for r in $(seq 1 "$RUNS"); do
  start_mock || exit 1
  art="$OUT/art-$TAG-r$r"
  rm -rf "$art"
  rc=0
  timeout 1800 "$BIN" profile --config "$cfg" --random-seed 42 \
    --export-level summary --artifact-dir "$art" \
    > "$OUT/run-$TAG-r$r.log" 2>&1 || rc=$?

  python3 - "$TAG" "$r" "$rc" "$art/profile_export_aiperf.json" <<'PY' >> "$TSV"
import json, os, sys
tag, r, rc, path = sys.argv[1:5]
def g(d, k, s="avg"):
    v = d.get(k)
    return v.get(s) if isinstance(v, dict) else None
if not os.path.exists(path):
    print("\t".join([tag, r, rc, "NA","NA","NA","NA","NA","NA","NA"])); raise SystemExit
d = json.load(open(path))
row = [tag, r, rc,
       f"{g(d,'request_throughput'):.0f}",
       f"{g(d,'input_sequence_length'):.2f}",
       f"{g(d,'input_sequence_length','std'):.2f}",
       f"{g(d,'output_sequence_length'):.2f}",
       f"{g(d,'request_count'):.0f}",
       f"{g(d,'request_error_rate'):.2f}",
       f"{g(d,'benchmark_duration'):.2f}"]
print("\t".join(str(x) for x in row))
# An invariant break means the run is not a datapoint, not a slow datapoint.
isl, isl_std, count, err = float(row[4]), float(row[5]), float(row[7]), float(row[8])
if abs(isl - 550.0) > 0.01 or isl_std > 0.01 or count != float(REQ := os.environ.get("REQUESTS", count)) or err > 0.0:
    print(f"  WARN: invariant break on {tag} r{r}: isl={isl} std={isl_std} count={count} err={err}", file=sys.stderr)
PY
done

echo "--- $TSV ---"
cat "$TSV"
