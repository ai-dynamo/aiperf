#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dispatch-mode A/B at OSL 1 on the c4-standard-144 rig. Runs INSIDE the pod.
#
#   MODES="sharded global-hop global-push" RUNS=2 rig-dispatch-ab.sh
#
# OSL 1 is deliberate: at OSL 150 the mock burns ~130 of 144 cores generating
# SSE frames and caps every client at ~23.5k rps, which hides the coordinator
# cost entirely. Neither side is pinned -- CPUs 72-143 are the SMT siblings of
# 0-71 on this box, so a naive `taskset` split silently overlaps physical cores.
set -uo pipefail

PORT=${PORT:-8151}
RUNS=${RUNS:-2}
MODES=${MODES:-"sharded global global-hop global-push"}
BIN=${BIN:-/work/target/release/aiperf}
MOCK=${MOCK:-/work/target/release/aiperf-mock-server}
TSV=${TSV:-/work/goal/dispatch-o1.tsv}

export HF_HOME=/work/hf50 HF_HUB_OFFLINE=1 NO_PROXY=127.0.0.1,localhost

mock_up() { curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; }

start_mock() {
  if mock_up; then return 0; fi
  setsid nohup "$MOCK" --port $PORT --fast --random-seed 42 \
    > /work/goal/mock.log 2>&1 < /dev/null & disown
  for _ in $(seq 1 30); do mock_up && return 0; sleep 1; done
  echo "FATAL: mock did not come up" >&2; return 1
}

[ -f "$TSV" ] || printf 'mode\trun\trc\trps\tisl\tisl_std\tosl\tcount\terr_pct\tbench_s\n' > "$TSV"

for MODE in $MODES; do
  cfg="/work/goal/ap-o1-$MODE.yaml"
  sed "s/^  dispatch: .*/  dispatch: $MODE/" /work/goal/ap-o1.yaml > "$cfg"
  grep -q "dispatch: $MODE" "$cfg" || { echo "  dispatch: $MODE" >> "$cfg"; }
  grep -q "dispatch: $MODE" "$cfg" || { echo "FATAL: dispatch not set for $MODE"; exit 1; }

  for r in $(seq 1 "$RUNS"); do
    start_mock || exit 1
    art="/work/goal/art-o1-$MODE-r$r"
    rm -rf "$art"
    timeout 1200 "$BIN" profile --config "$cfg" --random-seed 42 \
      --export-level summary --artifact-dir "$art" \
      > "/work/goal/run-o1-$MODE-r$r.log" 2>&1
    rc=$?
    mock_up || echo "WARN: mock died during $MODE r$r" >&2

    python3 - "$MODE" "$r" "$rc" "$art/profile_export_aiperf.json" <<'PY' >> "$TSV"
import json, os, sys
mode, r, rc, path = sys.argv[1:5]
def g(d, k, s="avg"):
    v = d.get(k)
    return v.get(s) if isinstance(v, dict) else None
if not os.path.exists(path):
    print("\t".join([mode, r, rc, "NA","NA","NA","NA","NA","NA","NA"])); raise SystemExit
d = json.load(open(path))
print("\t".join(str(x) for x in [
    mode, r, rc,
    f"{g(d,'request_throughput'):.0f}",
    f"{g(d,'input_sequence_length'):.2f}",
    f"{g(d,'input_sequence_length','std'):.2f}",
    f"{g(d,'output_sequence_length'):.2f}",
    f"{g(d,'request_count'):.0f}",
    f"{g(d,'request_error_rate'):.2f}",
    f"{g(d,'benchmark_duration'):.2f}"]))
PY
    tail -1 "$TSV"

    # A run that broke an invariant is not a datapoint.
    python3 - "$TSV" <<'PY'
import sys
last = open(sys.argv[1]).read().strip().split("\n")[-1].split("\t")
if last[3] == "NA":
    print("  !! NO RESULTS"); raise SystemExit
_, _, rc, rps, isl, isl_std, osl, count, err, *_ = last
bad = []
if rc != "0":          bad.append(f"rc={rc}")
if isl != "550.00":    bad.append(f"ISL={isl}")
if isl_std != "0.00":  bad.append(f"ISL_std={isl_std}")
if osl != "1.00":      bad.append(f"OSL={osl}")
if count != "200000":  bad.append(f"count={count}")
if err != "0.00":      bad.append(f"err={err}%")
print("  !! INVALID: " + ", ".join(bad) if bad else "  ok (invariants hold)")
PY
    find "$art" -name "profile_export.jsonl" -delete 2>/dev/null
  done
done
echo "DISPATCH_AB_DONE"
cat "$TSV"
