# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-exact replay verification: compare profile_export.jsonl ISL/OSL to
recorded weka trace in/out per turn.

This script is the manual receipt for the byte-exact replay claim:
  - OSL exact match: aiperf's per-record output_sequence_length must equal
    the recorded ``out[k]`` of the matching turn (Turn.max_tokens enforces
    this at loader emit time via ``_cap_output``).
  - ISL drift bound: aiperf's per-record input_sequence_length must be
    within ``MAX_TOKENIZER_DIVERGENCE_PER_MSG * n_msgs_in_turn`` of the
    recorded ``in[k]``. Post-P17 the reconstructor guarantees
    ``sum(len(seg.tokens)) == in[k]`` exactly per turn (block-aligned
    segment sizes, no terminator stamp); the residual bound only absorbs
    BPE-on-join residual at segment seams and cross-tokenizer translation
    differences (Claude tokenizer ↔ Qwen). Per-message empirical max is
    0.96 on the kv-cache-tester corpus; the bound is set to 3 with margin.

Usage:
  python tools/weka_byte_exact_verify.py \\
      --traces  /path/to/trace/dir \\
      --profile /path/to/profile_export.jsonl

Exit 0 if all checks pass, 1 if any drift bound is violated, 2 if any OSL
mismatch is found.

Reproduces the verification half of plan task P11 (end-to-end mock-server
replay sanity); see ``docs/tutorials/weka-byte-exact-replay-results.md`` for
the empirical numbers measured against the 8-trace subset on the AIPerf
mock server.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

MAX_TOKENIZER_DIVERGENCE_PER_MSG = 3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--traces",
        type=Path,
        required=True,
        help="Directory containing the original .json trace files.",
    )
    p.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="Path to aiperf's profile_export.jsonl (one record per request).",
    )
    p.add_argument(
        "--per-msg-bound",
        type=int,
        default=MAX_TOKENIZER_DIVERGENCE_PER_MSG,
        help=(
            f"Per-message ISL drift tolerance. Default {MAX_TOKENIZER_DIVERGENCE_PER_MSG}. "
            "Absorbs cross-tokenizer translation residual + chat-template overhead."
        ),
    )
    return p.parse_args()


def grab(field_value: dict | int | float | None) -> int | float | None:
    """Pull the actual numeric value out of an aiperf metric dict."""
    if isinstance(field_value, dict):
        return field_value.get("value", field_value.get("avg"))
    return field_value


def turn_index(rec: dict) -> int:
    """Turn index — may live at root level or under metadata depending on schema version."""
    if "turn_index" in rec:
        return rec["turn_index"]
    return rec.get("metadata", {}).get("turn_index", 0)


def load_recorded(traces_dir: Path) -> dict[str, tuple[list[int], list[int], int]]:
    """Return ``{trace_id: (in_per_turn, out_per_turn, n_normal_turns)}``."""
    recorded: dict[str, tuple[list[int], list[int], int]] = {}
    for path in sorted(traces_dir.glob("*.json")):
        blob = json.loads(path.read_text())
        ins, outs = [], []
        for r in blob["requests"]:
            if r.get("type") in ("n", "s"):
                ins.append(r["in"])
                outs.append(r["out"])
        recorded[blob["id"]] = (ins, outs, len(ins))
    return recorded


def main() -> int:
    args = parse_args()

    profile_lines = [
        json.loads(line)
        for line in args.profile.read_text().splitlines()
        if line.strip()
    ]
    print(f"records: {len(profile_lines)}")

    by_conv: dict[str, list[dict]] = defaultdict(list)
    for r in profile_lines:
        cid = r.get("conversation_id") or r.get("metadata", {}).get(
            "conversation_id", "?"
        )
        by_conv[cid].append(r)

    recorded = load_recorded(args.traces)

    isl_drift: list[tuple[int, str, int, int, int, int]] = []
    osl_match = 0
    osl_total = 0
    osl_mismatches: list[tuple[str, int, int, int]] = []
    errors = 0

    isl_actual_all: list[int] = []
    osl_actual_all: list[int] = []
    isl_recorded_all: list[int] = []
    osl_recorded_all: list[int] = []

    for conv_id, recs in by_conv.items():
        if conv_id not in recorded:
            continue
        ins, outs, n_turns = recorded[conv_id]
        recs.sort(key=turn_index)

        for k, rec in enumerate(recs):
            if k >= len(ins):
                break
            metrics = rec.get("metrics", rec)
            actual_isl = grab(metrics.get("input_sequence_length"))
            actual_osl = grab(metrics.get("output_sequence_length"))
            if actual_isl is None or actual_osl is None:
                continue

            recorded_in = ins[k]
            recorded_out = outs[k]

            # n_msgs in raw_messages: turn 0 = system + user (≤2);
            # turn k = system + user_0 + (asst_i + user_{i+1}) for i in 0..k-1
            #        = 2 + 2*k. Conservative upper bound for non-template-corrected
            # bound; a more precise count would inspect inputs.json per turn.
            n_msgs_bound = 2 if k == 0 else 2 + 2 * k
            bound = args.per_msg_bound * n_msgs_bound
            drift = abs(actual_isl - recorded_in)

            isl_drift.append((drift, conv_id, k, int(actual_isl), recorded_in, bound))
            osl_total += 1
            if actual_osl == recorded_out:
                osl_match += 1
            else:
                osl_mismatches.append((conv_id, k, int(actual_osl), recorded_out))

            isl_actual_all.append(int(actual_isl))
            osl_actual_all.append(int(actual_osl))
            isl_recorded_all.append(recorded_in)
            osl_recorded_all.append(recorded_out)

            meta = rec.get("metadata", {})
            if meta.get("was_cancelled") or rec.get("was_cancelled"):
                errors += 1

    if not isl_drift:
        print("no comparable records — recorded trace ids not found in profile")
        return 1

    print(
        f"\nISL drift: median={statistics.median(d[0] for d in isl_drift)}, "
        f"mean={statistics.mean(d[0] for d in isl_drift):.2f}, "
        f"max={max(d[0] for d in isl_drift)}"
    )
    print(f"OSL match: {osl_match} / {osl_total}")
    print(f"errors / cancellations: {errors}")

    print("\n--- Aggregate stats ---")
    print(
        f"ISL actual:   avg={statistics.mean(isl_actual_all):.1f}  "
        f"min={min(isl_actual_all)}  max={max(isl_actual_all)}"
    )
    print(
        f"ISL recorded: avg={statistics.mean(isl_recorded_all):.1f}  "
        f"min={min(isl_recorded_all)}  max={max(isl_recorded_all)}"
    )
    print(
        f"OSL actual:   avg={statistics.mean(osl_actual_all):.1f}  "
        f"min={min(osl_actual_all)}  max={max(osl_actual_all)}"
    )
    print(
        f"OSL recorded: avg={statistics.mean(osl_recorded_all):.1f}  "
        f"min={min(osl_recorded_all)}  max={max(osl_recorded_all)}"
    )

    violations = [d for d in isl_drift if d[0] > d[5]]
    print()
    if violations:
        print(f"BOUND VIOLATIONS ({len(violations)}):")
        for v in violations[:10]:
            print(
                f"  {v[1]} turn {v[2]}: drift={v[0]} bound={v[5]} "
                f"(actual={v[3]}, recorded={v[4]})"
            )
    else:
        print("All within bound — byte-exact contract holds.")

    print("\n--- Worst 10 ISL drift (informational) ---")
    for v in sorted(isl_drift, key=lambda x: -x[0])[:10]:
        print(
            f"  {v[1]} turn {v[2]}: drift={v[0]} bound={v[5]} "
            f"(actual={v[3]}, recorded={v[4]})"
        )

    if osl_mismatches:
        print(f"\nOSL MISMATCHES ({len(osl_mismatches)}):")
        for m in osl_mismatches[:10]:
            print(f"  {m[0]} turn {m[1]}: actual={m[2]} recorded={m[3]}")
        return 2

    if violations:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
