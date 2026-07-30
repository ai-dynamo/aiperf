#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gate dry-run recorded-latency timing parity against WEKA corpus api_time.

Graph-IR exports share one trajectory ``conversation_id`` across parent and
subagent chains, so per-(conversation, turn) joins are ambiguous. Instead this
compares the **multiset** of ``request_latency`` (ms) values in
``profile_export.jsonl`` to the multiset of expected latencies derived from
every WEKA leaf ``api_time`` in the selected oracle traces — the same µs
round-trip the graph-IR path uses.

Example::

    python3 tools/weka_timing_parity_check.py \\
        --records artifacts/weka-timing-parity/phase1-raw/profile_export.jsonl \\
        --hf-dataset semianalysisai/cc-traces-weka-062126 \\
        --first-n-traces 393
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def _round_ties_even_us(seconds: float) -> int:
    """Match Rust ``(seconds * 1_000_000.0).round_ties_even()`` for finite values."""
    scaled = seconds * 1_000_000.0
    if not math.isfinite(scaled) or scaled < 0.0 or scaled >= float(2**64):
        return 0
    return int(round(scaled))


def expected_latency_ms(api_time_s: float) -> float:
    api_time_us = _round_ties_even_us(api_time_s)
    return (api_time_us * 1000) / 1_000_000.0


def _flatten_entries(
    entries: list[Any],
    scope: str,
    out: dict[tuple[str, int], float],
) -> None:
    """Mirror ``flatten_entries`` in ``graph/recorded/weka/mod.rs``."""
    turn_index = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        etype = entry.get("type")
        if etype == "subagent":
            _flatten_entries(entry.get("requests") or [], entry["agent_id"], out)
            continue
        if etype in ("n", "s"):
            api_time = entry.get("api_time")
            out[(scope, turn_index)] = 0.0 if api_time is None else float(api_time)
            turn_index += 1
            continue
        raise SystemExit(f"unrecognized WEKA entry type {etype!r} in scope {scope!r}")


def flatten_weka_trace(trace: dict[str, Any]) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    _flatten_entries(trace.get("requests") or [], trace["id"], out)
    return out


def load_traces_from_hf(dataset_id: str, split: str | None) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {}
    if split:
        kwargs["split"] = split
    ds = load_dataset(dataset_id, **kwargs)
    if hasattr(ds, "keys"):
        first = split or next(iter(ds.keys()))
        ds = ds[first]
    traces: list[dict[str, Any]] = []
    for row in ds:
        if "id" in row and "requests" in row:
            traces.append(dict(row))
        else:
            raise SystemExit(f"unrecognized WEKA HF row keys: {list(row.keys())[:20]}")
    return traces


def load_traces_from_jsonl(path: Path) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traces.append(json.loads(line))
    return traces


def oracle_latency_counter(traces: list[dict[str, Any]]) -> Counter[float]:
    counts: Counter[float] = Counter()
    for trace in traces:
        for api_s in flatten_weka_trace(trace).values():
            counts[round(expected_latency_ms(api_s), 6)] += 1
    return counts


def record_latency_counter(records_path: Path) -> tuple[Counter[float], int]:
    counts: Counter[float] = Counter()
    skipped_error = 0
    with records_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error") is not None:
                skipped_error += 1
                continue
            cell = (rec.get("metrics") or {}).get("request_latency")
            if cell is None:
                raise SystemExit("record missing request_latency")
            value = cell.get("value") if isinstance(cell, dict) else cell
            if value is None:
                raise SystemExit("record request_latency has no value")
            counts[round(float(value), 6)] += 1
    return counts, skipped_error


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", type=Path, required=True, help="profile_export.jsonl")
    p.add_argument(
        "--hf-dataset",
        default="semianalysisai/cc-traces-weka-062126",
        help="HuggingFace WEKA corpus id",
    )
    p.add_argument("--hf-split", default="train", help="HF split name")
    p.add_argument(
        "--oracle-jsonl",
        type=Path,
        default=None,
        help="Local WEKA JSONL oracle (skips HF download)",
    )
    p.add_argument(
        "--first-n-traces",
        type=int,
        default=None,
        help="Use only the first N traces in HF/jsonl order (matches sequential selection)",
    )
    args = p.parse_args()

    if not args.records.is_file():
        print(f"records not found: {args.records}", file=sys.stderr)
        return 2

    traces = (
        load_traces_from_jsonl(args.oracle_jsonl)
        if args.oracle_jsonl is not None
        else load_traces_from_hf(args.hf_dataset, args.hf_split)
    )
    if args.first_n_traces is not None:
        traces = traces[: args.first_n_traces]

    expected = oracle_latency_counter(traces)
    got, skipped_error = record_latency_counter(args.records)

    print(f"oracle_traces={len(traces)}")
    print(f"oracle_leaves={sum(expected.values())}")
    print(f"record_latencies={sum(got.values())}")
    print(f"skipped_error={skipped_error}")
    print(f"multiset_equal={got == expected}")

    if got == expected and sum(got.values()) > 0:
        print("PASS")
        return 0

    only_got = got - expected
    only_exp = expected - got
    print(f"only_in_records={len(only_got)}", file=sys.stderr)
    for latency, count in list(only_got.items())[:20]:
        print(f"  records-only latency_ms={latency} count={count}", file=sys.stderr)
    print(f"only_in_oracle={len(only_exp)}", file=sys.stderr)
    for latency, count in list(only_exp.items())[:20]:
        print(f"  oracle-only latency_ms={latency} count={count}", file=sys.stderr)
    print("FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
