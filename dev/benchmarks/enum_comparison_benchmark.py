#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Microbenchmark: CaseInsensitiveStrEnum comparison vs identity/plain-string.

Measures the cost of SSEFieldType/SSEEventType comparisons that happen on
every SSE packet during streaming. The hot path was:

    packet.name == SSEFieldType.EVENT   # triggers _normalize_name() x2
    packet.value == SSEEventType.ERROR  # triggers _normalize_name() x2

After the fix:
    packet.name is SSEFieldType.EVENT   # identity check
    packet.value == "error"             # plain str == str

Usage:
    python dev/benchmarks/enum_comparison_benchmark.py
"""

import sys
import timeit
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from aiperf.common.enums import SSEEventType, SSEFieldType
from aiperf.common.enums.base_enums import _normalize_name

# ---------------------------------------------------------------------------
# Setup: simulate the SSE packet field values
# ---------------------------------------------------------------------------

# After the fix, SSEField.name stores the enum member directly
enum_name = SSEFieldType.DATA
enum_event = SSEFieldType.EVENT
enum_comment = SSEFieldType.COMMENT

# Before the fix, SSEField.name was a plain string from field_name.strip()
str_name_data = "data"
str_name_event = "event"

# SSEEventType comparison target
str_value_error = "error"

# Number of iterations per benchmark
N = 1_000_000
REPEAT = 5

# ---------------------------------------------------------------------------
# Benchmark functions
# ---------------------------------------------------------------------------


def bench_old_name_eq_match():
    """OLD: str == CaseInsensitiveStrEnum (match) -- triggers _normalize_name x2."""
    _ = str_name_data == SSEFieldType.DATA


def bench_old_name_eq_miss():
    """OLD: str == CaseInsensitiveStrEnum (miss) -- triggers _normalize_name x2."""
    _ = str_name_data == SSEFieldType.EVENT


def bench_new_name_is_match():
    """NEW: enum is enum (match) -- identity check."""
    _ = enum_name is SSEFieldType.DATA


def bench_new_name_is_miss():
    """NEW: enum is enum (miss) -- identity check."""
    _ = enum_name is SSEFieldType.EVENT


def bench_old_value_eq():
    """OLD: str == CaseInsensitiveStrEnum -- triggers _normalize_name x2."""
    _ = str_value_error == SSEEventType.ERROR


def bench_new_value_eq():
    """NEW: str == str -- plain string comparison."""
    _ = str_value_error == "error"


def bench_normalize_name():
    """Cost of a single _normalize_name() call."""
    _ = _normalize_name("data")


def bench_old_inspect_loop():
    """OLD: simulate inspect_message_for_error inner loop (3 packets, typical)."""
    packets = [
        (str_name_data, "chunk1"),
        (str_name_data, "chunk2"),
        (str_name_event, str_value_error),
    ]
    _ = any(
        name == SSEFieldType.EVENT and val == SSEEventType.ERROR
        for name, val in packets
    )


def bench_new_inspect_loop():
    """NEW: simulate inspect_message_for_error inner loop (3 packets, typical)."""
    packets = [
        (SSEFieldType.DATA, "chunk1"),
        (SSEFieldType.DATA, "chunk2"),
        (SSEFieldType.EVENT, "error"),
    ]
    _ = any(name is SSEFieldType.EVENT and val == "error" for name, val in packets)


def bench_old_extract_data():
    """OLD: simulate extract_data_content loop (5 data packets, typical)."""
    packets = [
        (str_name_data, "v1"),
        (str_name_data, "v2"),
        (str_name_data, "v3"),
        (str_name_data, "v4"),
        (str_name_data, "v5"),
    ]
    _ = [v for n, v in packets if n == SSEFieldType.DATA and v]


def bench_new_extract_data():
    """NEW: simulate extract_data_content loop (5 data packets, typical)."""
    packets = [
        (SSEFieldType.DATA, "v1"),
        (SSEFieldType.DATA, "v2"),
        (SSEFieldType.DATA, "v3"),
        (SSEFieldType.DATA, "v4"),
        (SSEFieldType.DATA, "v5"),
    ]
    _ = [v for n, v in packets if n is SSEFieldType.DATA and v]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("--- Single comparison: name field ---", None),
    ("OLD: str == SSEFieldType (match)", bench_old_name_eq_match),
    ("OLD: str == SSEFieldType (miss)", bench_old_name_eq_miss),
    ("NEW: enum is SSEFieldType (match)", bench_new_name_is_match),
    ("NEW: enum is SSEFieldType (miss)", bench_new_name_is_miss),
    ("", None),
    ("--- Single comparison: value field ---", None),
    ("OLD: str == SSEEventType", bench_old_value_eq),
    ("NEW: str == 'error'", bench_new_value_eq),
    ("", None),
    ("--- _normalize_name cost ---", None),
    ("_normalize_name('data')", bench_normalize_name),
    ("", None),
    ("--- Realistic: inspect_message_for_error (3 packets) ---", None),
    ("OLD: any(name == enum and val == enum)", bench_old_inspect_loop),
    ("NEW: any(name is enum and val == str)", bench_new_inspect_loop),
    ("", None),
    ("--- Realistic: extract_data_content (5 packets) ---", None),
    ("OLD: [v for n,v if n == enum]", bench_old_extract_data),
    ("NEW: [v for n,v if n is enum]", bench_new_extract_data),
]


def run_benchmarks() -> None:
    print(f"Iterations: {N:,} x {REPEAT} repeats (best-of-{REPEAT})")
    print(f"{'Benchmark':<50} {'ns/op':>8} {'speedup':>10}")
    print("-" * 70)

    prev_old_ns = None
    for label, func in BENCHMARKS:
        if func is None:
            if label:
                print(f"\n{label}")
            prev_old_ns = None
            continue

        times = timeit.repeat(func, number=N, repeat=REPEAT)
        best = min(times)
        ns_per_op = best / N * 1e9

        speedup = ""
        if "OLD:" in label:
            prev_old_ns = ns_per_op
        elif "NEW:" in label and prev_old_ns is not None:
            ratio = prev_old_ns / ns_per_op
            speedup = f"{ratio:.1f}x"

        print(f"  {label:<48} {ns_per_op:>7.1f} {speedup:>10}")


if __name__ == "__main__":
    run_benchmarks()
