# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author the golden `t*` parity vectors (Python numpy is authoritative).

Python numpy is the fixed reference the Rust `aiperf::graph::tstar` port must
match. This script recomputes `t*` with the EXACT agentx logic ported from
`src/aiperf/timing/graph_ir_source.py:113-150` (`_sample_t_star` and
`_seed_for_trace_lane`, viewed via `git show ajc/aiperf-graph-ir:...`) plus the
`_plan_trace` `max(t_star, 0.0)` clamp, and emits a committed JSON grid that
`rust/aiperf/tests/tstar_parity.rs` replays and asserts bit-exact.

`t_star_us` is stored as its IEEE-754 f64 bit pattern (`t_star_us_bits`, a u64)
to avoid decimal round-trip drift; the Rust side compares `got.to_bits()`.

Requires numpy 2.5.1 (the version the Rust `NumpyPcg64` port is pinned to).

Regenerate (run from the repo root):

    python tools/gen_tstar_parity_vectors.py

which overwrites `rust/aiperf/tests/data/tstar_parity_vectors.json`.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

# The grid, straight from the task brief.
BASE_SEEDS = [0, 1, 42]
TRACE_IDS = ["t-0", "t-1#0", "conv-9"]
LANES = [0, 1, 5]
WINDOWS = [(0.0, 0.0), (0.0, 1.0), (0.25, 0.75), (1.0, 1.0), (0.5, 0.5)]
DURATIONS_US = [0.0, 1_000.0, 5_000_000.0]

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "rust"
    / "aiperf"
    / "tests"
    / "data"
    / "tstar_parity_vectors.json"
)


def _seed_for_trace_lane(base_seed: int, trace_id: str, lane_index: int) -> int:
    """agentx `_seed_for_trace_lane` (graph_ir_source.py:138-150).

    SHA-256 the ASCII `"{base_seed}:{trace_id}:{lane_index}"` and take the low 8
    bytes big-endian.
    """
    digest = hashlib.sha256(f"{base_seed}:{trace_id}:{lane_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _sample_t_star(
    base_seed: int,
    trace_id: str,
    lane_index: int,
    start_min_ratio: float,
    start_max_ratio: float,
    duration_us: float,
) -> float:
    """agentx `_sample_t_star` (graph_ir_source.py:113-138) + `_plan_trace` clamp.

    `t* = uniform(lo, hi)` over `[start_min_ratio, start_max_ratio] * duration`
    with a lane-salted `np.random.default_rng` seed. A non-positive duration or a
    collapsed window (`hi <= lo`) draws nothing. The final `max(_, 0.0)` mirrors
    `_plan_trace` (graph_ir_source.py:107).
    """
    if duration_us <= 0:
        return 0.0
    lo = start_min_ratio * duration_us
    hi = start_max_ratio * duration_us
    if hi <= lo:
        t_star = float(lo)
    else:
        rng = np.random.default_rng(
            _seed_for_trace_lane(base_seed, trace_id, lane_index)
        )
        t_star = float(rng.uniform(lo, hi))
    # graph_ir_source.py:_plan_trace -> max(t_star_us, 0.0)
    return max(t_star, 0.0)


def _bits(value: float) -> int:
    """f64 -> its little-endian IEEE-754 u64 bit pattern."""
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def main() -> None:
    rows = []
    for base_seed in BASE_SEEDS:
        for trace_id in TRACE_IDS:
            for lane in LANES:
                for (lo_ratio, hi_ratio) in WINDOWS:
                    for duration_us in DURATIONS_US:
                        t_star = _sample_t_star(
                            base_seed,
                            trace_id,
                            lane,
                            lo_ratio,
                            hi_ratio,
                            duration_us,
                        )
                        rows.append(
                            {
                                "base_seed": base_seed,
                                "trace_id": trace_id,
                                "lane": lane,
                                "min": lo_ratio,
                                "max": hi_ratio,
                                "duration_us": duration_us,
                                "t_star_us_bits": _bits(t_star),
                            }
                        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
        fh.write("\n")
    print(f"wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
