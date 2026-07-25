# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden for the agentic-replay execution-order timing SCHEDULE.

Uses the REAL trajectory helpers (`_next_turn_index_at_or_after`, `_offset_ms`,
and the `ConversationState.warmup_turn_index` rule) to compute, for a stream's
recorded turn timestamps and a fixed t*, each turn's replay phase
(history / warmup / profiling) and PROFILING dispatch offset from t*. The Rust
port reproduces this from `trajectory_source` byte-for-byte — the deterministic
execution-order timing schedule that the async dispatch loop fires against.

Run: ``python tools/agentx_schedule_golden.py`` ->
``tests/fixtures/agentx/schedule_golden.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from aiperf.timing.trajectory_source import _next_turn_index_at_or_after, _offset_ms


class _Turn:
    def __init__(self, ts_ms):
        self.timestamp_ms = ts_ms


class _Meta:
    def __init__(self, ts_list):
        self.turns = [_Turn(t) for t in ts_list]


# (name, turn timestamps in ms, t*)
SCENARIOS = [
    ("simple", [0.0, 1000.0, 2000.0], 800.0),
    ("all_profiling", [500.0, 1500.0, 2500.0], 200.0),
    ("all_history", [0.0, 100.0, 200.0], 900.0),
    ("boundary", [0.0, 800.0, 1600.0], 800.0),
]


def schedule(ts_list, t_star):
    meta = _Meta(ts_list)
    resume = _next_turn_index_at_or_after(meta, t_star)
    warmup_idx = (resume - 1) if (resume is not None and resume >= 1) else None
    per_turn = []
    for k, ts in enumerate(ts_list):
        if resume is None:
            phase = "history"  # whole stream is pre-t*; last turn warms
            if k == len(ts_list) - 1:
                phase = "warmup"
            offset = None
        elif k == warmup_idx:
            phase = "warmup"
            offset = None
        elif k >= resume:
            phase = "profiling"
            offset = _offset_ms(ts, t_star)
        else:
            phase = "history"
            offset = None
        per_turn.append({"k": k, "phase": phase, "offset_ms": offset})
    return {"resume_index": resume, "warmup_index": warmup_idx, "per_turn": per_turn}


def main():
    out = []
    for name, ts, t_star in SCENARIOS:
        out.append({"name": name, "timestamps_ms": ts, "t_star_ms": t_star, **schedule(ts, t_star)})
    dest = Path(__file__).resolve().parents[1] / "tests/fixtures/agentx/schedule_golden.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} ({len(out)} scenarios)")


if __name__ == "__main__":
    main()
