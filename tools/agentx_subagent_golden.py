# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden for subagent expansion (real _expand_subagent_to_child_plans).

Run: ``python tools/agentx_subagent_golden.py`` ->
``tests/fixtures/agentx/subagent_golden.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from aiperf.dataset.loader.weka_trace import _expand_subagent_to_child_plans
from aiperf.dataset.loader.weka_trace_models import WekaSubagentEntry


def ireq(t, model, hash_ids, in_len, out_len, api_time=0.1):
    return {"t": t, "type": "n", "model": model, "in": in_len, "out": out_len,
            "hash_ids": hash_ids, "api_time": api_time}


def entry(agent_id, t, reqs, tool=0, system=0, models=("m",)):
    return WekaSubagentEntry(
        t=t, type="subagent", agent_id=agent_id, subagent_type="Explore",
        duration_ms=1000, status="completed", requests=reqs, models=list(models),
        tool_tokens=tool, system_tokens=system,
    )


SCENARIOS = [
    {
        "name": "single_chain",
        "trace_id": "t0", "sa_index": 0, "source_outer_idx": 5, "block_size": 4,
        "entry": entry("agent_001", 10.0, [ireq(10.0, "m", [1], 4, 4), ireq(11.0, "m", [1, 2], 8, 4)]),
    },
    {
        "name": "spawn_worker",
        "trace_id": "t0", "sa_index": 1, "source_outer_idx": 7, "block_size": 4,
        "entry": entry("agent_002", 0.0, [
            ireq(0.0, "m", [1, 2], 8, 8),
            ireq(1.0, "m", [1, 9], 8, 4),
            ireq(2.0, "m", [1, 2, 3], 12, 4),
        ]),
    },
    {
        "name": "cross_model_aux",
        "trace_id": "t0", "sa_index": 2, "source_outer_idx": 9, "block_size": 4,
        "entry": entry("agent_003", 0.0, [
            ireq(0.0, "opus", [1, 2], 8, 8),
            ireq(1.0, "haiku", [1, 9], 8, 4),
            ireq(2.0, "opus", [1, 2, 3], 12, 4),
        ], models=("opus", "haiku")),
    },
    {
        "name": "relative_ts",
        "trace_id": "t0", "sa_index": 3, "source_outer_idx": 11, "block_size": 4,
        "entry": entry("agent_004", 100.0, [ireq(0.5, "m", [1], 4, 4), ireq(1.5, "m", [1, 2], 8, 4)]),
    },
]


def dump_plan(p):
    return {
        "session_id": p.session_id,
        "chain_index": p.chain_index,
        "request_inner_indices": list(p.request_inner_indices),
        "request_ts": [r.t for r in p.requests],
        "init_tool_tokens": p.init_tool_tokens,
        "init_system_tokens": p.init_system_tokens,
        "is_aux": p.is_aux,
    }


def main():
    out = []
    for sc in SCENARIOS:
        plans = _expand_subagent_to_child_plans(
            sc["trace_id"], sc["sa_index"], sc["source_outer_idx"], sc["entry"], sc["block_size"]
        )
        out.append({
            "name": sc["name"],
            "trace_id": sc["trace_id"],
            "sa_index": sc["sa_index"],
            "source_outer_idx": sc["source_outer_idx"],
            "block_size": sc["block_size"],
            "agent_id": sc["entry"].agent_id,
            "entry_t": sc["entry"].t,
            "tool_tokens": sc["entry"].tool_tokens,
            "system_tokens": sc["entry"].system_tokens,
            "requests": [
                {"t": r.t if isinstance(r, dict) else r.t, "model": r.model, "hash_ids": list(r.hash_ids),
                 "in": r.input_length, "out": r.output_length, "api_time": r.api_time}
                for r in sc["entry"].requests
            ],
            "plans": [dump_plan(p) for p in plans],
        })
    dest = Path(__file__).resolve().parents[1] / "tests/fixtures/agentx/subagent_golden.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} ({len(out)} scenarios)")


if __name__ == "__main__":
    main()
