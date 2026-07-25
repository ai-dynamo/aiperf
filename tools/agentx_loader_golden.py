# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Golden for the main-conversation loader turn-loop.

Drives the REAL Python `ConversationReconstructor`, `compute_asst_block_caps`,
prefix-cache prepass, and the real loop helpers (`_classify_turn_input`,
`_end_to_start_delay_ms`, `_api_time_ms`, `_clamp_delay_ms`, `_cap_output`
logic) exactly as `WekaTraceLoader._reconstruct_serial` does for the main
(no-subagent) path, with a deterministic stub token generator matching the Rust
`StubSynth`. Dumps the full reconstructed conversation so the Rust
`reconstruct_main_conversation` can diff it.

Run: ``python tools/agentx_loader_golden.py`` ->
``tests/fixtures/agentx/loader_golden.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from aiperf.dataset.loader.weka_metric_prepass import (
    MetricRecord,
    compute_shared_prefix_cache_metrics,
)
from aiperf.dataset.loader.weka_synth_buf import (
    ConversationReconstructor,
    compute_asst_block_caps,
)
from aiperf.dataset.loader.weka_trace import (
    _api_time_ms,
    _classify_turn_input,
    _clamp_delay_ms,
    _end_to_start_delay_ms,
)


def make_stub(bs):
    def decode_block_tokens(hash_ids):
        out = []
        for h in hash_ids:
            out.extend(h * 1000 + i for i in range(bs))
        return out

    def sample_partial_tail_tokens(n, seed):
        return [900000 + i for i in range(n)]

    def decode_tokens_to_text(tokens):
        return " ".join(str(t) for t in tokens)

    return decode_block_tokens, sample_partial_tail_tokens, decode_tokens_to_text


def cap_output(out_len, max_osl):
    capped = out_len
    if max_osl is not None and capped > max_osl:
        capped = max_osl
    return capped if capped >= 1 else 1


# Each request: (outer_idx, dict of fields)
SCENARIOS = [
    {
        "name": "simple_two_turn",
        "trace_id": "t_simple",
        "block_size": 4,
        "tool_tokens": 0,
        "system_tokens": 0,
        "normals": [
            {"outer": 0, "t": 0.0, "api_time": 0.1, "think_time": None, "model": "m", "hash_ids": [0, 1], "in": 8, "out": 5, "input_types": [], "stop": "end_turn"},
            {"outer": 1, "t": 1.0, "api_time": 0.2, "think_time": None, "model": "m", "hash_ids": [0, 1, 2, 3], "in": 16, "out": 4, "input_types": [], "stop": ""},
        ],
    },
    {
        "name": "system_prefix_tool_result",
        "trace_id": "t_sys",
        "block_size": 4,
        "tool_tokens": 4,
        "system_tokens": 4,
        "normals": [
            {"outer": 0, "t": 0.0, "api_time": 0.05, "think_time": None, "model": "m", "hash_ids": [0, 1, 2, 3], "in": 16, "out": 3, "input_types": [], "stop": "tool_use"},
            {"outer": 1, "t": 2.0, "api_time": None, "think_time": None, "model": "m", "hash_ids": [0, 1, 2, 3, 4], "in": 20, "out": 8, "input_types": ["tool_result"], "stop": "end_turn"},
        ],
    },
    {
        "name": "think_time_and_pullback",
        "trace_id": "t_think",
        "block_size": 4,
        "tool_tokens": 0,
        "system_tokens": 0,
        "normals": [
            {"outer": 0, "t": 0.0, "api_time": 0.1, "think_time": None, "model": "m", "hash_ids": [0, 1, 2, 3], "in": 16, "out": 8, "input_types": [], "stop": ""},
            {"outer": 1, "t": 5.0, "api_time": 0.1, "think_time": 0.5, "model": "m", "hash_ids": [0, 1, 9, 10], "in": 16, "out": 4, "input_types": ["text"], "stop": ""},
        ],
    },
]


def run(sc, *, think_time_only=False, ignore_delays=False, max_osl=None, delay_cap=None):
    bs = sc["block_size"]
    tid = sc["trace_id"]
    dbt, spt, dtt = make_stub(bs)
    normals = sc["normals"]

    caps = compute_asst_block_caps([(r["hash_ids"], r["in"]) for r in normals], bs)
    records = [
        MetricRecord(sort_key=(r["t"], r["outer"], 0, k), session_id=tid, k=k, hash_ids=r["hash_ids"])
        for k, r in enumerate(normals)
    ]
    metrics = compute_shared_prefix_cache_metrics(records)

    recon = ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=dbt,
        sample_partial_tail_tokens=spt,
        decode_tokens_to_text=dtt,
    )
    turns = []
    for k, r in enumerate(normals):
        seed = f"{tid}:turn_{k}:partial_tail"
        prev = normals[k - 1] if k else None
        # Build lightweight objects with the attributes the helpers read.
        class _R:
            pass
        rr = _R(); rr.input_types = r["input_types"]; rr.stop = r["stop"]
        pr = None
        if prev is not None:
            pr = _R(); pr.input_types = prev["input_types"]; pr.stop = prev["stop"]
        ik = _classify_turn_input(rr, pr)
        is_tool = ik is not None and ik.value == "tool_result"
        if k == 0:
            recon.init_turn_0(r["hash_ids"], r["in"], sc["tool_tokens"], sc["system_tokens"], seed, is_tool)
        else:
            recon.advance_turn(prev["hash_ids"], prev["in"], prev["out"], r["hash_ids"], r["in"], seed, is_tool, max_asst_blocks=caps[k])

        t_ms = r["t"] * 1000.0
        if k == 0:
            delay = None
        elif think_time_only and r["think_time"] is not None:
            delay = r["think_time"] * 1000.0
        else:
            delay = _end_to_start_delay_ms(t_ms - prev["t"] * 1000.0, prev["api_time"])
        if delay is not None:
            delay = _clamp_delay_ms(delay, delay_cap)
            if delay is not None:
                delay = max(delay, 0.0)

        delta = recon.turn_delta()
        hit, total = metrics[(tid, k)]
        turns.append({
            "timestamp_ms": None if ignore_delays else t_ms,
            "delay_ms": None if ignore_delays else delay,
            "api_time_ms": None if ignore_delays else _api_time_ms(r["api_time"]),
            "source_outer_idx": r["outer"],
            "source_kind": "weka_main",
            "model": r["model"],
            "max_tokens": cap_output(r["out"], max_osl),
            "raw_messages": delta.delta_messages,
            "reset_context": delta.reset_context,
            "theoretical_prefix_cache_hit_blocks": hit,
            "theoretical_prefix_cache_total_blocks": total,
            "input_kind": None if ik is None else ik.value,
        })
    return {
        "name": sc["name"],
        "trace_id": tid,
        "block_size": bs,
        "tool_tokens": sc["tool_tokens"],
        "system_tokens": sc["system_tokens"],
        "think_time_only": think_time_only,
        "normals": normals,
        "turns": turns,
    }


def main():
    out = [run(SCENARIOS[0]), run(SCENARIOS[1]), run(SCENARIOS[2], think_time_only=True)]
    dest = Path(__file__).resolve().parents[1] / "tests/fixtures/agentx/loader_golden.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} ({len(out)} scenarios)")


if __name__ == "__main__":
    main()
