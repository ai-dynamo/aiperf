# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate byte-exact golden output from the Python ConversationReconstructor.

Runs a set of scenarios through the real
``aiperf.dataset.loader.weka_synth_buf.ConversationReconstructor`` with a
deterministic stub token generator (block ``h`` -> ``[h*1000 .. h*1000+bs)``,
tail -> ``[900000 + i]``, text -> space-joined token ids) that EXACTLY matches
the Rust ``StubSynth`` in ``rust/runtime/src/agentx/synth.rs``. After each step
it captures the emitted ``TurnDelta`` and the full segment state, so the Rust
port can replay the same scenarios and diff the result.

Run: ``python tools/agentx_synth_golden.py`` -> writes
``tests/fixtures/agentx/synth_golden.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from aiperf.dataset.loader.weka_synth_buf import ConversationReconstructor


def make_stub(bs: int):
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


SCENARIOS = [
    {
        "name": "turn0_partial_tail",
        "block_size": 4,
        "steps": [
            {"op": "init", "hash_ids": [0, 1], "in": 10, "tool": 0, "system": 0, "seed": "s0"},
        ],
    },
    {
        "name": "turn0_system_prefix",
        "block_size": 4,
        "steps": [
            {"op": "init", "hash_ids": [0, 1, 2, 3], "in": 16, "tool": 4, "system": 4, "seed": "s0"},
        ],
    },
    {
        "name": "append_only",
        "block_size": 4,
        "steps": [
            {"op": "init", "hash_ids": [0, 1], "in": 8, "tool": 0, "system": 0, "seed": "s0"},
            {"op": "advance", "prev_hash_ids": [0, 1], "prev_out": 4, "hash_ids": [0, 1, 2, 3], "in": 16, "seed": "s1", "is_tool_result": False, "max_asst_blocks": None},
            {"op": "advance", "prev_hash_ids": [0, 1, 2, 3], "prev_out": 8, "hash_ids": [0, 1, 2, 3, 4, 5], "in": 24, "seed": "s2", "is_tool_result": False, "max_asst_blocks": None},
        ],
    },
    {
        "name": "pull_back_midseq_replace",
        "block_size": 4,
        "steps": [
            {"op": "init", "hash_ids": [0, 1, 2, 3], "in": 16, "tool": 0, "system": 0, "seed": "s0"},
            {"op": "advance", "prev_hash_ids": [0, 1, 2, 3], "prev_out": 8, "hash_ids": [0, 1, 9, 10], "in": 16, "seed": "s1", "is_tool_result": False, "max_asst_blocks": None},
        ],
    },
    {
        "name": "tool_result_and_partials",
        "block_size": 4,
        "steps": [
            {"op": "init", "hash_ids": [0, 1], "in": 10, "tool": 0, "system": 0, "seed": "s0"},
            {"op": "advance", "prev_hash_ids": [0, 1], "prev_out": 3, "hash_ids": [0, 1, 2], "in": 14, "seed": "s1", "is_tool_result": True, "max_asst_blocks": None},
        ],
    },
]


def dump_segments(r: ConversationReconstructor):
    return [
        {
            "role": s.role,
            "block_start": s.block_start,
            "block_count": s.block_count,
            "tokens": list(s.tokens),
            "content": s.content,
            "tool_result_turn": s.tool_result_turn,
        }
        for s in r._segments
    ]


def run_scenario(sc):
    bs = sc["block_size"]
    dbt, spt, dtt = make_stub(bs)
    r = ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=dbt,
        sample_partial_tail_tokens=spt,
        decode_tokens_to_text=dtt,
    )
    steps_out = []
    for step in sc["steps"]:
        if step["op"] == "init":
            r.init_turn_0(step["hash_ids"], step["in"], step["tool"], step["system"], step["seed"])
        else:
            r.advance_turn(
                step["prev_hash_ids"],
                0,  # prev_in_tokens unused by the algorithm
                step["prev_out"],
                step["hash_ids"],
                step["in"],
                step["seed"],
                step["is_tool_result"],
                max_asst_blocks=step["max_asst_blocks"],
            )
        delta = r.turn_delta()
        steps_out.append(
            {
                "delta_messages": delta.delta_messages,
                "reset_context": delta.reset_context,
                "segments": dump_segments(r),
            }
        )
    return {
        "name": sc["name"],
        "block_size": bs,
        "steps_input": sc["steps"],
        "steps_output": steps_out,
        "trailing_non_user_turns": list(r._trailing_non_user_turns),
    }


def main():
    out = [run_scenario(sc) for sc in SCENARIOS]
    dest = Path(__file__).resolve().parents[1] / "tests/fixtures/agentx/synth_golden.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} ({len(out)} scenarios)")


if __name__ == "__main__":
    main()
