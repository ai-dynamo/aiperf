# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate byte-exact golden output from Python detect_agent_chains.

Runs a set of flat-request scenarios through the real
``aiperf.dataset.loader.weka_agent_chains.detect_agent_chains`` and dumps the
partition (main index, worker indices, per-chain requests + fork + spliced_into,
seams_merged, unclassified) so the Rust port can diff against it.

Run: ``python tools/agentx_chains_golden.py`` ->
``tests/fixtures/agentx/chains_golden.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

from aiperf.dataset.loader.weka_agent_chains import (
    compute_chain_prefix_blocks,
    detect_agent_chains,
    worker_group_assignment,
)
from aiperf.dataset.loader.weka_trace_models import WekaNormalRequest

GROUP_MIN = 2
DECLARED_PREFIX_BLOCKS = 1


def req(outer, t, model, hash_ids, in_len=None, out_len=4, api_time=0.1):
    r = WekaNormalRequest(
        t=t,
        type="n",
        model=model,
        **{"in": in_len if in_len is not None else len(hash_ids) * 4, "out": out_len},
        hash_ids=hash_ids,
        api_time=api_time,
    )
    return (outer, r)


SCENARIOS = [
    {
        "name": "single_chain_extension",
        "normals": [req(0, 0.0, "m", [1]), req(1, 1.0, "m", [1, 2]), req(2, 2.0, "m", [1, 2, 3])],
    },
    {
        "name": "seam_no_pullback",
        "normals": [req(0, 0.0, "m", [1, 2]), req(1, 1.0, "m", [1, 9])],
    },
    {
        "name": "spawn_future_pullback",
        "normals": [req(0, 0.0, "m", [1, 2]), req(1, 1.0, "m", [1, 9]), req(2, 2.0, "m", [1, 2, 3])],
    },
    {
        "name": "cross_model_spawn",
        "normals": [req(0, 0.0, "opus", [1, 2]), req(1, 1.0, "haiku", [1, 2, 3])],
    },
    {
        "name": "empty_hash_on_main",
        "normals": [req(0, 0.0, "m", [1]), req(1, 1.0, "m", []), req(2, 2.0, "m", [1, 2])],
    },
    {
        "name": "parallel_fanout",
        "normals": [
            req(0, 0.0, "m", [1, 2, 3]),
            req(1, 1.0, "m", [1, 2, 3, 10]),
            req(2, 1.1, "m", [1, 2, 3, 20]),
            req(3, 1.2, "m", [1, 2, 3, 30]),
            req(4, 5.0, "m", [1, 2, 3, 4]),
        ],
    },
]


def dump_result(res):
    def dump_chain(c):
        return {
            "requests": [oi for oi, _ in c.requests],
            "spliced_into": c.spliced_into,
            "fork": None
            if c.fork is None
            else {
                "parent_chain": c.fork.parent_chain,
                "fork_outer_idx": c.fork.fork_outer_idx,
                "depth": c.fork.depth,
            },
        }

    wg = worker_group_assignment(res, group_min=GROUP_MIN)
    prefixes = compute_chain_prefix_blocks(res, declared_prefix_blocks=DECLARED_PREFIX_BLOCKS)
    return {
        "main_index": res.main_index,
        "worker_indices": list(res.worker_indices),
        "seams_merged": res.seams_merged,
        "unclassified_empty_hash": res.unclassified_empty_hash,
        "chains": [dump_chain(c) for c in res.chains],
        # {chain_index: [group, member]} and {chain_index: prefix_blocks}
        "worker_group_assignment": {str(k): list(v) for k, v in sorted(wg.items())},
        "chain_prefix_blocks": {str(k): v for k, v in sorted(prefixes.items())},
    }


def main():
    out = []
    for sc in SCENARIOS:
        res = detect_agent_chains(
            sc["normals"], seam_max_gap_seconds=3600.0, seam_min_overlap_ratio=0.5
        )
        out.append(
            {
                "name": sc["name"],
                "normals": [
                    {"outer": oi, "t": r.t, "model": r.model, "hash_ids": list(r.hash_ids),
                     "in": r.input_length, "out": r.output_length, "api_time": r.api_time}
                    for oi, r in sc["normals"]
                ],
                "result": dump_result(res),
            }
        )
    dest = Path(__file__).resolve().parents[1] / "tests/fixtures/agentx/chains_golden.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} ({len(out)} scenarios)")


if __name__ == "__main__":
    main()
