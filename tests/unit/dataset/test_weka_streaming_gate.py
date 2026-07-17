# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The streaming store build must recover the per-node prefix-cache map from
the merged structural graphs, and the weka streaming entry point must drain
local directory sources. (The former eager-vs-streaming store-route pins are
superseded by the streaming builder's own branch tests:
``test_weka_content_knob_wiring``, ``test_slot_graph_eager_drain``, and
``test_dag_jsonl_streaming_store_parity`` -- every graph workload now takes
the one streaming store build.)"""

from __future__ import annotations

from pathlib import Path

from aiperf.dataset.graph.adapters.weka.trace import (
    from_weka_trace,
    stream_weka_trace_segment_payloads,
)
from aiperf.dataset.graph.codecs import decode_parsed_graph_msgpack
from aiperf.dataset.graph.merge import merge_parsed_graphs
from aiperf.dataset.graph.segment_ir.store_builder import (
    iter_trace_segment_payloads,
)
from aiperf.dataset.graph.store_build import GraphStoreBuilder

FIX_MIN = Path(__file__).parents[1] / "graph" / "fixtures" / "weka_min.json"
FIX_SUB = Path(__file__).parents[1] / "graph" / "fixtures" / "weka_subagent.json"


def test_streaming_structural_prefix_cache_matches_eager_parse() -> None:
    parsed = from_weka_trace(str(FIX_MIN), content_root_seed=42)
    eager_map = GraphStoreBuilder._build_graph_prefix_cache_by_trace(parsed)
    assert eager_map, "fixture must produce a non-empty prefix-cache map"

    structural_blobs = [
        p.structural_graph
        for p in iter_trace_segment_payloads(parsed)
        if p.structural_graph
    ]
    merged = merge_parsed_graphs(
        decode_parsed_graph_msgpack(b) for b in structural_blobs
    )
    streaming_map = GraphStoreBuilder._build_graph_prefix_cache_by_trace(merged)
    assert streaming_map == eager_map


def test_stream_segment_payloads_local_directory_yields_all_traces(tmp_path) -> None:
    """The streaming entry point drains a LOCAL directory of weka .json files.

    Pins the local-source branch the weka streaming store build depends on: a
    directory source must stream one payload per file, keyed by each file's
    own trace id.
    """
    weka_dir = tmp_path / "corpus"
    weka_dir.mkdir()
    (weka_dir / "a.json").write_bytes(FIX_MIN.read_bytes())
    (weka_dir / "b.json").write_bytes(FIX_SUB.read_bytes())

    payloads = list(
        stream_weka_trace_segment_payloads(str(weka_dir), content_root_seed=42)
    )

    assert sorted(p.trace_id for p in payloads) == [
        "trace_03_n3",
        "trace_sub_n2s1",
    ]
