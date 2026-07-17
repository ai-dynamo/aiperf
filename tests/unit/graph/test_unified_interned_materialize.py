# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)


@pytest.mark.asyncio
async def test_unified_interned_bytes_match_dict(tmp_path):
    store = GraphSegmentUnifiedBackingStore(tmp_path, "b")
    ha = store.put_segment("a", "system", "SYS")
    hb = store.put_segment("b", "user", "hi")
    store.add_node_manifest_interned(
        "t0", 0, "profiling", [ha, hb], {"model": "m"}, False
    )
    await store.finalize()

    from aiperf.graph.worker_materialize import (
        materialize_graph_request_unified,
        materialize_graph_request_unified_bytes,
    )

    class _Ep:
        streaming = False
        extra = []
        use_server_token_count = False
        use_legacy_max_tokens = False

    with GraphSegmentUnifiedClient(tmp_path, "b").open() as c:
        payload = materialize_graph_request_unified(
            c, "t0", 0, "profiling", use_legacy_max_tokens=False
        )
        built = materialize_graph_request_unified_bytes(
            c, "t0", 0, "profiling", use_legacy_max_tokens=False, endpoint=_Ep()
        )
    assert payload["messages"] == [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]
    body, model, effective_stream = built
    assert model == "m"
    # Node stream False + global (_Ep) streaming False -> effective wire False.
    assert effective_stream is False
    assert orjson.loads(body)["messages"] == payload["messages"]


@pytest.mark.asyncio
async def test_interned_bytes_self_consistent_on_real_fixture(tmp_path, monkeypatch):
    """Dict path == parsed bytes path == pool-derived messages on a real fixture.

    Grounds the interned bytes fast path three ways on a real weka parse: the
    materialized ``messages`` equal the segment pool's own ``(role, content)``
    walk of the node's hex path (not just fn-vs-fn agreement), and the bytes
    body's parsed JSON equals the dict path + run-level options EXACTLY (the
    worker's parity contract, cache-bust NONE = the verbatim-bytes fast path).
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from aiperf.common.enums import CacheBustTarget
    from aiperf.common.models import EndpointInfo
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
    from aiperf.dataset.graph.segment_ir.store_builder import (
        _prompt_segment_ids,
        _trie_llm_nodes,
        build_unified_trie_store_interned,
        trie_node_ordinals,
    )
    from aiperf.graph.worker_materialize import (
        apply_run_level_payload_options,
        materialize_graph_request_unified,
        materialize_graph_request_unified_bytes,
    )
    from aiperf.plugin.enums import EndpointType

    fx = Path(__file__).parent / "fixtures" / "weka_subagent.json"
    parsed = from_weka_trace(fx)
    trace = parsed.traces[0]
    nodes = _trie_llm_nodes(parsed, trace)
    ordinals = trie_node_ordinals(nodes)
    node_id = next(nid for nid, n in nodes.items() if _prompt_segment_ids(n))
    ordinal = ordinals[node_id]
    path = _prompt_segment_ids(nodes[node_id])

    # Pool-derived ground truth: the node's hex path walked directly against
    # the parse's content-addressed pool.
    pool = parsed.segment_pool._by_id
    expected_messages = [
        {"role": pool[sid].role, "content": pool[sid].content} for sid in path
    ]

    # A real EndpointInfo (NONE cache-bust = the verbatim-bytes fast path).
    # base_url is a read-only property on EndpointInfo -- the settable field is
    # base_urls.
    endpoint = EndpointInfo(
        type=EndpointType.CHAT,
        base_urls=["http://localhost:8000/v1/chat"],
        streaming=True,
        extra=[],
        use_server_token_count=False,
        use_legacy_max_tokens=False,
        cache_bust=CacheBustTarget.NONE,
    )

    us = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="b")
    await build_unified_trie_store_interned(parsed, us)

    with GraphSegmentUnifiedClient(tmp_path, "b").open() as uc:
        payload = materialize_graph_request_unified(uc, trace.id, ordinal, "profiling")
        built = materialize_graph_request_unified_bytes(
            uc,
            trace.id,
            ordinal,
            "profiling",
            use_legacy_max_tokens=False,
            endpoint=endpoint,
        )

    assert payload is not None
    assert payload["messages"] == expected_messages
    # Mirror the worker's dict path: extract the recorded per-node override from
    # the materialized payload BEFORE applying run-level options, so the dict
    # baseline resolves ``stream`` the same way the bytes path does.
    env_stream = payload.get("stream")
    stream_override = bool(env_stream) if env_stream is not None else None
    apply_run_level_payload_options(payload, endpoint, stream_override=stream_override)

    assert built is not None
    body, model, effective_stream = built
    assert orjson.loads(body) == payload
    assert model == payload.get("model")
    assert effective_stream == payload["stream"]
