# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker graph-IR bytes path: pre-serialized body parity + record correctness.

The interned unified store is the sole trie store shape: on a graph credit the
worker's segment reader is a :class:`GraphSegmentUnifiedClient` and, when no
content mutation is needed (``cache_bust == NONE``), the worker builds the
request body ONCE from content-pool slices via
:func:`materialize_graph_request_unified_bytes` and sends it verbatim as a
``Turn.raw_payload_bytes`` -- no per-segment ``orjson.loads`` and no
``orjson.dumps`` of the messages array.

Three gates this test protects:

1. **Body parity** -- the bytes the bytes path builds, when ``orjson.loads``-ed,
   equal the dict the dict path (``materialize_graph_request_unified`` +
   ``apply_run_level_payload_options``) produces for the same node + same
   endpoint settings (stream / extra / server-token-count / warmup all folded
   identically). This is the core correctness proof: a pre-serialized body can't
   be mutated after build, so every outer field must be folded in at build time.
2. **payload_bytes record correctness** -- ``inference_client`` records a bytes
   body VERBATIM (not ``orjson.dumps`` of it, which would corrupt the recorded
   JSON into a string and break ISL / raw-export); a dict ``raw_payload`` still
   records ``orjson.dumps(dict)``.
3. **cache-bust fallback** -- a cache-bust run mutates message CONTENT (prepends a
   marker to the first user message), which a pre-serialized body cannot do; so
   the dict path is the only one that carries the marker, and the worker gates the
   bytes path on ``cache_bust == NONE``.

Builds a 2-trace weka trie corpus (merged segment pool drained straight into a
``GraphSegmentUnifiedBackingStore`` with the node's interned manifest written
per-test).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from aiperf.common.enums import (
    CacheBustTarget,
    ModelSelectionStrategy,
)
from aiperf.common.models.dataset_models import Turn
from aiperf.common.models.model_endpoint_info import (
    EndpointInfo,
    ModelEndpointInfo,
    ModelInfo,
    ModelListInfo,
)
from aiperf.common.models.record_models import RequestRecord
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.segment_ir.pool import SegmentPool
from aiperf.dataset.graph.segment_ir.store_builder import (
    _prompt_segment_ids,
    _trie_llm_nodes,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from aiperf.graph.worker_materialize import (
    apply_run_level_payload_options,
    encode_overrides_inner,
    materialize_graph_request_unified,
    materialize_graph_request_unified_bytes,
    stamp_cache_bust_marker,
)
from aiperf.plugin.enums import EndpointType, TransportType
from aiperf.workers.inference_client import InferenceClient

FIXTURES = Path(__file__).parent / "fixtures"

_TRACE_ID = "t-1#0"
_NODE_ORDINAL = 7


def _merge_pool(*pools: SegmentPool) -> SegmentPool:
    """Union content-addressed pools (ids collide iff content matches)."""
    merged = SegmentPool()
    for pool in pools:
        merged._by_id.update(pool._by_id)
    return merged


@pytest.fixture
def two_trace_corpus(monkeypatch):
    """A 2-trace weka trie corpus: merged pool + one node's known prompt path."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    pg_a = from_weka_trace(FIXTURES / "weka_subagent.json")
    pg_b = from_weka_trace(FIXTURES / "weka_min.json")
    assert pg_a.segment_pool is not None
    assert pg_b.segment_pool is not None
    pool = _merge_pool(pg_a.segment_pool, pg_b.segment_pool)

    known_path: list[str] | None = None
    for trace in pg_a.traces:
        for node in _trie_llm_nodes(pg_a, trace).values():
            path = _prompt_segment_ids(node)
            if path and (known_path is None or len(path) > len(known_path)):
                known_path = path
    assert known_path, "fixture must yield a node with a prompt_segment_ids path"
    return pool, known_path


async def _build_unified_client(
    tmp_path: Path,
    pool: SegmentPool,
    known_path: list[str],
    benchmark_id: str,
    *,
    dispatch_overrides: dict[str, Any],
    stream: bool = True,
) -> GraphSegmentUnifiedClient:
    """Drain the pool + one interned node manifest into a unified store; open it.

    Mirrors ``build_unified_trie_store_interned``: every pool segment is
    ``put_segment``'d (assigning dense int handles), and the node's hex
    ``known_path`` is resolved to those handles for the interned manifest.
    """
    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id=benchmark_id
    )
    for segment in pool._by_id.values():
        store.put_segment(segment.id, segment.role, segment.content)
    handles = [store.segment_handle(sid) for sid in known_path]
    assert all(h is not None for h in handles)
    store.add_node_manifest_interned(
        _TRACE_ID, _NODE_ORDINAL, "profiling", handles, dispatch_overrides, stream
    )
    await store.finalize()
    return GraphSegmentUnifiedClient(tmp_path, benchmark_id).open()


def _endpoint(
    *,
    streaming: bool = True,
    extra: list[tuple[str, Any]] | None = None,
    use_server_token_count: bool = False,
    use_legacy_max_tokens: bool = False,
    cache_bust: CacheBustTarget = CacheBustTarget.NONE,
) -> EndpointInfo:
    return EndpointInfo(
        type=EndpointType.CHAT,
        base_url="http://localhost:8000/v1/chat",
        streaming=streaming,
        extra=extra or [],
        use_server_token_count=use_server_token_count,
        use_legacy_max_tokens=use_legacy_max_tokens,
        cache_bust=cache_bust,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "streaming,extra,use_server_token_count,use_legacy_max_tokens,phase_variant",
    [
        pytest.param(True, [], False, False, "profiling", id="stream_modern"),
        pytest.param(False, [], False, True, "profiling", id="nostream_legacy"),
        pytest.param(True, [], True, False, "profiling", id="stream_usage"),
        pytest.param(
            True, [("temperature", 0.0), ("top_p", 0.9)], False, False, "profiling",
            id="stream_extra",
        ),
        pytest.param(True, [], True, False, "warmup", id="warmup_usage"),
        pytest.param(False, [("seed", 5)], False, True, "warmup", id="warmup_extra_legacy"),
    ],
)  # fmt: skip
async def test_bytes_body_matches_dict_path(
    tmp_path,
    two_trace_corpus,
    streaming,
    extra,
    use_server_token_count,
    use_legacy_max_tokens,
    phase_variant,
):
    """``materialize_graph_request_unified_bytes`` bytes round-trip to the dict-path dict.

    The dict path = ``materialize_graph_request_unified`` +
    ``apply_run_level_payload_options`` (cache-bust off, so no
    ``stamp_cache_bust_marker`` mutation). The bytes path must produce a body
    whose ``orjson.loads`` equals that dict EXACTLY -- proving every outer field
    (token mapping, stream override, extra, include_usage, warmup cap) is folded
    into the pre-serialized tail identically.
    """
    pool, known_path = two_trace_corpus
    client = await _build_unified_client(
        tmp_path,
        pool,
        known_path,
        "bench",
        dispatch_overrides={"model": "m", "max_output_tokens": 7},
        stream=streaming,
    )
    endpoint = _endpoint(
        streaming=streaming,
        extra=extra,
        use_server_token_count=use_server_token_count,
        use_legacy_max_tokens=use_legacy_max_tokens,
    )

    try:
        dict_payload = materialize_graph_request_unified(
            client,
            _TRACE_ID,
            _NODE_ORDINAL,
            phase_variant,
            use_legacy_max_tokens=use_legacy_max_tokens,
        )
        assert dict_payload is not None
        apply_run_level_payload_options(dict_payload, endpoint)

        built = materialize_graph_request_unified_bytes(
            client,
            _TRACE_ID,
            _NODE_ORDINAL,
            phase_variant,
            use_legacy_max_tokens=use_legacy_max_tokens,
            endpoint=endpoint,
        )
    finally:
        client.close()

    assert built is not None
    body, model, effective_stream = built
    # Parsed-JSON parity is the correctness-relevant property (not raw-byte order).
    assert orjson.loads(body) == dict_payload
    # The per-node model is surfaced so the worker can stamp Turn.model = the same
    # value the dict path's payload.get("model") would.
    assert model == dict_payload.get("model")
    # The effective wire stream is surfaced so the worker carries the recorded
    # per-node mode onto RequestInfo; it matches the FINAL stamped dict value.
    assert effective_stream == dict_payload["stream"]


@pytest.mark.asyncio
async def test_bytes_path_returns_none_without_handles(tmp_path, two_trace_corpus):
    """A node manifest WITHOUT ``handles`` is a miss on both unified paths (None)."""
    pool, known_path = two_trace_corpus
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="bench")
    for segment in pool._by_id.values():
        store.put_segment(segment.id, segment.role, segment.content)
    # A handle-less envelope (no interned path) -- the unified fns must not
    # guess; they return None and the worker reports a genuine miss.
    store.add_node_manifest(
        _TRACE_ID,
        _NODE_ORDINAL,
        "profiling",
        orjson.dumps({"dispatch_overrides": {"model": "m"}, "stream": True}),
    )
    await store.finalize()

    with GraphSegmentUnifiedClient(tmp_path, "bench").open() as client:
        built = materialize_graph_request_unified_bytes(
            client,
            _TRACE_ID,
            _NODE_ORDINAL,
            "profiling",
            endpoint=_endpoint(),
        )
        payload = materialize_graph_request_unified(
            client, _TRACE_ID, _NODE_ORDINAL, "profiling"
        )
    assert built is None
    assert payload is None


@pytest.mark.asyncio
async def test_cache_bust_only_dict_path_can_inject_marker(tmp_path, two_trace_corpus):
    """Cache-bust mutates message CONTENT -> only the dict path can carry it.

    The bytes body is built once and cannot be mutated, which is exactly WHY the
    worker gates the bytes path on ``cache_bust == NONE``. This proves the dict
    path injects the marker into the first user message and the bytes body does
    NOT contain it (so taking the bytes path under cache-bust would silently drop
    the marker -- the bug the gate prevents).
    """
    pool, known_path = two_trace_corpus
    client = await _build_unified_client(
        tmp_path,
        pool,
        known_path,
        "bench",
        dispatch_overrides={"model": "m", "max_output_tokens": 7},
    )
    endpoint = _endpoint(cache_bust=CacheBustTarget.FIRST_TURN_PREFIX)
    try:
        dict_payload = materialize_graph_request_unified(
            client, _TRACE_ID, _NODE_ORDINAL, "profiling"
        )
        assert dict_payload is not None
        apply_run_level_payload_options(dict_payload, endpoint)
        stamp_cache_bust_marker(
            dict_payload,
            benchmark_id="bench",
            trace_instance_id=_TRACE_ID,
            target=endpoint.cache_bust,
        )
        built = materialize_graph_request_unified_bytes(
            client,
            _TRACE_ID,
            _NODE_ORDINAL,
            "profiling",
            endpoint=endpoint,
        )
    finally:
        client.close()

    first_user = next(m for m in dict_payload["messages"] if m["role"] == "user")
    assert first_user["content"].startswith("[rid:"), "dict path must inject the marker"
    # The bytes body carries the ORIGINAL content (no marker) -- proving it cannot
    # represent the cache-bust mutation, hence the worker's cache_bust==NONE gate.
    assert built is not None
    body, _, _ = built
    assert b"[rid:" not in body


@pytest.mark.asyncio
async def test_bytes_path_preserves_per_node_model_in_wire_body(
    tmp_path, two_trace_corpus
):
    """A node whose model != endpoint primary keeps its OWN model in the wire body.

    The bytes path surfaces the node's ``dispatch_overrides["model"]`` (a real
    multi-model weka pattern, e.g. Haiku WebFetch sidecars under a non-Haiku
    primary) and folds it into the body bytes, matching the dict path's
    ``payload.get("model")``. The worker does NOT stamp ``Turn.model`` with it:
    ``record.model_name`` falls back to the run ``--model`` in
    ``_finalize_request_record`` so tokenizer selection behaves like plain
    aiperf (recorded deployment ids are usually not resolvable tokenizer
    repos); the recorded model reaches the server via the verbatim body only.
    """
    pool, known_path = two_trace_corpus
    # Endpoint primary is "test-model"; the node overrides to a DIFFERENT model.
    per_node_model = "anthropic/claude-haiku"
    client = await _build_unified_client(
        tmp_path,
        pool,
        known_path,
        "bench",
        dispatch_overrides={"model": per_node_model, "max_output_tokens": 7},
    )
    endpoint = _endpoint()
    try:
        built = materialize_graph_request_unified_bytes(
            client,
            _TRACE_ID,
            _NODE_ORDINAL,
            "profiling",
            endpoint=endpoint,
        )
    finally:
        client.close()

    assert built is not None
    body, model, _ = built
    # The model is surfaced for dict-path parity checks.
    assert model == per_node_model
    # The wire body carries it (folded into the overrides tail).
    assert orjson.loads(body)["model"] == per_node_model


def test_encode_overrides_inner():
    assert encode_overrides_inner({"max_tokens": 256}) == b'"max_tokens":256'
    assert encode_overrides_inner({}) == b""
    assert encode_overrides_inner(None) == b""


# --- payload_bytes record correctness (inference_client selection) ----------


@pytest.fixture
def _http_transport_entry():
    entry = MagicMock()
    entry.name = TransportType.HTTP.value
    entry.metadata = {"url_schemes": ["http", "https"]}
    return entry


@pytest.fixture
def inference_client(_http_transport_entry):
    model_endpoint = ModelEndpointInfo(
        models=ModelListInfo(
            models=[ModelInfo(name="test-model")],
            model_selection_strategy=ModelSelectionStrategy.ROUND_ROBIN,
        ),
        endpoint=EndpointInfo(
            type=EndpointType.CHAT, base_url="http://localhost:8000/v1/chat"
        ),
    )
    mock_transport = MagicMock()
    mock_endpoint = MagicMock()
    mock_endpoint.get_endpoint_headers.return_value = {}
    mock_endpoint.get_endpoint_params.return_value = {}
    mock_endpoint.format_payload.return_value = {"rewritten": True}

    def mock_get_class(protocol, name):
        if protocol == "endpoint":
            return lambda **kwargs: mock_endpoint
        if protocol == "transport":
            return lambda **kwargs: mock_transport
        raise ValueError(f"Unknown protocol: {protocol}")

    with (
        patch(
            "aiperf.workers.inference_client.plugins.get_class",
            side_effect=mock_get_class,
        ),
        patch(
            "aiperf.workers.inference_client.plugins.list_entries",
            return_value=[_http_transport_entry],
        ),
    ):
        return InferenceClient(
            model_endpoint=model_endpoint, service_id="test-service-id"
        )


@pytest.mark.asyncio
async def test_inference_client_records_bytes_payload_verbatim(
    inference_client, sample_request_info
):
    """A ``raw_payload_bytes`` turn records ``payload_bytes`` VERBATIM (no dumps)."""
    body = b'{"messages":[]}'
    request_info = sample_request_info
    request_info.turns = [Turn(role="user", raw_payload_bytes=body)]
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=request_info)
    )

    await inference_client.send_request(request_info)

    inference_client.endpoint.format_payload.assert_not_called()
    # Recorded VERBATIM: the exact bytes, which round-trip back to valid JSON.
    assert request_info.payload_bytes == body
    assert orjson.loads(request_info.payload_bytes) == {"messages": []}
    call_args = inference_client.transport.send_request.call_args
    assert call_args.kwargs["payload"] == body


@pytest.mark.asyncio
async def test_inference_client_records_dict_payload_as_dumps(
    inference_client, sample_request_info
):
    """A dict ``raw_payload`` turn still records ``orjson.dumps(dict)`` (unchanged)."""
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    request_info = sample_request_info
    request_info.turns = [Turn(role="user", raw_payload=payload)]
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=request_info)
    )

    await inference_client.send_request(request_info)

    inference_client.endpoint.format_payload.assert_not_called()
    assert request_info.payload_bytes == orjson.dumps(payload)
    call_args = inference_client.transport.send_request.call_args
    assert call_args.kwargs["payload"] == payload


@pytest.mark.asyncio
async def test_graph_record_model_name_falls_back_to_run_model(
    inference_client, sample_request_info
):
    """A graph turn carries no ``Turn.model``, so ``record.model_name`` resolves
    to the run ``--model`` (primary), never the recorded body model — recorded
    deployment ids (e.g. ``dynamo/org/model-fp8``) are not tokenizer repos.
    """
    body = b'{"messages":[],"model":"dynamo/org/model-fp8"}'
    request_info = sample_request_info
    request_info.turns = [Turn(role="user", raw_payload_bytes=body)]
    inference_client.transport.send_request = AsyncMock(
        return_value=RequestRecord(request_info=request_info)
    )

    record = await inference_client.send_request(request_info)

    assert record.model_name == "test-model"
