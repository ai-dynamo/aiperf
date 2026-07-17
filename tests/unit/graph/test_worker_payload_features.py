# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""F3/F4 regression: graph credits must honor run-level payload options and
must surface a clear, fatal error (not a silent retry-loop) when the
graph store is absent or corrupt.

F3 (payload features dropped): the worker must layer ``endpoint.extra``
(``--extra-inputs``) and ``stream_options.include_usage`` (``--server-token-count``
on a streaming request) onto the materialized payload. The wire ``stream`` is
stamped from the RECORDED per-node mode (weka ``"n"``/``"s"``): a recorded
streaming node streams even when the global run is non-streaming, and a recorded
non-streaming node stays non-streaming even when the global run streams; the
global ``endpoint.streaming`` is only the fallback for a mode-less node.
``stream_options.include_usage`` keys on that FINAL stamped ``stream``. User
``extra`` CLOBBERS any colliding per-node dispatch key
(``payload.update(endpoint.extra)``); non-colliding per-node ``dispatch_overrides``
still pass through.

F4 (missing-store hang): when the unified store open raises because the
store files are missing, the worker must set ``credit_context.error`` (so the
run reports a real error, not zero-success/zero-error), and must not retry the
failed open on every subsequent credit.
"""

import time
from pathlib import Path
from unittest.mock import AsyncMock

import orjson
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.models import RequestRecord
from aiperf.common.models.dataset_models import GraphSegmentClientMetadata
from aiperf.credit.structs import Credit, CreditContext
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.dataset.graph.segment_ir.store_builder import (
    build_unified_trie_store_interned,
)
from aiperf.dataset.graph_segment_unified_store import (
    GraphSegmentUnifiedBackingStore,
    GraphSegmentUnifiedClient,
)
from aiperf.graph.worker_materialize import (
    materialize_graph_request_unified,
    strip_dynamo_session_headers,
    uniquify_dynamo_session_headers,
)
from aiperf.workers.worker import Worker
from tests.harness.fake_tokenizer import FakeTokenizer

FIX_MIN = Path(__file__).parent / "fixtures" / "weka_min.json"
# Mixed-mode fixture: node trace_first_token_anchor:0 is recorded streaming ("s"); the top-level "n" node is trace_first_token_anchor:1.
FIX_MIXED = Path(__file__).parent / "fixtures" / "weka_first_token_anchor.json"


def _graph_client_metadata(
    tmp_path: Path, benchmark_id: str
) -> GraphSegmentClientMetadata:
    """Broadcast-shaped store location the worker opens (Task 5: no env fallback)."""
    return GraphSegmentClientMetadata(
        store_base_path=tmp_path,
        benchmark_id=benchmark_id,
        sidecar_path=tmp_path
        / f"aiperf_graph_meta_{benchmark_id}"
        / "graph_meta.msgpack",
    )


@pytest.fixture
async def mock_worker(
    benchmark_run, fake_tokenizer: FakeTokenizer, skip_service_registration
):
    worker = Worker(run=benchmark_run, service_id="mock-service-id")
    await worker.initialize()
    await worker.start()
    yield worker
    await worker.stop()


async def _build_store_for_worker(
    worker: Worker, tmp_path: Path, *, fixture: Path = FIX_MIN
):
    """Build the interned unified trie store the worker will open.

    Mirrors ``dataset_manager._build_graph_trie_stores``: the content-addressed
    pool AND every node's ``prompt_segment_ids`` manifest land in the ONE
    unified store. The worker opens it from the graph-typed dataset broadcast
    (``GraphSegmentClientMetadata``), which we set directly here.
    """
    parsed = from_weka_trace(str(fixture))
    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id=worker.run.benchmark_id
    )
    addr = await build_unified_trie_store_interned(parsed, store)
    worker._graph_client_metadata = _graph_client_metadata(
        tmp_path, worker.run.benchmark_id
    )
    return parsed, addr


# ---------------------------------------------------------------------------
# F3 -- payload features carried onto the wire
# ---------------------------------------------------------------------------


async def _capture_graph_wire_payload(
    mock_worker,
    tmp_path,
    monkeypatch,
    *,
    global_streaming: bool,
    fixture: Path = FIX_MIN,
    node_key: str = "trace_03_n3:2",
) -> dict:
    """Drive one graph credit through the worker, formatting the final wire
    payload exactly as the inference client would, and return it.

    ``global_streaming`` sets the run-level ``endpoint.streaming`` -- the FALLBACK
    for a mode-less node; the recorded per-node mode (``node_key`` in ``fixture``)
    wins for the wire ``stream`` byte.
    """
    parsed, addr = await _build_store_for_worker(mock_worker, tmp_path, fixture=fixture)
    t0 = parsed.traces[0].id
    leaf_ordinal = addr[t0][node_key]

    endpoint = mock_worker.model_endpoint.endpoint
    monkeypatch.setattr(endpoint, "streaming", global_streaming)
    monkeypatch.setattr(endpoint, "use_server_token_count", True)
    monkeypatch.setattr(endpoint, "extra", [("guided_decoding_backend", "outlines")])

    captured: dict = {}

    async def capture_wire(request_info, first_token_callback=None):
        # Mirror the real three-way select in _send_request_to_transport:
        # raw_payload_bytes (verbatim) -> raw_payload (dict) -> format_payload.
        rpb = request_info.turns[-1].raw_payload_bytes
        raw_payload = request_info.turns[-1].raw_payload
        if rpb is not None:
            payload = orjson.loads(rpb)
        elif raw_payload is not None:
            payload = raw_payload
        else:
            payload = mock_worker.inference_client.endpoint.format_payload(request_info)
        request_info.payload_bytes = orjson.dumps(payload)
        captured["wire_payload"] = payload
        return RequestRecord(
            request_info=request_info,
            timestamp_ns=time.time_ns(),
            start_perf_ns=time.perf_counter_ns(),
            end_perf_ns=time.perf_counter_ns(),
        )

    monkeypatch.setattr(
        mock_worker.inference_client,
        "send_request",
        AsyncMock(side_effect=capture_wire),
    )
    mock_worker._send_inference_result_message = AsyncMock()

    credit = Credit(
        id=21,
        phase=CreditPhase.PROFILING,
        conversation_id=t0,
        x_correlation_id="x-21",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        trace_id=t0,
        node_ordinal=leaf_ordinal,
        phase_variant="profiling",
    )
    ctx = CreditContext(credit=credit, drop_perf_ns=time.perf_counter_ns())
    await mock_worker._process_credit(ctx)
    assert ctx.error is None
    return captured["wire_payload"]


@pytest.mark.asyncio
async def test_graph_payload_recorded_streaming_node_wins_over_global_off(
    mock_worker, tmp_path, monkeypatch
):
    """A recorded streaming (``"s"``) node streams on the wire even when the
    global run is NON-streaming; ``extra`` is merged and ``include_usage`` is
    layered because the FINAL stamped ``stream`` is True (server-token-count on).
    Asserted on the FINAL payload, not the intermediate materialized dict."""
    wire = await _capture_graph_wire_payload(
        mock_worker,
        tmp_path,
        monkeypatch,
        global_streaming=False,
        fixture=FIX_MIXED,
        node_key="trace_first_token_anchor:0",
    )
    assert wire["messages"], "materialized messages preserved"
    assert wire["guided_decoding_backend"] == "outlines", "endpoint.extra merged"
    assert wire["stream"] is True, "recorded 's' node streams despite global off"
    assert wire["stream_options"] == {"include_usage": True}, (
        "include_usage layered when the FINAL stream is on + server-token-count"
    )


@pytest.mark.asyncio
async def test_graph_payload_recorded_non_streaming_node_wins_over_global_on(
    mock_worker, tmp_path, monkeypatch
):
    """A recorded non-streaming (``"n"``) node stays non-streaming even when the
    global run streams; no ``stream_options`` is added even with
    server-token-count on (the FINAL stream is False), and ``extra`` is still
    merged."""
    wire = await _capture_graph_wire_payload(
        mock_worker, tmp_path, monkeypatch, global_streaming=True
    )
    assert wire["guided_decoding_backend"] == "outlines", "endpoint.extra still merged"
    assert wire["stream"] is False, (
        "recorded 'n' node stays non-streaming despite global on"
    )
    assert "stream_options" not in wire, "no usage opt-in when the FINAL stream is off"


@pytest.mark.asyncio
async def test_user_extra_clobbers_per_node_overrides(tmp_path):
    """User ``--extra-inputs`` CLOBBERS colliding per-node dispatch keys (agentx
    ``payload.update(endpoint.extra)`` precedence); non-colliding per-node keys
    pass through.

    On the trie path the per-node ``dispatch_overrides`` carry the recorded
    ``max_output_tokens`` cap (endpoint-mapped to ``max_completion_tokens``
    here) and ``model`` verbatim, so a run-level ``extra`` naming the mapped
    wire key collides; the user must win.
    """
    parsed = from_weka_trace(str(FIX_MIN))
    store = GraphSegmentUnifiedBackingStore(base_path=tmp_path, benchmark_id="ov")
    addr = await build_unified_trie_store_interned(parsed, store)

    t0 = parsed.traces[0].id
    ordinal = addr[t0]["trace_03_n3:0"]  # recorded out cap 25
    client = GraphSegmentUnifiedClient(base_path=tmp_path, benchmark_id="ov").open()

    from aiperf.common.models import EndpointInfo
    from aiperf.graph.worker_materialize import apply_run_level_payload_options

    payload = materialize_graph_request_unified(
        client,
        t0,
        ordinal,
        "profiling",
        use_legacy_max_tokens=False,
    )
    assert payload["max_completion_tokens"] == 25
    assert payload["model"] == "claude-opus-4-5-20251101"

    # Run-level extra collides with the per-node token cap + model; user wins.
    endpoint = EndpointInfo(
        type="chat",
        streaming=True,
        use_server_token_count=False,
        extra=[("max_completion_tokens", 999), ("model", "run-level-model")],
    )
    apply_run_level_payload_options(payload, endpoint)
    assert payload["max_completion_tokens"] == 999, "user extra clobbers node cap"
    assert payload["model"] == "run-level-model", "user extra clobbers node model"
    client.close()


@pytest.mark.asyncio
async def test_include_usage_skipped_when_global_not_streaming(tmp_path):
    """include_usage is only layered when the GLOBAL ``endpoint.streaming`` is on;
    a False per-node materialized stream does not suppress it once global is on,
    and a False global suppresses it regardless of the per-node value."""
    from aiperf.common.models import EndpointInfo
    from aiperf.graph.worker_materialize import apply_run_level_payload_options

    # Global off -> wire stream forced False -> no include_usage.
    endpoint_off = EndpointInfo(
        type="chat", streaming=False, use_server_token_count=True, extra=[]
    )
    payload = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
    apply_run_level_payload_options(payload, endpoint_off)
    assert payload["stream"] is False, "global streaming=False stamps false"
    assert "stream_options" not in payload

    # Global on -> wire stream forced True even though per-node was False ->
    # include_usage layered.
    endpoint_on = EndpointInfo(
        type="chat", streaming=True, use_server_token_count=True, extra=[]
    )
    payload2 = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    apply_run_level_payload_options(payload2, endpoint_on)
    assert payload2["stream"] is True, "global streaming=True stamps true"
    assert payload2["stream_options"] == {"include_usage": True}


def test_skip_endpoint_extra_leaves_adapter_owned_key_untouched() -> None:
    """``skip_endpoint_extra=True`` suppresses the ``endpoint.extra`` merge so an
    adapter that already folded ``--extra-inputs`` into ``dispatch_overrides`` at
    parse keeps its own value; the ``stream`` stamp and ``include_usage`` forcing
    are UNCHANGED by the flag."""
    from aiperf.common.models import EndpointInfo
    from aiperf.graph.worker_materialize import apply_run_level_payload_options

    endpoint = EndpointInfo(
        type="chat",
        streaming=True,
        use_server_token_count=True,
        extra=[("guided_decoding_backend", "run-level")],
    )
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "guided_decoding_backend": "adapter-owned",
    }
    apply_run_level_payload_options(payload, endpoint, skip_endpoint_extra=True)
    assert payload["guided_decoding_backend"] == "adapter-owned", (
        "skip_endpoint_extra leaves the adapter-owned key un-clobbered"
    )
    assert payload["stream"] is True, "stream stamp is unaffected by the flag"
    assert payload["stream_options"] == {"include_usage": True}, (
        "include_usage forcing is unaffected by the flag"
    )


def test_skip_endpoint_extra_default_false_still_clobbers() -> None:
    """Default ``skip_endpoint_extra=False`` is byte-identical to today: the
    user's ``endpoint.extra`` still clobbers a colliding per-node key."""
    from aiperf.common.models import EndpointInfo
    from aiperf.graph.worker_materialize import apply_run_level_payload_options

    endpoint = EndpointInfo(
        type="chat",
        streaming=False,
        use_server_token_count=False,
        extra=[("guided_decoding_backend", "run-level")],
    )
    payload = {
        "messages": [{"role": "user", "content": "hi"}],
        "guided_decoding_backend": "adapter-owned",
    }
    apply_run_level_payload_options(payload, endpoint)
    assert payload["guided_decoding_backend"] == "run-level", (
        "default behavior: user extra clobbers the per-node key"
    )


# ---------------------------------------------------------------------------
# F4 -- missing/corrupt store surfaces a clear fatal error (no silent retry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_store_surfaces_error_not_silent(
    mock_worker, tmp_path, monkeypatch
):
    """When the graph store is absent, the credit error is set with an
    actionable message; the inference client is NOT called."""
    # Broadcast points at an empty dir -- no store written -> open() will fail.
    mock_worker._graph_client_metadata = _graph_client_metadata(
        tmp_path, mock_worker.run.benchmark_id
    )
    monkeypatch.setattr(mock_worker.inference_client, "send_request", AsyncMock())
    mock_worker._send_inference_result_message = AsyncMock()

    credit = Credit(
        id=41,
        phase=CreditPhase.PROFILING,
        conversation_id="t-missing",
        x_correlation_id="x-41",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        trace_id="t-missing",
        node_ordinal=0,
        phase_variant="profiling",
    )
    ctx = CreditContext(credit=credit, drop_perf_ns=time.perf_counter_ns())
    await mock_worker._process_credit(ctx)

    mock_worker.inference_client.send_request.assert_not_awaited()
    assert ctx.error is not None, "missing store must set a fatal error, not swallow"
    # Actionable: names the store directory / base path so the operator can fix
    # the shared-FS / MMAP_BASE_PATH config.
    assert str(tmp_path) in ctx.error.message or "graph store" in ctx.error.message


@pytest.mark.asyncio
async def test_missing_store_does_not_retry_open_each_credit(
    mock_worker, tmp_path, monkeypatch
):
    """A failed store open is cached: subsequent credits do not re-attempt the
    open (no silent retry-loop)."""
    mock_worker._graph_client_metadata = _graph_client_metadata(
        tmp_path, mock_worker.run.benchmark_id
    )
    monkeypatch.setattr(mock_worker.inference_client, "send_request", AsyncMock())
    mock_worker._send_inference_result_message = AsyncMock()

    open_calls = {"n": 0}
    real_open = GraphSegmentUnifiedClient.open

    def counting_open(self):
        open_calls["n"] += 1
        return real_open(self)

    monkeypatch.setattr(GraphSegmentUnifiedClient, "open", counting_open)

    def make_credit(cid):
        return CreditContext(
            credit=Credit(
                id=cid,
                phase=CreditPhase.PROFILING,
                conversation_id="t-missing",
                x_correlation_id=f"x-{cid}",
                turn_index=0,
                num_turns=1,
                issued_at_ns=time.time_ns(),
                trace_id="t-missing",
                node_ordinal=0,
                phase_variant="profiling",
            ),
            drop_perf_ns=time.perf_counter_ns(),
        )

    await mock_worker._process_credit(make_credit(50))
    await mock_worker._process_credit(make_credit(51))
    await mock_worker._process_credit(make_credit(52))

    assert open_calls["n"] == 1, (
        "store open must be attempted once and the failure cached, "
        f"not retried every credit (got {open_calls['n']} opens)"
    )


@pytest.mark.asyncio
async def test_graph_envelope_extra_headers_reach_request_turn(mock_worker):
    """Envelope ``extra_headers`` (dynamo ``x-dynamo-*`` session identity) land
    on the synthetic Turn -- the transport merges the LAST turn's extra_headers
    into the request headers -- and never leak into the body payload."""
    headers = {
        "x-dynamo-session-id": "sess-1",
        "x-dynamo-parent-session-id": "root-1",
        "x-dynamo-session-final": "true",
    }
    credit = Credit(
        id=7,
        phase=CreditPhase.PROFILING,
        conversation_id="t-1#0",
        x_correlation_id="x-7",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        trace_id="t-1#0",
        node_ordinal=0,
        phase_variant="profiling",
    )
    ctx = CreditContext(credit=credit, drop_perf_ns=time.perf_counter_ns())

    payload = {"messages": [{"role": "user", "content": "hi"}], "model": "m"}
    info = mock_worker._build_graph_request_info(
        ctx, payload, "x-req-7", extra_headers=headers
    )
    assert info.turns[-1].extra_headers == headers
    assert "extra_headers" not in payload and "nvext" not in payload

    info_bytes = mock_worker._build_graph_request_info(
        ctx,
        None,
        "x-req-7",
        raw_payload_bytes=b"{}",
        extra_headers=headers,
    )
    assert info_bytes.turns[-1].extra_headers == headers

    info_none = mock_worker._build_graph_request_info(ctx, payload, "x-req-7")
    assert info_none.turns[-1].extra_headers is None


# ---------------------------------------------------------------------------
# N1 -- dynamo session identity uniquified per replay instance
# ---------------------------------------------------------------------------

RECORDED_SESSION_HEADERS = {
    "x-dynamo-session-id": "sess-X",
    "x-dynamo-parent-session-id": "root-X",
    "x-dynamo-session-final": "true",
}


def test_uniquify_two_instances_get_distinct_linked_session_ids() -> None:
    """Two replay instances of ONE trace must open DISTINCT server sessions
    (the first finisher's session-final would otherwise evict KV under the
    still-running sibling); parent linkage transforms with the SAME suffix and
    session-final is forwarded untouched."""
    h0 = uniquify_dynamo_session_headers(
        dict(RECORDED_SESSION_HEADERS),
        trace_instance_id="t-1::nonceA",
        phase_variant="profiling",
    )
    h1 = uniquify_dynamo_session_headers(
        dict(RECORDED_SESSION_HEADERS),
        trace_instance_id="t-1::nonceB",
        phase_variant="profiling",
    )
    assert h0["x-dynamo-session-id"] != h1["x-dynamo-session-id"]
    for h in (h0, h1):
        assert h["x-dynamo-session-id"] != "sess-X"
        suffix = h["x-dynamo-session-id"].removeprefix("sess-X")
        assert suffix, "instance suffix must be appended to the recorded id"
        # Parent gets the SAME suffix: intra-instance parent-child linkage holds.
        assert h["x-dynamo-parent-session-id"] == f"root-X{suffix}"
        # session-final is a per-turn flag, not an identity: untouched, so each
        # instance still closes (only) its own session on its final turn.
        assert h["x-dynamo-session-final"] == "true"


def test_uniquify_deterministic_within_instance_and_phase() -> None:
    """Every dispatch of one instance must agree on the transformed ids (the
    stateless worker recomputes per credit), and a warmup instance must not
    collide with the profiling instance of the same (lane, pass) slot."""
    kwargs = {"trace_instance_id": "t-1::nonce0", "phase_variant": "profiling"}
    first = uniquify_dynamo_session_headers(dict(RECORDED_SESSION_HEADERS), **kwargs)
    second = uniquify_dynamo_session_headers(dict(RECORDED_SESSION_HEADERS), **kwargs)
    assert first == second

    warmup = uniquify_dynamo_session_headers(
        dict(RECORDED_SESSION_HEADERS),
        trace_instance_id="t-1::nonce0",
        phase_variant="warmup",
    )
    assert warmup["x-dynamo-session-id"] != first["x-dynamo-session-id"]


@pytest.mark.parametrize(
    "extra_headers, trace_instance_id",
    [
        param(None, "t-1::n", id="no_headers"),
        param({}, "t-1::n", id="empty_headers"),
        param({"x-other": "v"}, "t-1::n", id="no_dynamo_identity_header"),
        param(dict(RECORDED_SESSION_HEADERS), "t-1", id="no_instance_suffix"),
        param(dict(RECORDED_SESSION_HEADERS), None, id="no_trace_id"),
    ],
)  # fmt: skip
def test_uniquify_noop_paths_return_input_unchanged(
    extra_headers: dict | None, trace_instance_id: str | None
) -> None:
    """Plain (non-instanced) replay and header-less nodes are unaffected."""
    result = uniquify_dynamo_session_headers(
        extra_headers,
        trace_instance_id=trace_instance_id,
        phase_variant="profiling",
    )
    assert result is extra_headers


def test_strip_dynamo_session_headers_removes_identity_keeps_rest() -> None:
    """With a live --session-routing plugin owning session identity, the
    RECORDED identity headers (session id / parent / session-final) are stale
    replay artifacts and must not ride the wire beside the plugin's live
    headers; unrelated recorded headers survive."""
    headers = {**RECORDED_SESSION_HEADERS, "x-custom": "keep-me"}
    stripped = strip_dynamo_session_headers(headers)
    assert stripped == {"x-custom": "keep-me"}
    # Identity-only header sets strip to None (no empty-dict envelope noise).
    assert strip_dynamo_session_headers(dict(RECORDED_SESSION_HEADERS)) is None
    # No-op paths return the input unchanged.
    assert strip_dynamo_session_headers(None) is None
    plain = {"x-custom": "keep-me"}
    assert strip_dynamo_session_headers(plain) is plain


async def _build_session_headers_store(worker: Worker, tmp_path: Path):
    """One-node unified store whose envelope carries recorded dynamo headers."""
    store = GraphSegmentUnifiedBackingStore(
        base_path=tmp_path, benchmark_id=worker.run.benchmark_id
    )
    handle = store.put_segment("seg0", "user", "hello")
    store.add_node_manifest(
        "t-1",
        0,
        "profiling",
        orjson.dumps({"handles": [handle], "extra_headers": RECORDED_SESSION_HEADERS}),
    )
    await store.finalize()
    worker._graph_client_metadata = _graph_client_metadata(
        tmp_path, worker.run.benchmark_id
    )


@pytest.mark.asyncio
async def test_graph_credit_flow_uniquifies_session_headers_per_instance(
    mock_worker, tmp_path, monkeypatch
) -> None:
    """End to end through ``_process_credit``: two credits addressing the SAME
    trace under different instance ids reach the wire with DIFFERENT
    x-dynamo-session-id values, consistently-transformed parent ids, and an
    untouched per-instance session-final."""
    await _build_session_headers_store(mock_worker, tmp_path)

    captured: list[dict] = []

    async def capture_headers(request_info, first_token_callback=None):
        captured.append(request_info.turns[-1].extra_headers)
        return RequestRecord(
            request_info=request_info,
            timestamp_ns=time.time_ns(),
            start_perf_ns=time.perf_counter_ns(),
            end_perf_ns=time.perf_counter_ns(),
        )

    monkeypatch.setattr(
        mock_worker.inference_client,
        "send_request",
        AsyncMock(side_effect=capture_headers),
    )
    mock_worker._send_inference_result_message = AsyncMock()

    for i, instance_id in enumerate(("t-1::inst0", "t-1::inst1")):
        credit = Credit(
            id=100 + i,
            phase=CreditPhase.PROFILING,
            conversation_id="t-1",
            x_correlation_id=f"x-{100 + i}",
            turn_index=0,
            num_turns=1,
            issued_at_ns=time.time_ns(),
            trace_id=instance_id,
            node_ordinal=0,
            phase_variant="profiling",
        )
        ctx = CreditContext(credit=credit, drop_perf_ns=time.perf_counter_ns())
        await mock_worker._process_credit(ctx)
        assert ctx.error is None

    h0, h1 = captured
    assert h0["x-dynamo-session-id"] != h1["x-dynamo-session-id"]
    for h in (h0, h1):
        assert h["x-dynamo-session-id"] != "sess-X"
        suffix = h["x-dynamo-session-id"].removeprefix("sess-X")
        assert h["x-dynamo-parent-session-id"] == f"root-X{suffix}"
        assert h["x-dynamo-session-final"] == "true"


@pytest.mark.asyncio
async def test_graph_credit_flow_strips_recorded_headers_when_routing_active(
    mock_worker, tmp_path, monkeypatch
) -> None:
    """With a live --session-routing plugin, the plugin OWNS session identity:
    the recorded x-dynamo-* identity headers are STRIPPED (not uniquified),
    and the graph RequestInfo carries the live routing facts the chokepoint
    stamps from (corr / root corr / recorded finality)."""
    await _build_session_headers_store(mock_worker, tmp_path)
    monkeypatch.setattr(
        type(mock_worker.inference_client),
        "session_routing_active",
        property(lambda self: True),
    )

    captured: list = []

    async def capture_request_info(request_info, first_token_callback=None):
        captured.append(request_info)
        return RequestRecord(
            request_info=request_info,
            timestamp_ns=time.time_ns(),
            start_perf_ns=time.perf_counter_ns(),
            end_perf_ns=time.perf_counter_ns(),
        )

    monkeypatch.setattr(
        mock_worker.inference_client,
        "send_request",
        AsyncMock(side_effect=capture_request_info),
    )
    mock_worker._send_inference_result_message = AsyncMock()

    credit = Credit(
        id=300,
        phase=CreditPhase.PROFILING,
        conversation_id="t-1",
        x_correlation_id="t-1::corr300",
        turn_index=0,
        num_turns=1,
        issued_at_ns=time.time_ns(),
        trace_id="t-1::inst0",
        node_ordinal=0,
        phase_variant="profiling",
        root_correlation_id="t-1::root-corr",
    )
    ctx = CreditContext(credit=credit, drop_perf_ns=time.perf_counter_ns())
    await mock_worker._process_credit(ctx)
    assert ctx.error is None

    (request_info,) = captured
    # Recorded dynamo identity headers are gone (identity-only set -> None).
    assert request_info.turns[-1].extra_headers is None
    # Routing identity facts ride the RequestInfo for the chokepoint.
    assert request_info.x_correlation_id == "t-1::corr300"
    assert request_info.root_correlation_id == "t-1::root-corr"
    assert request_info.is_final_turn is True  # num_turns=1: recorded final
    assert request_info.is_parent_final is None
    assert request_info.is_tree_final is False
