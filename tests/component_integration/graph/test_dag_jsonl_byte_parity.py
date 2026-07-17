# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process wire-parity golden test: ``--graph-format dag_jsonl`` vs legacy
``--custom-dataset-type dag_jsonl``.

The core acceptance gate for the dag_jsonl graph adapter: BOTH full production
paths run in one process against the SAME fixture with a deterministic fake
transport, and every wire request body the graph path produces must be
payload-identical to the legacy path's (canonical order-insensitive bytes).

Harness seams:

* Both sides run the full ``aiperf profile`` CLI in-process via the ``cli``
  fixture (FakeServiceManager + FakeCommunication). The wire seam is
  :class:`tests.harness.fake_transport.FakeTransport` -- ``send_request`` is
  monkeypatched to capture ``request_info.payload_bytes`` (the canonical
  pre-encoded body ``inference_client`` stamps before transport dispatch on
  the format_payload, raw_payload, AND raw_payload_bytes paths) and to return
  a deterministic non-streaming chat completion whose text is a pure function
  of the request's ``messages`` bytes. Identical request messages therefore
  produce identical replies on both harnesses, so live-reply splices (FORK
  children, multi-turn accumulators) compose inductively: parity of every
  parent request implies parity of every child request.
* Requests are keyed ``(session_id, turn_index)``. Legacy: straight off
  ``RequestInfo.conversation_id`` / ``turn_index``. Graph: recovered from the
  minted ``x_request_id`` (the worker's ``_mint_x_request_id``:
  ``{node_id}::{nonce}`` where the nonce is a ``uuid4().hex`` with no ``::``);
  split off the trailing ``::<nonce>`` from the RIGHT, then split the node id
  (``"<session>[#n]:<turn_idx>"``) on its trailing ``:<turn_idx>``. The graph
  ``x_correlation_id`` is now an opaque per-trajectory id carrying no node
  identity, so it is NOT parsed. Profiling-phase is checked via
  ``RequestInfo.credit_phase`` on both planes.

Pinned environment facts that make parity hold (asserted where cheap):

* ``AIPERF_GRAPH_MERGE_CONSECUTIVE_USER`` default False stays in effect.
* No ``--cache-bust`` (default NONE -- no ``[rid:...]`` marker, which also
  keeps the graph worker on the pre-serialized bytes path for lineage-free
  nodes).
* No warmup on either side (PROFILING-only comparison; every capture is
  asserted PROFILING via ``credit_phase`` on both planes).
* Single worker on both sides (legacy FORK seeding and the graph dynamic
  pool are both worker-local).
* ``--extra-inputs`` parity: the dag graph adapter folds ``endpoint.extra``
  into ``dispatch_overrides`` at parse (legacy position/precedence: turn
  ``extra`` wins on overlap, first insertion keeps position) and stamps
  ``endpoint_extra_applied`` so the worker skips its re-merge. The
  ``mixed-full-extra-inputs`` case proves it: one key overlapping the
  fixture's turn ``extra`` (different value) plus one fresh vendor key.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.models import RequestInfo, RequestRecord, TextResponse
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.fake_transport import FakeTransport
from tests.harness.utils import AIPerfCLI

pytestmark = pytest.mark.component_integration

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "dag" / "graph_parity"

# Markers the tool_calls responder keys on (matched against the LAST user
# message content only, so an inherited ancestor marker never re-triggers on a
# child request). ``[tool]`` is intentionally NOT a substring of ``[tool-only]``.
_TOOL_MARKER = "[tool]"
_TOOL_ONLY_MARKER = "[tool-only]"


@dataclass(frozen=True, slots=True)
class _WireRequest:
    """One captured transport dispatch: identity fields + canonical body bytes."""

    x_request_id: str
    """The credit's per-dispatch request id (legacy: opaque UUID; graph: minted
    ``{node_id}::{nonce}`` -- the per-node identity the graph key recovers)."""
    x_correlation_id: str
    """The credit's correlation id (legacy: per-session UUID; graph: an opaque
    per-trajectory ``{conversation}::{nonce}`` carrying NO node identity)."""
    conversation_id: str
    """``RequestInfo.conversation_id`` (legacy: the dag session id; graph: the
    trajectory template id -- NOT per-node, hence the x_request_id parse)."""
    turn_index: int
    """``RequestInfo.turn_index`` (legacy: turn within the session; graph: the
    node's 0-based turn)."""
    credit_phase: CreditPhase
    """The credit phase this request was dispatched under."""
    body: bytes
    """``request_info.payload_bytes`` -- the canonical wire body."""


def _deterministic_response_text(messages_bytes: bytes) -> str:
    """Reply text as a pure function of the request's ``messages`` bytes."""
    return f"resp-{hashlib.sha256(messages_bytes).hexdigest()[:8]}"


def _deterministic_tool_calls(messages_bytes: bytes) -> list[dict[str, Any]]:
    """One ``tool_calls`` entry as a pure function of the request's messages.

    The id/name/arguments all derive from the SAME hash that seeds the reply
    text, so identical requests synthesize an identical tool_call on BOTH
    planes and child prompts that splice the reply stay byte-identical.
    """
    digest = hashlib.sha256(messages_bytes).hexdigest()
    return [
        {
            "id": f"call_{digest[:12]}",
            "type": "function",
            "function": {
                "name": f"fn_{digest[:8]}",
                "arguments": orjson.dumps({"q": digest[8:16]}).decode("utf-8"),
            },
        }
    ]


def _last_user_content(messages: list[dict[str, Any]]) -> str:
    """The content of the LAST ``user`` message (``""`` when none / non-str)."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _install_capture_transport(
    monkeypatch: pytest.MonkeyPatch,
    sink: list[_WireRequest],
    *,
    tool_calls_mode: bool = False,
) -> None:
    """Replace ``FakeTransport.send_request`` with a deterministic responder.

    Captures every dispatched body into ``sink`` and answers with a
    non-streaming ``chat.completion`` whose content depends only on the
    request's ``messages`` bytes, so both harnesses see identical replies for
    identical requests (and therefore build identical child prompts).

    When ``tool_calls_mode`` is set (the ``fork_toolcalls`` fixture only), a
    request whose LAST user message carries a marker gets a structured reply:
    ``[tool-only]`` -> ``content: null`` + ``tool_calls`` (tool-only);
    ``[tool]`` -> the usual ``resp-<hash>`` content PLUS ``tool_calls``
    (mixed). Determinism is unchanged (pure function of the request bytes), and
    the responder is identical on both planes, so parity composes inductively.
    """

    async def _send_request(
        self: FakeTransport,
        request_info: RequestInfo,
        payload: Any,
        *,
        first_token_callback: Any = None,
    ) -> RequestRecord:
        body = request_info.payload_bytes
        assert body is not None, (
            "inference_client must stamp payload_bytes before transport dispatch"
        )
        sink.append(
            _WireRequest(
                x_request_id=request_info.x_request_id,
                x_correlation_id=request_info.x_correlation_id,
                conversation_id=request_info.conversation_id,
                turn_index=request_info.turn_index,
                credit_phase=request_info.credit_phase,
                body=bytes(body),
            )
        )
        parsed = orjson.loads(body)
        messages_bytes = orjson.dumps(parsed["messages"])
        text = _deterministic_response_text(messages_bytes)
        message: dict[str, Any] = {"role": "assistant", "content": text}
        finish_reason = "stop"
        if tool_calls_mode:
            marker = _last_user_content(parsed["messages"])
            if _TOOL_ONLY_MARKER in marker:
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": _deterministic_tool_calls(messages_bytes),
                }
                finish_reason = "tool_calls"
            elif _TOOL_MARKER in marker:
                message = {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": _deterministic_tool_calls(messages_bytes),
                }
                finish_reason = "tool_calls"
        response_data = {
            "id": "chatcmpl-parity",
            "object": "chat.completion",
            "created": 0,
            "model": parsed.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }
        start_perf_ns = time.perf_counter_ns()
        end_perf_ns = time.perf_counter_ns()
        return RequestRecord(
            start_perf_ns=start_perf_ns,
            end_perf_ns=end_perf_ns,
            timestamp_ns=time.time_ns(),
            status=200,
            responses=[
                TextResponse(
                    perf_ns=end_perf_ns,
                    content_type="application/json",
                    text=orjson.dumps(response_data).decode("utf-8"),
                )
            ],
        )

    monkeypatch.setattr(FakeTransport, "send_request", _send_request)


def _reset_inprocess_state() -> None:
    """Isolate consecutive in-process CLI runs within ONE test.

    The package conftest clears singletons / reseeds the RNG only BETWEEN
    tests; this test runs ``aiperf profile`` twice in one test body, so the
    second run must not see the first run's singleton comms/services or RNG
    stream.
    """
    from aiperf.common import random_generator as rng
    from aiperf.common.singleton import SingletonMeta

    SingletonMeta._instances.clear()
    rng.reset()
    rng.init(42)


def _run_and_capture(
    cli: AIPerfCLI, sink: list[_WireRequest], cmd: str
) -> list[_WireRequest]:
    """Run one in-process profile and return the requests it dispatched."""
    _reset_inprocess_state()
    sink.clear()
    cli.run_sync(cmd, timeout=120.0, assert_success=True)
    return list(sink)


def _keyed_legacy(captures: list[_WireRequest]) -> dict[tuple[str, int], list[bytes]]:
    """Key legacy captures by ``(session_id, turn_index)`` into a multiset.

    A SPAWN template instantiated N times keys N bodies under the same
    ``(session, turn)`` (legacy carries the TEMPLATE ``conversation_id``, so
    repeated spawns share the key), so the value is a LIST -- one body per
    dispatch under that key.
    """
    keyed: dict[tuple[str, int], list[bytes]] = {}
    for c in captures:
        assert c.credit_phase == CreditPhase.PROFILING, (
            f"legacy request outside PROFILING: {c.credit_phase} "
            f"(warmup must stay disabled for parity)"
        )
        keyed.setdefault((c.conversation_id, c.turn_index), []).append(c.body)
    return keyed


def _keyed_graph(captures: list[_WireRequest]) -> dict[tuple[str, int], list[bytes]]:
    """Key graph captures by ``(session_id, turn_index)`` from x_request_id.

    ``_mint_x_request_id`` folds the per-dispatch node identity into
    ``x_request_id`` as ``{node_id}::{nonce}``; the nonce is a ``uuid4().hex``
    with no ``::``, so split it off from the RIGHT. Node ids are
    ``"<session>[#n]:<turn_idx>"`` (``#n`` only for repeated SPAWN instances;
    ``#`` is gated out of session ids at load time). Repeated SPAWN instances
    strip their ``#n`` and accumulate under one key as a multiset -- one body
    per instance. Profiling-phase is enforced via ``credit_phase``.
    """
    keyed: dict[tuple[str, int], list[bytes]] = {}
    for c in captures:
        assert c.credit_phase == CreditPhase.PROFILING, (
            f"graph request outside PROFILING: {c.credit_phase} "
            f"(warmup must stay disabled for parity)"
        )
        assert "::" in c.x_request_id, (
            f"unparsable graph x_request_id {c.x_request_id!r} (no '::' nonce delimiter)"
        )
        node_id = c.x_request_id.rsplit("::", 1)[0]
        session_part, sep, turn_str = node_id.rpartition(":")
        assert sep and turn_str.isdigit(), (
            f"graph node id {node_id!r} does not end in ':<turn_idx>'"
        )
        key = (session_part.split("#", 1)[0], int(turn_str))
        keyed.setdefault(key, []).append(c.body)
    return keyed


def _fmt_body(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def _canonical_bytes(body: bytes) -> bytes:
    """Canonical order-insensitive form: re-serialize with sorted keys."""
    return orjson.dumps(orjson.loads(body), option=orjson.OPT_SORT_KEYS)


def _has_live_reply_message(body: bytes) -> bool:
    """True if ``body``'s ``messages`` embed a captured live assistant reply.

    The fake transport answers every request with an assistant message whose
    content is ``resp-<hash>`` (:func:`_deterministic_response_text`), so a
    spliced-in reply is any ``role == "assistant"`` message whose content
    starts with ``"resp-"``.
    """
    payload = orjson.loads(body)
    return any(
        msg.get("role") == "assistant"
        and str(msg.get("content", "")).startswith("resp-")
        for msg in payload.get("messages", [])
    )


def _has_tool_calls_message(body: bytes) -> bool:
    """True if ``body`` embeds an assistant message carrying ``tool_calls``.

    Anti-vacuous guard for the tool_calls parity case: a spliced-in reply that
    reassembled to a ``tool_calls``-bearing assistant message must actually
    reach the child's wire body, or a symmetric drop would pass the multiset
    comparison while gutting all tool_calls splice coverage.
    """
    payload = orjson.loads(body)
    return any(
        msg.get("role") == "assistant" and "tool_calls" in msg
        for msg in payload.get("messages", [])
    )


def _has_content_null_tool_calls_message(body: bytes) -> bool:
    """True if ``body`` embeds a ``content: null`` tool_calls-only assistant."""
    payload = orjson.loads(body)
    return any(
        msg.get("role") == "assistant"
        and "tool_calls" in msg
        and msg.get("content") is None
        for msg in payload.get("messages", [])
    )


@pytest.mark.parametrize(
    "fixture_name,expected_requests,tool_calls_mode,extra_inputs",
    [
        param("fork_minimal.dag.jsonl", 5, False, None, id="fork-minimal"),
        param("spawn_join.dag.jsonl", 5, False, None, id="spawn-join"),
        param("mixed_full.dag.jsonl", 6, False, None, id="mixed-full"),
        param("spawn_repeat.dag.jsonl", 4, False, None, id="spawn-repeat"),
        param("fork_toolcalls.dag.jsonl", 3, True, None, id="fork-toolcalls"),
        # --extra-inputs precedence parity: ``temperature`` OVERLAPS the
        # fixture's turn ``extra`` (root:0 authors 0.5, finisher:0 authors 0.1;
        # the run supplies 0.9 -- turn extra must win on BOTH planes) and
        # ``min_p`` is a fresh vendor key (must land at the endpoint-extra
        # position, before the turn-extra-only keys, on BOTH planes).
        param(
            "mixed_full.dag.jsonl",
            6,
            False,
            "temperature:0.9 min_p:0.05",
            id="mixed-full-extra-inputs",
        ),
    ],
)  # fmt: skip
def test_dag_jsonl_graph_adapter_wire_bytes_match_legacy(
    cli: AIPerfCLI,
    mmap_base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_name: str,
    expected_requests: int,
    tool_calls_mode: bool,
    extra_inputs: str | None,
) -> None:
    """Every PROFILING wire body from the graph path byte-matches legacy.

    Three criteria per request, keyed ``(session_id, turn_index)``:

    1. ``payload["messages"]`` byte-equal (roots, FORK children with live
       parent replies spliced in, SPAWN children, join turns).
    2. Full payload parsed-equal.
    3. Canonical (sorted-keys) body byte-equal -- order-insensitive on key
       position, byte-strict on values and types.

    Plus: request COUNTS match and no request matched on only one side.
    """
    # Pin the environment facts parity depends on.
    assert Environment.GRAPH.MERGE_CONSECUTIVE_USER is False, (
        "AIPERF_GRAPH_MERGE_CONSECUTIVE_USER must stay at its False default"
    )

    fixture = _FIXTURE_DIR / fixture_name
    assert fixture.is_file(), f"missing fixture {fixture}"

    sink: list[_WireRequest] = []
    _install_capture_transport(monkeypatch, sink, tool_calls_mode=tool_calls_mode)

    # Shared per-case flags MUST be appended identically to both planes.
    extra_args = f"--extra-inputs {extra_inputs} " if extra_inputs else ""

    # The legacy plane needs --concurrency 2: fork fanout dispatches children as
    # separate sessions, so a single slot would serialize a parent and its child
    # and stall the fork seeding. The graph plane omits it because its replay lanes
    # self-schedule fork/spawn children off the parent's completion.
    legacy_captures = _run_and_capture(
        cli,
        sink,
        f"""
        aiperf profile \
            --model {defaults.model} \
            --custom-dataset-type dag_jsonl \
            --input-file {fixture} \
            --concurrency 2 \
            --num-conversations 1 \
            --record-processor-service-count 1 \
            --workers-max 1 \
            {extra_args}--ui {defaults.ui}
        """,
    )
    graph_captures = _run_and_capture(
        cli,
        sink,
        f"""
        aiperf profile \
            --model {defaults.model} \
            --graph-format dag_jsonl \
            --input-file {fixture} \
            --num-conversations 1 \
            --record-processor-service-count 1 \
            --workers-max 1 \
            {extra_args}--ui {defaults.ui}
        """,
    )

    assert len(legacy_captures) == expected_requests, (
        f"legacy dispatched {len(legacy_captures)} requests, "
        f"expected {expected_requests}"
    )
    assert len(graph_captures) == expected_requests, (
        f"graph dispatched {len(graph_captures)} requests, expected {expected_requests}"
    )

    legacy_by_key = _keyed_legacy(legacy_captures)
    graph_by_key = _keyed_graph(graph_captures)

    only_legacy = sorted(set(legacy_by_key) - set(graph_by_key))
    only_graph = sorted(set(graph_by_key) - set(legacy_by_key))
    assert not only_legacy and not only_graph, (
        f"request sets diverge: only-legacy={only_legacy} only-graph={only_graph}"
    )

    for key in sorted(legacy_by_key):
        # Repeated SPAWN templates key multiple bodies under one (session, turn);
        # compare as a multiset -- sort each side and pair element-wise. Per-key
        # multiplicity must also match (a template fired N times on one plane and
        # M on the other is a divergence).
        legacy_bodies = sorted(legacy_by_key[key], key=_canonical_bytes)
        graph_bodies = sorted(graph_by_key[key], key=_canonical_bytes)
        assert len(legacy_bodies) == len(graph_bodies), (
            f"{key}: request multiplicity diverges "
            f"legacy={len(legacy_bodies)} graph={len(graph_bodies)}"
        )

        for idx, (legacy_body, graph_body) in enumerate(
            zip(legacy_bodies, graph_bodies, strict=True)
        ):
            where = f"{key} #{idx}" if len(legacy_bodies) > 1 else f"{key}"
            legacy_payload = orjson.loads(legacy_body)
            graph_payload = orjson.loads(graph_body)

            # Criterion 1: messages byte-equal for EVERY request.
            assert orjson.dumps(legacy_payload["messages"]) == orjson.dumps(
                graph_payload["messages"]
            ), (
                f"{where}: messages diverge\n"
                f"  legacy: {legacy_payload['messages']}\n"
                f"  graph:  {graph_payload['messages']}"
            )

            # Criterion 2: full payload parsed-equal.
            assert legacy_payload == graph_payload, (
                f"{where}: parsed payloads diverge\n"
                f"  legacy: {legacy_payload}\n"
                f"  graph:  {graph_payload}"
            )

            # Criterion 3: canonical (sorted-keys) body byte-equal. Key order is
            # deliberately NOT compared -- the graph plane authors stream / the
            # token cap / model through native node fields, so their wire
            # positions differ from legacy; canonical bytes stay strict on
            # values AND types (1 vs 1.0, True vs 1) that criterion 2 conflates.
            assert _canonical_bytes(legacy_body) == _canonical_bytes(graph_body), (
                f"{where}: parsed-equal but CANONICALLY byte-different "
                f"(value/type drift)\n"
                f"  legacy: {_fmt_body(legacy_body)}\n"
                f"  graph:  {_fmt_body(graph_body)}"
            )

    # Self-verification: the gate must NOT pass by both paths symmetrically
    # dropping the spliced-in live replies (both omit the captured turn when
    # build_assistant_turn returns None). At least one compared graph body must
    # actually embed a captured assistant reply (content ``resp-*`` from the
    # fake transport) -- otherwise a fake-response-shape or parse regression
    # would silently gut all splice-position coverage while parity still holds.
    assert any(
        _has_live_reply_message(body)
        for bodies in graph_by_key.values()
        for body in bodies
    ), (
        "no graph request embedded a live assistant reply (resp-*); reply "
        "capture/parse regressed and splice-position coverage was silently lost"
    )

    # Fixture-specific: fork_minimal's branch-a turn-0 is a FORK child whose
    # prompt must embed the parent root turn's spliced live reply.
    if fixture_name == "fork_minimal.dag.jsonl":
        branch_a_bodies = graph_by_key.get(("branch-a", 0))
        assert branch_a_bodies, (
            "fork_minimal: expected a graph request keyed ('branch-a', 0)"
        )
        assert _has_live_reply_message(branch_a_bodies[0]), (
            "fork_minimal ('branch-a', 0): FORK-child prompt is missing the "
            "spliced parent reply (resp-*)"
        )

    # Fixture-specific: fork_toolcalls proves tool_calls-bearing replies round
    # -trip byte-identically. The root reply is mixed (content + tool_calls) and
    # must splice into the fork child's turn-0 prompt; the child's own turn-0
    # reply is content:null tool_calls-only and must splice into turn-1's prompt.
    if fixture_name == "fork_toolcalls.dag.jsonl":
        child_t0_bodies = graph_by_key.get(("branch-a", 0))
        assert child_t0_bodies, (
            "fork_toolcalls: expected a graph request keyed ('branch-a', 0)"
        )
        assert _has_tool_calls_message(child_t0_bodies[0]), (
            "fork_toolcalls ('branch-a', 0): FORK-child prompt is missing the "
            "spliced parent assistant message carrying tool_calls"
        )
        child_t1_bodies = graph_by_key.get(("branch-a", 1))
        assert child_t1_bodies, (
            "fork_toolcalls: expected a graph request keyed ('branch-a', 1)"
        )
        assert _has_content_null_tool_calls_message(child_t1_bodies[0]), (
            "fork_toolcalls ('branch-a', 1): prompt is missing the spliced "
            "content:null tool_calls-only assistant message"
        )

    # Fixture-specific: the extra-inputs case must not pass vacuously (both
    # planes symmetrically dropping the run extras would keep parity). The
    # overlap key keeps the TURN value (legacy precedence) at the endpoint-extra
    # position, and the fresh vendor key lands on every request.
    if extra_inputs is not None:
        root_body = orjson.loads(graph_by_key[("root", 0)][0])
        assert root_body["temperature"] == 0.5, (
            "overlap key: the turn 'extra' value must beat --extra-inputs"
        )
        assert root_body["min_p"] == 0.05
        keys = list(root_body)
        assert keys.index("temperature") < keys.index("min_p") < keys.index("top_p"), (
            f"endpoint-extra position drifted: {keys}"
        )
        finisher_body = orjson.loads(graph_by_key[("finisher", 0)][0])
        assert finisher_body["temperature"] == 0.1, (
            "overlap key: the turn 'extra' value must beat --extra-inputs"
        )
        helper_body = orjson.loads(graph_by_key[("helper", 0)][0])
        assert helper_body["temperature"] == 0.9, (
            "no turn overlap: the run-level --extra-inputs value must land"
        )
        assert helper_body["min_p"] == 0.05
