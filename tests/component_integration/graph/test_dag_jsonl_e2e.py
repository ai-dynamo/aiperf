# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process RUNTIME end-to-end proof for ``--graph-format dag_jsonl``.

Where :mod:`test_dag_jsonl_byte_parity` proves the graph path builds the SAME
wire bytes as legacy, this module proves the graph path RUNS with the right
firing semantics: it drives the full ``aiperf profile`` CLI in-process (the
``GraphIRReplayStrategy`` -> per-trace ``TraceExecutor`` -> ``CreditDispatchAdapter``
-> worker materialize path) against a deterministic fake transport and asserts
on the observed dispatch/completion ordering and per-turn delays.

Harness seams (shared with the byte-parity test's pattern):

* The wire seam is :meth:`tests.harness.fake_transport.FakeTransport.send_request`,
  monkeypatched to (a) capture ``request_info.payload_bytes`` (the canonical
  pre-encoded body) plus a monotonic dispatch/completion TICK and a
  ``perf_counter_ns`` at entry/exit, and (b) answer with a deterministic
  non-streaming chat completion whose text is a pure function of the request's
  ``messages`` bytes -- so FORK-child live-reply splices are reproducible and
  assertable.
* Node identity is recovered from the minted ``x_request_id``
  (``{x_base}|{runtime_trace_id}|{node_id}|{phase_variant}``), anchored from the
  RIGHT because ``x_base`` itself may contain ``|``. The node id is
  ``"<session>[#n]:<turn_idx>"``.

Causality is asserted with COMPLETION-ORDER ticks (a global monotonic counter
bumped on ``send_request`` entry AND exit), never wall-clock deltas -- so the
ordering assertions are xdist-safe. The single delay assertion is a LOWER BOUND
only (authored 250 ms), measured at the transport seam.

Asymmetric latency: the responder slows a chosen SESSION's replies by
``_SLOW_RESPONSE_S`` via ``asyncio.sleep`` (deterministic TEXT unchanged) so that
a fast sibling turn can be observed dispatching WHILE a slow turn is still
in-flight (the "intermediate turn does not wait for its spawned child" and
"pre-session child does not wait for its owner" proofs).

Pinned run flags (mirroring the byte-parity graph run): ``--num-conversations 1``
(single pass, no recycle -> exactly one dispatch per node), single worker (the
dynamic splice pool is worker-local), single record processor, no warmup.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import pytest
from pytest import param

from aiperf.common.constants import IS_WINDOWS
from aiperf.common.enums import CreditPhase
from aiperf.common.models import RequestInfo, RequestRecord, TextResponse
from tests.component_integration.conftest import (
    ComponentIntegrationTestDefaults as defaults,
)
from tests.harness.fake_transport import FakeTransport
from tests.harness.utils import AIPerfCLI

pytestmark = pytest.mark.component_integration

_FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "dag" / "graph_parity"


# Artificial per-session reply latency for the asymmetric-latency proofs. Large
# enough that a fast sibling turn reliably lands its whole dispatch->completion
# window inside a slow turn's in-flight window, small enough to keep the suite
# quick.
_SLOW_RESPONSE_S = 0.3

# Authored ``delay`` on the gated turn of ``delayed_turn.dag.jsonl`` (ms) and the
# lower bound we assert at the transport seam. asyncio.sleep never returns early
# and the executor stamps the predecessor finish AFTER send_request returns, so
# the real gap is provably >= the authored delay; a small slack below absorbs
# only perf_counter sampling granularity.
_DELAY_TURN_MS = 250
# 5 ms scheduler slack on POSIX; Windows timers tick at ~15.6 ms granularity
# and the loop's timer clock can fire up to two ticks early relative to the
# perf_counter timestamps the seam records, so allow that much there.
_DELAY_LOWER_BOUND_S = 0.218 if IS_WINDOWS else 0.245


def _deterministic_response_text(messages_bytes: bytes) -> str:
    """Reply text as a pure function of the request's ``messages`` bytes.

    Byte-identical to the byte-parity harness's responder so a spliced FORK-child
    reply is reproducible: ``resp-<sha256(messages)[:8]>``.
    """
    return f"resp-{hashlib.sha256(messages_bytes).hexdigest()[:8]}"


def _node_key(x_request_id: str) -> tuple[str, int]:
    """Recover ``(session_id, turn_index)`` from a minted graph request id.

    The worker mints ``{node_id}::{nonce}`` (``_mint_x_request_id``); the node
    id is ``"<session>[#n]:<turn_idx>"`` (``#n`` only for repeated SPAWN
    instances). The correlation id is the trajectory-instance id
    (``{conversation}::{nonce}``) and carries no node identity.
    """
    node_id, sep, _nonce = x_request_id.rpartition("::")
    assert sep, f"unparsable graph x_request_id {x_request_id!r}"
    session_part, sep, turn_str = node_id.rpartition(":")
    assert sep and turn_str.isdigit(), (
        f"graph node id {node_id!r} does not end in ':<turn_idx>'"
    )
    return session_part.split("#", 1)[0], int(turn_str)


@dataclass(frozen=True, slots=True)
class _Fire:
    """One captured transport dispatch with ordering + timing instrumentation."""

    session: str
    """Recovered template session id (``#n`` SPAWN suffix stripped)."""
    turn: int
    """Recovered zero-based turn index within the session."""
    x_correlation_id: str
    """The minted sticky correlation id."""
    credit_phase: CreditPhase
    """Credit phase this dispatch ran under (asserted PROFILING everywhere)."""
    agent_depth: int
    """DAG identity stamped on the credit and carried by ``request_info`` into
    the record pipeline (0 = root chain; children owner depth + 1)."""
    parent_correlation_id: str | None
    """The spawning parent node's correlation id for fork/spawn children; None
    for roots and pre-session children. Rides ``request_info`` into records."""
    body: bytes
    """``request_info.payload_bytes`` -- the canonical wire body."""
    dispatch_tick: int
    """Global monotonic counter value at ``send_request`` ENTRY (fire order)."""
    complete_tick: int
    """Global monotonic counter value at ``send_request`` EXIT (completion order)."""
    dispatch_perf_ns: int
    """``perf_counter_ns`` at ``send_request`` entry (delay lower-bound only)."""
    complete_perf_ns: int
    """``perf_counter_ns`` at ``send_request`` exit (delay lower-bound only)."""

    @property
    def key(self) -> tuple[str, int]:
        return (self.session, self.turn)

    def messages(self) -> list[dict[str, Any]]:
        return orjson.loads(self.body)["messages"]


def _install_timing_transport(
    monkeypatch: pytest.MonkeyPatch,
    sink: list[_Fire],
    *,
    slow_sessions: dict[str, float],
) -> None:
    """Replace ``FakeTransport.send_request`` with an instrumented responder.

    Records a :class:`_Fire` per dispatch (identity + body + entry/exit tick +
    entry/exit perf clock) and answers with a deterministic chat completion.
    Replies for a session listed in ``slow_sessions`` are delayed by that many
    seconds (``asyncio.sleep``) BEFORE the record is finalized, so a concurrent
    fast sibling dispatch interleaves ahead of the slow completion tick. The
    reply TEXT is unaffected by the delay (pure function of ``messages`` bytes).
    """
    tick = itertools.count()

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
        session, turn = _node_key(request_info.x_request_id)
        dispatch_tick = next(tick)
        dispatch_perf_ns = time.perf_counter_ns()

        delay_s = slow_sessions.get(session, 0.0)
        if delay_s:
            await asyncio.sleep(delay_s)

        parsed = orjson.loads(body)
        text = _deterministic_response_text(orjson.dumps(parsed["messages"]))
        response_data = {
            "id": "chatcmpl-e2e",
            "object": "chat.completion",
            "created": 0,
            "model": parsed.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
        }
        complete_perf_ns = time.perf_counter_ns()
        complete_tick = next(tick)
        sink.append(
            _Fire(
                session=session,
                turn=turn,
                x_correlation_id=request_info.x_correlation_id,
                credit_phase=request_info.credit_phase,
                agent_depth=request_info.agent_depth,
                parent_correlation_id=request_info.parent_correlation_id,
                body=bytes(body),
                dispatch_tick=dispatch_tick,
                complete_tick=complete_tick,
                dispatch_perf_ns=dispatch_perf_ns,
                complete_perf_ns=complete_perf_ns,
            )
        )
        return RequestRecord(
            start_perf_ns=dispatch_perf_ns,
            end_perf_ns=complete_perf_ns,
            timestamp_ns=time.time_ns(),
            status=200,
            responses=[
                TextResponse(
                    perf_ns=complete_perf_ns,
                    content_type="application/json",
                    text=orjson.dumps(response_data).decode("utf-8"),
                )
            ],
        )

    monkeypatch.setattr(FakeTransport, "send_request", _send_request)


def _run_graph(cli: AIPerfCLI, sink: list[_Fire], fixture: Path) -> list[_Fire]:
    """Run one in-process ``--graph-format dag_jsonl`` profile; return its fires.

    Single pass (``--num-conversations 1``), single worker (the dynamic splice
    pool is worker-local), single record processor, no warmup -- the same run
    shape the byte-parity gate pins.
    """
    assert fixture.is_file(), f"missing fixture {fixture}"
    sink.clear()
    cli.run_sync(
        f"""
        aiperf profile \
            --model {defaults.model} \
            --graph-format dag_jsonl \
            --input-file {fixture} \
            --num-conversations 1 \
            --record-processor-service-count 1 \
            --workers-max 1 \
            --ui {defaults.ui}
        """,
        timeout=120.0,
        assert_success=True,
    )
    return list(sink)


def _by_key(fires: list[_Fire]) -> dict[tuple[str, int], _Fire]:
    """Index fires by ``(session, turn)``, asserting no node fired twice."""
    keyed: dict[tuple[str, int], _Fire] = {}
    for f in fires:
        assert f.key not in keyed, f"node {f.key} dispatched more than once"
        keyed[f.key] = f
    return keyed


@pytest.mark.parametrize(
    "fixture_name,expected_keys",
    [
        param(
            "fork_minimal.dag.jsonl",
            {
                ("root", 0),
                ("branch-a", 0),
                ("branch-a", 1),
                ("branch-b", 0),
                ("branch-b", 1),
            },
            id="fork-minimal",
        ),
        param(
            "spawn_join.dag.jsonl",
            {("parent", 0), ("parent", 1), ("parent", 2), ("worker-a", 0), ("worker-a", 1)},
            id="spawn-join",
        ),
        param(
            "mixed_full.dag.jsonl",
            {
                ("root", 0),
                ("root", 1),
                ("root", 2),
                ("helper", 0),
                ("helper", 1),
                ("finisher", 0),
            },
            id="mixed-full",
        ),
        param(
            "prespawn.dag.jsonl",
            {("root", 0), ("root", 1), ("bg-child", 0)},
            id="prespawn",
        ),
        param(
            "delayed_turn.dag.jsonl",
            {("seq", 0), ("seq", 1)},
            id="delayed-turn",
        ),
    ],
)  # fmt: skip
def test_every_node_dispatches_exactly_once(
    cli: AIPerfCLI,
    mmap_base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_name: str,
    expected_keys: set[tuple[str, int]],
) -> None:
    """(a) Every tree node dispatches EXACTLY once per single pass.

    No recycles (``--num-conversations 1`` is single-pass), so the captured
    ``(session, turn)`` multiset must equal the tree's node set with count 1
    each -- across fork children, spawn children, join turns, pre-session
    children, and delayed turns.
    """
    sink: list[_Fire] = []
    _install_timing_transport(monkeypatch, sink, slow_sessions={})
    fires = _run_graph(cli, sink, _FIXTURE_DIR / fixture_name)

    assert all(f.credit_phase == CreditPhase.PROFILING for f in fires), (
        "a dispatch ran outside PROFILING (warmup must stay disabled)"
    )
    counts = Counter(f.key for f in fires)
    assert set(counts) == expected_keys, (
        f"node set mismatch: only-observed={set(counts) - expected_keys} "
        f"only-expected={expected_keys - set(counts)}"
    )
    assert all(c == 1 for c in counts.values()), (
        f"nodes dispatched more than once: {[k for k, c in counts.items() if c != 1]}"
    )


def test_fork_children_fire_after_parent_and_embed_live_reply(
    cli: AIPerfCLI,
    mmap_base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) FORK children fire only AFTER the parent's response completes, and
    (c) a FORK child's materialized messages embed the parent's ACTUAL
    deterministic reply at the correct position (exact message-list shape)."""
    sink: list[_Fire] = []
    _install_timing_transport(monkeypatch, sink, slow_sessions={})
    fires = _run_graph(cli, sink, _FIXTURE_DIR / "fork_minimal.dag.jsonl")
    by_key = _by_key(fires)

    root = by_key[("root", 0)]
    branch_a = by_key[("branch-a", 0)]
    branch_b = by_key[("branch-b", 0)]

    # (b) Causality: each fork child's dispatch strictly follows the parent's
    # completion (its firing gate is a ChannelRequirement on ``root:0_out``,
    # written only after the parent request returns).
    assert branch_a.dispatch_tick > root.complete_tick, (
        f"branch-a fired before parent completed: "
        f"a.dispatch={branch_a.dispatch_tick} root.complete={root.complete_tick}"
    )
    assert branch_b.dispatch_tick > root.complete_tick, (
        f"branch-b fired before parent completed: "
        f"b.dispatch={branch_b.dispatch_tick} root.complete={root.complete_tick}"
    )

    # (c) Exact message-list shape: the fork child body is the parent's authored
    # messages, then the parent's ACTUAL captured reply as an assistant turn,
    # then the child's own authored messages -- in that order.
    parent_messages = root.messages()
    parent_reply = _deterministic_response_text(orjson.dumps(parent_messages))
    child_authored = [
        {"role": "user", "content": "Expand on the first section in more detail."},
        {"role": "user", "content": "Add a brief counter-argument."},
    ]
    expected = [
        *parent_messages,
        {"role": "assistant", "content": parent_reply},
        *child_authored,
    ]
    assert branch_a.messages() == expected, (
        "fork child ('branch-a', 0) message shape/positioning wrong\n"
        f"  expected: {expected}\n"
        f"  actual:   {branch_a.messages()}"
    )
    # The embedded reply is a genuine live capture, not a static echo.
    assert parent_reply.startswith("resp-"), parent_reply


def test_spawn_join_gates_on_children_but_intermediate_does_not_wait(
    cli: AIPerfCLI,
    mmap_base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) SPAWN-join gating and busy-parent independence in one run.

    ``worker-a`` replies are slowed so the parent's intermediate turn (turn 1,
    between the spawn turn 0 and the ``join_at: 2`` turn) can be observed
    dispatching WHILE the spawned child is still in-flight, while the join turn
    (turn 2) is held until the child's leaf response completes.
    """
    sink: list[_Fire] = []
    _install_timing_transport(
        monkeypatch, sink, slow_sessions={"worker-a": _SLOW_RESPONSE_S}
    )
    fires = _run_graph(cli, sink, _FIXTURE_DIR / "spawn_join.dag.jsonl")
    by_key = _by_key(fires)

    p0 = by_key[("parent", 0)]
    p1 = by_key[("parent", 1)]
    p2 = by_key[("parent", 2)]
    w0 = by_key[("worker-a", 0)]
    w1 = by_key[("worker-a", 1)]

    # Intermediate turn gates only on its own predecessor (parent turn 0), so it
    # dispatches after parent turn 0 completes...
    assert p1.dispatch_tick > p0.complete_tick, (
        f"intermediate fired before parent turn 0 completed: "
        f"p1.dispatch={p1.dispatch_tick} p0.complete={p0.complete_tick}"
    )
    # ...WITHOUT waiting for the (deliberately slowed) spawned child: the
    # intermediate dispatch lands before the slow child's turn-0 completion.
    assert p1.dispatch_tick < w0.complete_tick, (
        f"intermediate waited on the spawned child: "
        f"p1.dispatch={p1.dispatch_tick} w0.complete={w0.complete_tick}"
    )

    # Join turn is gated on ALL gating children's LEAF responses (worker-a:1) as
    # well as its own predecessor (parent turn 1).
    assert p2.dispatch_tick > w1.complete_tick, (
        f"join turn fired before the child leaf completed: "
        f"p2.dispatch={p2.dispatch_tick} w1.complete={w1.complete_tick}"
    )
    assert p2.dispatch_tick > w0.complete_tick, (
        f"join turn fired before the child's first turn completed: "
        f"p2.dispatch={p2.dispatch_tick} w0.complete={w0.complete_tick}"
    )
    assert p2.dispatch_tick > p1.complete_tick, (
        f"join turn fired before its own predecessor completed: "
        f"p2.dispatch={p2.dispatch_tick} p1.complete={p1.complete_tick}"
    )


def test_pre_session_spawn_child_fires_without_waiting_for_owner(
    cli: AIPerfCLI,
    mmap_base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) A ``pre_session_spawns`` child fires on the owner turn-0 trigger, NOT
    after the owner's turn-0 response.

    First runtime exercise of pre-spawn lowering. The owner (``root``) replies
    are slowed; the pre-session child (``bg-child``) shares the owner turn-0's
    (empty -> START) firing predecessors and carries no channel gate on the
    owner, so it must dispatch before the owner's slowed turn-0 completes.
    """
    sink: list[_Fire] = []
    _install_timing_transport(
        monkeypatch, sink, slow_sessions={"root": _SLOW_RESPONSE_S}
    )
    fires = _run_graph(cli, sink, _FIXTURE_DIR / "prespawn.dag.jsonl")
    by_key = _by_key(fires)

    owner_turn0 = by_key[("root", 0)]
    pre_child = by_key[("bg-child", 0)]

    assert pre_child.dispatch_tick < owner_turn0.complete_tick, (
        "pre-session child waited on the owner's turn-0 response: "
        f"child.dispatch={pre_child.dispatch_tick} "
        f"owner0.complete={owner_turn0.complete_tick}"
    )


def test_authored_turn_delay_honored_at_transport_seam(
    cli: AIPerfCLI,
    mmap_base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(d) A turn with ``delay: 250`` fires >= 250 ms after its predecessor's
    completion, measured at the transport seam (lower bound only)."""
    sink: list[_Fire] = []
    _install_timing_transport(monkeypatch, sink, slow_sessions={})
    fires = _run_graph(cli, sink, _FIXTURE_DIR / "delayed_turn.dag.jsonl")
    by_key = _by_key(fires)

    turn0 = by_key[("seq", 0)]
    turn1 = by_key[("seq", 1)]

    gap_s = (turn1.dispatch_perf_ns - turn0.complete_perf_ns) / 1e9
    assert gap_s >= _DELAY_LOWER_BOUND_S, (
        f"authored delay {_DELAY_TURN_MS}ms not honored: gap was {gap_s * 1000:.1f}ms "
        f"(predecessor completion -> delayed-turn dispatch)"
    )


def test_fork_children_records_carry_dag_identity(
    cli: AIPerfCLI,
    mmap_base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) FORK-child records carry the legacy DAG identity: ``agent_depth=1``
    and ``parent_correlation_id`` equal to the forking parent node's OWN minted
    ``x_correlation_id`` (not merely the same shape) -- root turns carry
    ``agent_depth=0`` / no parent. Asserted on ``request_info`` at the transport
    seam, the same object the record pipeline copies into
    ``MetricRecordMetadata.agent_depth`` / ``parent_correlation_id``."""
    sink: list[_Fire] = []
    _install_timing_transport(monkeypatch, sink, slow_sessions={})
    fires = _run_graph(cli, sink, _FIXTURE_DIR / "fork_minimal.dag.jsonl")
    by_key = _by_key(fires)

    root = by_key[("root", 0)]
    assert root.agent_depth == 0
    assert root.parent_correlation_id is None

    for key in (("branch-a", 0), ("branch-a", 1), ("branch-b", 0), ("branch-b", 1)):
        child = by_key[key]
        assert child.agent_depth == 1, (
            f"fork child {key} agent_depth={child.agent_depth}, expected 1"
        )
        # ALL turns of a child instance carry the SAME triggering parent's
        # correlation id -- byte-equal to the parent node's minted one.
        assert child.parent_correlation_id == root.x_correlation_id, (
            f"fork child {key} parent_correlation_id="
            f"{child.parent_correlation_id!r}, expected the parent node's "
            f"{root.x_correlation_id!r}"
        )


def test_prespawn_child_record_carries_depth_one_no_parent(
    cli: AIPerfCLI,
    mmap_base_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(e) A pre-session SPAWN child's record carries ``agent_depth=1`` with NO
    parent correlation id (legacy ``start_pre_session_child``: no parent session
    exists at pre-dispatch), while the owner root's turns stay depth 0."""
    sink: list[_Fire] = []
    _install_timing_transport(monkeypatch, sink, slow_sessions={})
    fires = _run_graph(cli, sink, _FIXTURE_DIR / "prespawn.dag.jsonl")
    by_key = _by_key(fires)

    for key in (("root", 0), ("root", 1)):
        owner = by_key[key]
        assert owner.agent_depth == 0, key
        assert owner.parent_correlation_id is None, key

    pre_child = by_key[("bg-child", 0)]
    assert pre_child.agent_depth == 1
    assert pre_child.parent_correlation_id is None
