# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R4 (slot bypass) — the graph-issue path skips session-slot acquisition.

Graph credits flow through real issuance -> router -> returns but BYPASS the
linear session-slot acquire (turn0) / release (is_final_turn && depth0)
arithmetic; the strategy owns trace concurrency. A normal credit still acquires
a session slot. Verified against the REAL ``CreditIssuer`` with recording fakes.
"""

from __future__ import annotations

import hashlib

import pytest

from aiperf.common.enums import CreditPhase
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.structs import TurnToSend

pytestmark = pytest.mark.asyncio


class RecordingConcurrency:
    """Records session/prefill acquisitions; always grants."""

    def __init__(self) -> None:
        self.session_acquires = 0
        self.session_releases = 0
        self.prefill_acquires = 0

    async def acquire_session_slot(self, phase, can_proceed_fn) -> bool:
        self.session_acquires += 1
        return can_proceed_fn()

    def release_session_slot(self, phase) -> None:
        self.session_releases += 1

    async def acquire_prefill_slot(self, phase, can_proceed_fn) -> bool:
        self.prefill_acquires += 1
        return can_proceed_fn()

    def release_prefill_slot(self, phase) -> None: ...


class FakeProgress:
    def increment_sent(self, turn) -> tuple[int, bool]:
        return (0, False)

    def freeze_sent_counts(self) -> None: ...


class FakeStopChecker:
    def can_start_new_session(self) -> bool:
        return True

    def can_send_any_turn(self) -> bool:
        return True

    def can_send_dag_child_turn(self) -> bool:
        return True


class FakeRouter:
    def __init__(self) -> None:
        self.sent: list = []

    async def send_credit(self, *, credit) -> None:
        self.sent.append(credit)

    async def end_graph_trace(self, *args) -> None: ...


class FakeCancellation:
    def next_cancellation_delay_ns(self, turn, phase) -> int | None:
        return None


class FakeLifecycle:
    started_at_ns = 0
    started_at_perf_ns = 0


def _issuer(
    concurrency: RecordingConcurrency,
    router: FakeRouter,
    url_selection_strategy=None,
) -> CreditIssuer:
    return CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=FakeStopChecker(),
        progress=FakeProgress(),
        concurrency_manager=concurrency,
        credit_router=router,
        cancellation_policy=FakeCancellation(),
        lifecycle=FakeLifecycle(),
        url_selection_strategy=url_selection_strategy,
    )


def _graph_turn() -> TurnToSend:
    return TurnToSend(
        conversation_id="t0",
        x_correlation_id="x-t0",
        turn_index=0,
        num_turns=1,
        trace_id="t0",
        node_ordinal=0,
        phase_variant="profiling",
    )


def _normal_turn() -> TurnToSend:
    return TurnToSend(
        conversation_id="c",
        x_correlation_id="x",
        turn_index=0,
        num_turns=1,
    )


async def test_graph_issue_skips_session_slot_acquire():
    concurrency = RecordingConcurrency()
    router = FakeRouter()
    issuer = _issuer(concurrency, router)

    await issuer.issue_graph_credit(_graph_turn())

    assert concurrency.session_acquires == 0  # bypassed
    assert concurrency.prefill_acquires == 1  # still rate-limited per request
    assert len(router.sent) == 1
    assert router.sent[0].trace_id == "t0"


async def test_graph_issue_propagates_node_ordinal_and_phase_variant():
    """TurnToSend -> ``_issue_credit_internal`` copy -> sent Credit (three-touch).

    ``node_ordinal`` and ``phase_variant`` are the worker's record-addressing
    keys into the unified store; dropping either copy in
    ``CreditIssuer._issue_credit_internal`` must fail HERE, not only at
    runtime. Non-default values (ordinal 5, warmup variant) so a
    default-passthrough cannot fake the propagation.
    """
    router = FakeRouter()
    issuer = _issuer(RecordingConcurrency(), router)

    turn = TurnToSend(
        conversation_id="t0",
        x_correlation_id="x-t0",
        turn_index=0,
        num_turns=1,
        trace_id="t0",
        node_ordinal=5,
        phase_variant="warmup",
    )
    assert await issuer.issue_graph_credit(turn) is True

    (credit,) = router.sent
    assert credit.trace_id == "t0"
    assert credit.node_ordinal == 5
    assert credit.phase_variant == "warmup"


async def test_normal_issue_still_acquires_session_slot():
    concurrency = RecordingConcurrency()
    router = FakeRouter()
    issuer = _issuer(concurrency, router)

    await issuer.issue_credit(_normal_turn())

    assert concurrency.session_acquires == 1  # turn0 of a normal session
    assert concurrency.prefill_acquires == 1
    assert len(router.sent) == 1


# ---------------------------------------------------------------------------
# TC6 — URL affinity is a stable HASH of the TEMPLATE id (nonce-stripped), NOT
# a round-robin mint and NOT the per-trajectory uuid-bearing sticky key, so
# every instance/recycle of one template lands on the backend that primed its
# KV -- across separate per-phase issuers (warmup vs profiling).
# ---------------------------------------------------------------------------

_URLS = ["http://u0", "http://u1", "http://u2"]


def _hash_url_index(template_id: str, n: int = len(_URLS)) -> int:
    """The issuer's stable template->URL mapping, recomputed independently."""
    digest = hashlib.sha256(template_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % n


def _sticky_graph_turn(
    instance_id: str, sticky_suffix: str, phase_variant: str = "profiling"
) -> TurnToSend:
    # x_correlation_id is the routing key AND the per-trajectory corr,
    # ``{conversation_id}::{nonce}``; URL affinity ignores it and keys on the
    # nonce-stripped ``trace_id`` template instead.
    x_corr = f"{instance_id}::{sticky_suffix}"
    return TurnToSend(
        conversation_id=instance_id,
        x_correlation_id=x_corr,
        turn_index=0,
        num_turns=1,
        trace_id=instance_id,
        node_ordinal=0,
        phase_variant=phase_variant,
    )


def _url_issuer(router: FakeRouter) -> CreditIssuer:
    from aiperf.timing.url_samplers import RoundRobinURLSampler

    return _issuer(
        RecordingConcurrency(),
        router,
        url_selection_strategy=RoundRobinURLSampler(_URLS),
    )


async def test_warmup_and_profiling_issuers_pick_same_url_for_same_template():
    """Two DISTINCT instances of one TEMPLATE (warmup nonce vs profiling nonce)
    map to the SAME url_index across two separate per-phase issuers, because URL
    affinity is a stable hash of the nonce-stripped template id -- NOT the
    per-trajectory uuid-bearing sticky key, and NOT a round-robin mint order."""
    warmup_router = FakeRouter()
    profiling_router = FakeRouter()
    warmup_issuer = _url_issuer(warmup_router)
    profiling_issuer = _url_issuer(profiling_router)

    # Unrelated traffic on the profiling issuer: a hash-based mapping is
    # order-independent, so this must not shift the shared template's url.
    await profiling_issuer.issue_graph_credit(_sticky_graph_turn("t-9", "deadbeef"))

    # Same template "t-1", DIFFERENT per-instance nonces / phase variants.
    await warmup_issuer.issue_graph_credit(
        _sticky_graph_turn("t-1", "aaaa1111", phase_variant="warmup")
    )
    await profiling_issuer.issue_graph_credit(
        _sticky_graph_turn("t-1", "bbbb2222", phase_variant="profiling")
    )

    warmup_url = warmup_router.sent[-1].url_index
    profiling_url = profiling_router.sent[-1].url_index
    assert warmup_url is not None
    assert warmup_url == profiling_url == _hash_url_index("t-1")


async def test_graph_url_is_stable_hash_of_template_across_lanes():
    """Distinct templates map deterministically via ``sha256(template) % urls``
    (nonce-stripped), NOT round-robin: each credit's url is a pure function of
    its template id."""
    router = FakeRouter()
    issuer = _url_issuer(router)

    for lane in range(4):
        await issuer.issue_graph_credit(_sticky_graph_turn(f"t-{lane}", f"nonce{lane}"))

    assert [credit.url_index for credit in router.sent] == [
        _hash_url_index(f"t-{lane}") for lane in range(4)
    ]


async def test_graph_credits_of_one_instance_share_one_url():
    """Every node credit of one instance rides the same backend (affinity),
    at the template's stable hash bucket."""
    router = FakeRouter()
    issuer = _url_issuer(router)

    # All three credits share template "t-2" (distinct per-turn nonces).
    for turn_nonce in range(3):
        await issuer.issue_graph_credit(_sticky_graph_turn("t-2", f"feed{turn_nonce}"))

    indices = {credit.url_index for credit in router.sent}
    assert len(indices) == 1
    assert indices == {_hash_url_index("t-2")}


async def test_end_graph_trace_keeps_template_affinity_for_recycles():
    """``end_graph_trace`` closes the router session but deliberately does NOT
    evict URL affinity: the entry keys on the nonce-stripped template, and a
    recycle of the template must land on the backend that primed its KV."""
    router = FakeRouter()
    issuer = _url_issuer(router)

    turn = _sticky_graph_turn("t-3", "0badf00d")
    template = turn.trace_id
    await issuer.issue_graph_credit(turn)
    assert template in issuer._graph_url_affinity
    first_url = router.sent[-1].url_index

    await issuer.end_graph_trace(template, "profiling")
    assert issuer._graph_url_affinity.get(template) == first_url

    # A recycle (fresh nonce) reuses the primed backend.
    await issuer.issue_graph_credit(_sticky_graph_turn("t-3", "5eed5eed"))
    assert router.sent[-1].url_index == first_url == _hash_url_index("t-3")
