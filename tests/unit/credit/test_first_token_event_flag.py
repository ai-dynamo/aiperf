# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-credit first-token event flag + graph first-token observer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest import param

from aiperf.common.enums import CreditPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.messages import FirstToken
from aiperf.credit.structs import Credit, TurnToSend
from aiperf.timing.concurrency import ConcurrencyManager

# =============================================================================
# Issuer path - flag copies TurnToSend -> Credit
# =============================================================================


class RecordingConcurrency:
    """Grants every slot and counts prefill acquisitions."""

    def __init__(self) -> None:
        self.prefill_acquires = 0

    async def acquire_session_slot(self, phase, can_proceed_fn) -> bool:
        return can_proceed_fn()

    def release_session_slot(self, phase) -> None: ...

    async def acquire_prefill_slot(self, phase, can_proceed_fn) -> bool:
        self.prefill_acquires += 1
        return can_proceed_fn()

    def release_prefill_slot(self, phase) -> None: ...


class FakeProgress:
    """Progress tracker stub that always reports a non-terminal send."""

    def increment_sent(self, turn) -> tuple[int, bool]:
        return (0, False)

    def freeze_sent_counts(self) -> None: ...


class FakeStopChecker:
    """Stop checker stub that permits every kind of send."""

    def can_start_new_session(self) -> bool:
        return True

    def can_send_any_turn(self) -> bool:
        return True

    def can_send_dag_child_turn(self) -> bool:
        return True


class FakeRouter:
    """Router stub that records the credits the issuer emits."""

    def __init__(self) -> None:
        self.sent: list = []

    async def send_credit(self, *, credit) -> None:
        self.sent.append(credit)


class FakeCancellation:
    """Cancellation policy stub that never schedules a cancellation."""

    def next_cancellation_delay_ns(self, turn, phase) -> int | None:
        return None


class FakeLifecycle:
    """Lifecycle stub anchored at time zero."""

    started_at_ns = 0
    started_at_perf_ns = 0


def _issuer(router: FakeRouter) -> CreditIssuer:
    """Issuer over the local fakes, emitting into ``router``."""
    return CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=FakeStopChecker(),
        progress=FakeProgress(),
        concurrency_manager=RecordingConcurrency(),
        credit_router=router,
        cancellation_policy=FakeCancellation(),
        lifecycle=FakeLifecycle(),
    )


def _graph_turn(*, first_token_event: bool | None = None) -> TurnToSend:
    """Graph turn (carries a trace id), optionally setting the first-token flag."""
    kwargs = (
        {} if first_token_event is None else {"first_token_event": first_token_event}
    )
    return TurnToSend(
        conversation_id="t0",
        x_correlation_id="x-t0",
        turn_index=0,
        num_turns=1,
        trace_id="t0#0",
        node_ordinal=0,
        **kwargs,
    )


def _linear_turn(*, first_token_event: bool) -> TurnToSend:
    """Non-graph turn with an explicit first-token flag."""
    return TurnToSend(
        conversation_id="c",
        x_correlation_id="x",
        turn_index=0,
        num_turns=1,
        first_token_event=first_token_event,
    )


class TestFirstTokenEventFlagCopy:
    """``first_token_event`` must survive the TurnToSend -> Credit hand-off."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "build_turn,issue_method,expected",
        [
            param(
                lambda: _graph_turn(first_token_event=True),
                "issue_graph_credit", True,
                id="graph_credit_copies_true",
            ),
            param(
                _graph_turn, "issue_graph_credit", False,
                id="graph_credit_defaults_false",
            ),
            param(
                lambda: _linear_turn(first_token_event=True),
                "issue_credit", True,
                id="linear_credit_copies_true",
            ),
        ],
    )  # fmt: skip
    async def test_issuance_paths_copy_flag_onto_credit(
        self, build_turn, issue_method: str, expected: bool
    ) -> None:
        """Both the graph and linear issuance paths carry the flag through verbatim."""
        router = FakeRouter()
        await getattr(_issuer(router), issue_method)(build_turn())
        assert router.sent[0].first_token_event is expected

    def test_from_previous_credit_propagates_flag(self) -> None:
        """A continuation turn built from a credit inherits that credit's flag."""
        credit = Credit(
            id=0,
            phase=CreditPhase.PROFILING,
            conversation_id="c",
            x_correlation_id="x",
            turn_index=0,
            num_turns=2,
            issued_at_ns=1,
            first_token_event=True,
        )

        assert TurnToSend.from_previous_credit(credit).first_token_event is True


# =============================================================================
# Callback handler - graph first-token observer + prefill-stat guard
# =============================================================================


def _make_handler(*, prefill_concurrency: int | None = None):
    """Build a handler with one registered PROFILING phase.

    The concurrency manager WRAPS a real :class:`ConcurrencyManager` configured
    for ``prefill_concurrency`` rather than being a bare ``MagicMock``. A bare
    mock auto-creates every attribute and returns a truthy value for any
    predicate, so a regression that re-gates the released counter on "is prefill
    limiting enabled?" would pass no matter which predicate it used. Wrapping
    keeps call assertions while letting real predicates answer honestly.
    """
    real = ConcurrencyManager()
    real.configure_for_phase(
        CreditPhase.PROFILING, concurrency=None, prefill_concurrency=prefill_concurrency
    )
    concurrency = MagicMock(wraps=real)

    progress = MagicMock()
    progress.increment_prefill_released = MagicMock()

    handler = CreditCallbackHandler(concurrency)
    handler.register_phase(
        phase=CreditPhase.PROFILING,
        progress=progress,
        lifecycle=MagicMock(is_complete=False),
        stop_checker=MagicMock(),
        strategy=MagicMock(handle_credit_return=AsyncMock()),
    )
    return handler, concurrency, progress


def _graph_first_token(**overrides) -> FirstToken:
    """First-token event carrying graph identity unless overridden."""
    kwargs = dict(
        credit_id=1,
        phase=CreditPhase.PROFILING,
        ttft_ns=5,
        trace_id="t0#0",
        x_correlation_id="x-t0",
        turn_index=0,
    )
    kwargs.update(overrides)
    return FirstToken(**kwargs)


class TestGraphFirstTokenObserver:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "trace_id,expect_observed",
        [
            param("t0#0", True, id="graph_first_token_observed"),
            param(None, False, id="non_graph_first_token_ignored"),
        ],
    )  # fmt: skip
    async def test_observer_fires_only_for_graph_first_tokens(
        self, trace_id: str | None, expect_observed: bool
    ) -> None:
        """The graph observer receives first tokens that carry a trace id, and only those."""
        handler, _, _ = _make_handler()
        seen: list[FirstToken] = []
        handler.set_graph_first_token_observer(seen.append)

        ft = _graph_first_token(trace_id=trace_id)
        await handler.on_first_token(ft)

        assert seen == ([ft] if expect_observed else [])

    @pytest.mark.asyncio
    async def test_observer_none_is_noop(self) -> None:
        """Clearing the observer leaves graph first tokens handled without error."""
        handler, _, _ = _make_handler()
        handler.set_graph_first_token_observer(None)

        await handler.on_first_token(_graph_first_token())

    @pytest.mark.asyncio
    async def test_observer_fires_even_for_unregistered_phase(self) -> None:
        """Observation is phase-independent, mirroring the credit-return observer."""
        handler = CreditCallbackHandler(MagicMock())
        seen: list[FirstToken] = []
        handler.set_graph_first_token_observer(seen.append)

        ft = _graph_first_token(phase=CreditPhase.WARMUP)
        await handler.on_first_token(ft)

        assert seen == [ft]


class TestFirstTokenPrefillStatGuard:
    @pytest.mark.asyncio
    async def test_prefill_counter_released_even_when_limiting_disabled(self) -> None:
        """The released counter advances without prefill limiting, and observation continues.

        ``in_flight_prefills`` is ``requests_sent - prefills_released``, so gating
        the release on whether prefill limiting is configured pinned it at
        ``requests_sent`` forever on any streaming run without
        ``--prefill-concurrency``. This site
        and the returned-without-TTFT site are mutually exclusive (at most one
        FirstToken per credit), so releasing unconditionally cannot double-count.
        """
        # prefill_concurrency=None -> the real limiter reports DISABLED, so a
        # re-gate on any genuine "is limiting enabled?" predicate fails here.
        handler, concurrency, progress = _make_handler(prefill_concurrency=None)
        seen: list[FirstToken] = []
        handler.set_graph_first_token_observer(seen.append)

        ft = _graph_first_token()
        await handler.on_first_token(ft)

        progress.increment_prefill_released.assert_called_once()
        assert seen == [ft]

    @pytest.mark.asyncio
    async def test_prefill_counter_touched_when_limiting_enabled(self) -> None:
        """With prefill limiting on, a first token releases the slot and bumps the counter."""
        handler, concurrency, progress = _make_handler(prefill_concurrency=4)

        await handler.on_first_token(_graph_first_token())

        progress.increment_prefill_released.assert_called_once()
        concurrency.release_prefill_slot.assert_called_once_with(CreditPhase.PROFILING)
