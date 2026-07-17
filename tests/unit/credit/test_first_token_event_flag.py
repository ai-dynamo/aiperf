# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-credit first-token event flag + graph first-token observer.

Credit-pipeline wiring for post-TTFT first-token anchoring:

- ``TurnToSend.first_token_event`` copies into the issued ``Credit`` on BOTH
  the linear (``issue_credit``) and graph (``issue_graph_credit``) paths, and
  propagates across turns via ``TurnToSend.from_previous_credit``. The known
  three-touch credit-pipeline trap: without the explicit copy in the
  ``Credit(...)`` construction the flag silently drops while tests that only
  assert on the ``TurnToSend`` still pass.
- ``CreditCallbackHandler.set_graph_first_token_observer`` fires for every
  ``FirstToken`` carrying a ``trace_id`` (a graph credit), independent of
  phase-handler gating -- mirroring the unconditional graph-return observer.
- ``on_first_token`` must NOT advance the prefill-released counter when prefill
  limiting is inactive: a graph first-token event arrives even then, and
  counting it would inflate the stat.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from aiperf.common.enums import CreditPhase
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.messages import FirstToken
from aiperf.credit.structs import Credit, TurnToSend

# asyncio_mode = "auto" (pyproject) runs the async tests without per-test marks.


# =============================================================================
# Issuer path — flag copies TurnToSend -> Credit
# =============================================================================


class RecordingConcurrency:
    """Grants every slot; records prefill acquisitions (mirrors the graph-path
    issuer test's recording fake)."""

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


class FakeCancellation:
    def next_cancellation_delay_ns(self, turn, phase) -> int | None:
        return None


class FakeLifecycle:
    started_at_ns = 0
    started_at_perf_ns = 0


def _issuer(router: FakeRouter) -> CreditIssuer:
    return CreditIssuer(
        phase=CreditPhase.PROFILING,
        stop_checker=FakeStopChecker(),
        progress=FakeProgress(),
        concurrency_manager=RecordingConcurrency(),
        credit_router=router,
        cancellation_policy=FakeCancellation(),
        lifecycle=FakeLifecycle(),
    )


class TestFirstTokenEventFlagCopy:
    """The per-credit ``first_token_event`` flag must survive the
    TurnToSend -> Credit hand-off on every issuance path."""

    async def test_graph_credit_copies_flag_true(self):
        router = FakeRouter()
        turn = TurnToSend(
            conversation_id="t0",
            x_correlation_id="x-t0",
            turn_index=0,
            num_turns=1,
            trace_id="t0#0",
            node_ordinal=0,
            first_token_event=True,
        )

        await _issuer(router).issue_graph_credit(turn)

        assert router.sent[0].first_token_event is True

    async def test_graph_credit_flag_defaults_false(self):
        router = FakeRouter()
        turn = TurnToSend(
            conversation_id="t0",
            x_correlation_id="x-t0",
            turn_index=0,
            num_turns=1,
            trace_id="t0#0",
            node_ordinal=0,
        )

        await _issuer(router).issue_graph_credit(turn)

        assert router.sent[0].first_token_event is False

    async def test_linear_credit_copies_flag_true(self):
        router = FakeRouter()
        turn = TurnToSend(
            conversation_id="c",
            x_correlation_id="x",
            turn_index=0,
            num_turns=1,
            first_token_event=True,
        )

        await _issuer(router).issue_credit(turn)

        assert router.sent[0].first_token_event is True

    def test_from_previous_credit_propagates_flag(self):
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

        nxt = TurnToSend.from_previous_credit(credit)

        assert nxt.first_token_event is True


# =============================================================================
# Callback handler — graph first-token observer + prefill-stat guard
# =============================================================================


def _make_handler(prefill_enabled: bool):
    """Build a handler with one registered PROFILING phase. ``concurrency`` and
    ``progress`` are MagicMocks so callers can assert on the prefill counters."""
    concurrency = MagicMock()
    concurrency.prefill_limiting_enabled = prefill_enabled
    concurrency.release_prefill_slot = MagicMock()

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
    async def test_observer_fires_for_graph_first_token(self):
        handler, _, _ = _make_handler(prefill_enabled=True)
        seen: list[FirstToken] = []
        handler.set_graph_first_token_observer(seen.append)

        ft = _graph_first_token()
        await handler.on_first_token(ft)

        assert seen == [ft]

    async def test_observer_not_fired_for_non_graph_first_token(self):
        handler, _, _ = _make_handler(prefill_enabled=True)
        seen: list[FirstToken] = []
        handler.set_graph_first_token_observer(seen.append)

        await handler.on_first_token(_graph_first_token(trace_id=None))

        assert seen == []

    async def test_observer_none_is_noop(self):
        handler, _, _ = _make_handler(prefill_enabled=True)
        handler.set_graph_first_token_observer(None)

        # Must not raise even for a graph-carrying first token.
        await handler.on_first_token(_graph_first_token())

    async def test_observer_fires_even_for_unregistered_phase(self):
        """Mirrors the return observer: fires independent of phase-handler
        registration so a graph first-token event is never stranded."""
        handler = CreditCallbackHandler(MagicMock())
        seen: list[FirstToken] = []
        handler.set_graph_first_token_observer(seen.append)

        ft = _graph_first_token(phase=CreditPhase.WARMUP)
        await handler.on_first_token(ft)

        assert seen == [ft]


class TestFirstTokenPrefillStatGuard:
    async def test_prefill_counter_not_touched_when_limiting_disabled(self):
        handler, concurrency, progress = _make_handler(prefill_enabled=False)
        seen: list[FirstToken] = []
        handler.set_graph_first_token_observer(seen.append)

        ft = _graph_first_token()
        await handler.on_first_token(ft)

        progress.increment_prefill_released.assert_not_called()
        # Observer still fires; slot release is internally guarded, so it is
        # harmless if invoked, but the counter must not advance.
        assert seen == [ft]

    async def test_prefill_counter_touched_when_limiting_enabled(self):
        handler, concurrency, progress = _make_handler(prefill_enabled=True)

        await handler.on_first_token(_graph_first_token())

        progress.increment_prefill_released.assert_called_once()
        concurrency.release_prefill_slot.assert_called_once_with(CreditPhase.PROFILING)
