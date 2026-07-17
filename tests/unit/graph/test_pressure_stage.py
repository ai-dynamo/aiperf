# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extended warmup (cache-pressure stage) -- config surface + strategy tests.

AgentX v1.0 parity: ``--agentic-cache-warmup-duration N`` continues the live
replay compressed (zero idle delay, 1-token outputs) for N seconds after the
boundary-priming warmup drains, then drains and hands the execution frontier
to PROFILING. This file pins the graph-native config plumbing and the
pressure stage itself.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import msgspec
import pytest

from aiperf.common.enums import CreditPhase
from aiperf.common.environment import Environment
from aiperf.common.scenario import TrajectoryWarmupFailedError
from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace
from aiperf.graph.credit_dispatch_adapter import CreditIssueRefusedError
from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

_FIX = Path(__file__).parent / "fixtures" / "weka_min.json"


def test_handoff_residual_cap_default():
    assert Environment.GRAPH.HANDOFF_RESIDUAL_CAP == 60.0


def test_cli_flag_threads_pressure_duration_to_config():
    """--agentic-cache-warmup-duration is per-run config, not env: it lands on
    cfg.agentic_cache_warmup_duration and rides TimingConfig onto the WARMUP
    CreditPhaseConfig as cache_pressure_duration (no process-global writes)."""
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.timing.config import TimingConfig
    from tests.unit.conftest import make_run_from_cli

    cli = CLIConfig(
        model_names=["test-model"],
        input_file=str(_FIX),
        request_count=3,
        agentic_cache_warmup_duration=45.0,
    )
    run = make_run_from_cli(cli)
    assert run.cfg.agentic_cache_warmup_duration == 45.0
    tc = TimingConfig.from_run(run)
    warmup = next(p for p in tc.phase_configs if p.phase == CreditPhase.WARMUP)
    assert warmup.cache_pressure_duration == 45.0


def test_cli_flag_unset_leaves_config_none():
    from aiperf.config.flags.cli_config import CLIConfig
    from tests.unit.conftest import make_run_from_cli

    cli = CLIConfig(
        model_names=["test-model"],
        input_file=str(_FIX),
        request_count=3,
    )
    run = make_run_from_cli(cli)
    assert run.cfg.agentic_cache_warmup_duration is None


def _corpus(n_traces: int) -> Any:
    parsed = from_weka_trace(str(_FIX))
    base = parsed.traces[0]
    traces = [msgspec.structs.replace(base, id=f"t-{i}") for i in range(n_traces)]
    return msgspec.structs.replace(parsed, traces=traces)


class _Config:
    timing_mode = None

    def __init__(
        self,
        *,
        concurrency: int | None = None,
        phase: CreditPhase = CreditPhase.WARMUP,
    ) -> None:
        self.phase = phase
        self.concurrency = concurrency
        self.expected_num_sessions = None
        self.total_expected_requests = None
        self.expected_duration_sec = None


class _ParkAfterIssuer:
    """Resolve the first ``park_after`` graph credits instantly, park the rest.

    Parking stalls the pressure lanes so the ``wait_for`` duration timer fires
    deterministically instead of recycling unboundedly fast under the
    instant-sleep test fixtures.

    ``error_pred`` / ``cancel_pred`` optionally inject a terminal failure into
    the ``observer(credit, error, cancelled)`` resolution: a credit matching
    ``error_pred(credit, issuer)`` resolves with ``error_text``, one matching
    ``cancel_pred(credit, issuer)`` resolves ``cancelled=True``. Both default to
    ``None`` (every credit resolves cleanly, the byte-identical prior behavior).
    The predicates receive the issuer so they can inspect ``issuer.strategy.
    _pressure_active`` to target a pressure-stage credit -- the identity no
    longer encodes priming-vs-pressure in the ``trace_id`` string.
    """

    def __init__(
        self,
        park_after: int | None = None,
        *,
        error_pred: Callable[[Any, Any], bool] | None = None,
        cancel_pred: Callable[[Any, Any], bool] | None = None,
        error_text: str = "boom",
    ) -> None:
        self.observer = None
        self.strategy: GraphIRReplayStrategy | None = None
        self.issued: list[Any] = []
        self._park_after = park_after
        self._error_pred = error_pred
        self._cancel_pred = cancel_pred
        self._error_text = error_text
        # Settable by the completeness-gate test: the strategy's teardown stash
        # skips when not every warmup credit return landed.
        self.all_returned = True

    def bind(self, strategy: GraphIRReplayStrategy) -> None:
        self.observer = strategy._on_graph_return
        self.strategy = strategy

    async def issue_graph_credit(self, credit: Any) -> bool:
        self.issued.append(credit)
        if self._park_after is not None and len(self.issued) > self._park_after:
            return True  # parked: never resolved
        observer = self.observer
        error = (
            self._error_text
            if self._error_pred and self._error_pred(credit, self)
            else None
        )
        cancelled = (
            bool(self._cancel_pred(credit, self)) if self._cancel_pred else False
        )
        asyncio.get_running_loop().call_soon(lambda: observer(credit, error, cancelled))
        return True

    def mark_graph_sending_complete(self) -> None: ...

    def graph_all_returned(self) -> bool:
        return self.all_returned

    def set_graph_all_returned_event(self) -> None: ...


def _error_first(pressure_only: bool = False) -> Callable[[Any, Any], bool]:
    """Predicate erroring the FIRST matching issued credit (fires once).

    ``pressure_only=False`` matches the first credit issued at all (a
    boundary-priming credit); ``pressure_only=True`` matches the first credit
    issued while the strategy's pressure stage is active
    (``issuer.strategy._pressure_active``) -- the priming/pressure split is no
    longer encoded in the ``trace_id`` string, so the stage flag is the signal.
    """
    state = {"fired": False}

    def pred(credit: Any, issuer: Any) -> bool:
        if state["fired"]:
            return False
        if pressure_only and not issuer.strategy._pressure_active:
            return False
        state["fired"] = True
        return True

    return pred


def _strategy(
    parsed: Any,
    *,
    duration: float | None,
    park_after: int | None = None,
    graph_channel: Any = None,
    concurrency: int = 1,
    phase: CreditPhase = CreditPhase.WARMUP,
    error_pred: Callable[[Any], bool] | None = None,
    cancel_pred: Callable[[Any], bool] | None = None,
    error_text: str = "boom",
) -> tuple[GraphIRReplayStrategy, _ParkAfterIssuer]:
    issuer = _ParkAfterIssuer(
        park_after=park_after,
        error_pred=error_pred,
        cancel_pred=cancel_pred,
        error_text=error_text,
    )
    strategy = GraphIRReplayStrategy(
        config=_Config(concurrency=concurrency, phase=phase),
        graph_channel=graph_channel,
        credit_issuer=issuer,
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        start_min_ratio=0.5,
        start_max_ratio=0.5,
        t_star_random_seed=1234,
        cache_pressure_duration_s=duration,
        # A mis-tuned park_after must fail in seconds, not ride out the 300s
        # adapter default (which collides with the global pytest --timeout).
        dispatch_timeout_s=2.0,
    )
    issuer.bind(strategy)
    return strategy, issuer


def _pressure_flag_spy(strategy: GraphIRReplayStrategy, sink: list) -> None:
    """Wrap ``_run_instance`` to record each call's (pressure, recycle_pass).

    The instance-id string no longer encodes the priming-vs-pressure flavor, so
    the ``pressure`` kwarg on ``_run_instance`` is the observable that
    distinguishes a boundary-priming instance from a pressure-stage one.
    """
    original = strategy._run_instance

    async def _spy(trace, lane, recycle_pass, **kwargs):
        sink.append((kwargs.get("pressure", False), recycle_pass))
        return await original(trace, lane, recycle_pass, **kwargs)

    strategy._run_instance = _spy


@pytest.mark.asyncio
async def test_execute_phase_without_duration_runs_no_pressure_instances():
    """Duration None: byte-identical warmup -- boundary priming only, no pressure."""
    parsed = _corpus(2)
    strategy, issuer = _strategy(parsed, duration=None)
    calls: list[tuple[bool, int]] = []
    _pressure_flag_spy(strategy, calls)
    await strategy.execute_phase()
    assert calls, "warmup must run at least one boundary-priming instance"
    assert not any(pressure for pressure, _pass in calls), "no pressure instances"
    assert strategy._pressure_active is False


@pytest.mark.asyncio
async def test_execute_phase_with_duration_runs_pressure_after_priming():
    """Pressure dispatches post-t* nodes AFTER priming drains (stage barrier).

    The priming/pressure split is no longer encoded in the ``trace_id`` string,
    so we observe ``_run_instance``'s ``pressure`` flag: every priming instance
    (pressure=False) runs before any pressure instance (pressure=True), the stage
    flips ``_pressure_active``, and credits are issued.
    """
    parsed = _corpus(2)
    strategy, issuer = _strategy(parsed, duration=0.2, park_after=50)
    calls: list[tuple[bool, int]] = []
    _pressure_flag_spy(strategy, calls)
    await strategy.execute_phase()

    flags = [pressure for pressure, _pass in calls]
    assert any(not f for f in flags), "boundary priming must still run first"
    assert any(flags), "pressure stage must dispatch at least one instance"
    # All priming (False) precedes all pressure (True) -- the stage barrier.
    first_pressure = flags.index(True)
    assert not any(flags[:first_pressure]), "priming must all precede pressure"
    assert strategy._pressure_active is True
    assert issuer.issued, "priming + pressure must have issued credits"


@pytest.mark.asyncio
async def test_pressure_ledger_records_return_walls_by_instance_and_node():
    parsed = _corpus(1)
    strategy, issuer = _strategy(parsed, duration=0.2, park_after=50)
    await strategy.execute_phase()

    assert strategy._return_walls, "ledger must be populated"
    for instance_id, walls in strategy._return_walls.items():
        # Instance ids are ``{template}::{nonce}``.
        assert "::" in instance_id
        for node_id, wall in walls.items():
            assert isinstance(node_id, str) and node_id in parsed.graph.nodes
            assert wall > 0.0


def _ledger_key_for(
    strategy: GraphIRReplayStrategy, credit: Any
) -> tuple[str, str] | None:
    """Replicate ``_record_return_wall``'s (instance_id, node_id) mapping.

    Returns None when the credit does not resolve to a catalog node (unmappable
    returns are never ledgered regardless of the cancelled gate).
    """
    instance_id = getattr(credit, "trace_id", None)
    ordinal = getattr(credit, "node_ordinal", None)
    if instance_id is None or ordinal is None:
        return None
    template_id = instance_id.split("::", 1)[0]
    inverse = {
        o: nid for nid, o in strategy._catalog.catalog.get(template_id, {}).items()
    }
    node_id = inverse.get(ordinal)
    return (instance_id, node_id) if node_id is not None else None


@pytest.mark.asyncio
async def test_cancelled_returns_never_enter_the_pressure_ledger():
    """Wire-cancelled turns are NOT executed: the server may never have
    completed them, so the handoff must let profiling refire them (and a
    successful grace-expiry cancel-drain then yields a VALID handoff).

    Keyed by (instance_id, node_id): a single-template pressure corpus recycles
    the same node across passes, so a bare node-id set would collide with a
    clean return of the same node on a later instance.
    """
    parsed = _corpus(1)
    cancelled: list[Any] = []

    def cancel_first_pressure(credit: Any, issuer: Any) -> bool:
        # Cancel exactly the FIRST pressure-stage return, identified by the
        # strategy's pressure-stage flag (the flavor is no longer in trace_id).
        if cancelled:
            return False
        if issuer.strategy._pressure_active:
            cancelled.append(credit)
            return True
        return False

    strategy, _issuer = _strategy(
        parsed, duration=0.2, park_after=50, cancel_pred=cancel_first_pressure
    )
    await strategy.execute_phase()

    assert cancelled, "the stub must have cancelled at least one pressure return"
    assert strategy._return_walls, "clean returns must still populate the ledger"

    cancelled_keys = {
        key for c in cancelled if (key := _ledger_key_for(strategy, c)) is not None
    }
    assert cancelled_keys, "the cancelled credit must map to a catalog node"
    recorded_keys = {
        (instance_id, node_id)
        for instance_id, walls in strategy._return_walls.items()
        for node_id in walls
    }
    assert not (cancelled_keys & recorded_keys), (
        "a wire-cancelled return must be excluded from the pressure ledger"
    )


@pytest.mark.asyncio
async def test_pressure_recycles_lane_after_instance_completion():
    """A completed pressure instance frees its lane for a recycle (pass >= 1) draw.

    The recycle pass is no longer id-encoded, so observe ``_run_instance``'s
    ``recycle_pass`` arg for pressure-stage instances directly.
    """
    parsed = _corpus(1)
    # park very late so at least one full instance completes and recycles
    strategy, issuer = _strategy(parsed, duration=0.2, park_after=500)
    calls: list[tuple[bool, int]] = []
    _pressure_flag_spy(strategy, calls)
    await strategy.execute_phase()
    pressure_passes = {recycle_pass for pressure, recycle_pass in calls if pressure}
    assert any(p >= 1 for p in pressure_passes), (
        f"expected a recycled pressure pass (>= 1), saw {pressure_passes}"
    )


class _RefusingIssuer:
    """``issue_graph_credit`` raises ``CreditIssueRefusedError`` (closed stop gate).

    Models the issuer stop gate refusing a fresh dispatch (request-count /
    duration cap reached, or run cancelled): the adapter surfaces the refusal to
    the executor, whose ``TaskGroup`` wraps it, and ``_leaf_credit_refusal``
    matches the (grouped) refusal so ``_run_instance`` reports a clean stop.
    """

    def __init__(self) -> None:
        self.observer = None
        self.issued: list[Any] = []

    def bind(self, strategy: GraphIRReplayStrategy) -> None:
        self.observer = strategy._on_graph_return

    async def issue_graph_credit(self, credit: Any) -> bool:
        self.issued.append(credit)
        raise CreditIssueRefusedError("stop gate closed for test")

    def mark_graph_sending_complete(self) -> None: ...

    def graph_all_returned(self) -> bool:
        return True

    def set_graph_all_returned_event(self) -> None: ...


@pytest.mark.asyncio
async def test_run_instance_returns_true_on_issuer_refusal():
    """A clean issuer refusal makes ``_run_instance`` report True (lane stop signal)."""
    parsed = _corpus(1)
    issuer = _RefusingIssuer()
    strategy = GraphIRReplayStrategy(
        config=_Config(concurrency=1, phase=CreditPhase.PROFILING),
        credit_issuer=issuer,
        parsed_graph=parsed,
        register_observer=lambda _obs: None,
        start_min_ratio=0.0,
        start_max_ratio=0.0,
        dispatch_timeout_s=2.0,
    )
    issuer.bind(strategy)

    refused = await strategy._run_instance(parsed.traces[0], 0, 0)

    assert refused is True
    # A clean refusal is a healthy stop, NOT a trace error.
    assert strategy._errored_traces == 0


@pytest.mark.asyncio
async def test_run_instance_returns_false_on_clean_success():
    """A normally-resolving instance reports False (no lane stop)."""
    parsed = _corpus(1)
    strategy, _issuer = _strategy(parsed, duration=None, phase=CreditPhase.PROFILING)

    refused = await strategy._run_instance(parsed.traces[0], 0, 0)

    assert refused is False


class _FakeSource:
    """Duck-typed graph channel (cross-phase warmup-handoff slot)."""

    def __init__(self) -> None:
        self.warmup_handoff = None


@pytest.mark.asyncio
async def test_teardown_stashes_handoff_for_stalled_lanes():
    parsed = _corpus(1)
    source = _FakeSource()
    strategy, issuer = _strategy(
        parsed, duration=0.2, park_after=3, graph_channel=source
    )
    await strategy.execute_phase()
    await strategy.teardown_phase()

    handoff = source.warmup_handoff
    assert handoff is not None
    assert handoff.lanes, "the parked lane must be live at drain"
    for _lane, entry in handoff.lanes.items():
        assert entry.template_trace_id == "t-0"
        assert entry.executed_node_ids <= set(parsed.graph.nodes)
        # every executed node has a residual anchor wall <= drain end
        for node_id in entry.executed_node_ids:
            assert entry.return_wall_us[node_id] <= handoff.drain_end_wall_us
    assert handoff.drain_end_wall_us > 0.0


@pytest.mark.asyncio
async def test_teardown_skips_stash_when_returns_incomplete():
    """Grace-timeout / cancelled drains must NOT hand profiling a wrong handoff.

    With finite drain grace (this task), a grace-timeout force-complete reaches
    teardown with graph_all_returned() False; the stash must skip so profiling
    falls back to plain t* plans instead of refiring server-executed nodes.
    """
    parsed = _corpus(1)
    source = _FakeSource()
    strategy, issuer = _strategy(
        parsed, duration=0.2, park_after=3, graph_channel=source
    )
    issuer.all_returned = False
    await strategy.execute_phase()
    await strategy.teardown_phase()
    assert source.warmup_handoff is None


@pytest.mark.asyncio
async def test_teardown_merges_priming_walls_into_pass0_handoff():
    """Pass-0 pressure lanes carry their boundary-priming return walls too.

    A chain the pressure stage never advanced still needs its residual anchored
    on the PRIMING return of its boundary turn (agentx baseline-return parity).
    """
    parsed = _corpus(1)
    source = _FakeSource()
    # park immediately after priming: pressure issues nothing, lane stalls on
    # its first pressure dispatch -> executed empty, priming walls present
    strategy, issuer = _strategy(
        parsed, duration=0.2, park_after=1, graph_channel=source
    )
    await strategy.execute_phase()
    await strategy.teardown_phase()

    handoff = source.warmup_handoff
    assert handoff is not None and handoff.lanes
    # concurrency=1 in the helper => exactly lane 0; assert on it explicitly
    # (dict insertion order is pop/re-insert-sensitive across recycles).
    entry = handoff.lanes[0]
    # The priming instance id is ``t-0::{nonce}`` -- recover it as the ledgered
    # instance that is NOT the live pressure instance.
    priming_walls = next(
        (
            walls
            for iid, walls in strategy._return_walls.items()
            if iid != entry.instance_id
        ),
        {},
    )
    assert priming_walls, "the priming instance must have ledgered returns"
    assert set(priming_walls) <= set(entry.return_wall_us)


@pytest.mark.asyncio
async def test_teardown_handoff_carries_recycle_cursor_past_pass0():
    """The stashed handoff carries the pressure stage's next recycle cursor.

    A 1-template corpus resolves pass-0 at cursor 1 (one template consumed); a
    pressure run that recycled at least once advances ``_pressure_next_index``
    strictly past that, and the stash persists it so profiling's bounded recycle
    continues from the pressure stage's last draw (agentx shared-sampler parity).
    """
    parsed = _corpus(1)
    source = _FakeSource()
    strategy, _issuer = _strategy(
        parsed, duration=0.2, park_after=500, graph_channel=source
    )
    await strategy.execute_phase()
    await strategy.teardown_phase()

    handoff = source.warmup_handoff
    assert handoff is not None
    assert handoff.corpus_cursor >= 2, (
        "corpus_cursor must advance strictly past the pass-0 resolution cursor "
        f"(1) after at least one recycle draw; saw {handoff.corpus_cursor}"
    )


@pytest.mark.asyncio
async def test_stash_records_pressure_lane_count():
    parsed = _corpus(1)
    source = _FakeSource()
    strategy, issuer = _strategy(
        parsed, duration=0.2, park_after=3, graph_channel=source
    )
    await strategy.execute_phase()
    await strategy.teardown_phase()
    assert source.warmup_handoff.pressure_lane_count == 1


@pytest.mark.asyncio
async def test_teardown_without_pressure_leaves_source_untouched():
    parsed = _corpus(1)
    source = _FakeSource()
    strategy, _issuer = _strategy(parsed, duration=None, graph_channel=source)
    await strategy.execute_phase()
    await strategy.teardown_phase()
    assert source.warmup_handoff is None


@pytest.mark.asyncio
async def test_report_warmup_failures_raises_on_errored_priming_return():
    """A terminal error on a boundary-priming return aborts the warmup (agentx parity)."""
    parsed = _corpus(1)
    strategy, issuer = _strategy(parsed, duration=None, error_pred=_error_first())
    await strategy.execute_phase()

    assert issuer.issued, "boundary priming must issue at least one credit"
    assert strategy._warmup_failure_count == 1
    with pytest.raises(TrajectoryWarmupFailedError) as excinfo:
        strategy.report_warmup_failures()
    msg = str(excinfo.value)
    assert "boom" in msg, f"abort message must surface the error text: {msg!r}"
    assert "1 trace" in msg, f"abort message must surface the count: {msg!r}"


@pytest.mark.asyncio
async def test_report_warmup_failures_counts_pressure_errors():
    """A terminal error on a pressure-stage return aborts the warmup."""
    parsed = _corpus(1)
    strategy, issuer = _strategy(
        parsed,
        duration=0.2,
        park_after=50,
        error_pred=_error_first(pressure_only=True),
    )
    await strategy.execute_phase()

    assert strategy._pressure_active, (
        "pressure stage must have run to error a pressure-stage credit"
    )
    assert strategy._warmup_failure_count >= 1
    with pytest.raises(TrajectoryWarmupFailedError):
        strategy.report_warmup_failures()


@pytest.mark.asyncio
async def test_report_warmup_failures_ignores_cancelled_returns():
    """Cancelled returns are excluded from the abort gate even with error text.

    The pressure drain cancels executor coroutines on the duration timer, so a
    cancellation surfacing at drain is self-inflicted teardown, not a server
    failure -- it must not abort an otherwise-healthy warmup.
    """
    parsed = _corpus(1)
    strategy, issuer = _strategy(
        parsed,
        duration=None,
        cancel_pred=lambda _c, _i: True,  # every return is cancelled
        error_pred=_error_first(),  # the first also carries error text
        error_text="err",
    )
    await strategy.execute_phase()

    assert issuer.issued
    assert strategy._warmup_failure_count == 0
    strategy.report_warmup_failures()  # must not raise


def test_report_warmup_failures_message_carries_true_count_past_sample_cap():
    """More failures than the 5-sample cap still surface the TRUE total in the message."""

    class _StubCredit:
        def __init__(self, i: int) -> None:
            self.trace_id = f"t-{i}#0.0"
            self.node_ordinal = i

    parsed = _corpus(1)
    strategy, _issuer = _strategy(parsed, duration=None)
    for i in range(7):
        strategy._record_warmup_failure(_StubCredit(i), f"boom-{i}")

    assert strategy._warmup_failure_count == 7
    assert len(strategy._warmup_failure_samples) == 5
    with pytest.raises(TrajectoryWarmupFailedError) as excinfo:
        strategy.report_warmup_failures()
    msg = str(excinfo.value)
    assert "total 7" in msg, f"abort message must carry the true count: {msg!r}"
    assert "boom-0" in msg, f"abort message must keep the real samples: {msg!r}"


@pytest.mark.asyncio
async def test_report_warmup_failures_noop_for_profiling_phase_and_clean_warmup():
    """PROFILING phases and clean warmups never abort the run."""
    parsed = _corpus(1)

    # A PROFILING phase with an errored return does NOT record or abort.
    prof, _issuer = _strategy(
        parsed,
        duration=None,
        phase=CreditPhase.PROFILING,
        error_pred=_error_first(),
    )
    await prof.execute_phase()
    assert prof._warmup_failure_count == 0
    prof.report_warmup_failures()  # profiling: no-op

    # A clean warmup (no terminal failures) does NOT abort.
    warm, _issuer2 = _strategy(parsed, duration=None)
    await warm.execute_phase()
    assert warm._warmup_failure_count == 0
    warm.report_warmup_failures()  # clean: no-op
