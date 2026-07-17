# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail-loud wrap-guard for EXPLICIT concurrent over-subscription.

Component-level: no real worker / ZMQ / mock server. A fake ``CreditIssuer``
echoes each ``issue_graph_credit`` back to the strategy's installed graph-return
observer, so ``execute_phase`` drives the REAL dispatch/setup path (the guard
fires early in ``execute_phase``, before any lane starts).

Asserts the contract: resolved ``concurrency`` exceeding the
distinct loaded traces is a HARD configuration error UNLESS dataset wrapping is
allowed (explicit ``--allow-dataset-wrap`` or, unset, cache-bust ON) -- never a
silent clone-to-fill.

- (a) concurrency 100 over 15 distinct, cache-bust OFF, wrap unset -> RAISES;
      message carries the cache-bust suggestion.
- (b) same but cache-bust ON -> wraps (no raise).
- (c) explicit ``allow_dataset_wrap=False`` + cache-bust ON -> RAISES; message
      OMITS the cache-bust suggestion (already on) and notes the explicit disable.
- (d) default concurrency 1 over any corpus -> NEVER raises.
- (e) a selection knob (num_dataset_entries) capping the corpus -> the message
      phrases the shortfall as CAPPED, not EXHAUSTED.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import msgspec
import pytest

from aiperf.common.enums import CacheBustTarget, CreditPhase
from aiperf.common.exceptions import ConfigurationError

pytestmark = [pytest.mark.component_integration, pytest.mark.asyncio]

_FIX_DIR = Path(__file__).parents[2] / "unit" / "graph" / "fixtures"
_MIN = _FIX_DIR / "weka_min.json"


@dataclass
class _EchoIssuer:
    """Fake CreditIssuer echoing each issued credit to the return observer."""

    observer: Any = None
    issued: int = 0
    returned: int = 0
    sending_complete_calls: int = 0
    all_returned_event_set: bool = False
    sent: list[Any] = field(default_factory=list)

    async def issue_graph_credit(self, turn: Any) -> bool:
        self.issued += 1
        self.sent.append(turn)
        asyncio.get_running_loop().call_soon(self._echo, turn)
        return True

    def _echo(self, turn: Any) -> None:
        self.returned += 1
        if self.observer is not None:
            self.observer(turn, None, False)

    def mark_graph_sending_complete(self) -> None:
        self.sending_complete_calls += 1

    def graph_all_returned(self) -> bool:
        return self.returned >= self.issued

    def set_graph_all_returned_event(self) -> None:
        self.all_returned_event_set = True


class _PhaseCfg:
    """Per-phase config stub carrying only the fields the guard/strategy read.

    ``num_dataset_entries`` / ``max_context_length`` mirror the graph-plane
    selection knobs the guard reads off the config (via ``getattr``) to phrase
    the shortfall as CAPPED rather than EXHAUSTED.
    """

    def __init__(
        self,
        *,
        concurrency: int | None = None,
        num_dataset_entries: int | None = None,
        max_context_length: int | None = None,
        expected_num_sessions: int | None = None,
    ) -> None:
        self.phase = CreditPhase.PROFILING
        self.concurrency = concurrency
        self.expected_num_sessions = expected_num_sessions
        self.total_expected_requests = None
        self.expected_duration_sec = None
        self.num_dataset_entries = num_dataset_entries
        self.max_context_length = max_context_length


def _corpus(n: int, *, gap_free: bool = False):
    """Return a ParsedGraph whose ``traces`` holds ``n`` distinct-id instances.

    Replicates the single ``weka_min`` template with ``#N`` suffixes; every clone
    resolves to the base graph (single-graph fallback). ``gap_free`` zeros edge
    delays so a dispatching (non-raising) run replays instantly instead of parking
    on ``weka_min``'s recorded ~1s idle gaps (real sleeps in component tests).
    """
    from aiperf.dataset.graph.adapters.weka.trace import from_weka_trace

    base = from_weka_trace(str(_MIN))
    if gap_free:
        graph = base.graph
        zeroed = [
            msgspec.structs.replace(
                e,
                **{
                    f: 0.0
                    for f in ("delay_after_predecessor_us", "min_start_delay_us")
                    if getattr(e, f, None) is not None
                },
            )
            for e in graph.edges
        ]
        base = msgspec.structs.replace(
            base, graph=msgspec.structs.replace(graph, edges=zeroed)
        )
    t0 = base.traces[0]
    clones = [t0]
    clones.extend(msgspec.structs.replace(t0, id=f"{t0.id}#{i}") for i in range(1, n))
    return msgspec.structs.replace(base, traces=clones)


def _make_strategy(parsed, issuer: _EchoIssuer, **overrides):
    from aiperf.timing.strategies.graph_ir_replay import GraphIRReplayStrategy

    overrides.setdefault("start_min_ratio", 0.0)
    overrides.setdefault("start_max_ratio", 0.0)
    return GraphIRReplayStrategy(
        config=overrides.pop("config", None),
        conversation_source=None,
        scheduler=None,
        stop_checker=None,
        credit_issuer=issuer,
        lifecycle=overrides.pop("lifecycle", None),
        parsed_graph=parsed,
        register_observer=lambda obs: setattr(issuer, "observer", obs),
        **overrides,
    )


async def test_oversubscription_cache_bust_off_wrap_unset_raises():
    """(a) concurrency 100 over 15, cache-bust OFF, wrap unset -> RAISES."""
    parsed = _corpus(15)
    strategy = _make_strategy(
        parsed,
        _EchoIssuer(),
        max_concurrent_traces=100,
        cache_bust=CacheBustTarget.NONE,
        allow_dataset_wrap=None,
    )
    # The guard raises on the AWAITED setup_phase path (PhaseRunner launches
    # execute_phase fire-and-forget, so a raise there would be swallowed).
    with pytest.raises(ConfigurationError) as exc:
        await strategy.setup_phase()

    msg = str(exc.value)
    assert "concurrency 100 exceeds 15 distinct loaded traces" in msg
    # EXHAUSTED phrasing (no selection knob was set).
    assert "only 15 distinct traces available" in msg
    # Cache-bust suggestion present ONLY because cache-bust is OFF.
    assert "--cache-bust first_turn_prefix" in msg
    assert "clones do not collide on identical prefixes" in msg


async def test_oversubscription_cache_bust_on_wraps_no_raise():
    """(b) same over-subscription but cache-bust ON -> wraps (no raise)."""
    parsed = _corpus(15, gap_free=True)
    issuer = _EchoIssuer()
    strategy = _make_strategy(
        parsed,
        issuer,
        max_concurrent_traces=100,
        cache_bust=CacheBustTarget.FIRST_TURN_PREFIX,
        allow_dataset_wrap=None,
    )
    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=15.0)

    # No raise: the run wraps/covers the corpus and completes.
    assert strategy.completed_traces == 15
    assert issuer.sending_complete_calls >= 1


async def test_explicit_wrap_false_raises_and_omits_cache_bust_suggestion():
    """(c) explicit allow_dataset_wrap=False + cache-bust ON -> RAISES, no cache-bust hint."""
    parsed = _corpus(15)
    strategy = _make_strategy(
        parsed,
        _EchoIssuer(),
        max_concurrent_traces=100,
        cache_bust=CacheBustTarget.FIRST_TURN_PREFIX,
        allow_dataset_wrap=False,
    )
    # The guard raises on the AWAITED setup_phase path (PhaseRunner launches
    # execute_phase fire-and-forget, so a raise there would be swallowed).
    with pytest.raises(ConfigurationError) as exc:
        await strategy.setup_phase()

    msg = str(exc.value)
    assert "concurrency 100 exceeds 15 distinct loaded traces" in msg
    # The real distinguisher between (a) and (c): cache-bust already ON -> the
    # cache-bust suggestion must be OMITTED (no cache-bust mention at all).
    assert "--cache-bust" not in msg
    assert "clones do not collide" not in msg
    # The disabled-wrapping note is present but NEUTRAL -- it must never claim the
    # user passed --allow-dataset-wrap false (``GraphDispatchResolver`` derives that
    # False from an UNSET flag on the default path).
    assert "Dataset wrapping is disabled" in msg
    assert "--allow-dataset-wrap false" not in msg


async def test_default_concurrency_one_never_raises():
    """(d) default concurrency 1 over a corpus -> never raises (1 <= corpus)."""
    parsed = _corpus(3, gap_free=True)
    issuer = _EchoIssuer()
    strategy = _make_strategy(
        parsed,
        issuer,
        config=_PhaseCfg(concurrency=None),
        max_concurrent_traces=None,  # resolves to the aiperf default of 1
        cache_bust=CacheBustTarget.NONE,
        allow_dataset_wrap=False,
    )
    assert strategy._max_concurrent == 1
    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=15.0)

    assert strategy.completed_traces == 3


async def test_session_budget_within_corpus_stands_down():
    """(f) an explicit session budget <= distinct traces -> NO raise even with
    concurrency over-provisioned: total instances are bounded below any
    cloning need, so concurrency is a mere ceiling (e.g. the common
    ``--num-conversations 1 --concurrency 4`` single-pass shape)."""
    parsed = _corpus(3, gap_free=True)
    issuer = _EchoIssuer()
    strategy = _make_strategy(
        parsed,
        issuer,
        config=_PhaseCfg(concurrency=100, expected_num_sessions=3),
        max_concurrent_traces=100,
        cache_bust=CacheBustTarget.NONE,
        allow_dataset_wrap=False,
    )
    await strategy.setup_phase()
    await asyncio.wait_for(strategy.execute_phase(), timeout=15.0)

    assert strategy.completed_traces == 3
    assert issuer.sending_complete_calls >= 1


async def test_session_budget_beyond_corpus_still_raises():
    """(f') a session budget EXCEEDING the distinct corpus still needs clones,
    so the guard keeps failing loud without wrapping."""
    parsed = _corpus(3)
    strategy = _make_strategy(
        parsed,
        _EchoIssuer(),
        config=_PhaseCfg(concurrency=100, expected_num_sessions=6),
        max_concurrent_traces=100,
        cache_bust=CacheBustTarget.NONE,
        allow_dataset_wrap=None,
    )
    with pytest.raises(ConfigurationError) as exc:
        await strategy.setup_phase()
    assert "concurrency 100 exceeds 3 distinct loaded traces" in str(exc.value)


async def test_capped_corpus_phrases_message_as_capped():
    """(e) a selection knob capping the corpus -> CAPPED phrasing, not EXHAUSTED."""
    parsed = _corpus(15)
    strategy = _make_strategy(
        parsed,
        _EchoIssuer(),
        config=_PhaseCfg(concurrency=100, num_dataset_entries=15),
        max_concurrent_traces=100,
        cache_bust=CacheBustTarget.NONE,
        allow_dataset_wrap=None,
    )
    # The guard raises on the AWAITED setup_phase path (PhaseRunner launches
    # execute_phase fire-and-forget, so a raise there would be swallowed).
    with pytest.raises(ConfigurationError) as exc:
        await strategy.setup_phase()

    msg = str(exc.value)
    assert "capped to 15 by" in msg
    assert "--num-dataset-entries" in msg
    assert "only 15 distinct traces available" not in msg


async def test_from_run_threads_num_dataset_entries_onto_phase_config():
    """from_run seam: --num-dataset-entries N lands on the graph CreditPhaseConfig.

    Closes the three-touch wiring trap the CAPPED branch depends on. The guard
    reads ``num_dataset_entries`` straight off ``self._config`` (the
    CreditPhaseConfig the runner hands the strategy), so if ``from_run`` does
    not thread the resolved cap onto that config the CAPPED phrasing silently
    dies and every real run says EXHAUSTED even when the user capped the corpus.
    Mirrors ``test_duplication_report.test_cache_bust_seam_reaches_strategy_end_to_end``.
    """
    from aiperf.config.flags.cli_config import CLIConfig
    from aiperf.plugin.enums import TimingMode
    from aiperf.timing.config import TimingConfig
    from tests.unit.conftest import make_run_from_cli

    cfg = CLIConfig(
        model_names=["test-model"],
        input_file=str(_MIN),
        request_count=3,
        conversation_num_dataset_entries=7,
    )
    run = make_run_from_cli(cfg)

    tc = TimingConfig.from_run(run)
    profiling = [p for p in tc.phase_configs if p.phase == CreditPhase.PROFILING]
    assert profiling, "expected a graph profiling phase"
    phase_cfg = profiling[0]
    assert phase_cfg.timing_mode == TimingMode.GRAPH_IR
    # from_run resolves the explicit --num-dataset-entries cap (the default
    # dataset's ``entries``) onto the CreditPhaseConfig, feeding the guard's
    # CAPPED branch.
    assert phase_cfg.num_dataset_entries == 7
    # No synthesis.max_context_length set -> the sibling cap resolves to None.
    assert phase_cfg.max_context_length is None
