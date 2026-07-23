# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic-replay end-to-end component-integration tests."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import ConversationBranchMode, CreditPhase
from aiperf.common.models import DatasetMetadata
from aiperf.credit.structs import Credit
from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.exporters.aggregate import (
    AggregateConfidenceJsonExporter,
    AggregateExporterConfig,
)
from aiperf.orchestrator.aggregation.base import AggregateResult
from aiperf.plugin.enums import DatasetSamplingStrategy
from aiperf.timing.strategies.agentic_replay import AgenticReplayStrategy
from aiperf.timing.trajectory_source import TrajectorySource

pytestmark = pytest.mark.component_integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "weka_traces_small"


@dataclass
class _DispatchLog:
    """Capture every credit issued through the strategy for ordering checks."""

    entries: list[tuple[CreditPhase, str, int]] = field(default_factory=list)
    """List of (phase, conversation_id, turn_index) per dispatched credit."""

    def by_phase(self, phase: CreditPhase) -> list[tuple[str, int]]:
        return [(cid, idx) for ph, cid, idx in self.entries if ph == phase]

    def trace_ids_in_phase(self, phase: CreditPhase) -> list[str]:
        return [cid for ph, cid, _ in self.entries if ph == phase]


class _SequentialSampler:
    """Deterministic sampler over a fixed conversation_id list (rooted only)."""

    def __init__(self, conversation_ids: list[str]) -> None:
        self._ids = list(conversation_ids)
        self._idx = 0

    def next_conversation_id(self) -> str:
        if self._idx >= len(self._ids):
            self._idx = 0
        cid = self._ids[self._idx]
        self._idx += 1
        return cid


def _make_weka_run():
    """Build a real ``BenchmarkRun`` adequate for WekaTraceLoader."""
    from tests.unit.dataset.loader.conftest import make_weka_run

    return make_weka_run(
        model_names=["claude-opus-4-5-20251101"],
        tokenizer_name="test-tok",
    )


def _load_small_weka_dataset(monkeypatch, *, parallel: bool = False) -> DatasetMetadata:
    """Load the synthetic weka fixture into a DatasetMetadata."""
    from aiperf.common.environment import Environment

    run = _make_weka_run()
    loader = WekaTraceLoader(filename=str(FIXTURES), run=run, default_block_size=64)
    monkeypatch.setattr(
        loader, "synthesize_prompts_from_hash_ids", lambda rs: {r.key: "p" for r in rs}
    )
    monkeypatch.setattr(
        loader,
        "sample_partial_tail_tokens",
        lambda n_tokens, seed: [0] * max(n_tokens, 0),
    )
    monkeypatch.setattr(
        loader, "sample_partial_tail", lambda n_tokens, seed: "x" * max(n_tokens, 0)
    )
    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    if not parallel:
        monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_WORKERS", 1)
        monkeypatch.setattr(
            loader,
            "_decode_block_tokens",
            lambda hash_ids: [0] * (len(hash_ids) * loader._block_size),
        )
        monkeypatch.setattr(
            loader, "_decode_tokens_to_text", lambda tokens: "x" * len(tokens)
        )
    else:
        from aiperf.common.hash_id_random_generator import HashIdRandomGenerator

        loader.prompt_generator._tokenized_corpus = list(range(10000, 11000))
        loader.prompt_generator._corpus_size = 1000
        loader.prompt_generator._bpe_stable_terminator_tokens = []
        loader.prompt_generator._hash_id_corpus_rng = HashIdRandomGenerator(
            12345, _internal=True
        )
        loader.prompt_generator.tokenizer.decode.side_effect = (
            lambda toks: f"<dec:{len(toks)}>"
        )

        monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_WORKERS", 2)
        monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_THRESHOLD", 1)
        _install_inproc_pool(monkeypatch, loader)

    convs = loader.convert_to_conversations(loader.load_dataset())
    return DatasetMetadata(
        conversations=[c.to_metadata() for c in convs],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _install_inproc_pool(monkeypatch, loader) -> None:
    """Replace the multiprocessing Pool with a synchronous in-process stub."""
    from aiperf.dataset.loader import weka_parallel_convert as wpc

    pg = loader.prompt_generator

    class _InProcPool:
        def __init__(self, num_workers, init_fn, init_args) -> None:
            init_fn(init_args[0])

        def imap(self, fn, items, chunksize=1):
            return [fn(it) for it in items]

        def close(self) -> None:
            return None

        def join(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

    class _FakeCtx:
        Pool = _InProcPool

    monkeypatch.setattr(wpc, "get_loader_mp_context", lambda **kw: _FakeCtx())
    monkeypatch.setattr(wpc.Tokenizer, "from_pretrained", lambda *a, **kw: pg.tokenizer)


def _make_recording_issuer(log: _DispatchLog, current_phase: list[CreditPhase]):
    """Build an AsyncMock credit issuer that records dispatches into ``log``."""
    issuer = AsyncMock()
    cid_to_xcorr: dict[str, str] = {}

    async def _issue(turn) -> bool:
        log.entries.append((current_phase[0], turn.conversation_id, turn.turn_index))
        cid_to_xcorr[turn.conversation_id] = turn.x_correlation_id
        return True

    issuer.issue_credit.side_effect = _issue
    issuer.cid_to_xcorr = cid_to_xcorr
    return issuer


def _make_stop_checker(allow_new_sessions: bool = True):
    sc = MagicMock()
    sc.can_start_new_session.return_value = allow_new_sessions
    return sc


def _make_credit(
    *,
    conversation_id: str,
    turn_index: int,
    num_turns: int,
    x_correlation_id: str = "xcorr",
    phase: CreditPhase = CreditPhase.PROFILING,
) -> Credit:
    return Credit(
        id=0,
        phase=phase,
        conversation_id=conversation_id,
        x_correlation_id=x_correlation_id,
        turn_index=turn_index,
        num_turns=num_turns,
        issued_at_ns=0,
        branch_mode=ConversationBranchMode.FORK,
    )


def _make_running_scheduler() -> MagicMock:
    """Build a scheduler mock whose ``schedule_later`` actually runs coroutines."""
    scheduler = MagicMock()
    scheduled: list[asyncio.Task] = []

    def _schedule_later(_delay, coro):
        scheduled.append(asyncio.ensure_future(coro))

    scheduler.schedule_later.side_effect = _schedule_later
    scheduler._scheduled_tasks = scheduled
    return scheduler


async def _flush_scheduled(strategy: AgenticReplayStrategy) -> None:
    """Await every task the strategy's scheduler queued via ``schedule_later``."""
    scheduled: list[asyncio.Task] = strategy.scheduler._scheduled_tasks
    for _ in range(50):
        await asyncio.sleep(0)
        pending = [t for t in scheduled if not t.done()]
        if not pending:
            break
        await asyncio.gather(*pending)
    for task in scheduled:
        if task.done() and not task.cancelled():
            task.result()


def _build_phase_strategy(
    *,
    phase: CreditPhase,
    source: TrajectorySource,
    issuer,
    stop_checker=None,
):
    cfg = MagicMock()
    cfg.phase = phase
    cfg.concurrency = len(source.trajectories)
    return AgenticReplayStrategy(
        config=cfg,
        conversation_source=source,
        scheduler=_make_running_scheduler(),
        stop_checker=stop_checker if stop_checker is not None else _make_stop_checker(),
        credit_issuer=issuer,
        lifecycle=MagicMock(),
        run=None,
    )


async def _export_aggregate(aggregate: AggregateResult, tmp_path: Path) -> dict:
    config = AggregateExporterConfig(result=aggregate, output_dir=tmp_path)
    exporter = AggregateConfidenceJsonExporter(config)
    out_path = await exporter.export()
    with open(out_path) as f:
        return json.load(f)


def _make_aggregate_with_carriers(
    *,
    scenario_name: str | None,
    validator_valid: bool | None,
    validator_reasons: list[str],
    total_responses: int,
    context_overflow_count: int,
) -> AggregateResult:
    """Build an AggregateResult carrying the cli_runner stamps."""
    md: dict = {}
    if scenario_name is not None:
        md["_scenario_name"] = scenario_name
        md["_validator_submission_valid"] = validator_valid
        md["_validator_submission_invalid_reasons"] = list(validator_reasons)
        md["_total_responses"] = total_responses
        md["_context_overflow_count"] = context_overflow_count
    return AggregateResult(
        aggregation_type="confidence",
        num_runs=2,
        num_successful_runs=2,
        failed_runs=[],
        metrics={},
        metadata=md,
    )


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.asyncio
async def test_agentic_replay_e2e_clean_run_under_scenario(
    tmp_path: Path, monkeypatch, parallel: bool
) -> None:
    """Spec section 8.2 #1: clean scenario run."""
    dataset = _load_small_weka_dataset(monkeypatch, parallel=parallel)
    assert len(dataset.conversations) == 10, (
        "small fixture should produce exactly 10 traces"
    )

    sampler = _SequentialSampler([c.conversation_id for c in dataset.conversations])
    source = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=sampler,
        concurrency=4,
        random_seed=12345,
    )
    assert len(source.trajectories) == 4, "trajectory = min(concurrency, pool) = 4"

    log = _DispatchLog()
    current_phase = [CreditPhase.WARMUP]
    issuer = _make_recording_issuer(log, current_phase)

    warmup = _build_phase_strategy(
        phase=CreditPhase.WARMUP, source=source, issuer=issuer
    )
    await warmup.setup_phase()
    await warmup.execute_phase()
    await _flush_scheduled(warmup)
    warmup.report_warmup_failures()

    warmup_dispatched = log.by_phase(CreditPhase.WARMUP)
    expected_warmup = {
        (state.conversation_id, state.warmup_turn_index)
        for trajectory in source.trajectories
        if trajectory.snapshot is not None
        for state in trajectory.snapshot.states
        if state.warmup_turn_index is not None
    }
    assert set(warmup_dispatched) == expected_warmup, (
        f"WARMUP must dispatch each warmable stream once at its warmup_turn_index; "
        f"got {warmup_dispatched}, expected {sorted(expected_warmup)}"
    )
    assert len(warmup_dispatched) == source.warmup_credit_count

    assert log.by_phase(CreditPhase.PROFILING) == [], (
        "Warmup barrier violated: PROFILING dispatched before WARMUP completed"
    )

    current_phase[0] = CreditPhase.PROFILING
    profiling = _build_phase_strategy(
        phase=CreditPhase.PROFILING, source=source, issuer=issuer
    )
    await profiling.setup_phase()
    assert profiling.config.phase == CreditPhase.PROFILING
    assert len(profiling.conversation_source.trajectories) == len(source.trajectories)

    await profiling.execute_phase()
    await _flush_scheduled(profiling)

    profiling_dispatched = log.by_phase(CreditPhase.PROFILING)
    trajectory_ks = {
        trajectory.conversation_id: trajectory.start_turn_index
        for trajectory in source.trajectories
    }
    metadata_lookup = source._metadata_lookup
    for trajectory_id, k in trajectory_ks.items():
        n = len(metadata_lookup[trajectory_id].turns)
        if k + 1 < n:
            assert (trajectory_id, k + 1) in profiling_dispatched, (
                f"trajectory {trajectory_id} should resume at k+1={k + 1}"
            )
        else:
            recycled_at_zero = [cid for cid, idx in profiling_dispatched if idx == 0]
            assert recycled_at_zero, (
                f"trajectory {trajectory_id} (N={n}, k={k}) should trigger an "
                "immediate recycle dispatch but none observed"
            )

    pre_recycle_count = len(profiling_dispatched)

    def _finalize(cid: str) -> Credit:
        n = len(metadata_lookup[cid].turns)
        return _make_credit(
            conversation_id=cid,
            turn_index=n - 1,
            num_turns=n,
            x_correlation_id=issuer.cid_to_xcorr[cid],
        )

    trajectories_to_finalize = list(source.trajectories)
    for trajectory in trajectories_to_finalize:
        await profiling.handle_credit_return(_finalize(trajectory.conversation_id))
    await _flush_scheduled(profiling)

    after_round1 = log.by_phase(CreditPhase.PROFILING)
    assert len(after_round1) > pre_recycle_count, (
        "round 1: recycle should have produced new turn-0 dispatches"
    )

    last_seen = pre_recycle_count
    finalized_so_far: set[str] = {m.conversation_id for m in trajectories_to_finalize}
    safety = 0
    while safety < 8:
        safety += 1
        snapshot = log.by_phase(CreditPhase.PROFILING)
        if len(snapshot) == last_seen:
            break
        new_dispatches = snapshot[last_seen:]
        last_seen = len(snapshot)
        for cid, _idx in new_dispatches:
            if cid in finalized_so_far:
                continue
            n = len(metadata_lookup[cid].turns)
            await profiling.handle_credit_return(
                _make_credit(
                    conversation_id=cid,
                    turn_index=n - 1,
                    num_turns=n,
                    x_correlation_id=issuer.cid_to_xcorr[cid],
                )
            )
            finalized_so_far.add(cid)
        full = log.trace_ids_in_phase(CreditPhase.PROFILING)
        if any(full.count(tid) > 1 for tid in set(full)):
            break

    full_profiling_ids = log.trace_ids_in_phase(CreditPhase.PROFILING)
    duplicates = [
        tid for tid in set(full_profiling_ids) if full_profiling_ids.count(tid) > 1
    ]
    assert duplicates, (
        "RECYCLE not observed: no trace_id appeared more than once in PROFILING "
        f"dispatch log over {len(full_profiling_ids)} dispatches; ids={full_profiling_ids}"
    )

    profiling.stop_checker.can_start_new_session.return_value = False
    pre_post_stop = len(log.by_phase(CreditPhase.PROFILING))
    in_flight_xcorrs = list(profiling._correlation_to_lane.keys())
    assert in_flight_xcorrs, (
        "Post-stop gate test requires at least one in-flight session"
    )
    safe_xcorr = in_flight_xcorrs[0]
    safe_cid = next(cid for cid, xc in issuer.cid_to_xcorr.items() if xc == safe_xcorr)
    safe_n = len(metadata_lookup[safe_cid].turns)
    await profiling.handle_credit_return(
        _make_credit(
            conversation_id=safe_cid,
            turn_index=safe_n - 1,
            num_turns=safe_n,
            x_correlation_id=safe_xcorr,
        )
    )
    assert len(log.by_phase(CreditPhase.PROFILING)) == pre_post_stop, (
        "Metrics window: handle_credit_return after stop must not spawn new sessions"
    )

    aggregate = _make_aggregate_with_carriers(
        scenario_name="inferencex-agentx-mvp",
        validator_valid=True,
        validator_reasons=[],
        total_responses=len(full_profiling_ids),
        context_overflow_count=0,
    )
    data = await _export_aggregate(aggregate, tmp_path)

    md = data["metadata"]
    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is True
    assert "submission_invalid_reasons" not in md
    for key in (
        "_scenario_name",
        "_validator_submission_valid",
        "_validator_submission_invalid_reasons",
        "_total_responses",
        "_context_overflow_count",
    ):
        assert key not in md, f"carrier key {key!r} leaked into output"


@pytest.mark.asyncio
async def test_agentic_replay_e2e_unsafe_override_stamps_false(
    tmp_path: Path,
) -> None:
    """Spec section 8.2 #2: --unsafe-override + violation -> submission_valid: false."""
    aggregate = _make_aggregate_with_carriers(
        scenario_name="inferencex-agentx-mvp",
        validator_valid=False,
        validator_reasons=["unsafe_override"],
        total_responses=500,
        context_overflow_count=0,
    )

    data = await _export_aggregate(aggregate, tmp_path)
    md = data["metadata"]

    assert md["scenario"] == "inferencex-agentx-mvp"
    assert md["submission_valid"] is False, (
        "Under --unsafe-override + duration<floor, submission_valid must be False"
    )
    assert "unsafe_override" in md["submission_invalid_reasons"]


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.asyncio
async def test_agentic_replay_e2e_no_scenario_omits_submission_valid(
    tmp_path: Path, monkeypatch, parallel: bool
) -> None:
    """Spec section 8.2 #3: bare agentic_replay timing mode without scenario."""
    dataset = _load_small_weka_dataset(monkeypatch, parallel=parallel)
    assert len(dataset.conversations) == 10

    sampler = _SequentialSampler([c.conversation_id for c in dataset.conversations])
    source = TrajectorySource(
        dataset_metadata=dataset,
        dataset_sampler=sampler,
        concurrency=3,
        random_seed=42,
    )
    assert len(source.trajectories) == 3

    log = _DispatchLog()
    current_phase = [CreditPhase.WARMUP]
    issuer = _make_recording_issuer(log, current_phase)

    warmup = _build_phase_strategy(
        phase=CreditPhase.WARMUP, source=source, issuer=issuer
    )
    await warmup.setup_phase()
    await warmup.execute_phase()
    await _flush_scheduled(warmup)
    warmup.report_warmup_failures()
    warmup_dispatched = log.by_phase(CreditPhase.WARMUP)
    expected_warmup = {
        (state.conversation_id, state.warmup_turn_index)
        for trajectory in source.trajectories
        if trajectory.snapshot is not None
        for state in trajectory.snapshot.states
        if state.warmup_turn_index is not None
    }
    assert set(warmup_dispatched) == expected_warmup, (
        f"WARMUP must dispatch each warmable stream once at its warmup_turn_index; "
        f"got {warmup_dispatched}, expected {sorted(expected_warmup)}"
    )
    assert len(warmup_dispatched) == source.warmup_credit_count

    current_phase[0] = CreditPhase.PROFILING
    profiling = _build_phase_strategy(
        phase=CreditPhase.PROFILING, source=source, issuer=issuer
    )
    await profiling.setup_phase()
    await profiling.execute_phase()
    await _flush_scheduled(profiling)

    aggregate = _make_aggregate_with_carriers(
        scenario_name=None,
        validator_valid=None,
        validator_reasons=[],
        total_responses=0,
        context_overflow_count=0,
    )
    aggregate.metadata["confidence_level"] = 0.95
    aggregate.metadata["cooldown_seconds"] = 5

    data = await _export_aggregate(aggregate, tmp_path)
    md = data["metadata"]

    assert "submission_valid" not in md, (
        "Bare agentic_replay timing mode (no --scenario) must omit submission_valid"
    )
    assert "submission_invalid_reasons" not in md
    assert "scenario" not in md
    assert md["confidence_level"] == 0.95
    assert md["cooldown_seconds"] == 5
