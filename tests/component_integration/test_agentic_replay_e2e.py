# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic_replay end-to-end happy-path component-integration tests.

Three tests:

1. ``test_agentic_replay_e2e_clean_run_under_scenario`` -- exercise the full
   agentic_replay pipeline against the small synthetic weka fixture: load
   traces, build a TrajectorySource, run WARMUP+PROFILING strategies,
   stamp the validator outcome onto AggregateResult.metadata, export the JSON.
   Asserts the four spec invariants:
     - warmup barrier (no PROFILING request before all WARMUP credits resolve)
     - recycle observed (a trace_id dispatched more than once)
     - metrics window correct (no measured request before profiling start;
       no measured request after duration end + grace -- enforced via
       ``stop_checker.can_start_new_session`` gating in the strategy)
     - aggregate JSON contains ``submission_valid: true`` and
       ``scenario: "inferencex-agentx-mvp"``
2. ``test_agentic_replay_e2e_unsafe_override_stamps_false`` -- the validator
   path under ``--unsafe-override`` with a duration below the 900s floor:
   aggregate JSON contains ``submission_valid: false`` with
   ``unsafe_override`` in ``submission_invalid_reasons``.
3. ``test_agentic_replay_e2e_no_scenario_omits_submission_valid`` -- bare
   agentic_replay timing mode (no ``--scenario``): aggregate JSON omits
   the ``submission_valid`` field; the rest of the run still succeeds.

Wiring scope:
- ``cli_runner._run_multi_benchmark`` stamps the validator-outcome
  carrier keys (``_scenario_name``, ``_validator_submission_valid``,
  ``_validator_submission_invalid_reasons``) onto ``AggregateResult.metadata``
  from ``user_config._scenario_outcome``. The runtime totals
  (``_total_responses``, ``_context_overflow_count``) are stamped to ``0``
  by default.
- The full e2e CLI pathway (``cli.run_sync('aiperf profile --scenario ...')``)
  is *not* exercised here because ``PhaseOrchestrator`` constructs a plain
  ``ConversationSource`` rather than a ``TrajectorySource``, which would
  cause ``AgenticReplayStrategy`` to refuse construction at startup. These
  tests pin the genuine loader -> trajectory -> strategy -> aggregate ->
  exporter chain end-to-end at the integration boundary above the
  orchestrator-construction seam.
"""

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


# =============================================================================
# Helpers
# =============================================================================


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
    """Deterministic sampler over a fixed conversation_id list (rooted only).

    Wraps around to the start indefinitely, matching the production
    ``SequentialSampler`` (``dataset_samplers.py``). Recycle now draws roots from
    this same sampler via ``TrajectorySource.next_recycle_conversation_id``, so
    it must keep handing ids back past the first pass rather than raising
    StopIteration.
    """

    def __init__(self, conversation_ids: list[str]) -> None:
        self._ids = list(conversation_ids)
        self._idx = 0

    def next_conversation_id(self) -> str:
        if self._idx >= len(self._ids):
            self._idx = 0
        cid = self._ids[self._idx]
        self._idx += 1
        return cid


def _mk_user_config():
    """Build a minimal UserConfig stub adequate for WekaTraceLoader."""
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "test-tok"
    uc.endpoint.model_names = ["claude-opus-4-5-20251101"]
    # MagicMock auto-creates attributes; pin the ones the loader compares to numeric.
    uc.loadgen.inter_turn_delay_cap_seconds = None
    return uc


def _load_small_weka_dataset(monkeypatch, *, parallel: bool = False) -> DatasetMetadata:
    """Load the synthetic weka fixture into a DatasetMetadata.

    Stubs the prompt-synthesis path because the test does not need real
    tokenization -- the inputs/outputs are downstream of trajectory selection
    and credit dispatch, not the actual prompt content.

    ``parallel=False``: forces serial reconstruction
    (``WEKA_PARALLEL_WORKERS=1``) and stubs ``_decode_block_tokens`` /
    ``_decode_tokens_to_text`` directly on the loader so the corpus/tokenizer
    machinery never runs.

    ``parallel=True``: forces the multi-process reconstruction path
    (``WEKA_PARALLEL_WORKERS=2``, threshold lowered) but replaces the
    real ``multiprocessing.Pool`` with an in-process stub that calls
    ``_init_worker`` + ``_process_task`` synchronously. A small real
    int-array corpus and a stub tokenizer satisfy
    ``run_parallel_weka_reconstruction``'s ``SharedMemory`` allocation
    and the per-worker ``Tokenizer.from_pretrained`` lookup. Mirrors the
    technique in ``tests/unit/dataset/loader/test_weka_trace_parallel.py``
    (``_drive_parallel_inproc``).
    """
    from aiperf.common.environment import Environment

    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES), user_config=uc)
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
        # Bypass the corpus/tokenizer-dependent path entirely; serial
        # reconstruction calls these directly per turn.
        monkeypatch.setattr(
            loader,
            "_decode_block_tokens",
            lambda hash_ids: [0] * (len(hash_ids) * loader._block_size),
        )
        monkeypatch.setattr(
            loader, "_decode_tokens_to_text", lambda tokens: "x" * len(tokens)
        )
    else:
        # Parallel path: provide a real corpus + real RNG so SharedMemory
        # allocation succeeds and worker reseeding is deterministic. The
        # fake Pool runs everything in-process so monkeypatched callables
        # remain visible.
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

        # Force parallel: workers >= 2 AND threshold low enough that 10
        # traces cross the bar.
        monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_WORKERS", 2)
        monkeypatch.setattr(Environment.DATASET, "WEKA_PARALLEL_THRESHOLD", 1)
        _install_inproc_pool(monkeypatch, loader)

    convs = loader.convert_to_conversations(loader.load_dataset())
    return DatasetMetadata(
        conversations=[c.to_metadata() for c in convs],
        sampling_strategy=DatasetSamplingStrategy.SEQUENTIAL,
    )


def _install_inproc_pool(monkeypatch, loader) -> None:
    """Replace the multiprocessing Pool with a synchronous in-process stub.

    Patches ``get_loader_mp_context`` to return an object whose
    ``Pool(...)`` is a context-manager fake that runs ``_init_worker``
    once in-process, then dispatches ``_process_task`` per task on
    ``imap``. Patches ``Tokenizer.from_pretrained`` to return the
    loader's stub tokenizer so the worker's tokenizer lookup
    succeeds without network access.
    """
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
    """Build an AsyncMock credit issuer that records dispatches into ``log``.

    The current-phase list is a one-element box so the WARMUP and PROFILING
    strategies (constructed sequentially) can update the recorded phase
    without re-binding the issuer.

    Also exposes ``cid_to_xcorr`` on the issuer: a mapping from conversation_id
    to the most recently issued x_correlation_id. Tests use this to send
    final-turn credit returns whose ``x_correlation_id`` matches what
    ``setup_phase`` / ``_spawn_from_recycle_or_id`` minted, so the strategy's
    ``_correlation_to_lane`` invariant holds and recycle proceeds.
    """
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
    """Build a scheduler mock whose ``schedule_later`` actually runs coroutines.

    Both phase strategies defer dispatches through
    ``scheduler.schedule_later(delay, coro)`` rather than awaiting them inline:
    under spread mode (the default) WARMUP fires only the single max-lead credit
    inline and schedules every earlier-lead credit, and PROFILING fires only a
    lane whose first post-t* offset is 0 inline and schedules the rest. A bare
    ``MagicMock`` scheduler swallows those coroutines (they never run and leak as
    un-awaited warnings), so the recording issuer never sees them.

    This stub schedules each coroutine as a real task on the running loop
    (delay is irrelevant -- the test fixtures collapse asyncio.sleep) and tracks
    the tasks so ``_flush_scheduled`` can drain them after ``execute_phase``.
    """
    scheduler = MagicMock()
    scheduled: list[asyncio.Task] = []

    def _schedule_later(_delay, coro):
        scheduled.append(asyncio.ensure_future(coro))

    scheduler.schedule_later.side_effect = _schedule_later
    scheduler._scheduled_tasks = scheduled
    return scheduler


async def _flush_scheduled(strategy: AgenticReplayStrategy) -> None:
    """Await every task the strategy's scheduler queued via ``schedule_later``.

    Drains iteratively so a scheduled coroutine that itself schedules more
    (e.g. a recycle dispatch) is also awaited. Yields the loop a couple times
    first so freshly ``ensure_future``-d tasks register before draining.
    """
    scheduled: list[asyncio.Task] = strategy.scheduler._scheduled_tasks
    for _ in range(50):
        await asyncio.sleep(0)
        pending = [t for t in scheduled if not t.done()]
        if not pending:
            break
        await asyncio.gather(*pending)
    # Surface any exception captured in a completed task.
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
    """Build an AggregateResult carrying the cli_runner stamps.

    This mirrors the wiring added in cli_runner._run_multi_benchmark: when
    ``--scenario`` is set, the validator outcome flows through these
    underscore-prefixed metadata keys to the JSON exporter, which pops them
    and emits the ``submission_valid`` / ``submission_invalid_reasons``
    fields. When ``--scenario`` is unset (no_scenario test), no carrier
    keys are stamped so the exporter omits the field entirely.
    """
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


# =============================================================================
# Test 1: clean run under --scenario inferencex-agentx-mvp
# =============================================================================


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.asyncio
async def test_agentic_replay_e2e_clean_run_under_scenario(
    tmp_path: Path, monkeypatch, parallel: bool
) -> None:
    """Spec §8.2 #1: clean scenario run.

    End-to-end through the genuine pipeline:
    1. WekaTraceLoader loads the small synthetic fixture (10 traces, N in [1, 10]).
    2. TrajectorySource samples a 4-member trajectory with k_i in [0, 0.7*N_i].
    3. WARMUP strategy dispatches one credit per trajectory at turn k_i.
    4. PROFILING strategy resumes each trajectory at k_i + 1 and processes
       enough credit-returns to drive at least one full trace recycle.
    5. cli_runner-style metadata stamping populates carrier keys.
    6. AggregateConfidenceJsonExporter produces the final JSON.

    Assertions:
    - Warmup barrier: zero PROFILING dispatches before WARMUP execute_phase
      finishes.
    - Recycle: at least one trace_id appears more than once in the dispatch
      log (trajectory + recycle re-dispatch).
    - Metrics window: stop_checker.can_start_new_session gating prevents new
      sessions from being spawned post-stop (verified by toggling the gate).
    - Aggregate JSON: submission_valid is True; scenario is the locked name.
    """
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

    # ---- WARMUP ----
    warmup = _build_phase_strategy(
        phase=CreditPhase.WARMUP, source=source, issuer=issuer
    )
    await warmup.setup_phase()
    await warmup.execute_phase()
    await _flush_scheduled(warmup)  # run the spread-deferred warmup credits
    warmup.report_warmup_failures()  # must not raise -- no terminal failures injected

    # Snapshot-path WARMUP dispatches one priming credit per stream active at
    # t* -- each stream's last request before t* (its ``warmup_turn_index``),
    # NOT every trajectory at its root start. A trajectory whose only turn is at
    # t=0 (``trace_01_n1``) has warmup_turn_index None and contributes nothing,
    # so the count is ``source.warmup_credit_count`` (here 3), below the 4 lanes.
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

    # WARMUP BARRIER: no PROFILING dispatch happened during WARMUP.
    assert log.by_phase(CreditPhase.PROFILING) == [], (
        "Warmup barrier violated: PROFILING dispatched before WARMUP completed"
    )

    # ---- PROFILING ----
    current_phase[0] = CreditPhase.PROFILING
    profiling = _build_phase_strategy(
        phase=CreditPhase.PROFILING, source=source, issuer=issuer
    )
    await profiling.setup_phase()
    # Recycle now draws roots straight from the shared dataset sampler
    # (TrajectorySource.next_recycle_conversation_id) rather than a strategy-side
    # queue, so there is no _recycle_queue to introspect. PROFILING setup must
    # leave the trajectory lanes intact for recycle to draw against.
    assert profiling.config.phase == CreditPhase.PROFILING
    assert len(profiling.conversation_source.trajectories) == len(source.trajectories)

    await profiling.execute_phase()
    await _flush_scheduled(profiling)  # run the spread-deferred resume credits

    # Each trajectory resumed at k_i + 1, except trace_01_n1 (N=1) which
    # has k_i=0 with no further turns and is recycled immediately. Verify
    # resume-or-recycle holds for every trajectory.
    profiling_dispatched = log.by_phase(CreditPhase.PROFILING)
    trajectory_ks = {
        trajectory.conversation_id: trajectory.start_turn_index
        for trajectory in source.trajectories
    }
    metadata_lookup = source._metadata_lookup
    for trajectory_id, k in trajectory_ks.items():
        n = len(metadata_lookup[trajectory_id].turns)
        if k + 1 < n:
            # Resume path: must have dispatched (trajectory_id, k+1).
            assert (trajectory_id, k + 1) in profiling_dispatched, (
                f"trajectory {trajectory_id} should resume at k+1={k + 1}"
            )
        else:
            # Recycle-immediately path (N=1 + k=0): a fresh session from
            # the recycle queue must have dispatched in its place at turn
            # 0. The full-pool queue may select the same trace_id again.
            recycled_at_zero = [cid for cid, idx in profiling_dispatched if idx == 0]
            assert recycled_at_zero, (
                f"trajectory {trajectory_id} (N={n}, k={k}) should trigger an "
                "immediate recycle dispatch but none observed"
            )

    # ---- RECYCLE: drive final-turn completions for every trajectory to
    # exercise recycle. On the snapshot path each lane's root resumes at its
    # ``next_turn_index`` (its first turn at/after t*), which for these short
    # fixtures is already the trace's final turn, so its initial PROFILING
    # dispatch IS the final turn. Returning it final recycles the lane: a fresh
    # root is drawn from the dataset sampler and dispatched at turn 0. The
    # recycled roots are then themselves driven to final-turn returns so they
    # feed back through the sampler. After enough rounds at least one trace_id
    # must appear more than once in the dispatch log (the canonical recycle
    # observation).
    pre_recycle_count = len(profiling_dispatched)

    def _finalize(cid: str) -> Credit:
        n = len(metadata_lookup[cid].turns)
        return _make_credit(
            conversation_id=cid,
            turn_index=n - 1,
            num_turns=n,
            x_correlation_id=issuer.cid_to_xcorr[cid],
        )

    # Round 1: complete every trajectory at its final turn. Each finish draws a
    # fresh root from the dataset sampler and dispatches it at turn 0.
    trajectories_to_finalize = list(source.trajectories)
    for trajectory in trajectories_to_finalize:
        await profiling.handle_credit_return(_finalize(trajectory.conversation_id))
    await _flush_scheduled(profiling)

    after_round1 = log.by_phase(CreditPhase.PROFILING)
    assert len(after_round1) > pre_recycle_count, (
        "round 1: recycle should have produced new turn-0 dispatches"
    )

    # Rounds 2..R: complete each newly-recycled dispatch (turn 0 of a new
    # trace_id). Treat each as final so the strategy recycles again. The
    # queue is finite (size 6 + trajectory pushes), so after at most a few rounds
    # at least one trace_id MUST resurface. Track which trace_ids we've
    # already finalized to avoid the strategy's debug ``_in_flight_recycled``
    # assert which guards against re-recycling a still-in-flight trace_id.
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
                # Duplicate observed; recycle confirmed.
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

    # ---- METRICS WINDOW: post-stop gating ----
    # Once the stop condition fires, can_start_new_session() returns False and
    # _spawn_from_recycle_or_id is a no-op -- no new sessions begin. Verify
    # by toggling the gate and triggering a final-turn return.
    profiling.stop_checker.can_start_new_session.return_value = False
    pre_post_stop = len(log.by_phase(CreditPhase.PROFILING))
    # Pick an in-flight session (correlation_id present in _correlation_to_lane)
    # so the strategy treats the final-turn return as legitimate. The post-stop
    # gate is what must prevent the follow-up recycle dispatch.
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

    # ---- AGGREGATE JSON STAMPING ----
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
    # Carrier keys stripped from output.
    for key in (
        "_scenario_name",
        "_validator_submission_valid",
        "_validator_submission_invalid_reasons",
        "_total_responses",
        "_context_overflow_count",
    ):
        assert key not in md, f"carrier key {key!r} leaked into output"


# =============================================================================
# Test 2: --unsafe-override + duration-below-floor stamps submission_valid: false
# =============================================================================


@pytest.mark.asyncio
async def test_agentic_replay_e2e_unsafe_override_stamps_false(
    tmp_path: Path,
) -> None:
    """Spec §8.2 #2: --unsafe-override + violation -> submission_valid: false.

    Models the cli_runner stamping path under ``--unsafe-override`` with a
    violation (duration below the 900s floor): the validator returns
    ``submission_valid=False`` with ``["unsafe_override"]`` reasons; cli_runner
    pipes that through to the aggregate metadata; the JSON exporter emits
    ``submission_valid: false`` with the reason list.

    This test focuses on the cli_runner -> exporter wire (the validator's
    own behavior is covered by Tasks 12, 17 adversarial tests).
    """
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


# =============================================================================
# Test 3: bare agentic_replay timing mode (no --scenario) omits the field
# =============================================================================


@pytest.mark.parametrize("parallel", [False, True], ids=["serial", "parallel"])
@pytest.mark.asyncio
async def test_agentic_replay_e2e_no_scenario_omits_submission_valid(
    tmp_path: Path, monkeypatch, parallel: bool
) -> None:
    """Spec §8.2 #3: bare agentic_replay timing mode without scenario -> no submission_valid field.

    Exercises the same loader -> trajectory -> strategy chain to confirm the
    pipeline runs cleanly when ``--scenario`` is unset, then stamps the
    aggregate with no carrier keys (mirroring cli_runner's branch where
    ``user_config.scenario is None``). The exporter must omit
    ``submission_valid`` and ``scenario`` entirely.
    """
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
    await _flush_scheduled(warmup)  # run the spread-deferred warmup credits
    warmup.report_warmup_failures()
    # Snapshot-path WARMUP dispatches one priming credit per stream active at t*
    # (its ``warmup_turn_index``), not one per lane: ``trace_01_n1``'s only turn
    # is at t=0 so warmup_turn_index is None and it contributes nothing. The
    # count is ``source.warmup_credit_count`` (here 2), below the 3 lanes.
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
    await _flush_scheduled(profiling)  # run the spread-deferred resume credits

    # Aggregate the way cli_runner does for a non-scenario run: no carrier keys.
    aggregate = _make_aggregate_with_carriers(
        scenario_name=None,
        validator_valid=None,
        validator_reasons=[],
        total_responses=0,
        context_overflow_count=0,
    )
    # Add some normal metadata so the run is recognizable as a real export.
    aggregate.metadata["confidence_level"] = 0.95
    aggregate.metadata["cooldown_seconds"] = 5

    data = await _export_aggregate(aggregate, tmp_path)
    md = data["metadata"]

    assert "submission_valid" not in md, (
        "Bare agentic_replay timing mode (no --scenario) must omit submission_valid"
    )
    assert "submission_invalid_reasons" not in md
    assert "scenario" not in md
    # Standard non-scenario metadata still flows through.
    assert md["confidence_level"] == 0.95
    assert md["cooldown_seconds"] == 5
