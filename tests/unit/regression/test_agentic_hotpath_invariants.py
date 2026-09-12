# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic / WEKA hot-path invariants; each test fails if the corresponding production behavior is reverted."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.enums import CacheBustTarget, ConversationBranchMode, CreditPhase
from aiperf.common.models import Text, Turn
from aiperf.config.flags._converter_dataset import build_dataset
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.credit.callback_handler import CreditCallbackHandler
from aiperf.credit.issuer import CreditIssuer
from aiperf.credit.messages import CreditReturn
from aiperf.credit.sticky_router import StickyCreditRouter, _StickyEntry
from aiperf.credit.structs import Credit
from aiperf.metrics.theoretical_prefix_cache import TheoreticalPrefixCacheAccumulator
from aiperf.plugin import plugins
from aiperf.plugin.enums import AccumulatorType, PluginType
from aiperf.records.records_manager_processing import load_accumulators
from aiperf.timing.session_tree import SessionTreeRegistry
from aiperf.timing.strategies.cache_bust import build_cache_bust_marker
from aiperf.workers.worker import _inject_marker_into_first_user_text


def test_theoretical_prefix_cache_registered_as_accumulator_plugin() -> None:
    names = [e.name for e in plugins.iter_entries(PluginType.ACCUMULATOR)]
    assert "theoretical_prefix_cache" in names, (
        "theoretical_prefix_cache missing from plugins.yaml "
        "accumulator section — metric never ships"
    )
    assert AccumulatorType.THEORETICAL_PREFIX_CACHE == "theoretical_prefix_cache"
    cls = plugins.get_class(
        PluginType.ACCUMULATOR, AccumulatorType.THEORETICAL_PREFIX_CACHE
    )
    assert cls is TheoreticalPrefixCacheAccumulator
    entry = plugins.get_entry(PluginType.ACCUMULATOR, "theoretical_prefix_cache")
    assert entry.metadata is not None
    assert entry.metadata.get("record_types") == ["metric_records"]


def test_load_accumulators_constructs_theoretical_prefix_cache(
    benchmark_run,
) -> None:
    host = MagicMock()
    host.service_id = "records-manager"
    host.run = benchmark_run
    host.pub_client = MagicMock()
    host.attach_child_lifecycle = MagicMock()
    host.debug = MagicMock()
    host.error = MagicMock()

    accumulators = load_accumulators(host)

    assert AccumulatorType.THEORETICAL_PREFIX_CACHE in accumulators, (
        "load_accumulators did not construct TheoreticalPrefixCacheAccumulator"
    )
    assert isinstance(
        accumulators[AccumulatorType.THEORETICAL_PREFIX_CACHE],
        TheoreticalPrefixCacheAccumulator,
    )
    host.error.assert_not_called()


def _make_warmup_credit() -> Credit:
    return Credit(
        id=1,
        phase=CreditPhase.WARMUP,
        conversation_id="trace-warmup",
        x_correlation_id="corr-warmup",
        turn_index=0,
        num_turns=5,
        issued_at_ns=time.time_ns(),
        agent_depth=0,
    )


@pytest.mark.asyncio
async def test_non_overflow_warmup_gated_intercept_records_warmup_failure() -> None:
    """Reverts to RED if early-return skips ``_handle_warmup_failure``."""
    concurrency = MagicMock()
    concurrency.release = AsyncMock()
    concurrency.release_prefill = MagicMock()
    concurrency.release_session_slot = MagicMock()
    concurrency.release_prefill_slot = MagicMock()
    registry = MagicMock()
    registry.has_tree.return_value = True
    orch = MagicMock()
    orch.intercept = AsyncMock(return_value=True)
    orch.has_pending_branch_work = MagicMock(return_value=False)
    strategy = MagicMock()
    strategy.handle_credit_return = AsyncMock()
    strategy.record_warmup_failure = MagicMock()
    strategy.wants_returns_after_sending_complete = False
    progress = MagicMock()
    progress.increment_returned = MagicMock(return_value=False)
    progress.increment_prefill_released = MagicMock()
    progress.all_credits_returned_event = asyncio.Event()
    progress.in_flight = 1
    lifecycle = MagicMock()
    lifecycle.is_complete = False
    stop = MagicMock()
    stop.can_send_any_turn = MagicMock(return_value=True)
    stop.can_send_child_turn = MagicMock(return_value=True)

    handler = CreditCallbackHandler(
        concurrency,
        branch_orchestrator=orch,
        session_tree_registry=registry,
    )
    handler.register_phase(
        phase=CreditPhase.WARMUP,
        progress=progress,
        lifecycle=lifecycle,
        stop_checker=stop,
        strategy=strategy,
    )
    credit = _make_warmup_credit()
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(
            credit=credit,
            cancelled=False,
            first_token_sent=True,
            error="Internal server error: pool exhausted",
        ),
    )

    orch.intercept.assert_awaited_once_with(credit)
    strategy.record_warmup_failure.assert_called_once_with(credit.conversation_id)
    strategy.handle_credit_return.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_overflow_warmup_gated_intercept_fires_live_abort() -> None:
    """Live abort must also survive the gated early-return."""
    concurrency = MagicMock()
    concurrency.release = AsyncMock()
    concurrency.release_prefill = MagicMock()
    concurrency.release_session_slot = MagicMock()
    concurrency.release_prefill_slot = MagicMock()
    registry = MagicMock()
    registry.has_tree.return_value = True
    orch = MagicMock()
    orch.intercept = AsyncMock(return_value=True)
    orch.has_pending_branch_work = MagicMock(return_value=False)
    strategy = MagicMock()
    strategy.handle_credit_return = AsyncMock()
    strategy.record_warmup_failure = MagicMock()
    strategy.wants_returns_after_sending_complete = False
    progress = MagicMock()
    progress.increment_returned = MagicMock(return_value=False)
    progress.increment_prefill_released = MagicMock()
    progress.all_credits_returned_event = asyncio.Event()
    progress.in_flight = 1
    lifecycle = MagicMock()
    lifecycle.is_complete = False
    stop = MagicMock()
    stop.can_send_any_turn = MagicMock(return_value=True)
    stop.can_send_child_turn = MagicMock(return_value=True)
    on_abort = AsyncMock()

    handler = CreditCallbackHandler(
        concurrency,
        branch_orchestrator=orch,
        session_tree_registry=registry,
        on_warmup_abort=on_abort,
    )
    handler.register_phase(
        phase=CreditPhase.WARMUP,
        progress=progress,
        lifecycle=lifecycle,
        stop_checker=stop,
        strategy=strategy,
    )
    credit = _make_warmup_credit()
    await handler.on_credit_return(
        "worker-1",
        CreditReturn(
            credit=credit,
            cancelled=False,
            first_token_sent=True,
            error="Internal server error: pool exhausted",
        ),
    )

    strategy.record_warmup_failure.assert_called_once_with(credit.conversation_id)
    on_abort.assert_awaited_once()


@pytest.mark.asyncio
async def test_spawn_colocates_on_live_parent_sticky_not_least_loaded(
    benchmark_run,
) -> None:
    """SPAWN sticky-hits parent while entry lives (graph-mode co-locate)."""
    router = StickyCreditRouter(run=benchmark_run, service_id="spawn-sticky-router")
    router._router_client.send_to = AsyncMock()
    router._register_worker("worker-A")
    router._register_worker("worker-B")
    # Bias B as least-loaded so a non-sticky pick would choose B.
    router._workers["worker-A"].active_sessions = 5
    router._workers["worker-A"].virtual_sent_credits = 100
    router._workers["worker-B"].active_sessions = 0
    router._workers["worker-B"].virtual_sent_credits = 1
    by_load: dict[int, set[str]] = defaultdict(set)
    for wid, load in router._workers.items():
        by_load[load.active_sessions].add(wid)
    router._workers_by_load = by_load
    router._min_load = min(by_load)
    router._sticky_sessions["parent-corr"] = _StickyEntry(worker_id="worker-A")

    spawn = Credit(
        id=2,
        phase=CreditPhase.PROFILING,
        conversation_id="child",
        x_correlation_id="child-corr",
        turn_index=0,
        num_turns=2,
        issued_at_ns=0,
        parent_correlation_id="parent-corr",
        branch_mode=ConversationBranchMode.SPAWN,
    )
    await router.send_credit(spawn)

    target = router._router_client.send_to.call_args[0][0]
    assert target == "worker-A", (
        "SPAWN with live parent sticky must co-locate on "
        f"worker-A, got {target!r} (least-loaded would be worker-B)"
    )


def test_empty_texts_cache_bust_seed_is_idempotent() -> None:
    marker = build_cache_bust_marker(
        "bench", 0, 0, "trace", target=CacheBustTarget.FIRST_TURN_PREFIX
    )
    assert marker is not None
    assert marker.endswith("\n\n")

    turn = Turn(texts=[])
    _inject_marker_into_first_user_text(turn, marker, is_prefix=True)
    _inject_marker_into_first_user_text(turn, marker, is_prefix=True)

    assert turn.texts[0].contents == [marker], (
        "empty-texts seed must store the full marker including "
        f"\\n\\n; got {turn.texts[0].contents[0]!r}"
    )
    assert turn.texts[0].contents[0].count("[rid:") == 1


def test_empty_contents_cache_bust_seed_is_idempotent() -> None:
    marker = build_cache_bust_marker(
        "bench", 0, 0, "trace", target=CacheBustTarget.FIRST_TURN_PREFIX
    )
    assert marker is not None

    turn = Turn(texts=[Text(contents=[])])
    _inject_marker_into_first_user_text(turn, marker, is_prefix=True)
    _inject_marker_into_first_user_text(turn, marker, is_prefix=True)

    assert turn.texts[0].contents == [marker], (
        "empty-contents seed must store the full marker; "
        f"got {turn.texts[0].contents[0]!r}"
    )
    assert turn.texts[0].contents[0].count("[rid:") == 1


@pytest.mark.asyncio
async def test_phase_keys_lane_credit_uses_runtime_index_not_enum() -> None:
    """Multi-phase runs key concurrency/trees by phase_index, not CreditPhase."""
    phase_index = 7
    concurrency = MagicMock()
    concurrency.acquire_session_slot = AsyncMock(return_value=True)
    concurrency.release_session_slot = MagicMock()
    registry = SessionTreeRegistry(concurrency)

    stop = MagicMock()
    stop.can_start_new_session = MagicMock(return_value=True)
    progress = MagicMock()
    progress.all_credits_sent_event = asyncio.Event()
    lifecycle = MagicMock()
    lifecycle.started_at_ns = time.time_ns()
    lifecycle.started_at_perf_ns = time.perf_counter_ns()

    issuer = CreditIssuer(
        phase=CreditPhase.PROFILING,
        phase_index=phase_index,
        stop_checker=stop,
        progress=progress,
        concurrency_manager=concurrency,
        credit_router=MagicMock(),
        cancellation_policy=MagicMock(),
        lifecycle=lifecycle,
        session_tree_registry=registry,
        session_tree_registry_enabled=True,
    )

    acquired = await issuer.acquire_lane_credit("root-phase-7", root_pending=False)
    assert acquired is True

    acquired_key = concurrency.acquire_session_slot.call_args.args[0]
    assert acquired_key == phase_index, (
        "acquire_session_slot must use phase_index "
        f"{phase_index}, got {acquired_key!r} (enum leak would be "
        f"{CreditPhase.PROFILING!r})"
    )
    assert registry._trees["root-phase-7"].phase == phase_index, (
        "open_tree must store the runtime phase key"
    )
    released = registry.release_all(phase_index)
    assert released == 1, (
        "release_all(phase_index) must find the tree "
        f"(got {released}); enum-keyed trees would miss index {phase_index}"
    )
    assert concurrency.release_session_slot.call_args.args[0] == phase_index


def test_build_dataset_does_not_call_apply_file_block_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_apply_block_size`` owns file hash-id block size; do not double-apply."""
    import aiperf.config.flags._converter_dataset as cd

    calls: list[object] = []
    real = cd._apply_file_block_size

    def _spy(d: dict, cli: CLIConfig) -> None:
        calls.append(d.get("type"))
        return real(d, cli)

    monkeypatch.setattr(cd, "_apply_file_block_size", _spy)

    trace = tmp_path / "trace.jsonl"
    trace.touch()
    build_dataset(
        CLIConfig(
            model_names=["m"],
            input_file=str(trace),
            custom_dataset_type="mooncake_trace",
            prompt_input_tokens_block_size=256,
            prompt_output_tokens_mean=64,
        )
    )
    assert calls == [], (
        f"build_dataset must not call _apply_file_block_size (got calls={calls!r})"
    )


def test_loader_mp_context_rejects_tokenizer_preload_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forkserver helper is process-global; switching preload must raise."""
    import aiperf.dataset._mp_context as mpc

    mpc._loader_ctx = None
    mpc._loader_ctx_key = None
    fake_ctx = MagicMock(name="loader-ctx")
    monkeypatch.setattr(mpc.multiprocessing, "get_context", lambda method: fake_ctx)
    monkeypatch.setattr(mpc, "_eagerly_start_forkserver", lambda: None)
    monkeypatch.setattr(mpc, "IS_LINUX", True)

    mpc.get_loader_mp_context(preload_tokenizer="tok-a")
    with pytest.raises(ValueError, match=r"tok-a|tok-b|preload"):
        mpc.get_loader_mp_context(preload_tokenizer="tok-b")
    mpc._loader_ctx = None
    mpc._loader_ctx_key = None
