# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sweep-robustness features: jitter, goodput collapse, disconnects."""

import asyncio
import contextlib
import random
import time

import pytest
from aiperf_mock_server import scheduler as scheduler_module
from aiperf_mock_server.config import MockServerConfig
from aiperf_mock_server.metrics import DYNAMO_FRONTEND_DISCONNECTED_CLIENTS
from aiperf_mock_server.scheduler import BatchScheduler, _DecodeWaiter, _PrefillWaiter
from aiperf_mock_server.utils import (
    LatencySimulator,
    _lognormal_jitter,
    _positive_jitter_extra_seconds,
)

# ============================================================================
# Jitter helpers
# ============================================================================


def test_lognormal_jitter_zero_cv_returns_one():
    assert _lognormal_jitter(0.0) == 1.0
    assert _lognormal_jitter(-0.5) == 1.0


def test_lognormal_jitter_mean_approx_one():
    """E[lognormal_jitter(cv)] ~= 1 for any cv >= 0."""
    random.seed(123)
    samples = [_lognormal_jitter(0.3) for _ in range(20_000)]
    mean = sum(samples) / len(samples)
    assert 0.97 < mean < 1.03, f"mean={mean:.4f} drifted too far from 1.0"


def test_lognormal_jitter_cv_matches_target():
    """Coefficient of variation of samples ~= cv argument."""
    random.seed(456)
    cv_target = 0.25
    samples = [_lognormal_jitter(cv_target) for _ in range(20_000)]
    mean = sum(samples) / len(samples)
    var = sum((x - mean) ** 2 for x in samples) / len(samples)
    cv = (var**0.5) / mean
    assert abs(cv - cv_target) < 0.03, f"observed cv={cv:.4f} target={cv_target}"


def test_positive_jitter_extra_zero_when_disabled():
    assert _positive_jitter_extra_seconds(50.0, 0.0) == 0.0
    assert _positive_jitter_extra_seconds(0.0, 0.5) == 0.0


def test_positive_jitter_extra_nonneg_and_nonzero_under_noise():
    """With cv>0 the extra time is always >=0 and is sometimes positive."""
    random.seed(789)
    samples = [_positive_jitter_extra_seconds(10.0, 0.4) for _ in range(2000)]
    assert all(s >= 0 for s in samples)
    assert any(s > 0 for s in samples), "lognormal should sometimes exceed 1.0"


# ============================================================================
# LatencySimulator jitter integration
# ============================================================================


@pytest.mark.asyncio
async def test_open_loop_itl_jitter_produces_variance():
    """With itl_jitter_cv>0 successive ITLs vary; with cv=0 they're constant."""
    cfg_jit = MockServerConfig(ttft=1.0, itl=10.0, itl_jitter_cv=0.5)
    cfg_det = MockServerConfig(ttft=1.0, itl=10.0, itl_jitter_cv=0.0)

    async def measure_itls(cfg) -> list[float]:
        sim = LatencySimulator("/x", "m", time.perf_counter(), cfg, isl=1, osl=8)
        deltas: list[float] = []
        prev = time.perf_counter()
        for _ in range(8):
            await sim.wait_for_next_token()
            now = time.perf_counter()
            deltas.append(now - prev)
            prev = now
        return deltas[1:]  # drop TTFT delta

    random.seed(1)
    jit = await measure_itls(cfg_jit)
    random.seed(1)
    det = await measure_itls(cfg_det)

    # Deterministic case: ITLs hug the configured 10ms (allow scheduler slack).
    det_spread = max(det) - min(det)
    assert det_spread < 0.005, (
        f"det spread {det_spread * 1000:.2f}ms unexpectedly noisy"
    )
    # Jittered case: at cv=0.5 we expect at least 2x the spread.
    jit_spread = max(jit) - min(jit)
    assert jit_spread > det_spread * 2, (
        f"jit spread {jit_spread * 1000:.2f}ms not larger than det {det_spread * 1000:.2f}ms"
    )


@pytest.mark.asyncio
async def test_open_loop_ttft_jitter_disabled_is_deterministic():
    """ttft_jitter_cv=0.0 produces tight, reproducible TTFT."""
    cfg = MockServerConfig(ttft=20.0, itl=0.0)
    samples = []
    for _ in range(5):
        sim = LatencySimulator("/x", "m", time.perf_counter(), cfg, isl=1, osl=1)
        await sim.wait_for_next_token()
        samples.append(sim.measured_ttft)
    spread = max(samples) - min(samples)
    assert spread < 0.005, f"deterministic ttft drifted {spread * 1000:.2f}ms"


# ============================================================================
# Goodput collapse
# ============================================================================


def _seed_decode_queue(sched: BatchScheduler, n: int) -> None:
    for i in range(n):
        sched._decode_queue.append(_DecodeWaiter(request_id=f"r{i}"))


def test_goodput_collapse_disabled_returns_full_batch():
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_max_batch_size=10,
        scheduler_goodput_collapse_enabled=False,
    )
    sched = BatchScheduler(cfg)
    _seed_decode_queue(sched, 100)  # 10x oversubscribed
    assert sched._effective_decode_budget() == 10


def test_goodput_collapse_below_threshold_returns_full_batch():
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_max_batch_size=10,
        scheduler_goodput_collapse_enabled=True,
        scheduler_goodput_collapse_threshold=1.5,
    )
    sched = BatchScheduler(cfg)
    _seed_decode_queue(sched, 12)  # ratio 1.2 < threshold 1.5
    assert sched._effective_decode_budget() == 10


def test_goodput_collapse_shrinks_admit_past_threshold():
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_max_batch_size=10,
        scheduler_goodput_collapse_enabled=True,
        scheduler_goodput_collapse_threshold=1.0,
        scheduler_goodput_collapse_slope=0.5,
        scheduler_goodput_collapse_floor=0.3,
    )
    sched = BatchScheduler(cfg)
    _seed_decode_queue(sched, 30)  # ratio 3.0, overload 2.0, shrink min(0.7, 1.0) = 0.7
    # shrink capped at 1 - floor = 0.7  →  effective batch = 10 * 0.3 = 3
    assert sched._effective_decode_budget() == 3


def test_goodput_collapse_at_floor_under_extreme_overload():
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_max_batch_size=20,
        scheduler_goodput_collapse_enabled=True,
        scheduler_goodput_collapse_threshold=1.0,
        scheduler_goodput_collapse_slope=10.0,
        scheduler_goodput_collapse_floor=0.25,
    )
    sched = BatchScheduler(cfg)
    _seed_decode_queue(sched, 1000)
    # slope is huge so we hit floor immediately: 20 * 0.25 = 5
    assert sched._effective_decode_budget() == 5


def test_goodput_collapse_never_drops_below_one():
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_max_batch_size=2,
        scheduler_goodput_collapse_enabled=True,
        scheduler_goodput_collapse_threshold=0.0,
        scheduler_goodput_collapse_slope=10.0,
        scheduler_goodput_collapse_floor=0.0,
    )
    sched = BatchScheduler(cfg)
    _seed_decode_queue(sched, 100)
    assert sched._effective_decode_budget() >= 1


# ============================================================================
# Scheduler cancel()
# ============================================================================


def test_scheduler_cancel_removes_pending_decode_waiters():
    cfg = MockServerConfig(scheduler_enabled=True)
    sched = BatchScheduler(cfg)
    sched._decode_queue.extend(
        [
            _DecodeWaiter(request_id="keep-1"),
            _DecodeWaiter(request_id="drop-me"),
            _DecodeWaiter(request_id="keep-2"),
            _DecodeWaiter(request_id="drop-me"),
        ]
    )
    sched.cancel("drop-me")
    remaining = [w.request_id for w in sched._decode_queue]
    assert remaining == ["keep-1", "keep-2"]


def test_scheduler_cancel_removes_pending_prefill_waiters():
    cfg = MockServerConfig(scheduler_enabled=True)
    sched = BatchScheduler(cfg)
    sched._prefill_queue.extend(
        [
            _PrefillWaiter(request_id="a"),
            _PrefillWaiter(request_id="b"),
            _PrefillWaiter(request_id="a"),
        ]
    )
    sched.cancel("a")
    remaining = [w.request_id for w in sched._prefill_queue]
    assert remaining == ["b"]


def test_scheduler_cancel_wakes_orphaned_event():
    """Any awaiter still suspended on the waiter event gets unblocked."""
    cfg = MockServerConfig(scheduler_enabled=True)
    sched = BatchScheduler(cfg)
    w = _DecodeWaiter(request_id="x")
    sched._decode_queue.append(w)
    assert not w.event.is_set()
    sched.cancel("x")
    assert w.event.is_set()


@pytest.mark.asyncio
async def test_scheduler_cancel_unblocks_pending_next_decode_step():
    """A coroutine awaiting next_decode_step returns promptly when cancelled."""
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_step_ms=1000.0,  # ticks once per second so the test must rely on cancel
        scheduler_max_batch_size=1,
    )
    sched = BatchScheduler(cfg)
    await sched.start()
    try:
        # Saturate the single batch slot so subsequent waiters block.
        first = asyncio.create_task(sched.next_decode_step("first"))
        second = asyncio.create_task(sched.next_decode_step("second"))
        # Give the loop a tick to enqueue both.
        await asyncio.sleep(0.01)
        sched.cancel("second")
        # second should now be unblocked (admitted_step stays -1 since it was cancelled).
        await asyncio.wait_for(second, timeout=0.5)
        # first is still queued waiting for the first tick — cancel it too to clean up.
        sched.cancel("first")
        await asyncio.wait_for(first, timeout=0.5)
    finally:
        await sched.stop()


# ============================================================================
# LatencySimulator + streaming disconnect handling
# ============================================================================


def test_latency_simulator_cancel_is_idempotent_after_finish():
    cfg = MockServerConfig(scheduler_enabled=False)
    sim = LatencySimulator("/x", "m", time.perf_counter(), cfg)
    sim.mark_finished()
    sim.cancel()  # should be a no-op
    assert sim._cancelled is False


def test_latency_simulator_cancel_records_disconnect_metric():
    cfg = MockServerConfig(scheduler_enabled=False)
    sim = LatencySimulator("/x", "model-x", time.perf_counter(), cfg)
    before = DYNAMO_FRONTEND_DISCONNECTED_CLIENTS.labels(model="model-x")._value.get()
    sim.cancel()
    after = DYNAMO_FRONTEND_DISCONNECTED_CLIENTS.labels(model="model-x")._value.get()
    assert after == before + 1
    # Calling again is idempotent.
    sim.cancel()
    after2 = DYNAMO_FRONTEND_DISCONNECTED_CLIENTS.labels(model="model-x")._value.get()
    assert after2 == after


@pytest.mark.asyncio
async def test_scheduler_cancel_called_on_simulator_cancel_when_enabled():
    """LatencySimulator.cancel() routes to the active scheduler."""
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_step_ms=1000.0,
        scheduler_max_batch_size=1,
    )
    sched = await scheduler_module.init_scheduler(cfg)
    assert sched is not None
    try:
        sim = LatencySimulator("/x", "m", time.perf_counter(), cfg, isl=1, osl=4)
        # Start the request — it'll block on prefill admit (chunk pool=8, step=1s).
        task = asyncio.create_task(sim.wait_for_next_token())
        await asyncio.sleep(0.01)
        # Now seed a "stuck" decode waiter to verify cancel removes it.
        sched._decode_queue.append(_DecodeWaiter(request_id=sim.request_key))
        sim.cancel()
        # The decode waiter for our key should be gone.
        assert sim.request_key not in {w.request_id for w in sched._decode_queue}
        task.cancel()
        with contextlib.suppress(BaseException):
            await task
    finally:
        await scheduler_module.shutdown_scheduler()


@pytest.mark.asyncio
async def test_streaming_chat_disconnect_triggers_cancel():
    """Closing a chat-completion async generator mid-stream invokes sim.cancel()."""
    from aiperf_mock_server.tokens import TokenizedText
    from aiperf_mock_server.utils import RequestCtx, stream_chat_completion

    cfg = MockServerConfig(ttft=1.0, itl=50.0, scheduler_enabled=False)
    tokenized = TokenizedText(
        text="hi",
        tokens=["a", "b", "c", "d", "e"],
        prompt_token_count=1,
        finish_reason="stop",
    )
    sim = LatencySimulator(
        "/v1/chat/completions",
        "disco-model",
        time.perf_counter(),
        cfg,
        isl=1,
        osl=5,
    )
    ctx = RequestCtx(
        request_id="r-1",
        model="disco-model",
        tokenized=tokenized,
        usage=tokenized.create_usage(),
        latency_sim=sim,
    )

    before = DYNAMO_FRONTEND_DISCONNECTED_CLIENTS.labels(
        model="disco-model"
    )._value.get()

    gen = stream_chat_completion(ctx, "/v1/chat/completions", include_usage=False)
    # Pull the first chunk so we're mid-stream.
    chunk = await gen.__anext__()
    assert chunk.startswith(b"data: ")
    # Simulate the client disconnecting.
    await gen.aclose()

    after = DYNAMO_FRONTEND_DISCONNECTED_CLIENTS.labels(
        model="disco-model"
    )._value.get()
    assert sim._cancelled is True
    assert sim._finished is False
    assert after == before + 1


@pytest.mark.asyncio
async def test_streaming_chat_normal_completion_does_not_record_disconnect():
    """Full stream consumption marks finished and does NOT bump the disconnect counter."""
    from aiperf_mock_server.tokens import TokenizedText
    from aiperf_mock_server.utils import RequestCtx, stream_chat_completion

    cfg = MockServerConfig(ttft=0.0, itl=0.0, scheduler_enabled=False)
    tokenized = TokenizedText(
        text="hi",
        tokens=["a", "b"],
        prompt_token_count=1,
        finish_reason="stop",
    )
    sim = LatencySimulator(
        "/v1/chat/completions",
        "ok-model",
        time.perf_counter(),
        cfg,
        isl=1,
        osl=2,
    )
    ctx = RequestCtx(
        request_id="r-2",
        model="ok-model",
        tokenized=tokenized,
        usage=tokenized.create_usage(),
        latency_sim=sim,
    )

    before = DYNAMO_FRONTEND_DISCONNECTED_CLIENTS.labels(model="ok-model")._value.get()

    chunks = []
    async for chunk in stream_chat_completion(
        ctx, "/v1/chat/completions", include_usage=False
    ):
        chunks.append(chunk)

    after = DYNAMO_FRONTEND_DISCONNECTED_CLIENTS.labels(model="ok-model")._value.get()
    assert sim._finished is True
    assert sim._cancelled is False
    assert after == before
    assert chunks[-1] == b"data: [DONE]\n\n"
