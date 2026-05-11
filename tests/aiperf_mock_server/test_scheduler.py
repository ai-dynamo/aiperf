# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the mock server's batched step scheduler."""

import asyncio
import time

import pytest
from aiperf_mock_server import scheduler as scheduler_module
from aiperf_mock_server.config import MockServerConfig
from aiperf_mock_server.scheduler import BatchScheduler
from aiperf_mock_server.utils import LatencySimulator


@pytest.mark.asyncio
async def test_scheduler_single_request_one_step_admission():
    """Single decoder gets admitted on the next step tick."""
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_step_ms=5.0,
        scheduler_max_batch_size=4,
        scheduler_max_prefill_chunks_per_step=2,
        scheduler_prefill_chunk_tokens=512,
    )
    sched = BatchScheduler(cfg)
    await sched.start()
    try:
        token_idx = await sched.next_decode_step("req-1")
        assert token_idx >= 1, "first admitted decode step is >= 1"
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_scheduler_oversubscription_serializes_admission():
    """At concurrency = 2 * max_batch_size, half wait an extra step each tick."""
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_step_ms=2.0,
        scheduler_max_batch_size=4,
        scheduler_max_prefill_chunks_per_step=64,
        scheduler_prefill_chunk_tokens=512,
    )
    sched = BatchScheduler(cfg)
    await sched.start()
    try:
        results = await asyncio.gather(
            *[sched.next_decode_step(f"r{i}") for i in range(8)]
        )
        early = [s for s in results if s == min(results)]
        late = [s for s in results if s > min(results)]
        assert len(early) == 4
        assert len(late) == 4
        assert max(late) - min(early) == 1
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_scheduler_prefill_chunks_split_long_prompts():
    """A 1500-token prompt with chunk_tokens=512 needs 3 chunks."""
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_step_ms=1.0,
        scheduler_max_batch_size=64,
        scheduler_max_prefill_chunks_per_step=64,
        scheduler_prefill_chunk_tokens=512,
    )
    sched = BatchScheduler(cfg)
    await sched.start()
    try:
        steps_consumed = await sched.run_prefill("req-long", prompt_tokens=1500)
        assert steps_consumed == 3
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_scheduler_disabled_returns_passthrough():
    """When scheduler_enabled=False the scheduler refuses to start."""
    cfg = MockServerConfig(scheduler_enabled=False)
    sched = BatchScheduler(cfg)
    with pytest.raises(RuntimeError, match="not enabled"):
        await sched.start()


@pytest.mark.asyncio
async def test_latency_simulator_uses_scheduler_when_enabled():
    """When scheduler_enabled, wait_for_next_token blocks on scheduler ticks."""
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_step_ms=10.0,
        scheduler_max_batch_size=2,
        scheduler_max_prefill_chunks_per_step=64,
        scheduler_prefill_chunk_tokens=512,
        ttft=0.0,
        itl=0.0,
    )
    sched = await scheduler_module.init_scheduler(cfg)
    assert sched is not None
    try:
        sim = LatencySimulator(
            endpoint="/v1/chat/completions",
            model="m",
            start_time=time.perf_counter(),
            config=cfg,
            isl=10,
            osl=2,
        )
        t0 = time.perf_counter()
        await sim.wait_for_next_token()  # token 0 (TTFT, gated by prefill)
        await sim.wait_for_next_token()  # token 1 (decode step)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms >= 15.0, (
            f"too fast: {elapsed_ms:.1f}ms — scheduler not gating"
        )
    finally:
        await scheduler_module.shutdown_scheduler()


@pytest.mark.asyncio
async def test_latency_simulator_passthrough_when_scheduler_disabled():
    """When scheduler_enabled=False, original open-loop sleep is used."""
    cfg = MockServerConfig(scheduler_enabled=False, ttft=0.0, itl=0.0)
    sim = LatencySimulator(
        endpoint="/v1/chat/completions",
        model="m",
        start_time=time.perf_counter(),
        config=cfg,
    )
    t0 = time.perf_counter()
    await sim.wait_for_next_token()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert elapsed_ms < 5.0, (
        f"too slow: {elapsed_ms:.1f}ms — scheduler shouldn't be running"
    )
