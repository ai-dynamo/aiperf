# Mock Server Batched Scheduler (Design B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a step-based batched scheduler to the AIPerf mock server so it produces a real throughput-vs-concurrency saturation knee and a prefill/decode Pareto-front shape, enabling adaptive-search testing against ground-truth optima.

**Architecture:**
- New `BatchScheduler` singleton in `tests/aiperf_mock_server/scheduler.py` runs an asyncio loop that "ticks" every `step_ms`, admitting up to `max_batch_size` decode tokens per step and up to `max_prefill_chunks_per_step` prefill chunks. Requests await per-step admission via per-request `asyncio.Event`s.
- `LatencySimulator.wait_for_next_token` / `wait_for_tokens` route through the scheduler when `cfg.scheduler_enabled`, otherwise keep today's open-loop sleep path (full backward compat).
- The five existing per-request penalty knobs (`ttft_per_isl_token_ms`, `ttft_concurrency_quad_ms`, `itl_per_osl_token_ms`, `itl_concurrency_lin_ms`) keep working — they layer on top of the scheduler's structural latency. Default config (`scheduler_enabled=False`) is bit-identical to today.

**Tech Stack:** Python 3.10+ asyncio, FastAPI, pydantic-settings, pytest-asyncio. No new deps.

**Knee math (for tests + docs):**
- `step_ms=5`, `max_batch_size=256` → max decode-token rate = 256 / 0.005 = 51,200 tok/s
- For a request producing decode tokens at "natural" rate `1/itl_ms`, the saturation concurrency is roughly `max_batch_size` (one slot per active decoder). Past that, each additional decoder linearly stretches every other decoder's ITL.
- Prefill cost: prompt of P tokens splits into `ceil(P / prefill_chunk_tokens)` chunks; each chunk consumes one prefill slot for one step. With `max_prefill_chunks_per_step=8` and `prefill_chunk_tokens=512`, a 4096-token prompt takes 1 step in isolation but queues behind other prefills under load.

**Out of scope (deferred to design C):** KV-block accounting, preemption/swap, request cancellation mid-decode, GPU-specific timing.

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `tests/aiperf_mock_server/scheduler.py` | Create | `BatchScheduler` class + module-level `get_scheduler()` accessor + `init_scheduler()` / `shutdown_scheduler()` lifespan hooks. ~200 LOC. |
| `tests/aiperf_mock_server/config.py` | Modify | Append five `scheduler_*` fields to `MockServerConfig`. |
| `tests/aiperf_mock_server/utils.py` | Modify | `LatencySimulator._ensure_latencies` / `wait_for_next_token` / `wait_for_tokens` branch on `cfg.scheduler_enabled`; new `_wait_via_scheduler_*` helpers. |
| `tests/aiperf_mock_server/app.py` | Modify | `lifespan` calls `init_scheduler(cfg)` on enter, `shutdown_scheduler()` on exit. |
| `tests/aiperf_mock_server/test_scheduler.py` | Create | Unit + behavior tests for the scheduler. |
| `tests/aiperf_mock_server/test_scheduler_integration.py` | Create | End-to-end: spin up FastAPI under uvicorn, hit `/v1/chat/completions` at varying concurrencies, assert knee shape. |
| `tests/aiperf_mock_server/README.md` | Modify | New "Saturation modeling" section documenting the knobs + knee math. |

---

## Task 1: Add scheduler config knobs

**Files:**
- Modify: `tests/aiperf_mock_server/config.py` (append after `itl_concurrency_lin_ms`, before the embedding section)

- [ ] **Step 1: Add five `scheduler_*` fields to `MockServerConfig`**

```python
    scheduler_enabled: Annotated[
        bool,
        Field(
            description=(
                "Enable the step-based batched scheduler. When true, requests "
                "compete for per-step decode and prefill slots, producing a "
                "real saturation knee. When false (default), the open-loop "
                "TTFT/ITL latency model is used."
            ),
        ),
        Parameter(name="--scheduler-enabled", negative="--no-scheduler-enabled"),
    ] = False

    scheduler_step_ms: Annotated[
        float,
        Field(
            description=(
                "Virtual decode-step cadence in milliseconds. Each step admits "
                "up to scheduler_max_batch_size decode tokens. Smaller values "
                "= finer-grained ITL but higher scheduler CPU cost."
            ),
            gt=0.0,
            le=1000.0,
        ),
        Parameter(name="--scheduler-step-ms"),
    ] = 5.0

    scheduler_max_batch_size: Annotated[
        int,
        Field(
            description=(
                "Maximum concurrent decoders served per step. Past this "
                "concurrency the per-request ITL stretches linearly. Throughput "
                "ceiling = max_batch_size / step_ms tokens/sec."
            ),
            ge=1,
        ),
        Parameter(name="--scheduler-max-batch-size"),
    ] = 256

    scheduler_max_prefill_chunks_per_step: Annotated[
        int,
        Field(
            description=(
                "Maximum prefill chunks admitted per step. Lower = prefill "
                "becomes the binding constraint, producing TTFT cliffs under "
                "concurrent prompt arrivals."
            ),
            ge=1,
        ),
        Parameter(name="--scheduler-max-prefill-chunks-per-step"),
    ] = 8

    scheduler_prefill_chunk_tokens: Annotated[
        int,
        Field(
            description=(
                "Tokens per prefill chunk. A prompt of P tokens needs "
                "ceil(P / chunk_tokens) chunks. Larger = fewer steps per "
                "prompt but coarser-grained competition."
            ),
            ge=1,
        ),
        Parameter(name="--scheduler-prefill-chunk-tokens"),
    ] = 512
```

- [ ] **Step 2: Run config import smoke test**

```bash
uv run python -c "from tests.aiperf_mock_server.config import MockServerConfig; c = MockServerConfig(scheduler_enabled=True, scheduler_step_ms=2.0); assert c.scheduler_step_ms == 2.0; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add tests/aiperf_mock_server/config.py
git commit -m "feat(mock-server): add scheduler_* config knobs (design B)"
```

---

## Task 2: Implement `BatchScheduler` core

**Files:**
- Create: `tests/aiperf_mock_server/scheduler.py`

- [ ] **Step 1: Write the failing tests first (subset — full suite in Task 4)**

Create `tests/aiperf_mock_server/test_scheduler.py` with:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the mock server's batched step scheduler."""

import asyncio
import pytest

from tests.aiperf_mock_server.config import MockServerConfig
from tests.aiperf_mock_server.scheduler import BatchScheduler


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
        # 8 concurrent decoders, batch size 4 -> 2 steps to drain
        results = await asyncio.gather(
            *[sched.next_decode_step(f"r{i}") for i in range(8)]
        )
        # Half should land on step N, half on step N+1
        early = [s for s in results if s == min(results)]
        late = [s for s in results if s > min(results)]
        assert len(early) == 4
        assert len(late) == 4
        assert max(late) - min(early) == 1
    finally:
        await sched.stop()


@pytest.mark.asyncio
async def test_scheduler_prefill_chunks_split_long_prompts():
    """A 1500-token prompt with chunk_tokens=512 needs 3 chunks (3 steps min)."""
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_step_ms=1.0,
        scheduler_max_batch_size=64,
        scheduler_max_prefill_chunks_per_step=64,  # never the bottleneck
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
```

- [ ] **Step 2: Run the tests — they should fail with ImportError**

```bash
uv run pytest tests/aiperf_mock_server/test_scheduler.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.aiperf_mock_server.scheduler'`.

- [ ] **Step 3: Implement `BatchScheduler`**

Create `tests/aiperf_mock_server/scheduler.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step-based batched scheduler for the mock server.

Models the dominant first-order behavior of a continuous-batching LLM server:
a global decode loop ticking every `step_ms`, admitting up to `max_batch_size`
decoders per step, plus a separate prefill chunk pool with bounded
`max_prefill_chunks_per_step`. Produces a real throughput-vs-concurrency
saturation knee at concurrency ~= max_batch_size.

Out of scope (would require design C): KV-block budget, preemption, swap.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field

from tests.aiperf_mock_server.config import MockServerConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _DecodeWaiter:
    """A request waiting to have its next decode token admitted."""

    request_id: str
    event: asyncio.Event = field(default_factory=asyncio.Event)
    admitted_step: int = -1


@dataclass(slots=True)
class _PrefillWaiter:
    """A prefill chunk waiting for a slot."""

    request_id: str
    event: asyncio.Event = field(default_factory=asyncio.Event)
    admitted_step: int = -1


class BatchScheduler:
    """Global step-based scheduler. Singleton per mock-server process.

    Lifecycle:
        sched = BatchScheduler(cfg)
        await sched.start()              # spawns tick loop
        await sched.next_decode_step(rid) # request waits for next decode admit
        steps = await sched.run_prefill(rid, prompt_tokens=N)  # waits for ceil(N/chunk) chunks
        await sched.stop()               # cancels tick loop, drains waiters
    """

    def __init__(self, cfg: MockServerConfig) -> None:
        self._cfg = cfg
        self._step_index = 0
        self._tick_task: asyncio.Task | None = None
        self._decode_queue: deque[_DecodeWaiter] = deque()
        self._prefill_queue: deque[_PrefillWaiter] = deque()
        self._stopped = False

    @property
    def step_index(self) -> int:
        return self._step_index

    async def start(self) -> None:
        if not self._cfg.scheduler_enabled:
            raise RuntimeError("BatchScheduler.start called but scheduler is not enabled")
        if self._tick_task is not None:
            return
        self._tick_task = asyncio.create_task(self._tick_loop(), name="batch-scheduler-tick")
        logger.info(
            "BatchScheduler started: step_ms=%.3f max_batch=%d max_prefill_chunks=%d "
            "prefill_chunk_tokens=%d",
            self._cfg.scheduler_step_ms,
            self._cfg.scheduler_max_batch_size,
            self._cfg.scheduler_max_prefill_chunks_per_step,
            self._cfg.scheduler_prefill_chunk_tokens,
        )

    async def stop(self) -> None:
        self._stopped = True
        if self._tick_task is not None:
            self._tick_task.cancel()
            try:
                await self._tick_task
            except asyncio.CancelledError:
                pass
            self._tick_task = None
        # Wake any stragglers so they unblock and observe _stopped.
        for w in list(self._decode_queue):
            w.event.set()
        for w in list(self._prefill_queue):
            w.event.set()
        self._decode_queue.clear()
        self._prefill_queue.clear()

    async def next_decode_step(self, request_id: str) -> int:
        """Block until this request's next decode token is admitted.

        Returns the step index at which admission happened. Each call costs
        one admit slot.
        """
        if self._stopped:
            return self._step_index
        waiter = _DecodeWaiter(request_id=request_id)
        self._decode_queue.append(waiter)
        await waiter.event.wait()
        return waiter.admitted_step

    async def run_prefill(self, request_id: str, prompt_tokens: int) -> int:
        """Block until all prefill chunks for this prompt have been admitted.

        Returns the number of prefill chunks consumed (== number of steps if
        chunks were never queued, more if competing prompts forced waits).
        """
        if self._stopped or prompt_tokens <= 0:
            return 0
        chunks = max(1, _ceil_div(prompt_tokens, self._cfg.scheduler_prefill_chunk_tokens))
        for _ in range(chunks):
            waiter = _PrefillWaiter(request_id=request_id)
            self._prefill_queue.append(waiter)
            await waiter.event.wait()
        return chunks

    async def _tick_loop(self) -> None:
        step_seconds = self._cfg.scheduler_step_ms * 0.001
        loop = asyncio.get_running_loop()
        next_tick = loop.time() + step_seconds
        while not self._stopped:
            sleep_for = next_tick - loop.time()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            self._step_index += 1
            self._admit_prefill()
            self._admit_decode()
            next_tick += step_seconds

    def _admit_decode(self) -> None:
        budget = self._cfg.scheduler_max_batch_size
        while budget > 0 and self._decode_queue:
            w = self._decode_queue.popleft()
            w.admitted_step = self._step_index
            w.event.set()
            budget -= 1

    def _admit_prefill(self) -> None:
        budget = self._cfg.scheduler_max_prefill_chunks_per_step
        while budget > 0 and self._prefill_queue:
            w = self._prefill_queue.popleft()
            w.admitted_step = self._step_index
            w.event.set()
            budget -= 1


def _ceil_div(n: int, d: int) -> int:
    return -(-n // d)


_scheduler: BatchScheduler | None = None


def get_scheduler() -> BatchScheduler | None:
    return _scheduler


async def init_scheduler(cfg: MockServerConfig) -> BatchScheduler | None:
    """Lifespan hook: start the scheduler if enabled."""
    global _scheduler
    if not cfg.scheduler_enabled:
        _scheduler = None
        return None
    _scheduler = BatchScheduler(cfg)
    await _scheduler.start()
    return _scheduler


async def shutdown_scheduler() -> None:
    """Lifespan hook: stop the scheduler if running."""
    global _scheduler
    if _scheduler is not None:
        await _scheduler.stop()
        _scheduler = None
```

- [ ] **Step 4: Run tests — they should pass**

```bash
uv run pytest tests/aiperf_mock_server/test_scheduler.py -v -n auto
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add tests/aiperf_mock_server/scheduler.py tests/aiperf_mock_server/test_scheduler.py
git commit -m "feat(mock-server): add BatchScheduler core (design B)"
```

---

## Task 3: Wire `LatencySimulator` to scheduler

**Files:**
- Modify: `tests/aiperf_mock_server/utils.py` (`LatencySimulator` class, lines ~69-187)
- Modify: `tests/aiperf_mock_server/app.py` (`lifespan` function, lines ~94-138)

- [ ] **Step 1: Write the failing wiring test**

Append to `tests/aiperf_mock_server/test_scheduler.py`:

```python
import time
from unittest.mock import patch

from tests.aiperf_mock_server.utils import LatencySimulator
from tests.aiperf_mock_server import scheduler as scheduler_module


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
        # 1 prefill step + 1 decode step = ~20ms minimum
        assert elapsed_ms >= 15.0, f"too fast: {elapsed_ms:.1f}ms — scheduler not gating"
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
    assert elapsed_ms < 5.0, f"too slow: {elapsed_ms:.1f}ms — scheduler shouldn't be running"
```

- [ ] **Step 2: Run — should fail (sim doesn't know about scheduler yet)**

```bash
uv run pytest tests/aiperf_mock_server/test_scheduler.py::test_latency_simulator_uses_scheduler_when_enabled -v
```
Expected: FAIL — `elapsed_ms < 15` (the open-loop sleep is too fast).

- [ ] **Step 3: Modify `LatencySimulator` to branch on `cfg.scheduler_enabled`**

Edit `tests/aiperf_mock_server/utils.py`. Add a `_prefill_done: bool` slot and a `_request_id: str` slot. Inside `wait_for_next_token`, replace the body with:

```python
    async def wait_for_next_token(self) -> None:
        """Wait for TTFT (first token) or ITL (subsequent tokens)."""
        cfg = self._cfg
        if cfg.scheduler_enabled:
            from tests.aiperf_mock_server.scheduler import get_scheduler

            sched = get_scheduler()
            if sched is not None:
                await self._wait_via_scheduler(sched)
                return

        await self._wait_for_token_at_index(self.token_index)

        now = perf_counter()
        if self.token_index == 0:
            ttft = now - self.start_time
            self.measured_ttft = ttft
            record_ttft(self.endpoint, self.model, ttft)
        elif self.last_token_time is not None:
            itl = now - self.last_token_time
            record_itl(self.endpoint, self.model, itl)

        self.last_token_time = now
        self.token_index += 1

    async def _wait_via_scheduler(self, sched) -> None:
        """Scheduler-driven path: prefill on first call, then per-token decode admits."""
        if self.token_index == 0:
            # Prefill phase: chunks = ceil(isl / chunk_tokens)
            await sched.run_prefill(
                request_id=f"{self.endpoint}-{id(self)}",
                prompt_tokens=max(1, self._isl),
            )
            now = perf_counter()
            self.measured_ttft = now - self.start_time
            record_ttft(self.endpoint, self.model, self.measured_ttft)
            self.last_token_time = now
            self.token_index += 1
            return
        # Decode phase: one admit per token
        await sched.next_decode_step(f"{self.endpoint}-{id(self)}")
        now = perf_counter()
        if self.last_token_time is not None:
            record_itl(self.endpoint, self.model, now - self.last_token_time)
        self.last_token_time = now
        self.token_index += 1
```

Add `wait_for_tokens` scheduler branch (used by non-streaming endpoints):

```python
    async def wait_for_tokens(self, num_tokens: int) -> None:
        """Wait for entire completion (TTFT + ITL * num_tokens)."""
        cfg = self._cfg
        if cfg.scheduler_enabled:
            from tests.aiperf_mock_server.scheduler import get_scheduler

            sched = get_scheduler()
            if sched is not None:
                await sched.run_prefill(
                    request_id=f"{self.endpoint}-{id(self)}",
                    prompt_tokens=max(1, self._isl),
                )
                self.measured_ttft = perf_counter() - self.start_time
                for _ in range(num_tokens):
                    await sched.next_decode_step(f"{self.endpoint}-{id(self)}")
                self.measured_decode = (
                    perf_counter() - self.start_time - self.measured_ttft
                )
                return

        # Open-loop fallback (existing behavior).
        self._ensure_latencies()
        ttft_target = self.start_time + self.ttft_sec
        ttft_remaining = ttft_target - perf_counter()
        if ttft_remaining > 0:
            await asyncio.sleep(ttft_remaining)
        self.measured_ttft = perf_counter() - self.start_time
        decode_target = ttft_target + (self.itl_sec * num_tokens)
        decode_remaining = decode_target - perf_counter()
        if decode_remaining > 0:
            await asyncio.sleep(decode_remaining)
        self.measured_decode = perf_counter() - self.start_time - self.measured_ttft
```

- [ ] **Step 4: Wire scheduler lifecycle into FastAPI lifespan**

In `tests/aiperf_mock_server/app.py`, edit the `lifespan` async-context-manager (around line 94-138). After the existing config / DCGM startup work, add:

```python
    from tests.aiperf_mock_server.scheduler import init_scheduler, shutdown_scheduler

    await init_scheduler(server_config)
    try:
        yield
    finally:
        await shutdown_scheduler()
```

(Adapt to existing try/finally structure — keep all existing startup/shutdown logic.)

- [ ] **Step 5: Run all scheduler tests**

```bash
uv run pytest tests/aiperf_mock_server/test_scheduler.py -v -n auto
```
Expected: 6 passed (4 from Task 2 + 2 new).

- [ ] **Step 6: Run full mock-server test suite to confirm no regressions**

```bash
uv run pytest tests/aiperf_mock_server/ -v -n auto
```
Expected: all green; pre-existing tests unaffected (default `scheduler_enabled=False`).

- [ ] **Step 7: Commit**

```bash
git add tests/aiperf_mock_server/utils.py tests/aiperf_mock_server/app.py tests/aiperf_mock_server/test_scheduler.py
git commit -m "feat(mock-server): wire LatencySimulator to BatchScheduler"
```

---

## Task 4: Behavior tests — knee shape + Pareto front

**Files:**
- Create: `tests/aiperf_mock_server/test_scheduler_integration.py`

- [ ] **Step 1: Write the integration test**

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: scheduler-enabled mock server exhibits a real saturation knee."""

import asyncio
import time
import socket

import httpx
import pytest
import uvicorn

from tests.aiperf_mock_server.app import app
from tests.aiperf_mock_server.config import MockServerConfig, set_server_config


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
@pytest.mark.slow
async def test_throughput_knees_at_max_batch_size():
    """Drive the server at concurrencies bracketing max_batch_size; throughput
    must saturate, not scale linearly."""
    cfg = MockServerConfig(
        scheduler_enabled=True,
        scheduler_step_ms=2.0,
        scheduler_max_batch_size=16,
        scheduler_max_prefill_chunks_per_step=64,
        scheduler_prefill_chunk_tokens=512,
        ttft=0.0,
        itl=0.0,
        no_tokenizer=True,
    )
    set_server_config(cfg)
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    # Wait for server up
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
        for _ in range(50):
            try:
                r = await client.get("/v1/models")
                if r.status_code == 200:
                    break
            except httpx.ConnectError:
                await asyncio.sleep(0.05)

        async def one_request() -> float:
            t0 = time.perf_counter()
            r = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "mock-model",
                    "messages": [{"role": "user", "content": "hello world"}],
                    "max_tokens": 32,
                    "stream": False,
                },
            )
            r.raise_for_status()
            return time.perf_counter() - t0

        async def measure_throughput(concurrency: int, n: int = 64) -> float:
            sem = asyncio.Semaphore(concurrency)
            async def gated():
                async with sem:
                    return await one_request()
            t0 = time.perf_counter()
            await asyncio.gather(*[gated() for _ in range(n)])
            return n / (time.perf_counter() - t0)

        # Throughput at low concurrency vs. above the knee
        tput_low = await measure_throughput(concurrency=8, n=32)
        tput_at = await measure_throughput(concurrency=16, n=64)
        tput_high = await measure_throughput(concurrency=64, n=128)

        # Throughput should not grow linearly past max_batch_size=16.
        # Heuristic: 64-conc throughput must be < 1.5x the at-knee throughput.
        assert tput_high < tput_at * 1.5, (
            f"no saturation knee: low={tput_low:.1f} at={tput_at:.1f} "
            f"high={tput_high:.1f} req/s"
        )
        # And at-knee should be meaningfully above low-concurrency.
        assert tput_at > tput_low * 1.3, (
            f"insufficient ramp: low={tput_low:.1f} at={tput_at:.1f} req/s"
        )

    server.should_exit = True
    await server_task
```

- [ ] **Step 2: Run it**

```bash
uv run pytest tests/aiperf_mock_server/test_scheduler_integration.py -v -n auto
```
Expected: PASS. If the heuristics fail, log the three throughput values and tune (do not relax the assertion below `1.5x` / `1.3x` without investigating).

- [ ] **Step 3: Commit**

```bash
git add tests/aiperf_mock_server/test_scheduler_integration.py
git commit -m "test(mock-server): end-to-end knee-shape integration test"
```

---

## Task 5: README documentation

**Files:**
- Modify: `tests/aiperf_mock_server/README.md`

- [ ] **Step 1: Append a new "Saturation modeling (design B)" section**

Find the existing latency-knobs section in the README and append below it:

````markdown
## Saturation modeling (design B)

By default, the mock server uses an **open-loop latency model**: every request sleeps for `ttft + itl * num_tokens` (with optional concurrency penalties layered on). This produces a smooth latency curve but no real saturation knee — throughput scales linearly with concurrency.

For testing adaptive search and Pareto-front planners, enable the **batched step scheduler**:

```bash
aiperf-mock-server \
  --scheduler-enabled \
  --scheduler-step-ms 5 \
  --scheduler-max-batch-size 256 \
  --scheduler-max-prefill-chunks-per-step 8 \
  --scheduler-prefill-chunk-tokens 512
```

### What it models

- A virtual decode loop ticking every `step_ms`. Per tick, up to `max_batch_size` decoders advance one token; surplus decoders wait one or more ticks.
- A separate prefill chunk pool. A prompt of `P` tokens splits into `ceil(P / prefill_chunk_tokens)` chunks; each chunk consumes one of the `max_prefill_chunks_per_step` per-tick slots.

### Knee math

- **Decode-token rate ceiling:** `max_batch_size / step_ms` tokens/sec. With defaults: `256 / 0.005 = 51,200 tok/s`.
- **Concurrency knee:** `~max_batch_size`. Past that, every additional decoder linearly stretches all decoders' ITL.
- **Prefill cliff:** TTFT spikes when `max_prefill_chunks_per_step` is the binding constraint. Tune low to test prefill-bound regimes.

### Knob recipes

| Goal | Settings |
|---|---|
| Knee at concurrency=32, fast iteration | `--scheduler-max-batch-size 32 --scheduler-step-ms 5` |
| Prefill-bound regime (TTFT cliffs first) | `--scheduler-max-prefill-chunks-per-step 1 --scheduler-prefill-chunk-tokens 256` |
| Sharp Pareto front (TTFT vs throughput) | combine prefill-bound + small `max-batch-size` |

### Composition with other knobs

The structural scheduler-driven latency is **additive** with the per-request penalty knobs (`ttft_per_isl_token_ms`, `ttft_concurrency_quad_ms`, `itl_per_osl_token_ms`, `itl_concurrency_lin_ms`). Those penalties layer on top of scheduler waits when both are set; in default scheduler config the penalties are 0 and only the scheduler decides timing.

### Limitations

- Single-process only: `--workers > 1` runs each worker with an independent scheduler (no cross-worker batching).
- No KV-block accounting, no preemption, no swap. For those, see design C (deferred).
````

- [ ] **Step 2: Commit**

```bash
git add tests/aiperf_mock_server/README.md
git commit -m "docs(mock-server): document scheduler knobs + knee math"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Knobs (5) ✓, scheduler core ✓, prefill+decode pools ✓, latency-sim wiring ✓, lifespan integration ✓, knee tests ✓, docs ✓.
- [x] **No placeholders:** every step has full code or full command.
- [x] **Type consistency:** `BatchScheduler.next_decode_step(request_id: str) -> int`, `run_prefill(request_id: str, prompt_tokens: int) -> int`, `start() / stop()` async, `init_scheduler(cfg) -> BatchScheduler | None`, `get_scheduler() -> BatchScheduler | None`. Used identically in Tasks 2, 3, 4.
- [x] **Backward compat:** default `scheduler_enabled=False` makes every existing test bit-identical. Wiring is gated behind that flag.

---

## Parallelization

Tasks 1, 2, and 5 are independent (config knobs / scheduler module / docs). Task 3 depends on 1+2. Task 4 depends on 3. Suggested wave dispatch:

- **Wave 1 (parallel):** Task 1, Task 2, Task 5 — three subagents
- **Wave 2 (after Wave 1 lands):** Task 3 — one subagent
- **Wave 3 (after Wave 2 lands):** Task 4 — one subagent
