# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process client for the out-of-process LCB codegen grading worker.

Owned by ``CodeExecutionGrader``. Lazily spawns a single persistent worker
subprocess, serializes grading requests through an ``asyncio.Lock``, and
enforces per-grade timeouts with kill+restart. See issue #1145 and
``_codegen_worker.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from collections import deque
from typing import Any

import orjson

from aiperf.common.aiperf_logger import AIPerfLogger

_log = AIPerfLogger(__name__)

_DEFAULT_WORKER_CMD = [sys.executable, "-m", "aiperf.accuracy.graders._codegen_worker"]

# asyncio's StreamReader defaults to a 64 KiB line limit; readline() raises
# ValueError past it. A grading error string can legitimately be large, so give
# the reader generous headroom while still bounding memory. The worker also caps
# its error strings, so this limit should never be reached in practice.
_STREAM_LIMIT = 16 * 1024 * 1024

# Bound the retained worker-stderr diagnostics so a chatty worker can't grow
# memory unbounded; the tail is logged on fault to explain why a worker died.
_STDERR_TAIL_LINES = 64


class CodegenWorkerError(Exception):
    """Raised when the grading worker cannot produce a result (timeout, crash,
    or repeated startup failure). The grader maps this to a grading failure."""


class CodegenGradingWorker:
    def __init__(
        self,
        worker_cmd: list[str] | None = None,
        max_start_failures: int = 3,
    ) -> None:
        self._cmd = worker_cmd or _DEFAULT_WORKER_CMD
        self._max_start_failures = max_start_failures
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._next_id = 0
        self._start_failures = 0
        self._worker_proven = False
        self._stderr_tail: deque[str] = deque(maxlen=_STDERR_TAIL_LINES)
        self._stderr_task: asyncio.Task[None] | None = None

    async def grade_codegen(
        self,
        evaluation_sample: list[dict[str, str]],
        generated_code: list[list[str]],
        timeout: float,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._start_failures >= self._max_start_failures:
                raise CodegenWorkerError(
                    f"grading worker unavailable after {self._start_failures} start failures"
                )
            await self._ensure_worker()
            self._next_id += 1
            req = {
                "id": self._next_id,
                "evaluation_sample": evaluation_sample,
                "generated_code": generated_code,
            }
            return await self._request(req, timeout)

    async def _ensure_worker(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            return
        self._worker_proven = False
        self._stderr_tail.clear()
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_STREAM_LIMIT,
            )
        except Exception as exc:
            self._start_failures += 1
            raise CodegenWorkerError(f"failed to spawn grading worker: {exc}") from exc
        # Drain stderr continuously so the pipe never fills (which would block the
        # worker) and the last output is retained to explain a fault.
        self._stderr_task = asyncio.create_task(self._drain_stderr(self._proc.stderr))

    async def _drain_stderr(self, reader: asyncio.StreamReader | None) -> None:
        """Continuously copy worker stderr into a bounded tail. Best-effort:
        diagnostics must never disrupt grading, so all read errors are swallowed."""
        if reader is None:
            return
        # Best-effort: a reader overrun or closed transport ends diagnostics
        # silently; they must never disrupt grading.
        with contextlib.suppress(Exception):
            async for line in reader:
                self._stderr_tail.append(line.decode("utf-8", "replace").rstrip("\n"))

    async def _request(self, req: dict[str, Any], timeout: float) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        try:
            self._proc.stdin.write(orjson.dumps(req) + b"\n")
            await self._proc.stdin.drain()
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout)
        except (TimeoutError, ConnectionError, BrokenPipeError, ValueError) as exc:
            # ValueError covers readline() overrunning the StreamReader limit
            # (asyncio.LimitOverrunError); route it through the fault path so the
            # worker is killed and counted rather than desyncing the next grade.
            await self._handle_fault()
            raise CodegenWorkerError(f"grading worker fault: {exc!r}") from exc

        if not line:  # EOF: worker died
            await self._handle_fault()
            raise CodegenWorkerError("grading worker exited before responding")

        try:
            resp = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            # Garbage on stdout means the worker desynced; fault it like an EOF
            # so it is killed and counted rather than silently reused.
            await self._handle_fault()
            raise CodegenWorkerError(
                f"grading worker emitted non-JSON output: {line!r}"
            ) from exc

        if not isinstance(resp, dict):
            # Valid JSON that is not an object would make the ok/metrics lookups
            # below raise; fault it like the other malformed-response classes.
            await self._handle_fault()
            raise CodegenWorkerError(
                f"grading worker emitted a non-object response: {line!r}"
            )

        if resp.get("id") != req["id"]:
            # A mismatched id means the response no longer correlates to the
            # request; the stream is desynced, so fault it before trusting
            # ok/metrics from a stale or wrong frame.
            await self._handle_fault()
            raise CodegenWorkerError(
                f"grading worker response id mismatch: expected {req['id']}, "
                f"got {resp.get('id')!r}"
            )

        if not resp.get("ok"):
            # A clean error response is a proven worker; do not restart.
            self._worker_proven = True
            self._start_failures = 0
            raise CodegenWorkerError(resp.get("error", "unknown grading error"))

        metrics = resp.get("metrics")
        if not isinstance(metrics, dict):
            # ok:true without a usable metrics dict is a broken worker, not a
            # clean result; fault it rather than returning junk to the grader.
            await self._handle_fault()
            raise CodegenWorkerError(
                "grading worker reported success without valid metrics"
            )

        self._worker_proven = True
        self._start_failures = 0
        return metrics

    async def _handle_fault(self) -> None:
        if not self._worker_proven:
            self._start_failures += 1
        tail = await self._kill()
        _log.debug(
            lambda: f"codegen worker fault (proven={self._worker_proven}, "
            f"start_failures={self._start_failures}); killed + respawning next grade"
            + (f"; stderr tail:\n{chr(10).join(tail)}" if tail else "")
        )

    async def _kill(self) -> list[str]:
        """Kill the worker and return its captured stderr tail (for diagnostics)."""
        proc, self._proc = self._proc, None
        task, self._stderr_task = self._stderr_task, None
        if proc is not None and proc.returncode is None:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
        tail: list[str] = []
        if task is not None:
            # The dead worker's stderr hits EOF, so the drain task finishes; bound
            # the wait, then snapshot whatever it captured. wait_for cancels the
            # task on timeout.
            with contextlib.suppress(TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=2.0)
            tail = list(self._stderr_tail)
        return tail

    async def aclose(self) -> None:
        await self._kill()
