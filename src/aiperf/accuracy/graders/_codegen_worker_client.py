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
import os
import signal
import sys
from collections import deque
from typing import Any

import orjson

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import IS_WINDOWS

_log = AIPerfLogger(__name__)

_DEFAULT_WORKER_CMD = [sys.executable, "-m", "aiperf.accuracy.graders._codegen_worker"]

# Env var carrying the death-pipe read fd to the worker. The client holds the
# write end open for the worker's life; its close (including via the parent's
# os._exit force-kill, which the graceful aclose() path can't cover) tells the
# worker's death watcher to reap itself and its sandbox children. Must match
# _codegen_worker._DEATH_FD_ENV.
_DEATH_FD_ENV = "AIPERF_CODEGEN_DEATH_FD"

# asyncio's StreamReader defaults to a 64 KiB line limit; readline() raises
# ValueError past it. A grading error string can legitimately be large, so give
# the reader generous headroom while still bounding memory. The worker also caps
# its error strings, so this limit should never be reached in practice.
_STREAM_LIMIT = 16 * 1024 * 1024

# Bound the retained worker-stderr diagnostics so a chatty worker can't grow
# memory unbounded; the tail is logged on fault to explain why a worker died.
_STDERR_TAIL_LINES = 64


def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the worker's whole process group so lighteval's forked sandbox
    grandchildren die with it, not just the worker. Falls back to killing the
    worker alone where process groups are unavailable (e.g. Windows)."""
    if hasattr(os, "killpg"):
        with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


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
        self._death_w: int | None = None

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
        self._close_death_pipe()
        # Death pipe: the child inherits the read end; we keep the write end open
        # for the worker's life so its close (even via the parent's os._exit)
        # tells the worker to reap itself. os.pipe fds are non-inheritable by
        # default, so mark the read end inheritable before passing it through.
        # Windows has no process groups (nothing for the worker to killpg) and
        # subprocess rejects pass_fds there, so skip the pipe and rely on the
        # stdin-EOF teardown; the worker's death watcher likewise no-ops there.
        death_r: int | None = None
        death_w: int | None = None
        pass_fds: tuple[int, ...] = ()
        death_env: dict[str, str] = {}
        if not IS_WINDOWS:
            death_r, death_w = os.pipe()
            os.set_inheritable(death_r, True)
            pass_fds = (death_r,)
            death_env = {_DEATH_FD_ENV: str(death_r)}
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_STREAM_LIMIT,
                # Own process group so _kill can reap lighteval's forked sandbox
                # grandchildren, not just the worker (no-op on Windows).
                start_new_session=True,
                pass_fds=pass_fds,
                env={**os.environ, **death_env},
            )
        except Exception as exc:
            if death_r is not None:
                os.close(death_r)
            if death_w is not None:
                os.close(death_w)
            self._start_failures += 1
            raise CodegenWorkerError(f"failed to spawn grading worker: {exc}") from exc
        if death_r is not None:
            os.close(death_r)  # the child holds it now; we keep only the write end
        self._death_w = death_w
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
        except asyncio.CancelledError:
            # Shutdown/cancellation while awaiting the worker: kill it (and its
            # sandbox group) so a pending request can't desync the next grade or
            # leave orphaned children, then propagate. Not a worker failure.
            await self._handle_fault(count_start_failure=False)
            raise
        except TimeoutError as exc:
            # The worker is alive but this grade is too slow — a per-grade fault,
            # not a worker-startup failure. Kill+respawn, but do NOT count it
            # toward the readiness cap, or a few slow problems at the start of a
            # run would trip the cap and disable all grading.
            await self._handle_fault(count_start_failure=False)
            raise CodegenWorkerError(f"grading worker timed out: {exc!r}") from exc
        except (ConnectionError, BrokenPipeError, ValueError) as exc:
            # Transport broke, or the response overran the StreamReader limit
            # (ValueError covers asyncio.LimitOverrunError): the worker died or
            # desynced, so this counts toward the startup/readiness cap.
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

    async def _handle_fault(self, count_start_failure: bool = True) -> None:
        # Only failures that mean the worker never became usable (it died or
        # emitted garbage before ever succeeding) count toward the startup cap.
        # Per-grade timeouts and shutdown cancellation pass count_start_failure=
        # False so a slow grade never disables the whole run.
        if count_start_failure and not self._worker_proven:
            self._start_failures += 1
        tail = await self._kill()
        _log.debug(
            lambda: f"codegen worker fault (proven={self._worker_proven}, "
            f"start_failures={self._start_failures}); killed + respawning next grade"
            + (f"; stderr tail:\n{chr(10).join(tail)}" if tail else "")
        )

    def _close_death_pipe(self) -> None:
        if self._death_w is not None:
            with contextlib.suppress(OSError):
                os.close(self._death_w)
            self._death_w = None

    async def _kill(self) -> list[str]:
        """Kill the worker and return its captured stderr tail (for diagnostics)."""
        proc, self._proc = self._proc, None
        task, self._stderr_task = self._stderr_task, None
        self._close_death_pipe()
        if proc is not None and proc.returncode is None:
            _kill_process_group(proc)
            with contextlib.suppress(ProcessLookupError):
                await proc.wait()
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
        # Acquire the lock so teardown cannot race with an in-flight _request
        # clearing/reading _proc across an await.
        async with self._lock:
            await self._kill()
