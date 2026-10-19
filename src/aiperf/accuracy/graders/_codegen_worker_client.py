# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process client for the out-of-process LCB codegen grading worker.

Owned by ``CodeExecutionGrader``. Lazily spawns a single persistent worker
subprocess, routes concurrent grading requests via an ``id → Future`` demux
table, and enforces per-grade timeouts with kill+restart. See issue #1094 and
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
        self._spawn_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
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
        if self._start_failures >= self._max_start_failures:
            raise CodegenWorkerError(
                f"grading worker unavailable after {self._start_failures} start failures"
            )
        await self._ensure_worker()
        self._next_id += 1
        req_id = self._next_id
        req = {
            "id": req_id,
            "evaluation_sample": evaluation_sample,
            "generated_code": generated_code,
        }
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        self._pending[req_id] = fut
        proc = self._proc
        if proc is None or proc.stdin is None:
            self._pending.pop(req_id, None)
            raise CodegenWorkerError("grading worker is not running")
        # Guard drain() with the caller's deadline so a worker that stops consuming
        # stdin (pipe buffer full) cannot block indefinitely. CancelledError during
        # drain() is also handled here so the pending future is cleaned up.
        try:
            proc.stdin.write(orjson.dumps(req) + b"\n")
            await asyncio.wait_for(proc.stdin.drain(), max(0.0, deadline - loop.time()))
        except TimeoutError as exc:
            # TimeoutError is an OSError subclass (PEP 3151), so it must be caught
            # before the OSError clause below or it is silently misrouted there and
            # _handle_fault() never runs, leaving the wedged worker alive.
            self._pending.pop(req_id, None)
            await self._handle_fault(count_start_failure=False)
            raise CodegenWorkerError(f"grading worker timed out: {exc!r}") from exc
        except (OSError, ConnectionError) as exc:
            self._pending.pop(req_id, None)
            raise CodegenWorkerError(
                f"failed to submit grading request: {exc}"
            ) from exc
        except asyncio.CancelledError:
            self._pending.pop(req_id, None)
            raise
        try:
            return await asyncio.wait_for(fut, max(0.0, deadline - loop.time()))
        except TimeoutError as exc:
            self._pending.pop(req_id, None)
            await self._handle_fault(count_start_failure=False)
            raise CodegenWorkerError(f"grading worker timed out: {exc!r}") from exc
        except asyncio.CancelledError:
            # Cancellation does not desync the protocol — the request was already
            # written and the late response will be dropped as stale by
            # _dispatch_response. Kill the worker only on real faults.
            self._pending.pop(req_id, None)
            raise

    async def _ensure_worker(self) -> None:
        async with self._spawn_lock:
            if self._proc is not None and self._proc.returncode is None:
                return
            # The old worker exited; tear down its reader/stderr tasks so a late
            # EOF from the dead process cannot fault the replacement worker.
            if self._proc is not None or self._reader_task is not None:
                await self._kill()
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
                raise CodegenWorkerError(
                    f"failed to spawn grading worker: {exc}"
                ) from exc
            if death_r is not None:
                os.close(death_r)  # the child holds it now; we keep only the write end
            self._death_w = death_w
            # Drain stderr continuously so the pipe never fills (which would block the
            # worker) and the last output is retained to explain a fault.
            self._stderr_task = asyncio.create_task(
                self._drain_stderr(self._proc.stderr)
            )
            self._reader_task = asyncio.create_task(self._run_reader())

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

    def _dispatch_response(self, resp: dict[str, Any]) -> bool:
        """Resolve the pending future for resp['id'].

        Returns True on success (including stale/timed-out ids, which are
        silently skipped). Returns False when the worker has desynced and the
        caller should fault: unhashable id, or ok=True with no metrics dict.
        """
        req_id = resp.get("id")
        if req_id is None:
            # The client only writes integer-keyed requests, so a null id means
            # the worker could not parse a request — the stream has desynced.
            return False
        try:
            fut = self._pending.pop(req_id, None)
        except TypeError:
            # req_id is unhashable (e.g., a list); the worker desynced
            return False
        if fut is None or fut.done():
            return True  # stale id (caller already timed out or cancelled)
        if not resp.get("ok"):
            fut.set_exception(
                CodegenWorkerError(resp.get("error", "unknown grading error"))
            )
            self._mark_proven()
            return True
        metrics = resp.get("metrics")
        if not isinstance(metrics, dict):
            return False
        fut.set_result(metrics)
        self._mark_proven()
        return True

    async def _run_reader(self) -> None:
        """Read worker responses and resolve the corresponding pending futures by id."""
        if self._proc is None or self._proc.stdout is None:
            return
        reader = self._proc.stdout
        try:
            while True:
                try:
                    line = await reader.readline()
                except (ValueError, ConnectionError, BrokenPipeError):
                    # ValueError covers asyncio.LimitOverrunError (oversized line);
                    # the other two indicate a broken transport. All mean the worker
                    # desynced; fault it so callers are unblocked.
                    await self._handle_fault()
                    return
                if not line:
                    await self._handle_fault()
                    return
                try:
                    resp = orjson.loads(line)
                except orjson.JSONDecodeError:
                    await self._handle_fault()
                    return
                if not isinstance(resp, dict) or not self._dispatch_response(resp):
                    await self._handle_fault()
                    return
        except asyncio.CancelledError:
            pass

    def _mark_proven(self) -> None:
        self._worker_proven = True
        self._start_failures = 0

    async def _handle_fault(self, count_start_failure: bool = True) -> None:
        if self._proc is None:
            return  # already handled; _handle_fault is idempotent
        if count_start_failure and not self._worker_proven:
            self._start_failures += 1
        # Kill and drain stderr before setting futures' exceptions. This ensures
        # the debug log (which includes the stderr tail) is written before callers
        # are unblocked — otherwise the event loop may schedule waiting coroutines
        # between set_exception() and the log, causing the tail to appear empty in
        # tests and diagnostics.
        tail = await self._kill()
        _log.debug(
            lambda: (
                f"codegen worker fault (proven={self._worker_proven}, "
                f"start_failures={self._start_failures}); killed + respawning next grade"
                + (f"; stderr tail:\n{chr(10).join(tail)}" if tail else "")
            )
        )
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(CodegenWorkerError("grading worker fault"))
        self._pending.clear()

    def _close_death_pipe(self) -> None:
        if self._death_w is not None:
            with contextlib.suppress(OSError):
                os.close(self._death_w)
            self._death_w = None

    async def _kill(self) -> list[str]:
        """Kill the worker and return its captured stderr tail (for diagnostics)."""
        proc, self._proc = self._proc, None
        task, self._stderr_task = self._stderr_task, None
        reader_task, self._reader_task = self._reader_task, None
        self._close_death_pipe()
        if proc is not None:
            # Kill the process group regardless of returncode: the worker leader may
            # have already exited while lighteval's forked sandbox grandchildren are
            # still alive in the same dedicated process group. ProcessLookupError
            # means the group is already gone — treat that as successful cleanup.
            _kill_process_group(proc)
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    await proc.wait()
        # Skip cancel/await if _reader_task is calling _kill from within itself to
        # avoid self-awaiting deadlock when the reader detects a fault condition.
        current = asyncio.current_task()
        if reader_task is not None and reader_task is not current:
            reader_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader_task
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
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(CodegenWorkerError("grading worker closed"))
        self._pending.clear()
        # Acquire _spawn_lock so teardown cannot race with an in-flight
        # _ensure_worker awaiting create_subprocess_exec. Without the lock,
        # _kill() sees _proc=None and skips cleanup; the spawn then completes
        # and leaves an orphaned start_new_session worker with no owner.
        async with self._spawn_lock:
            await self._kill()
