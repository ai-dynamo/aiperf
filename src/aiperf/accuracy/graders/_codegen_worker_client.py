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
import sys
from typing import Any

import orjson

from aiperf.common.aiperf_logger import AIPerfLogger

_log = AIPerfLogger(__name__)

_DEFAULT_WORKER_CMD = [sys.executable, "-m", "aiperf.accuracy.graders._codegen_worker"]


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
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception as exc:
            self._start_failures += 1
            raise CodegenWorkerError(f"failed to spawn grading worker: {exc}") from exc

    async def _request(self, req: dict[str, Any], timeout: float) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin and self._proc.stdout
        try:
            self._proc.stdin.write(orjson.dumps(req) + b"\n")
            await self._proc.stdin.drain()
            line = await asyncio.wait_for(self._proc.stdout.readline(), timeout)
        except (TimeoutError, ConnectionError, BrokenPipeError) as exc:
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
        _log.debug(
            lambda: f"codegen worker fault (proven={self._worker_proven}, "
            f"start_failures={self._start_failures}); killing + respawning next grade"
        )
        if not self._worker_proven:
            self._start_failures += 1
        await self._kill()

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        try:
            proc.kill()
            await proc.wait()
        except ProcessLookupError:
            pass

    async def aclose(self) -> None:
        await self._kill()
