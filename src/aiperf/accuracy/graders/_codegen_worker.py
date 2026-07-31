# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-process LCB codegen grading worker.

Runs as ``python -m aiperf.accuracy.graders._codegen_worker``. A fresh,
single-threaded interpreter that forces the ``fork`` start method once at
startup, then reads JSONL grading requests from stdin (batching queued
requests) and writes one JSONL response per request to a private protocol fd.
Executing lighteval's ``codegen_metrics`` here (not in the multithreaded
record-processor daemon) avoids both the nested-function pickle failure under
spawn/forkserver and the multithreaded-fork hang. See issue #1145.
"""

from __future__ import annotations

import contextlib
import math
import multiprocessing as mp
import os
import signal
import sys
import threading
from collections.abc import Callable
from typing import Any, BinaryIO

import orjson

_LCB_PASS_AT_K = (1,)
_LCB_NUM_PROCESSES = 8

# The client passes the read end of a "death pipe" here; the write end stays open
# in the client for the worker's whole life, so its close (even via the parent's
# os._exit force-kill) signals the worker to reap itself. See _start_death_watcher.
_DEATH_FD_ENV = "AIPERF_CODEGEN_DEATH_FD"

# A pathological lighteval exception could stringify to megabytes; bound the
# error so a single response line stays well under the client's stream limit.
_MAX_ERROR_CHARS = 4096


def _truncate_error(error: str) -> str:
    """Bound an error string so it cannot produce a multi-MB response line."""
    if len(error) <= _MAX_ERROR_CHARS:
        return error
    return error[:_MAX_ERROR_CHARS] + "...[truncated]"


def _coerce_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """orjson can't serialize numpy scalars lighteval returns; coerce to float.

    lighteval returns pass@1 as a scalar on some pins and a list on others; both
    must survive so the client's scalar/list extractor sees a real value rather
    than falling back to 0.0. Genuinely non-numeric values are dropped."""
    coerced: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, list) and all(_is_number(x) for x in value):
            coerced[key] = [float(x) for x in value]
        elif not isinstance(value, list) and _is_number(value):
            coerced[key] = float(value)
    return coerced


def _is_number(value: Any) -> bool:
    # Metrics cross the JSONL boundary, so only finite values may pass; NaN/Inf
    # are rejected per the repo's NaN/Inf discipline.
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def handle_batch(
    reqs: list[Any],
    codegen_fn: Callable[..., tuple[dict[str, Any], Any]],
    compute_metrics_fn: Callable[..., dict[str, Any]],
) -> list[dict[str, Any]]:
    """Grade a batch of requests with a single codegen_fn call.

    Calls codegen_fn once with all well-formed requests batched together so
    lighteval's ProcessPoolExecutor can process multiple problems in parallel.
    Never raises: all failures become error responses so a bad batch cannot
    kill the worker loop.
    """
    all_samples: list[Any] = []
    all_generations: list[Any] = []
    id_map: list[tuple[int, Any]] = []  # (batch_position, req_id)
    responses: list[dict[str, Any] | None] = [None] * len(reqs)

    for i, req in enumerate(reqs):
        if isinstance(req, dict) and "_parse_error" in req:
            responses[i] = {
                "id": None,
                "ok": False,
                "error": f"bad json: {req['_parse_error']}",
            }
            continue
        if not isinstance(req, dict):
            responses[i] = {
                "id": None,
                "ok": False,
                "error": "malformed request: expected object",
            }
            continue
        req_id = req.get("id")
        try:
            all_samples.append(req["evaluation_sample"])
            all_generations.append(req["generated_code"])
            id_map.append((i, req_id))
        except (KeyError, TypeError) as exc:
            responses[i] = {
                "id": req_id,
                "ok": False,
                "error": f"malformed request: {exc!r}",
            }

    if all_samples:
        batch_error: str | None = None
        raw_results: dict[int, Any] = {}
        try:
            _, raw_results = codegen_fn(
                all_samples,
                all_generations,
                k_list=list(_LCB_PASS_AT_K),
                num_process_evaluate=_LCB_NUM_PROCESSES,
            )
        except Exception as exc:
            batch_error = _truncate_error(f"{type(exc).__name__}: {exc}")

        for pos, (req_idx, req_id) in enumerate(id_map):
            if batch_error is not None:
                responses[req_idx] = {"id": req_id, "ok": False, "error": batch_error}
            else:
                try:
                    metrics = compute_metrics_fn(
                        {0: raw_results[pos]},
                        k_list=list(_LCB_PASS_AT_K),
                    )
                    responses[req_idx] = {
                        "id": req_id,
                        "ok": True,
                        "metrics": _coerce_metrics(metrics),
                    }
                except Exception as exc:
                    responses[req_idx] = {
                        "id": req_id,
                        "ok": False,
                        "error": _truncate_error(f"{type(exc).__name__}: {exc}"),
                    }

    return [r for r in responses if r is not None]


def _drain_buffered(stdin: BinaryIO) -> list[bytes]:
    """Drain all lines already buffered in stdin without blocking.

    For BufferedReader (sys.stdin.buffer in production), uses peek() to check
    the userspace buffer; an empty peek means the next readline() would block,
    so we stop. This is the correct non-blocking check because readline() pulls
    kernel data into userspace first, making select() on the raw fd unreliable.

    For seekable streams without peek() (e.g. BytesIO used in tests), reads all
    remaining data at once — safe because BytesIO is already fully in memory.
    """
    lines: list[bytes] = []
    if hasattr(stdin, "peek"):
        while True:
            if not stdin.peek(0):
                break
            line = stdin.readline()
            if not line:
                break
            line = line.strip()
            if line:
                lines.append(line)
    else:
        for raw_line in stdin.read().split(b"\n"):
            raw_line = raw_line.strip()
            if raw_line:
                lines.append(raw_line)
    return lines


def run_worker_loop(
    stdin: BinaryIO,
    out: BinaryIO,
    codegen_fn: Callable[..., tuple[dict[str, Any], Any]],
    compute_metrics_fn: Callable[..., dict[str, Any]],
) -> None:
    """Serve JSONL grading requests until stdin EOF.

    Blocks on the first request of each cycle, then non-blocking drains any
    already-queued requests to form a batch. Calls codegen_fn once per batch so
    lighteval's ProcessPoolExecutor can process multiple problems in parallel.
    """
    while True:
        first = stdin.readline()
        if not first:
            break  # EOF: client closed stdin, clean exit
        first = first.strip()
        if not first:
            continue
        batch_raw: list[bytes] = [first]
        batch_raw.extend(_drain_buffered(stdin))
        reqs: list[Any] = []
        for raw in batch_raw:
            try:
                reqs.append(orjson.loads(raw))
            except orjson.JSONDecodeError as exc:
                reqs.append({"_parse_error": str(exc)})
        for resp in handle_batch(reqs, codegen_fn, compute_metrics_fn):
            out.write(orjson.dumps(resp) + b"\n")
        out.flush()


def _force_fork() -> None:
    """Force the fork start method so lighteval's nested-function Process target
    works. Safe here: this is a fresh single-threaded interpreter, so there is no
    restore dance and no sibling thread can hold a lock across the fork."""
    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)


def _close_fd_quietly(fd: int) -> None:
    with contextlib.suppress(OSError):
        os.close(fd)


def _install_stdout_guard() -> BinaryIO:
    """Return a private binary stream that writes to the real stdout, then point
    fd 1 at fd 2 so any library writes to stdout land on stderr instead of
    corrupting the protocol stream."""
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    # lighteval forks sandbox children that run untrusted generated code. They
    # must not inherit the protocol fd (fork copies fds, and it isn't exec'd so
    # close-on-exec wouldn't help), or that code could write to the client's
    # JSONL response channel and spoof/desync grading. Close it in every forked
    # child; only the worker parent keeps it.
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=lambda: _close_fd_quietly(protocol_fd))
    return os.fdopen(protocol_fd, "wb", buffering=0)


def _start_death_watcher() -> None:
    """Reap this worker (and its lighteval sandbox children) if the parent dies
    abruptly — the aiperf ``os._exit`` force-kill path, where the parent can't run
    the client's ``aclose()`` and, because the worker has its own session
    (``start_new_session``), receives no signal from the parent's exit.

    The parent holds the write end of a pipe whose read end is passed here as
    ``AIPERF_CODEGEN_DEATH_FD``. A dedicated thread blocks reading it; when the
    parent exits the write end closes, the read returns EOF, and the thread
    ``killpg``s this worker's process group so the worker and its forked sandbox
    grandchildren die together. The thread stays blocked in ``os.read`` for its
    whole life, holding no lock, so it does not reintroduce the multithreaded-fork
    hazard when lighteval forks from the main thread.
    """
    # Process groups (killpg) are Unix-only; without them there is nothing to
    # reap, so skip the watcher entirely (LCB can't run on Windows anyway).
    if not hasattr(os, "killpg"):
        return
    fd_str = os.environ.get(_DEATH_FD_ENV)
    if not fd_str:
        return
    death_fd = int(fd_str)
    # Untrusted sandbox children shouldn't inherit the death fd either; close it
    # in every forked child (the parent's watcher keeps its own copy).
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=lambda: _close_fd_quietly(death_fd))

    def _watch() -> None:
        with contextlib.suppress(OSError):
            while os.read(death_fd, 4096):
                pass  # parent alive; ignore any bytes and wait for EOF
        # EOF: the parent's write end closed (it exited). Kill our own process
        # group (pgid 0 == this session, since the worker was start_new_session'd).
        with contextlib.suppress(OSError):
            os.killpg(0, signal.SIGKILL)

    threading.Thread(target=_watch, daemon=True).start()


def main() -> None:
    protocol_out = _install_stdout_guard()
    # Start the death watcher before importing lighteval so an abrupt parent exit
    # reaps the worker even if the (heavy) import is still in flight.
    _start_death_watcher()
    _force_fork()
    from lighteval.tasks.tasks.lcb.codegen_metrics import (
        codegen_metrics,
        compute_metrics_from_results,
    )

    run_worker_loop(
        sys.stdin.buffer, protocol_out, codegen_metrics, compute_metrics_from_results
    )


if __name__ == "__main__":
    main()
