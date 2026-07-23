# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Out-of-process LCB codegen grading worker.

Runs as ``python -m aiperf.accuracy.graders._codegen_worker``. A fresh,
single-threaded interpreter that forces the ``fork`` start method once at
startup, then reads one JSONL grading request per line from stdin and writes one
JSONL response per line to a private protocol fd. Executing lighteval's
``codegen_metrics`` here (not in the multithreaded record-processor daemon)
avoids both the nested-function pickle failure under spawn/forkserver and the
multithreaded-fork hang. See issue #1145.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from collections.abc import Callable
from typing import Any, BinaryIO

import orjson

_LCB_PASS_AT_K = (1,)
_LCB_NUM_PROCESSES = 8


def handle_request(
    req: dict[str, Any],
    codegen_fn: Callable[..., tuple[dict[str, Any], Any]],
) -> dict[str, Any]:
    """Run one grading request. Never raises: all failures become an error
    response so a single bad problem cannot kill the worker loop."""
    req_id = req.get("id")
    try:
        evaluation_sample = req["evaluation_sample"]
        generated_code = req["generated_code"]
    except (KeyError, TypeError) as exc:
        return {"id": req_id, "ok": False, "error": f"malformed request: {exc!r}"}

    try:
        metrics, _ = codegen_fn(
            evaluation_sample,
            generated_code,
            k_list=list(_LCB_PASS_AT_K),
            num_process_evaluate=_LCB_NUM_PROCESSES,
        )
    except Exception as exc:
        # A single bad problem must never crash the worker loop.
        return {"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"}

    return {"id": req_id, "ok": True, "metrics": _coerce_metrics(metrics)}


def _coerce_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """orjson can't serialize numpy scalars lighteval returns; coerce to float."""
    return {k: float(v) for k, v in metrics.items() if _is_number(v)}


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def run_worker_loop(
    stdin: BinaryIO,
    out: BinaryIO,
    codegen_fn: Callable[..., tuple[dict[str, Any], Any]],
) -> None:
    """Serve JSONL grading requests until stdin EOF. One response per request."""
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = orjson.loads(line)
        except orjson.JSONDecodeError as exc:
            resp: dict[str, Any] = {
                "id": None,
                "ok": False,
                "error": f"bad json: {exc}",
            }
        else:
            resp = handle_request(req, codegen_fn)
        out.write(orjson.dumps(resp) + b"\n")
        out.flush()


def _force_fork() -> None:
    """Force the fork start method so lighteval's nested-function Process target
    works. Safe here: this is a fresh single-threaded interpreter, so there is no
    restore dance and no sibling thread can hold a lock across the fork."""
    if "fork" in mp.get_all_start_methods():
        mp.set_start_method("fork", force=True)


def _install_stdout_guard() -> BinaryIO:
    """Return a private binary stream that writes to the real stdout, then point
    fd 1 at fd 2 so any library writes to stdout land on stderr instead of
    corrupting the protocol stream."""
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    return os.fdopen(protocol_fd, "wb", buffering=0)


def main() -> None:
    protocol_out = _install_stdout_guard()
    _force_fork()
    from lighteval.tasks.tasks.lcb.codegen_metrics import codegen_metrics

    run_worker_loop(sys.stdin.buffer, protocol_out, codegen_metrics)


if __name__ == "__main__":
    main()
