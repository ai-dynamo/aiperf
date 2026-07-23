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

from collections.abc import Callable
from typing import Any

import orjson  # noqa: F401  # used by run_worker_loop stream loop added in a later task

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
