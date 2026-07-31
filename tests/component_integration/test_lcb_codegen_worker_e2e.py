# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""End-to-end regression for issue #1145: the real out-of-process worker grades a
correct stdin/stdout LCB solution as pass@1=1.0, driving the actual lighteval
codegen path. Before the fix there was no worker and in-process grading under a
``spawn`` default scored every problem 0.000."""

from __future__ import annotations

import asyncio

import orjson
import pytest

pytestmark = pytest.mark.component_integration

lighteval = pytest.importorskip("lighteval")

from aiperf.accuracy.graders._codegen_worker_client import CodegenGradingWorker


def _sample_and_solution() -> tuple[list[dict[str, str]], list[list[str]]]:
    inputs = ["1 2\n", "10 20\n", "3 4\n"]
    outputs = ["3\n", "30\n", "7\n"]
    sample = [
        {
            "input_output": orjson.dumps(
                {"inputs": inputs, "outputs": outputs, "fn_name": None}
            ).decode()
        }
    ]
    return sample, [["a, b = map(int, input().split())\nprint(a + b)"]]


@pytest.mark.slow
@pytest.mark.asyncio
async def test_worker_grades_correct_stdin_solution() -> None:
    # The worker is a fresh single-threaded interpreter that forces fork
    # internally, so it grades correctly regardless of the parent's start method
    # (the old in-process path scored 0.000 under a spawn default before this fix).
    worker = CodegenGradingWorker()
    sample, code = _sample_and_solution()
    try:
        metrics = await worker.grade_codegen(sample, code, timeout=120)
        assert float(metrics["pass@1"]) == 1.0
    finally:
        await worker.aclose()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_worker_grades_multiple_problems_concurrently() -> None:
    """N concurrent grade_codegen() calls all resolve correctly.

    This exercises the batch-drain path: all N requests are sent before the
    worker responds, so they are drained into a single codegen_metrics call and
    processed in parallel by lighteval's ProcessPoolExecutor.
    """
    worker = CodegenGradingWorker()
    sample, code = _sample_and_solution()
    n = 4
    try:
        results = await asyncio.gather(
            *[worker.grade_codegen(sample, code, timeout=240) for _ in range(n)]
        )
        assert len(results) == n
        assert all(float(r["pass@1"]) == 1.0 for r in results), results
    finally:
        await worker.aclose()
