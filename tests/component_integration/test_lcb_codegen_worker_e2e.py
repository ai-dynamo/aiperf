# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""End-to-end regression for issue #1145: the real out-of-process worker grades a
correct stdin/stdout LCB solution as pass@1=1.0 even when the parent's default
multiprocessing start method is spawn (the macOS default and the exact condition
that scored every problem 0.000 in-process)."""

from __future__ import annotations

import json
import multiprocessing as mp

import pytest

pytestmark = pytest.mark.component_integration

lighteval = pytest.importorskip("lighteval")

from aiperf.accuracy.graders._codegen_worker_client import CodegenGradingWorker


def _sample_and_solution() -> tuple[list[dict[str, str]], list[list[str]]]:
    inputs = ["1 2\n", "10 20\n", "3 4\n"]
    outputs = ["3\n", "30\n", "7\n"]
    sample = [
        {
            "input_output": json.dumps(
                {"inputs": inputs, "outputs": outputs, "fn_name": None}
            )
        }
    ]
    return sample, [["a, b = map(int, input().split())\nprint(a + b)"]]


@pytest.mark.slow
@pytest.mark.asyncio
async def test_worker_grades_correct_stdin_solution_under_spawn_default() -> None:
    # The worker is spawned as a fresh interpreter; even if this parent is on the
    # spawn default, the worker forces fork internally, so grading must be 1.0.
    original = mp.get_start_method(allow_none=True)
    try:
        if "spawn" in mp.get_all_start_methods():
            mp.set_start_method("spawn", force=True)
        worker = CodegenGradingWorker()
        sample, code = _sample_and_solution()
        try:
            metrics = await worker.grade_codegen(sample, code, timeout=120)
            assert float(metrics["pass@1"]) == 1.0
        finally:
            await worker.aclose()
    finally:
        if original is not None:
            mp.set_start_method(original, force=True)
