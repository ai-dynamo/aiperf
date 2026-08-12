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


def _make_problem(
    io_pairs: list[tuple[str, str]], solution: str
) -> tuple[list[dict[str, str]], list[list[str]]]:
    sample = [
        {
            "input_output": orjson.dumps(
                {
                    "inputs": [i for i, _ in io_pairs],
                    "outputs": [o for _, o in io_pairs],
                    "fn_name": None,
                }
            ).decode()
        }
    ]
    return sample, [[solution]]


# Four distinct problems with distinct expected verdicts so per-problem
# misalignment (e.g. from a batching bug) can't pass all assertions.
_CONCURRENT_PROBLEMS: list[tuple[list[dict[str, str]], list[list[str]], float]] = [
    (
        *_make_problem(
            [("1 2\n", "3\n"), ("10 20\n", "30\n")],
            "a,b=map(int,input().split());print(a+b)",
        ),
        1.0,
    ),
    (
        *_make_problem([("5\n", "25\n"), ("3\n", "9\n")], "n=int(input());print(n*n)"),
        1.0,
    ),
    (
        *_make_problem([("4\n", "16\n")], "print('wrong')"),  # deliberately wrong
        0.0,
    ),
    (
        *_make_problem([("7\n", "49\n"), ("2\n", "4\n")], "n=int(input());print(n**2)"),
        1.0,
    ),
]


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
    """Concurrent grade_codegen() calls with distinct problems all return the
    correct per-problem verdict.

    Uses four problems with different expected pass@1 values (including one
    deliberately wrong) so per-problem misalignment from a batching bug cannot
    pass all assertions — identical problems would mask misattributed results.
    """
    worker = CodegenGradingWorker()
    try:
        results = await asyncio.gather(
            *[
                worker.grade_codegen(sample, code, timeout=240)
                for sample, code, _ in _CONCURRENT_PROBLEMS
            ]
        )
        assert len(results) == len(_CONCURRENT_PROBLEMS)
        for i, ((_, _, expected), result) in enumerate(
            zip(_CONCURRENT_PROBLEMS, results, strict=True)
        ):
            assert float(result["pass@1"]) == expected, (
                f"problem {i}: expected pass@1={expected}, got {result}"
            )
    finally:
        await worker.aclose()
