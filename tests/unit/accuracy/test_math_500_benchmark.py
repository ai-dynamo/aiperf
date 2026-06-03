# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``Math500Benchmark`` after lighteval alignment.

The recipe's ``acc_bench_lighteval.py:math_500`` uses ``prompt_fn``
which produces ``Doc(query=line["problem"], choices=[line["solution"]],
gold_index=0)``. Aiperf mirrors this: prompt is bare problem text;
ground_truth is the full solution (containing the boxed answer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.accuracy.benchmarks.math_500 import (
    DEFAULT_GENERATION_SIZE,
    TASK_NAME,
    Math500Benchmark,
)
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


def _make_run() -> BenchmarkRun:
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.MATH_500},
    )


def _make_row(
    problem: str = "What is 1+1?",
    solution: str = "The answer is $\\boxed{2}$.",
    subject: str = "Algebra",
    level: int | None = 1,
) -> dict[str, Any]:
    return {
        "problem": problem,
        "solution": solution,
        "subject": subject,
        "level": level,
    }


def _make_fake_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    ds = MagicMock()
    ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return ds


class TestPromptIsBareProblemText:
    @pytest.mark.asyncio
    async def test_flat_prompt_is_problem_text(self) -> None:
        rows = [_make_row(problem="Find x.")]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].prompt == "Find x."

    @pytest.mark.asyncio
    async def test_no_instruction_prefix(self) -> None:
        rows = [_make_row(problem="Q?")]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        prompt = problems[0].prompt
        assert "Solve the following" not in prompt
        assert "boxed" not in prompt
        assert "Let's think" not in prompt

    @pytest.mark.asyncio
    async def test_chat_message_is_single_user_message(self) -> None:
        rows = [_make_row()]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        msgs = problems[0].raw_messages
        assert msgs is not None
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"


class TestGroundTruthIsFullSolution:
    """Lighteval's ``prompt_fn`` puts ``line["solution"]`` in
    ``choices[0]``; ``latex_gold_metric`` extracts the boxed answer
    from it. Aiperf stores the full solution in
    ``BenchmarkProblem.ground_truth`` so the lighteval grader can do
    the same extraction at grade time."""

    @pytest.mark.asyncio
    async def test_ground_truth_is_full_solution(self) -> None:
        rows = [_make_row(solution="Step one: simplify. Step two: \\boxed{42}.")]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].ground_truth == (
            "Step one: simplify. Step two: \\boxed{42}."
        )


class TestTaskFieldIsSubject:
    """Per-row ``subject`` becomes the ``task`` so the accuracy CSV
    breaks down by MATH subject."""

    @pytest.mark.asyncio
    async def test_subject_used_as_task_name(self) -> None:
        rows = [
            _make_row(subject="Geometry"),
            _make_row(subject="Algebra"),
            _make_row(subject="Number Theory"),
        ]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert {p.task for p in problems} == {
            "Geometry",
            "Algebra",
            "Number Theory",
        }

    @pytest.mark.asyncio
    async def test_missing_subject_falls_back_to_task_name(self) -> None:
        rows = [{"problem": "Q", "solution": "S"}]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].task == TASK_NAME


class TestNShotsAndCoTAreIgnored:
    @pytest.mark.asyncio
    async def test_n_shots_argument_does_not_affect_prompt(self) -> None:
        rows = [_make_row()]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            zero_shot = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
            five_shot = await bench.load_problems(
                tasks=None, n_shots=5, enable_cot=False
            )
        assert zero_shot[0].prompt == five_shot[0].prompt

    @pytest.mark.asyncio
    async def test_enable_cot_does_not_affect_prompt(self) -> None:
        rows = [_make_row()]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            no_cot = await bench.load_problems(tasks=None, n_shots=0, enable_cot=False)
            with_cot = await bench.load_problems(tasks=None, n_shots=0, enable_cot=True)
        assert no_cot[0].prompt == with_cot[0].prompt


class TestLoadProblemsCore:
    @pytest.mark.asyncio
    async def test_returns_one_problem_per_row(self) -> None:
        rows = [_make_row(problem=f"q{i}") for i in range(5)]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert len(problems) == 5
        assert all(isinstance(p, BenchmarkProblem) for p in problems)

    @pytest.mark.asyncio
    async def test_metadata_carries_subject_level_gen_size(self) -> None:
        rows = [_make_row(subject="Geometry", level=4)]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        meta = problems[0].metadata
        assert meta["subject"] == "Geometry"
        assert meta["level"] == 4
        assert meta["generation_size"] == DEFAULT_GENERATION_SIZE
        assert DEFAULT_GENERATION_SIZE == 32768


class TestPathologicalDatasetRows:
    @pytest.mark.asyncio
    async def test_empty_dataset_returns_empty_list(self) -> None:
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset([]),
        ):
            bench = Math500Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems == []

    @pytest.mark.asyncio
    async def test_unicode_in_problem_preserved(self) -> None:
        rows = [_make_row(problem="∫ x dx = ?")]
        with patch(
            "aiperf.accuracy.benchmarks.math_500.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = Math500Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert "∫" in problems[0].prompt
