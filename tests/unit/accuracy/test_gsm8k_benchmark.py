# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``GSM8KBenchmark``.

lighteval's ``gsm8k_leaderboard`` uses ``prompt.gsm8k`` which produces
``Doc(query="Question: {question}\\nAnswer:", choices=[" {answer}"],
gold_index=0)``. Aiperf mirrors this: prompt is the
``"Question: ...\\nAnswer:"`` template; ground_truth is the raw
``answer`` field (which ends in ``#### <number>``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.accuracy.benchmarks.gsm8k import (
    DEFAULT_GENERATION_SIZE,
    TASK_NAME,
    GSM8KBenchmark,
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
        accuracy={"benchmark": AccuracyBenchmarkType.GSM8K},
    )


def _make_row(
    question: str = "Natalia sold clips. How many?",
    answer: str = "She sold 48/2 = <<48/2=24>>24 in May.\n#### 24",
) -> dict[str, Any]:
    return {"question": question, "answer": answer}


def _make_fake_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    ds = MagicMock()
    ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return ds


async def _load(rows: list[dict[str, Any]], **kwargs: Any) -> list[BenchmarkProblem]:
    with patch(
        "aiperf.accuracy.benchmarks.gsm8k.load_dataset",
        return_value=_make_fake_dataset(rows),
    ) as mock_load:
        bench = GSM8KBenchmark(run=_make_run())
        problems = await bench.load_problems(
            tasks=kwargs.get("tasks"),
            n_shots=kwargs.get("n_shots", 0),
            enable_cot=kwargs.get("enable_cot", False),
        )
    return problems, mock_load


class TestPromptFormat:
    @pytest.mark.asyncio
    async def test_prompt_uses_question_answer_template(self) -> None:
        problems, _ = await _load([_make_row(question="What is 1+1?")])
        assert problems[0].prompt == "Question: What is 1+1?\nAnswer:"

    @pytest.mark.asyncio
    async def test_chat_message_is_single_user_message(self) -> None:
        problems, _ = await _load([_make_row()])
        msgs = problems[0].raw_messages
        assert msgs is not None
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == problems[0].prompt


class TestGroundTruthIsRawAnswer:
    @pytest.mark.asyncio
    async def test_ground_truth_keeps_full_answer_with_marker(self) -> None:
        answer = "Steps... <<48/2=24>>24.\n#### 24"
        problems, _ = await _load([_make_row(answer=answer)])
        # Gold is stored raw, NOT pre-extracted — the grader normalizes it.
        assert problems[0].ground_truth == answer
        assert "####" in problems[0].ground_truth


class TestDatasetLoading:
    @pytest.mark.asyncio
    async def test_loads_main_subset_test_split(self) -> None:
        _, mock_load = await _load([_make_row()])
        mock_load.assert_called_once_with("gsm8k", "main", split="test")

    @pytest.mark.asyncio
    async def test_returns_one_problem_per_row(self) -> None:
        rows = [_make_row(question=f"q{i}") for i in range(5)]
        problems, _ = await _load(rows)
        assert len(problems) == 5
        assert all(isinstance(p, BenchmarkProblem) for p in problems)

    @pytest.mark.asyncio
    async def test_task_is_constant_gsm8k(self) -> None:
        problems, _ = await _load([_make_row(), _make_row(question="q2")])
        assert {p.task for p in problems} == {TASK_NAME}

    @pytest.mark.asyncio
    async def test_metadata_carries_generation_size(self) -> None:
        problems, _ = await _load([_make_row()])
        assert problems[0].metadata["generation_size"] == DEFAULT_GENERATION_SIZE
        assert DEFAULT_GENERATION_SIZE == 256


class TestNShotsAndCoTIgnored:
    @pytest.mark.asyncio
    async def test_n_shots_does_not_affect_prompt(self) -> None:
        zero_shot, _ = await _load([_make_row()], n_shots=0)
        eight_shot, _ = await _load([_make_row()], n_shots=8)
        assert zero_shot[0].prompt == eight_shot[0].prompt

    @pytest.mark.asyncio
    async def test_enable_cot_does_not_affect_prompt(self) -> None:
        no_cot, _ = await _load([_make_row()], enable_cot=False)
        with_cot, _ = await _load([_make_row()], enable_cot=True)
        assert no_cot[0].prompt == with_cot[0].prompt


class TestPathologicalRows:
    @pytest.mark.asyncio
    async def test_empty_dataset_returns_empty_list(self) -> None:
        problems, _ = await _load([])
        assert problems == []

    @pytest.mark.asyncio
    async def test_unicode_in_question_preserved(self) -> None:
        problems, _ = await _load([_make_row(question="½ of 8 is what?")])
        assert "½" in problems[0].prompt
