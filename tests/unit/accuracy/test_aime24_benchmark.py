# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``AIME24Benchmark`` after lighteval alignment.

The recipe's ``acc_bench_lighteval.py`` uses ``aime_prompt_fn`` which
makes ``Doc(query=line["problem"], choices=[line["answer"]],
gold_index=0)`` — lighteval's prompt manager then renders the bare
problem text as a single user message with no instruction prefix and
no few-shot priming (``few_shots_split=None``). These tests pin that
behavior so any future divergence fails loudly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.accuracy.benchmarks.aime24 import (
    DEFAULT_GENERATION_SIZE,
    TASK_NAME,
    AIME24Benchmark,
)
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_run():
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.AIME24},
    )


def _make_row(problem: str = "What is 1+1?", answer: int = 2) -> dict[str, Any]:
    return {"problem": problem, "answer": answer}


def _make_fake_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    ds = MagicMock()
    ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return ds


class TestPromptIsBareProblemText:
    """Lighteval's ``aime_prompt_fn`` + ``PromptManager`` produces the
    bare problem text as the user message — no instruction prefix, no
    decoration."""

    @pytest.mark.asyncio
    async def test_flat_prompt_is_problem_text(self) -> None:
        rows = [_make_row("Compute the answer.", 42)]
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIME24Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].prompt == "Compute the answer."

    @pytest.mark.asyncio
    async def test_no_instruction_prefix(self) -> None:
        """Any of the instruction phrasings used by the old or new
        AIME loader should NOT be in an AIME24 prompt."""
        rows = [_make_row("Q?", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIME24Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        prompt = problems[0].prompt
        assert "**Problem**" not in prompt
        assert "competition math" not in prompt
        assert "Let's think" not in prompt
        assert "boxed" not in prompt

    @pytest.mark.asyncio
    async def test_chat_message_is_single_user_message(self) -> None:
        rows = [_make_row("Q?", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIME24Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        msgs = problems[0].raw_messages
        assert msgs is not None
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Q?"


class TestNShotsAndCoTAreIgnored:
    """The lighteval reference is zero-shot, no CoT trigger. Aiperf
    keeps the parameters in the signature for protocol uniformity but
    must not let them change the prompt — that would diverge from the
    reference."""

    @pytest.mark.asyncio
    async def test_n_shots_argument_does_not_affect_prompt(self) -> None:
        rows = [_make_row(f"q{i}", i) for i in range(3)]
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIME24Benchmark(run=_make_run())
            zero_shot = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
            five_shot = await bench.load_problems(
                tasks=None, n_shots=5, enable_cot=False
            )
        assert [p.prompt for p in zero_shot] == [p.prompt for p in five_shot]

    @pytest.mark.asyncio
    async def test_enable_cot_does_not_affect_prompt(self) -> None:
        rows = [_make_row("Q?", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIME24Benchmark(run=_make_run())
            no_cot = await bench.load_problems(tasks=None, n_shots=0, enable_cot=False)
            with_cot = await bench.load_problems(tasks=None, n_shots=0, enable_cot=True)
        assert no_cot[0].prompt == with_cot[0].prompt


class TestLoadProblemsCore:
    @pytest.mark.asyncio
    async def test_returns_one_problem_per_row(self) -> None:
        rows = [_make_row(f"q{i}", i) for i in range(5)]
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIME24Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert len(problems) == 5
        assert all(isinstance(p, BenchmarkProblem) for p in problems)

    @pytest.mark.asyncio
    async def test_ground_truth_is_string_form_of_answer(self) -> None:
        rows = [_make_row("q", 42)]
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIME24Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].ground_truth == "42"

    @pytest.mark.asyncio
    async def test_task_name_is_aime24(self) -> None:
        rows = [_make_row("q", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIME24Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].task == TASK_NAME

    @pytest.mark.asyncio
    async def test_generation_size_is_32k(self) -> None:
        """Lighteval's aime24 task config sets ``generation_size=32768``."""
        rows = [_make_row("q", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIME24Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].metadata["generation_size"] == DEFAULT_GENERATION_SIZE
        assert DEFAULT_GENERATION_SIZE == 32768


class TestPathologicalDatasetRows:
    @pytest.mark.asyncio
    async def test_empty_dataset_returns_empty_list(self) -> None:
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset([]),
        ):
            bench = AIME24Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems == []

    @pytest.mark.asyncio
    async def test_unicode_problem_text_preserved(self) -> None:
        rows = [_make_row("Solve ∑₁ⁿ k² for n=10. ✓", 385)]
        with patch(
            "aiperf.accuracy.benchmarks.aime24.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIME24Benchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert "∑₁ⁿ" in problems[0].prompt
