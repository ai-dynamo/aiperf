# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``AIMEBenchmark`` after trt-llm reference alignment.

The expected prompt strings in this suite are byte-equal to the output
of the recipe's ``AIMETemplate.generate_output``
(``trt-llm-benchmark-recipe/src/accuracy/aime/template.py``) — any
divergence between aiperf and the reference would be caught here.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.accuracy.benchmarks.aime import (
    COT_SUFFIX,
    DEFAULT_GENERATION_SIZE,
    DEFAULT_N_SHOTS,
    FEW_SHOT_HEADER,
    MAX_N_SHOTS,
    NO_COT_SUFFIX,
    TASK_NAME,
    AIMEBenchmark,
)
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType


def _make_user_config() -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.COMPLETIONS,
            streaming=False,
        ),
        accuracy=AccuracyConfig(benchmark=AccuracyBenchmarkType.AIME),
    )


def _make_row(
    problem: str = "What is 1+1?",
    answer: int = 2,
    solution: str = "Add the numbers.",
) -> dict[str, Any]:
    return {"Problem": problem, "Answer": answer, "Solution": solution}


def _make_fake_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    ds = MagicMock()
    ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return ds


@pytest.fixture
def bench() -> AIMEBenchmark:
    return AIMEBenchmark(user_config=_make_user_config())


class TestDefaults:
    """Defaults must mirror the trt-llm recipe."""

    def test_n_shots_default_is_8(self) -> None:
        assert DEFAULT_N_SHOTS == 8

    def test_max_n_shots_is_8(self) -> None:
        """Recipe asserts ``n_shots <= 8`` — we mirror that cap."""
        assert MAX_N_SHOTS == 8

    def test_generation_size_default_is_32k(self) -> None:
        """Recipe runs unbounded; lighteval-aligned tasks use 32768."""
        assert DEFAULT_GENERATION_SIZE == 32768


class TestFormatPrompt:
    """Flat prompt must be byte-equal to recipe ``AIMETemplate.generate_output``."""

    def test_zero_shot_no_cot_matches_recipe(self, bench: AIMEBenchmark) -> None:
        """Zero shots, CoT off — recipe emits no header, no examples,
        and the no-CoT suffix after **Answer**:."""
        row = _make_row(problem="Compute 2+2.", answer=4)
        prompt = bench._format_prompt(row, few_shots=[], enable_cot=False)
        expected = (
            "**Problem**: Compute 2+2.\n"
            "**Answer**: \n"
            "\n"
            "No explanation needed. Just return a number."
        )
        assert prompt == expected

    def test_zero_shot_with_cot_matches_recipe(self, bench: AIMEBenchmark) -> None:
        row = _make_row(problem="Compute 2+2.", answer=4)
        prompt = bench._format_prompt(row, few_shots=[], enable_cot=True)
        expected = (
            "**Problem**: Compute 2+2.\n**Answer**: \n\nLet's think step-by-step."
        )
        assert prompt == expected

    def test_one_shot_with_cot_includes_solution(self, bench: AIMEBenchmark) -> None:
        """Recipe includes **Solution**: in few-shot blocks ONLY when CoT is on."""
        shot = bench._format_example(
            _make_row(problem="What is 1+1?", answer=2, solution="Trivial.")
        )
        row = _make_row(problem="What is 2+2?", answer=4)
        prompt = bench._format_prompt(row, few_shots=[shot], enable_cot=True)
        expected = (
            FEW_SHOT_HEADER
            + "**Problem**: What is 1+1?\n"
            + "**Solution**: Trivial.\n"
            + "**Answer**: 2\n"
            + "\n"
            + "**Problem**: What is 2+2?\n"
            + "**Answer**: \n"
            + "\n"
            + COT_SUFFIX
        )
        assert prompt == expected

    def test_one_shot_no_cot_omits_solution(self, bench: AIMEBenchmark) -> None:
        shot = bench._format_example(
            _make_row(problem="What is 1+1?", answer=2, solution="Trivial.")
        )
        row = _make_row(problem="What is 2+2?", answer=4)
        prompt = bench._format_prompt(row, few_shots=[shot], enable_cot=False)
        # No **Solution**: line in few-shot block
        assert "**Solution**:" not in prompt
        # No-CoT suffix at the end
        assert prompt.endswith(NO_COT_SUFFIX)

    def test_zero_shot_emits_no_header(self, bench: AIMEBenchmark) -> None:
        """When n_shots=0 the recipe emits no header at all (the
        ``FEW_SHOT_HEADER`` only appears for n_shots > 0)."""
        prompt = bench._format_prompt(_make_row(), few_shots=[], enable_cot=False)
        assert FEW_SHOT_HEADER not in prompt

    def test_few_shot_header_present_when_shots_present(
        self, bench: AIMEBenchmark
    ) -> None:
        shot = bench._format_example(_make_row())
        prompt = bench._format_prompt(_make_row(), few_shots=[shot], enable_cot=False)
        assert prompt.startswith(FEW_SHOT_HEADER)


class TestFormatExample:
    def test_collects_problem_solution_answer(self, bench: AIMEBenchmark) -> None:
        ex = bench._format_example(_make_row(problem="Q?", answer=42, solution="S"))
        assert ex["problem"] == "Q?"
        assert ex["solution"] == "S"
        assert ex["answer"] == "42"
        assert isinstance(ex["answer"], str)

    def test_missing_solution_field_handled(self, bench: AIMEBenchmark) -> None:
        """Some upstream rows omit Solution; we tolerate that with empty."""
        row = {"Problem": "Q?", "Answer": 1}
        ex = bench._format_example(row)
        assert ex["solution"] == ""


class TestBuildChatMessages:
    """Chat-message form is the recipe's flat prompt wrapped in one user message.

    The trt-llm recipe sends AIME prompts as a single string to DeepEval
    (no multi-turn conversation), so our chat representation does the
    same: one user message containing the recipe-rendered prompt.
    """

    def test_zero_shot_zero_cot_is_single_user_message(
        self, bench: AIMEBenchmark
    ) -> None:
        msgs = bench._build_chat_messages(_make_row(), few_shots=[], enable_cot=False)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_chat_content_equals_flat_prompt(self, bench: AIMEBenchmark) -> None:
        row = _make_row(problem="Sample.", answer=7)
        flat = bench._format_prompt(row, few_shots=[], enable_cot=True)
        msgs = bench._build_chat_messages(row, few_shots=[], enable_cot=True)
        assert msgs[0]["content"] == flat


class TestBuildFewShots:
    def test_zero_shots_returns_empty(self, bench: AIMEBenchmark) -> None:
        ds = _make_fake_dataset([_make_row("a", 1), _make_row("b", 2)])
        assert bench._build_few_shots(ds, n_shots=0) == []

    def test_negative_shots_returns_empty(self, bench: AIMEBenchmark) -> None:
        ds = _make_fake_dataset([_make_row("a", 1)])
        assert bench._build_few_shots(ds, n_shots=-3) == []

    def test_n_shots_clamped_to_dataset_size(self, bench: AIMEBenchmark) -> None:
        ds = _make_fake_dataset([_make_row("only", 1)])
        shots = bench._build_few_shots(ds, n_shots=5)
        assert len(shots) == 1

    def test_shots_drawn_from_start_in_order(self, bench: AIMEBenchmark) -> None:
        rows = [_make_row(f"problem-{i}", i) for i in range(10)]
        ds = _make_fake_dataset(rows)
        shots = bench._build_few_shots(ds, n_shots=3)
        assert [s["problem"] for s in shots] == [
            "problem-0",
            "problem-1",
            "problem-2",
        ]


class TestLoadProblems:
    @pytest.mark.asyncio
    async def test_returns_one_problem_per_row(self) -> None:
        rows = [_make_row(f"q{i}", i) for i in range(5)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert len(problems) == 5
        assert all(isinstance(p, BenchmarkProblem) for p in problems)

    @pytest.mark.asyncio
    async def test_ground_truth_is_string_form_of_integer_answer(self) -> None:
        rows = [_make_row("q", 42)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].ground_truth == "42"
        assert isinstance(problems[0].ground_truth, str)

    @pytest.mark.asyncio
    async def test_task_name_is_aime(self) -> None:
        rows = [_make_row("q", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].task == TASK_NAME

    @pytest.mark.asyncio
    async def test_metadata_carries_default_generation_size(self) -> None:
        rows = [_make_row("q", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].metadata["generation_size"] == DEFAULT_GENERATION_SIZE

    @pytest.mark.asyncio
    async def test_raw_messages_populated(self) -> None:
        rows = [_make_row("q", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].raw_messages is not None
        assert len(problems[0].raw_messages) == 1
        assert problems[0].raw_messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_max_n_shots_enforced(self) -> None:
        """The recipe asserts ``n_shots <= 8``; we raise ``ValueError``
        rather than silently accepting more."""
        bench = AIMEBenchmark(user_config=_make_user_config())
        with pytest.raises(ValueError, match="at most 8"):
            await bench.load_problems(tasks=None, n_shots=9, enable_cot=False)

    @pytest.mark.asyncio
    async def test_tasks_argument_is_ignored(self) -> None:
        rows = [_make_row("a", 1), _make_row("b", 2)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            none_problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
            named_problems = await bench.load_problems(
                tasks=["aime"], n_shots=0, enable_cot=False
            )
        assert len(none_problems) == len(named_problems) == 2

    @pytest.mark.asyncio
    async def test_cot_propagates_to_every_problem(self) -> None:
        rows = [_make_row(f"q{i}", i) for i in range(3)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(tasks=None, n_shots=0, enable_cot=True)
        # Every prompt ends with the CoT suffix.
        assert all(p.prompt.endswith(COT_SUFFIX) for p in problems)


class TestPathologicalDatasetRows:
    @pytest.mark.asyncio
    async def test_empty_dataset_returns_empty_list(self) -> None:
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset([]),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems == []

    @pytest.mark.asyncio
    async def test_unicode_problem_text_preserved(self) -> None:
        rows = [_make_row("Solve ∑₁ⁿ k² for n=10. ✓", 385)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert "∑₁ⁿ" in problems[0].prompt

    @pytest.mark.asyncio
    async def test_very_long_problem_text_does_not_crash(self) -> None:
        long_problem = "Q. " + ("blah " * 50_000) + "Find x."
        rows = [_make_row(long_problem, 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert len(problems) == 1
        assert long_problem in problems[0].prompt

    @pytest.mark.asyncio
    async def test_zero_padded_three_digit_answer_stringifies_cleanly(
        self,
    ) -> None:
        rows = [_make_row("q", 7)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].ground_truth == "7"
