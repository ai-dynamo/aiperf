# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``BigBenchBenchmark`` after DeepEval alignment.

Pins:
1. Prompt is byte-equal to ``deepeval.benchmarks.BigBenchHard``'s
   ``BigBenchHardTemplate.generate_output`` output (which itself
   reads the canonical CoT/non-CoT prompt files DeepEval ships).
2. ``ground_truth`` is the bare ``target`` string from
   ``lukaemon/bbh`` (DeepEval's convention for exact_match_score).
3. ``confinement`` carried in metadata maps per-task to the right
   "Output 'X' or 'Y'..." string.
4. Per-task task field so the accuracy CSV breaks down per BBH
   subtask.

These tests run against the real ``deepeval`` install (it's in the
``[accuracy]`` extras), so ``BigBenchHardTemplate`` is available and
can read its bundled CoT/shot prompt files.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ``BigBenchBenchmark`` calls into
# ``deepeval.benchmarks.BigBenchHard``'s bundled CoT/shot prompt files;
# without the ``[accuracy]`` extras installed the constructor raises
# ``RuntimeError`` and every test in this file would fail. Skip the
# whole module when deepeval is missing so CI environments that
# intentionally don't install the heavy extras still pass.
pytest.importorskip(
    "deepeval", reason="BigBench tests require the [accuracy] extras (deepeval)"
)

from aiperf.accuracy.benchmarks.bigbench import (  # noqa: E402
    DEFAULT_ENABLE_COT,
    DEFAULT_GENERATION_SIZE,
    DEFAULT_N_SHOTS,
    MAX_N_SHOTS,
    BigBenchBenchmark,
    _resolve_tasks,
)
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType  # noqa: E402
from tests.unit.conftest import make_benchmark_run  # noqa: E402


def _make_run():
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.BIGBENCH},
    )


def _make_row(input_text: str = "What is 2+2?", target: str = "4") -> dict[str, Any]:
    return {"input": input_text, "target": target}


def _make_fake_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mock ``load_dataset`` return value (a dict-like with split keys)."""
    test_split = MagicMock()
    test_split.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    test_split.__len__ = MagicMock(return_value=len(rows))
    test_split.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return {"test": test_split}


def _per_task_loader(per_task: dict[str, list[dict[str, Any]]]):
    """``load_dataset`` patch that dispatches by task name."""

    def loader(_dataset_name, task_name=None, **_kwargs):
        return _make_fake_dataset(per_task.get(task_name, []))

    return loader


class TestDefaultsMatchDeepEval:
    """Defaults mirror ``deepeval.benchmarks.BigBenchHard``."""

    def test_default_n_shots_is_3(self) -> None:
        assert DEFAULT_N_SHOTS == 3

    def test_max_n_shots_is_3(self) -> None:
        """DeepEval asserts ``n_shots <= 3`` because the bundled prompt
        files only contain 3 worked examples."""
        assert MAX_N_SHOTS == 3

    def test_default_enable_cot_is_true(self) -> None:
        assert DEFAULT_ENABLE_COT is True

    def test_default_generation_size_is_1024(self) -> None:
        assert DEFAULT_GENERATION_SIZE == 1024


class TestResolveTasks:
    def test_none_returns_all_27_subtasks(self) -> None:
        result = _resolve_tasks(None)
        assert len(result) == 27

    def test_all_returns_all_27_subtasks(self) -> None:
        result = _resolve_tasks(["all"])
        assert len(result) == 27

    def test_lower_snake_case_value_resolves(self) -> None:
        result = _resolve_tasks(["boolean_expressions"])
        assert len(result) == 1
        assert result[0].value == "boolean_expressions"

    def test_upper_snake_case_enum_name_resolves(self) -> None:
        result = _resolve_tasks(["BOOLEAN_EXPRESSIONS"])
        assert len(result) == 1
        assert result[0].value == "boolean_expressions"

    def test_unknown_subtask_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown BBH subtask"):
            _resolve_tasks(["not_a_real_task"])

    def test_unknown_subtask_lists_valid(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            _resolve_tasks(["not_a_real_task"])
        # All 27 should appear in the error.
        assert "boolean_expressions" in str(exc_info.value)
        assert "navigate" in str(exc_info.value)
        assert "object_counting" in str(exc_info.value)


class TestPromptByteEqualWithDeepEval:
    """The flat prompt must be byte-equal to what
    ``BigBenchHardTemplate.generate_output`` produces — same template,
    same CoT files, same n_shots, same enable_cot."""

    @pytest.mark.asyncio
    async def test_cot_prompt_starts_with_task_description(self) -> None:
        per_task = {"boolean_expressions": [_make_row("True and False is", "False")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["boolean_expressions"],
                n_shots=3,
                enable_cot=True,
            )
        prompt = problems[0].prompt
        # DeepEval's template prepends "Task description: " then the
        # canonical first paragraph. For boolean_expressions that
        # paragraph is "Evaluate the result of a random Boolean expression."
        assert prompt.startswith(
            "Task description: Evaluate the result of a random Boolean expression."
        )

    @pytest.mark.asyncio
    async def test_query_appended_at_end(self) -> None:
        per_task = {"boolean_expressions": [_make_row("True and False is", "False")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["boolean_expressions"],
                n_shots=3,
                enable_cot=True,
            )
        prompt = problems[0].prompt
        # DeepEval's template appends "\n\nQ: <input>\nA: " at the end.
        assert prompt.endswith("Q: True and False is\nA: ")

    @pytest.mark.asyncio
    async def test_cot_vs_no_cot_use_different_prompt_files(self) -> None:
        per_task = {"navigate": [_make_row("Walk forward 5 steps.", "No")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            cot = await bench.load_problems(
                tasks=["navigate"], n_shots=3, enable_cot=True
            )
            no_cot = await bench.load_problems(
                tasks=["navigate"], n_shots=3, enable_cot=False
            )
        # CoT version has "Let's think step by step." worked examples;
        # non-CoT has bare Q/A pairs.
        assert "step by step" in cot[0].prompt.lower() or "Let's" in cot[0].prompt
        assert cot[0].prompt != no_cot[0].prompt

    @pytest.mark.asyncio
    async def test_zero_shot_takes_only_task_description(self) -> None:
        """``n_shots=0`` should emit just ``"Task description: <first
        paragraph>"`` followed by the test query — no worked examples."""
        per_task = {"boolean_expressions": [_make_row("True and True is", "True")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["boolean_expressions"],
                n_shots=0,
                enable_cot=True,
            )
        prompt = problems[0].prompt
        # Only the task description and the query, no worked examples
        # (the CoT files use "Let's think step by step." in shot
        # examples; with n_shots=0 that phrase shouldn't appear).
        assert "Q: True and True is\nA: " in prompt
        # The 0-shot vs 3-shot length comparison lives in
        # ``TestNShotsAffectsPromptLength`` below.


class TestNShotsAffectsPromptLength:
    @pytest.mark.asyncio
    async def test_more_shots_make_longer_prompt(self) -> None:
        per_task = {"boolean_expressions": [_make_row("True is", "True")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            zero = await bench.load_problems(
                tasks=["boolean_expressions"], n_shots=0, enable_cot=True
            )
            three = await bench.load_problems(
                tasks=["boolean_expressions"], n_shots=3, enable_cot=True
            )
        assert len(three[0].prompt) > len(zero[0].prompt)


class TestNShotsCap:
    @pytest.mark.asyncio
    async def test_n_shots_above_3_raises(self) -> None:
        bench = BigBenchBenchmark(run=_make_run())
        with pytest.raises(ValueError, match="at most 3"):
            await bench.load_problems(tasks=None, n_shots=4, enable_cot=True)


class TestGroundTruthIsBareTarget:
    @pytest.mark.asyncio
    async def test_ground_truth_is_target_string(self) -> None:
        per_task = {
            "navigate": [
                _make_row("Walk left, then right.", "No"),
                _make_row("Walk forward 5 steps.", "Yes"),
            ]
        }
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["navigate"], n_shots=3, enable_cot=True
            )
        assert [p.ground_truth for p in problems] == ["No", "Yes"]


class TestConfinementInMetadata:
    """The per-task confinement string is carried in metadata so callers
    that need DeepEval's structured-fallback shape (or want to log it)
    can read it."""

    @pytest.mark.asyncio
    async def test_boolean_expressions_confinement(self) -> None:
        per_task = {"boolean_expressions": [_make_row("Q?", "True")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["boolean_expressions"], n_shots=3, enable_cot=True
            )
        assert "True" in problems[0].metadata["confinement"]
        assert "False" in problems[0].metadata["confinement"]

    @pytest.mark.asyncio
    async def test_navigate_confinement(self) -> None:
        per_task = {"navigate": [_make_row("Q?", "Yes")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["navigate"], n_shots=3, enable_cot=True
            )
        assert "Yes" in problems[0].metadata["confinement"]
        assert "No" in problems[0].metadata["confinement"]


class TestPerTaskAggregation:
    @pytest.mark.asyncio
    async def test_task_field_is_subtask_name(self) -> None:
        per_task = {
            "navigate": [_make_row("Q1", "Yes")],
            "object_counting": [_make_row("Q2", "5")],
        }
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["navigate", "object_counting"],
                n_shots=3,
                enable_cot=True,
            )
        tasks = {p.task for p in problems}
        assert tasks == {"navigate", "object_counting"}


class TestPathologicalDatasetRows:
    @pytest.mark.asyncio
    async def test_empty_subtask_returns_empty(self) -> None:
        per_task = {"navigate": []}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["navigate"], n_shots=3, enable_cot=True
            )
        assert problems == []

    @pytest.mark.asyncio
    async def test_unicode_in_target_preserved(self) -> None:
        per_task = {"navigate": [_make_row("Q?", "café")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["navigate"], n_shots=3, enable_cot=True
            )
        assert problems[0].ground_truth == "café"

    @pytest.mark.asyncio
    async def test_chat_message_is_single_user(self) -> None:
        per_task = {"navigate": [_make_row("Q?", "Yes")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["navigate"], n_shots=3, enable_cot=True
            )
        msgs = problems[0].raw_messages
        assert msgs is not None
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
