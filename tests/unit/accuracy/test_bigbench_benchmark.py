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

Most tests in this file run against ``tests.harness.fake_deepeval`` — a
small stand-in that mirrors the 27-task enum and confinement dict
exactly but generates a synthetic (non-byte-equal) prompt template.
Tests that pin the real upstream prompt bytes are marked
``@pytest.mark.requires_deepeval`` and skip when only the fake is
registered. The fake is wired in ``tests/unit/accuracy/conftest.py``
(autouse, function scope) so the ``aiperf[accuracy]`` extras are no
longer a hard prerequisite for running this file.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.accuracy.benchmarks.bigbench import (
    DEFAULT_ENABLE_COT,
    DEFAULT_GENERATION_SIZE,
    DEFAULT_N_SHOTS,
    MAX_N_SHOTS,
    BigBenchBenchmark,
    _resolve_tasks,
)
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


def _make_run() -> BenchmarkRun:
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


def _per_task_loader(
    per_task: dict[str, list[dict[str, Any]]],
) -> Callable[..., dict[str, Any]]:
    """``load_dataset`` patch that dispatches by task name."""

    def loader(
        _dataset_name: str,
        task_name: str | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        return _make_fake_dataset(
            per_task.get(task_name, []) if task_name is not None else []
        )

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


@pytest.mark.requires_deepeval
class TestPromptByteEqualWithDeepEval:
    """The flat prompt must be byte-equal to what
    ``BigBenchHardTemplate.generate_output`` produces — same template,
    same CoT files, same n_shots, same enable_cot.

    These assertions read specific strings out of DeepEval's bundled CoT
    and shot prompt ``.txt`` files (e.g. ``"Task description: Evaluate
    the result of a random Boolean expression."``). The fake harness
    cannot reproduce those bytes, so the class is tagged
    ``requires_deepeval``; the marker skips it when only the fake is
    registered (i.e. when the ``[accuracy]`` extras aren't installed).
    """

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
    async def test_query_appended_before_confinement(self) -> None:
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
        # DeepEval's template appends "\n\nQ: <input>\nA: " at the end of
        # its output. The loader then appends the per-task confinement
        # statement so the LLM sees the constraint as part of the prompt
        # (matches the trt-llm benchmark recipe's flow). For
        # boolean_expressions that confinement starts with "\n\nOutput
        # 'True' or 'False'." so the Q/A pair sits immediately before it.
        assert "Q: True and False is\nA: \n\nOutput 'True' or 'False'." in prompt
        assert prompt.endswith("Full answer not needed.")

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


class TestResolveTasksAdversarial:
    """Edge cases on ``--accuracy-tasks`` parsing not covered by
    ``TestResolveTasks``."""

    def test_empty_list_returns_all_27_subtasks(self) -> None:
        """A bare ``--accuracy-tasks`` with no values reaches the resolver
        as ``[]`` (falsy) — equivalent to ``None`` / ``["all"]``."""
        assert len(_resolve_tasks([])) == 27

    def test_mixed_case_all_returns_all_27_subtasks(self) -> None:
        """``"All"`` / ``"ALL"`` must match case-insensitively. The
        docstring promises this; pin it so a future case-sensitive
        refactor breaks the test loudly."""
        assert len(_resolve_tasks(["All"])) == 27
        assert len(_resolve_tasks(["ALL"])) == 27

    def test_all_mixed_with_typo_raises(self) -> None:
        """``["all", "NOT_A_REAL_TASK"]`` used to silently return every
        subtask and swallow the typo (the parallel HellaSwag bug AIP-877
        fixed). Must now raise so a user typo fails loudly instead of
        running the whole 27-task benchmark."""
        with pytest.raises(ValueError, match="'all' cannot be mixed"):
            _resolve_tasks(["all", "not_a_real_task"])

    def test_all_mixed_with_valid_name_also_raises(self) -> None:
        """Even when both names would individually be accepted, mixing
        ``"all"`` with anything else is ambiguous and must fail."""
        with pytest.raises(ValueError, match="'all' cannot be mixed"):
            _resolve_tasks(["all", "navigate"])

    def test_whitespace_in_task_name_raises(self) -> None:
        """A whitespace-bearing name is not silently trimmed — pin the
        loud-failure mode so accidental YAML spacing is caught."""
        with pytest.raises(ValueError, match="Unknown BBH subtask"):
            _resolve_tasks([" boolean_expressions "])

    def test_hyphenated_task_name_raises(self) -> None:
        """Hyphens aren't normalized. ``"boolean-expressions"`` upper-
        cases to ``"BOOLEAN-EXPRESSIONS"`` which is not a valid enum
        attribute, so the resolver raises."""
        with pytest.raises(ValueError, match="Unknown BBH subtask"):
            _resolve_tasks(["boolean-expressions"])

    def test_mixed_valid_and_invalid_lists_only_invalid(self) -> None:
        """When some names resolve and others don't, the unknown-list
        portion of the error must contain only the unknown name — no
        false positive on the valid one."""
        with pytest.raises(ValueError) as exc_info:
            _resolve_tasks(["navigate", "not_a_real"])
        msg = str(exc_info.value)
        assert "not_a_real" in msg
        # The error also lists the full valid set after "Valid subtasks:"
        # for guidance, so narrow the check to the unknown-list portion.
        unknown_portion = msg.split("Valid subtasks:")[0]
        assert "'navigate'" not in unknown_portion

    def test_duplicate_task_names_resolve_to_duplicate_enums(self) -> None:
        """The resolver does not deduplicate. Passing the same task
        twice yields two entries and will trigger ``load_dataset``
        twice for the same subtask — pin the behavior so callers know
        the cost."""
        result = _resolve_tasks(["navigate", "navigate"])
        assert len(result) == 2
        assert result[0] is result[1]


class TestConstructorWithoutDeepEval:
    """The constructor refuses to build when the ``[accuracy]`` extras
    aren't available — otherwise downstream ``BigBenchHardTemplate``
    calls would crash with an unhelpful ``NameError``."""

    def test_missing_deepeval_raises_with_install_hint(self) -> None:
        with (
            patch("aiperf.accuracy.benchmarks.bigbench._HAS_DEEPEVAL", False),
            pytest.raises(RuntimeError, match=r"aiperf\[accuracy\]"),
        ):
            BigBenchBenchmark(run=_make_run())


class TestOutputInvariants:
    """Per-problem fields that should always agree do agree."""

    @pytest.mark.asyncio
    async def test_prompt_equals_first_chat_message_content(self) -> None:
        """``prompt`` (the flat completions string) and the lone chat
        message's ``content`` must be byte-equal — drift here would
        mean completions vs chat endpoints render different prompts
        for the same problem."""
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
        assert problems[0].prompt == msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_metadata_bbh_task_matches_task_field(self) -> None:
        """``problem.task`` and ``problem.metadata['bbh_task']`` must
        match for every problem — the accuracy CSV reads the former,
        downstream tooling the latter, and both refer to the same
        subtask."""
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
        for p in problems:
            assert p.task == p.metadata["bbh_task"]

    @pytest.mark.asyncio
    async def test_generation_size_is_plumbed_through_metadata(self) -> None:
        """``DEFAULT_GENERATION_SIZE=1024`` is carried in per-problem
        metadata so request-level overrides can read it without
        round-tripping the module constant."""
        per_task = {"navigate": [_make_row("Q?", "Yes")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["navigate"], n_shots=3, enable_cot=True
            )
        assert problems[0].metadata["generation_size"] == DEFAULT_GENERATION_SIZE

    @pytest.mark.asyncio
    async def test_multitask_order_preserves_task_input_order(self) -> None:
        """When ``tasks=[A, B]``, every problem for A precedes every
        problem for B in the output list. The accuracy CSV's per-task
        grouping depends on this contiguity."""
        per_task = {
            "navigate": [_make_row("nav-q", "Yes")],
            "object_counting": [
                _make_row("oc-q1", "1"),
                _make_row("oc-q2", "2"),
            ],
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
        assert [p.task for p in problems] == [
            "navigate",
            "object_counting",
            "object_counting",
        ]


class TestLoadDatasetInvocation:
    """Pin the ``load_dataset(DATASET_NAME, task.value)`` call shape so
    a rename of ``DATASET_NAME`` or an accidental kwarg/positional
    reorder is caught."""

    @pytest.mark.asyncio
    async def test_load_dataset_called_once_per_task_with_canonical_args(
        self,
    ) -> None:
        per_task = {
            "navigate": [_make_row("Q1", "Yes")],
            "object_counting": [_make_row("Q2", "1")],
        }
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ) as mock_load:
            bench = BigBenchBenchmark(run=_make_run())
            await bench.load_problems(
                tasks=["navigate", "object_counting"],
                n_shots=3,
                enable_cot=True,
            )
        # Two requested tasks → exactly two load_dataset calls, each
        # with the canonical dataset name positional and the subtask
        # value positional. Asserting via call_args_list catches both a
        # rename of DATASET_NAME and a future swap to kwargs.
        assert [c.args for c in mock_load.call_args_list] == [
            ("lukaemon/bbh", "navigate"),
            ("lukaemon/bbh", "object_counting"),
        ]


class TestPathologicalRowContent:
    """Hostile row content the upstream dataset could theoretically
    ship."""

    @pytest.mark.asyncio
    async def test_empty_input_string_still_renders_prompt(self) -> None:
        """A blank ``input`` is unusual but shouldn't crash — DeepEval's
        template just appends it verbatim. Pin the passthrough so we
        notice if DeepEval ever rejects empty inputs."""
        per_task = {"navigate": [_make_row(input_text="", target="Yes")]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["navigate"], n_shots=3, enable_cot=True
            )
        # Prompt still rendered — the task description and shot
        # examples are always present even when the input is empty.
        assert len(problems) == 1
        assert len(problems[0].prompt) > 0

    @pytest.mark.asyncio
    async def test_numeric_target_coerced_to_string(self) -> None:
        """A numeric ``target`` (e.g. an int from a future BBH schema
        change) is coerced to ``str`` by the loader before constructing
        the ``BenchmarkProblem``. ``BenchmarkProblem.ground_truth`` is
        in strict mode, so the loader's defensive ``str(...)`` is the
        contract callers rely on for string equality in graders."""
        per_task = {"object_counting": [{"input": "Count items", "target": 42}]}
        with patch(
            "aiperf.accuracy.benchmarks.bigbench.load_dataset",
            side_effect=_per_task_loader(per_task),
        ):
            bench = BigBenchBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["object_counting"], n_shots=3, enable_cot=True
            )
        assert problems[0].ground_truth == "42"
