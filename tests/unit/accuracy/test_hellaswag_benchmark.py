# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``HellaSwagBenchmark`` after DeepEval alignment.

Pins:
1. Prompt is byte-equal to ``deepeval.benchmarks.HellaSwag``'s
   ``HellaSwagTemplate.generate_output`` output.
2. Few-shot draw rule is "one per unique activity_label" (matches
   DeepEval's ``categories_seen`` dedupe loop).
3. Validation split is filtered per task by ``activity_label ==
   task.value``.
4. ``ground_truth`` is a bare ``A``/``B``/``C``/``D`` letter
   (DeepEval's convention for ``Scorer.exact_match_score``).

These tests run against the real ``deepeval`` install (it's in the
``[accuracy]`` extras), so ``HellaSwagTemplate`` is available.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pytest import param

# ``HellaSwagBenchmark`` calls into ``deepeval.benchmarks.HellaSwag``'s
# bundled prompt template; without the ``[accuracy]`` extras installed,
# the constructor raises ``RuntimeError`` and every test in this file
# would fail. Skip the whole module when deepeval is missing so CI
# environments that intentionally don't install the heavy extras still
# pass.
pytest.importorskip(
    "deepeval", reason="HellaSwag tests require the [accuracy] extras (deepeval)"
)

from aiperf.accuracy.benchmarks.hellaswag import (  # noqa: E402
    DEEPEVAL_CONFINEMENT,
    DEFAULT_GENERATION_SIZE,
    DEFAULT_N_SHOTS,
    MAX_N_SHOTS,
    HellaSwagBenchmark,
    _build_unique_activity_label_shots_set,
    _resolve_tasks,
)
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType  # noqa: E402
from tests.unit.conftest import make_benchmark_run  # noqa: E402


def _make_run():
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.HELLASWAG},
    )


def _make_row(
    activity_label: str = "Applying sunscreen",
    ctx: str = "[header] A man is in the bathroom. [step] He",
    endings: list[str] | None = None,
    label: str | int = 0,
) -> dict[str, Any]:
    return {
        "activity_label": activity_label,
        "ctx": ctx,
        "endings": endings
        if endings is not None
        else [
            "applies sunscreen.",
            "watches TV.",
            "starts singing.",
            "cooks breakfast.",
        ],
        "label": label,
    }


def _make_fake_split(rows: list[dict[str, Any]]) -> MagicMock:
    split = MagicMock()
    split.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    split.__len__ = MagicMock(return_value=len(rows))
    split.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return split


def _make_fake_dataset_dict(
    train_rows: list[dict[str, Any]] | None = None,
    validation_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "train": _make_fake_split(train_rows or []),
        "validation": _make_fake_split(validation_rows or []),
    }


class TestDefaultsMatchDeepEval:
    """Defaults mirror ``deepeval.benchmarks.HellaSwag``."""

    def test_default_n_shots_is_10(self) -> None:
        assert DEFAULT_N_SHOTS == 10

    def test_max_n_shots_is_15(self) -> None:
        """DeepEval asserts ``n_shots <= 15``."""
        assert MAX_N_SHOTS == 15

    def test_default_generation_size_is_5(self) -> None:
        """A bare A/B/C/D answer fits in a few tokens."""
        assert DEFAULT_GENERATION_SIZE == 5


class TestUniqueActivityLabelShotsSet:
    """``_build_unique_activity_label_shots_set`` mirrors DeepEval's
    ``shots_dataset`` construction (one row per unique activity_label,
    in first-seen order)."""

    def test_picks_first_row_per_unique_label(self) -> None:
        train = [
            _make_row(activity_label="A", ctx="row0"),
            _make_row(activity_label="A", ctx="row1"),  # duplicate label
            _make_row(activity_label="B", ctx="row2"),
            _make_row(activity_label="A", ctx="row3"),  # duplicate
            _make_row(activity_label="C", ctx="row4"),
        ]
        shots = _build_unique_activity_label_shots_set(train)
        assert [s["ctx"] for s in shots] == ["row0", "row2", "row4"]

    def test_empty_train_returns_empty(self) -> None:
        assert _build_unique_activity_label_shots_set([]) == []


class TestResolveTasks:
    """``_resolve_tasks`` accepts None / 'all' / activity_label values
    / upper-snake-case enum names. Unknowns raise."""

    def test_none_returns_all_tasks(self) -> None:
        result = _resolve_tasks(None)
        # DeepEval's HellaSwagTask has ~190 entries.
        assert len(result) > 100

    def test_all_returns_all_tasks(self) -> None:
        result = _resolve_tasks(["all"])
        assert len(result) > 100

    @pytest.mark.parametrize(
        "name",
        [
            param("all", id="lowercase"),
            param("ALL", id="uppercase"),
            param("All", id="titlecase"),
            param("aLl", id="mixed_case"),
        ],
    )  # fmt: skip
    def test_all_alone_is_case_insensitive(self, name: str) -> None:
        """Any casing of the bare ``"all"`` sentinel selects every task."""
        result = _resolve_tasks([name])
        assert len(result) > 100

    @pytest.mark.parametrize(
        "tasks",
        [
            param(["all", "NOT_A_REAL_TASK"], id="all_with_typo"),
            param(["ALL", "Applying sunscreen"], id="all_with_real_task"),
            param(["Applying sunscreen", "all"], id="all_after_other"),
            param(["all", "all"], id="duplicate_all"),
        ],
    )  # fmt: skip
    def test_all_mixed_with_other_selectors_raises(self, tasks: list[str]) -> None:
        """``"all"`` mixed with any other selector raises rather than
        silently returning every task. Pin the regression: previously
        ``["all", "NOT_A_REAL_TASK"]`` returned all 192 tasks and the
        typo was swallowed."""
        with pytest.raises(ValueError, match="'all' cannot be mixed"):
            _resolve_tasks(tasks)

    def test_activity_label_value_resolves(self) -> None:
        result = _resolve_tasks(["Applying sunscreen"])
        assert len(result) == 1
        assert result[0].value == "Applying sunscreen"

    def test_upper_snake_case_enum_name_resolves(self) -> None:
        result = _resolve_tasks(["APPLYING_SUNSCREEN"])
        assert len(result) == 1
        assert result[0].value == "Applying sunscreen"

    @pytest.mark.parametrize(
        "name",
        [
            param("applying sunscreen", id="lowercase"),
            param("APPLYING SUNSCREEN", id="uppercase_with_space"),
            param("ApPlYiNg SuNsCrEeN", id="mixed_case"),
            param("Applying sunscreen", id="exact_case_baseline"),
        ],
    )  # fmt: skip
    def test_activity_label_value_case_insensitive(self, name: str) -> None:
        """The docstring promises case-insensitive activity_label
        matching. The previous implementation used ``name in
        valid_values`` (set membership), which is strictly
        case-sensitive and rejected ``"applying sunscreen"``. Pin that
        all reasonable casings resolve to the same enum member."""
        result = _resolve_tasks([name])
        assert len(result) == 1
        assert result[0].value == "Applying sunscreen"

    def test_unknown_task_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown HellaSwag task"):
            _resolve_tasks(["NOT_A_REAL_TASK"])


class TestTaskValidationPrecedesDatasetDownload:
    """Pin that an invalid ``--accuracy-tasks`` value fails BEFORE the
    HuggingFace dataset is fetched.

    Previously ``load_problems`` called ``load_dataset()`` before
    ``_resolve_tasks(tasks)``, so a typo in ``--accuracy-tasks`` would
    trigger a multi-MB HellaSwag download (and could fail on a
    network/cache error) just to surface the validation error.
    """

    @pytest.mark.asyncio
    async def test_unknown_task_does_not_call_load_dataset(self) -> None:
        with patch("aiperf.accuracy.benchmarks.hellaswag.load_dataset") as mock_load:
            bench = HellaSwagBenchmark(run=_make_run())
            with pytest.raises(ValueError, match="Unknown HellaSwag task"):
                await bench.load_problems(
                    tasks=["NOT_A_REAL_TASK"], n_shots=0, enable_cot=False
                )
            mock_load.assert_not_called()


class TestPromptByteEqualWithDeepEval:
    """The flat prompt must be byte-equal to what
    ``HellaSwagTemplate.generate_output`` produces — same template, same
    input, same shots, same n_shots."""

    @pytest.mark.asyncio
    async def test_zero_shot_prompt_starts_with_template_header(self) -> None:
        rows = [_make_row(activity_label="Applying sunscreen", label=0)]
        ds = _make_fake_dataset_dict(train_rows=rows, validation_rows=rows)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Applying sunscreen"], n_shots=0, enable_cot=False
            )
        prompt = problems[0].prompt
        # DeepEval's verbatim opening (note the awkward "questions
        # (with answers) are sentence completion" grammar).
        assert prompt.startswith(
            "The following are multiple choice questions (with answers) "
            "are sentence completion problems about Applying sunscreen.\n\n"
        )

    @pytest.mark.asyncio
    async def test_question_format_matches_deepeval(self) -> None:
        rows = [
            _make_row(
                activity_label="Applying sunscreen",
                ctx="A man is in the bathroom. He",
                endings=["applies", "watches", "sings", "cooks"],
                label=0,
            )
        ]
        ds = _make_fake_dataset_dict(train_rows=rows, validation_rows=rows)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Applying sunscreen"], n_shots=0, enable_cot=False
            )
        prompt = problems[0].prompt
        # Per DeepEval, the question block formats endings on
        # labelled lines: ``ctx`` then ``A.``, ``B.``, ``C.``, ``D.``,
        # each preceded by a literal newline, then ``Answer:``. The
        # D-line half is built via ``+`` rather than a single string
        # literal so codespell 2.2's dictionary doesn't false-positive
        # on the (newline + D) substring as a typo for "and" (ruff
        # format collapses adjacent literals but leaves binary ``+``
        # intact).
        expected_endings_block = (
            "A. applies\nB. watches\nC. sings\n" + "D. cooks\nAnswer:"
        )
        assert "A man is in the bathroom. He\n" in prompt
        assert expected_endings_block in prompt

    @pytest.mark.asyncio
    async def test_few_shots_drawn_from_train_with_one_per_label(self) -> None:
        train = [
            _make_row(activity_label="Applying sunscreen", ctx="train_AS_0"),
            _make_row(activity_label="Applying sunscreen", ctx="train_AS_1"),
            _make_row(activity_label="Sailing", ctx="train_SAIL_0"),
        ]
        val = [_make_row(activity_label="Applying sunscreen", ctx="VAL_0")]
        ds = _make_fake_dataset_dict(train_rows=train, validation_rows=val)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Applying sunscreen"], n_shots=2, enable_cot=False
            )
        prompt = problems[0].prompt
        # First shot should be train_AS_0 (first unique AS).
        # Second shot should be train_SAIL_0 (first unique SAIL).
        # train_AS_1 must NOT appear (it's the duplicate).
        assert "train_AS_0" in prompt
        assert "train_SAIL_0" in prompt
        assert "train_AS_1" not in prompt


class TestDeepEvalConfinementInstructionAppended:
    """Pin that aiperf appends DeepEval's default ``confinement_instructions``
    string to the rendered prompt.

    DeepEval's ``HellaSwag.predict()`` falls back to appending
    ``"Output 'A', 'B', 'C', or 'D'. Full answer not needed."`` when the
    model doesn't accept ``model.generate(..., schema=MultipleChoiceSchema)``
    — which is the only path aiperf has against OpenAI-compatible
    endpoints. Without the append, ``ExactMatchGrader`` (which mirrors
    ``Scorer.exact_match_score``) under-grades verbose-but-correct
    responses (e.g. ``"The answer is A."`` vs gold ``"A"``).
    """

    def test_constant_matches_deepeval_default(self) -> None:
        """The constant must byte-match DeepEval's hardcoded default —
        sourced from ``HellaSwag.__init__`` when
        ``confinement_instructions=None``."""
        assert (
            DEEPEVAL_CONFINEMENT
            == "Output 'A', 'B', 'C', or 'D'. Full answer not needed."
        )

    @pytest.mark.asyncio
    async def test_prompt_ends_with_confinement(self) -> None:
        rows = [_make_row(activity_label="Applying sunscreen", label=0)]
        ds = _make_fake_dataset_dict(train_rows=rows, validation_rows=rows)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Applying sunscreen"], n_shots=0, enable_cot=False
            )
        prompt = problems[0].prompt
        # Confinement is appended with a blank line separator after the
        # template's trailing "Answer:" — matches DeepEval's
        # ``prompt += f"\n\n{self.confinement_instructions}"``.
        assert prompt.endswith(f"\n\n{DEEPEVAL_CONFINEMENT}")

    @pytest.mark.asyncio
    async def test_template_output_is_a_prefix_of_prompt(self) -> None:
        """The HellaSwagTemplate output is preserved byte-for-byte as
        the prefix of the final prompt; only the confinement suffix is
        new. Pins parity with DeepEval's ``predict()`` flow:
        ``template.generate_output()`` then ``prompt += confinement``."""
        from deepeval.benchmarks.hellaswag.task import HellaSwagTask
        from deepeval.benchmarks.hellaswag.template import HellaSwagTemplate

        rows = [
            _make_row(
                activity_label="Applying sunscreen",
                ctx="A man is in the bathroom. He",
                endings=["applies", "watches", "sings", "cooks"],
                label=0,
            )
        ]
        ds = _make_fake_dataset_dict(train_rows=rows, validation_rows=rows)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Applying sunscreen"], n_shots=0, enable_cot=False
            )
        prompt = problems[0].prompt
        expected_template = HellaSwagTemplate.generate_output(
            input=HellaSwagTemplate.format_question(rows[0], include_answer=False),
            train_set=rows,
            task=HellaSwagTask.APPLYING_SUNSCREEN,
            n_shots=0,
        )
        assert prompt == f"{expected_template}\n\n{DEEPEVAL_CONFINEMENT}"


class TestGroundTruthIsBareLetter:
    @pytest.mark.asyncio
    async def test_ground_truth_is_letter_from_label(self) -> None:
        rows = [
            _make_row(activity_label="Sailing", label=0),
            _make_row(activity_label="Sailing", label=1),
            _make_row(activity_label="Sailing", label=2),
            _make_row(activity_label="Sailing", label=3),
        ]
        ds = _make_fake_dataset_dict(train_rows=rows, validation_rows=rows)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Sailing"], n_shots=0, enable_cot=False
            )
        assert [p.ground_truth for p in problems] == ["A", "B", "C", "D"]

    @pytest.mark.asyncio
    async def test_string_label_coerced_to_int(self) -> None:
        rows = [_make_row(activity_label="Sailing", label="2")]
        ds = _make_fake_dataset_dict(train_rows=rows, validation_rows=rows)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Sailing"], n_shots=0, enable_cot=False
            )
        assert problems[0].ground_truth == "C"


class TestActivityLabelFiltering:
    """Validation rows whose activity_label doesn't match the
    selected task are excluded — matches DeepEval's
    ``val_set.filter(lambda data: data['activity_label'] == task.value)``.
    """

    @pytest.mark.asyncio
    async def test_validation_rows_filtered_by_activity_label(self) -> None:
        train = [
            _make_row(activity_label="Sailing"),
            _make_row(activity_label="Ballet"),
        ]
        val = [
            _make_row(activity_label="Sailing", ctx="match"),
            _make_row(activity_label="Ballet", ctx="other"),
            _make_row(activity_label="Sailing", ctx="match2"),
        ]
        ds = _make_fake_dataset_dict(train_rows=train, validation_rows=val)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Sailing"], n_shots=0, enable_cot=False
            )
        assert len(problems) == 2
        assert all(p.task == "Sailing" for p in problems)


class TestEnableCotIgnored:
    """DeepEval's HellaSwag has no CoT variant. Aiperf accepts the
    parameter for protocol uniformity but ignores it."""

    @pytest.mark.asyncio
    async def test_enable_cot_does_not_affect_prompt(self) -> None:
        rows = [_make_row(activity_label="Sailing", label=0)]
        ds = _make_fake_dataset_dict(train_rows=rows, validation_rows=rows)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            no_cot = await bench.load_problems(
                tasks=["Sailing"], n_shots=0, enable_cot=False
            )
            with_cot = await bench.load_problems(
                tasks=["Sailing"], n_shots=0, enable_cot=True
            )
        assert no_cot[0].prompt == with_cot[0].prompt


class TestNShotsCap:
    @pytest.mark.asyncio
    async def test_n_shots_above_15_raises(self) -> None:
        bench = HellaSwagBenchmark(run=_make_run())
        with pytest.raises(ValueError, match="at most 15"):
            await bench.load_problems(tasks=None, n_shots=16, enable_cot=False)


class TestPathologicalDatasetRows:
    @pytest.mark.asyncio
    async def test_empty_validation_returns_empty(self) -> None:
        train = [_make_row(activity_label="Sailing")]
        ds = _make_fake_dataset_dict(train_rows=train, validation_rows=[])
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Sailing"], n_shots=0, enable_cot=False
            )
        assert problems == []

    @pytest.mark.asyncio
    async def test_unlabeled_rows_dropped(self) -> None:
        rows = [
            _make_row(activity_label="Sailing", label=0),
            _make_row(activity_label="Sailing", label=""),
            _make_row(activity_label="Sailing", label=None),
            _make_row(activity_label="Sailing", label=2),
        ]
        ds = _make_fake_dataset_dict(train_rows=rows, validation_rows=rows)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Sailing"], n_shots=0, enable_cot=False
            )
        assert len(problems) == 2

    @pytest.mark.asyncio
    async def test_per_problem_chat_message_is_single_user(self) -> None:
        rows = [_make_row(activity_label="Sailing", label=0)]
        ds = _make_fake_dataset_dict(train_rows=rows, validation_rows=rows)
        with patch(
            "aiperf.accuracy.benchmarks.hellaswag.load_dataset",
            return_value=ds,
        ):
            bench = HellaSwagBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=["Sailing"], n_shots=0, enable_cot=False
            )
        msgs = problems[0].raw_messages
        assert msgs is not None
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
