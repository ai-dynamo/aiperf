# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``ExactMatchGrader`` after DeepEval alignment.

The grader is now strict — ``pred.strip() == gold.strip()``, no other
normalization — to match ``Scorer.exact_match_score`` from
``deepeval.scorer``. These tests pin every reproducible aspect of
that contract.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.accuracy.graders.exact_match import ExactMatchGrader
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_run():
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.HELLASWAG},
    )


@pytest.fixture
def grader() -> ExactMatchGrader:
    return ExactMatchGrader(run=_make_run())


class TestStrictEquality:
    """``pred.strip() == gold.strip()``, byte-for-byte case-sensitive."""

    @pytest.mark.parametrize(
        "pred,gold,expected_correct",
        [
            param("A", "A", True, id="bare-letter-match"),
            param(" A ", "A", True, id="prediction-whitespace-stripped"),
            param("A", " A ", True, id="gold-whitespace-stripped"),
            param("A", "B", False, id="mismatch"),
            param("a", "A", False, id="case-sensitive-lower-vs-upper"),
            param("A.", "A", False, id="trailing-period-NOT-forgiven"),
            param('"A"', "A", False, id="surrounding-quotes-NOT-stripped"),
            param("yes", "Yes", False, id="case-mismatch-not-equal"),
            param("Yes", "Yes", True, id="case-exact-match"),
        ],
    )  # fmt: skip
    @pytest.mark.asyncio
    async def test_strict_equality_cases(
        self,
        grader: ExactMatchGrader,
        pred: str,
        gold: str,
        expected_correct: bool,
    ) -> None:
        result = await grader.grade(pred, gold)
        assert result.correct is expected_correct
        assert result.confidence == (1.0 if expected_correct else 0.0)


class TestEmptyAndUnparsed:
    @pytest.mark.asyncio
    async def test_empty_response_unparsed_and_incorrect(
        self, grader: ExactMatchGrader
    ) -> None:
        result = await grader.grade("", "A")
        assert result.correct is False
        assert result.unparsed is True

    @pytest.mark.asyncio
    async def test_whitespace_only_response_unparsed(
        self, grader: ExactMatchGrader
    ) -> None:
        result = await grader.grade("   \n\t  ", "A")
        assert result.correct is False
        assert result.unparsed is True

    @pytest.mark.asyncio
    async def test_empty_pred_and_empty_gold_neither_correct_nor_unparsed(
        self, grader: ExactMatchGrader
    ) -> None:
        """If both are empty, gold is meaningless — not unparsed."""
        result = await grader.grade("", "")
        assert result.correct is False
        assert result.unparsed is False


class TestMultiLineNotForgiven:
    """DeepEval doesn't take "first non-empty line" — it strips the
    full response. So multi-line responses fail unless the entire
    stripped content matches the gold."""

    @pytest.mark.asyncio
    async def test_multi_line_response_does_not_match_single_letter(
        self, grader: ExactMatchGrader
    ) -> None:
        result = await grader.grade("A\nbecause...", "A")
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_explanation_prefix_does_not_match(
        self, grader: ExactMatchGrader
    ) -> None:
        result = await grader.grade("The answer is A.", "A")
        assert result.correct is False


class TestUnicodeAndNonAscii:
    @pytest.mark.asyncio
    async def test_unicode_match(self, grader: ExactMatchGrader) -> None:
        result = await grader.grade("café", "café")
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_unicode_case_sensitive(self, grader: ExactMatchGrader) -> None:
        result = await grader.grade("Café", "café")
        assert result.correct is False


class TestExtractAnswerInterface:
    def test_extract_answer_strips_only(self, grader: ExactMatchGrader) -> None:
        assert grader.extract_answer("  A  ") == "A"

    def test_extract_answer_preserves_inner(self, grader: ExactMatchGrader) -> None:
        """No first-line / no quote-strip / no punct-strip — unlike
        the previous over-engineered ExactMatchGrader."""
        assert grader.extract_answer("hello world") == "hello world"

    def test_extract_answer_empty(self, grader: ExactMatchGrader) -> None:
        assert grader.extract_answer("") == ""
        assert grader.extract_answer("   ") == ""


class TestGradingResultFields:
    @pytest.mark.asyncio
    async def test_reasoning_includes_stripped_forms(
        self, grader: ExactMatchGrader
    ) -> None:
        result = await grader.grade("  A  ", "A")
        assert "stripped pred 'A'" in result.reasoning
        assert "gold 'A'" in result.reasoning

    @pytest.mark.asyncio
    async def test_extracted_answer_is_stripped_pred(
        self, grader: ExactMatchGrader
    ) -> None:
        result = await grader.grade("  Yes  ", "Yes")
        assert result.extracted_answer == "Yes"
