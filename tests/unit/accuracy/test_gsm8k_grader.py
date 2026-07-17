# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``LightevalGSM8KGrader``.

Covers the gold-side ``gsm8k_normalizer`` (the number after ``####``)
and the prediction-side extraction that deviates from lighteval to
handle chat-model outputs that lack the ``####`` marker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pytest import param

from aiperf.accuracy.graders.gsm8k_grader import (
    LightevalGSM8KGrader,
    _numbers_match,
    gsm8k_normalizer,
)
from tests.unit.conftest import make_benchmark_run

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

_RAW_GOLD = "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n#### 24"


def _make_run() -> BenchmarkRun:
    return make_benchmark_run(model_names=["test-model"])


def _grader() -> LightevalGSM8KGrader:
    return LightevalGSM8KGrader(run=_make_run())


class TestGsm8kNormalizer:
    @pytest.mark.parametrize(
        "text,expected",
        [
            param("Solution.\n#### 24", "24", id="simple"),
            param("#### -5", "-5", id="negative"),
            param("#### 1,234", "1234", id="strips-commas"),
            param("#### 18.5", "18.5", id="decimal"),
            param("no marker here", "[invalid]", id="no-marker"),
            param("", "[invalid]", id="empty"),
        ],
    )  # fmt: skip
    def test_extracts_number_after_marker(self, text: str, expected: str) -> None:
        assert gsm8k_normalizer(text) == expected


class TestGradingCorrect:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response,extracted,unparsed",
        [
            # Only a response carrying the #### marker matches the expected
            # format; everything else is a correct-but-fallback extraction
            # and is flagged unparsed (still scored correct).
            param("#### 24", "24", False, id="model-emits-marker"),
            param("The answer is 24.", "24", True, id="trailing-sentence"),
            param("... so 48/2 = 24", "24", True, id="last-number"),
            param("24", "24", True, id="bare-number"),
            param("The total is 24.0", "24.0", True, id="decimal-equiv"),
            param("That gives 1,024 - 1000 = 24 clips", "24", True, id="commas-and-last"),
        ],
    )  # fmt: skip
    async def test_correct_predictions_match_gold(
        self, response: str, extracted: str, unparsed: bool
    ) -> None:
        result = await _grader().grade(response, _RAW_GOLD)
        assert result.correct is True
        assert result.unparsed is unparsed
        assert result.extracted_answer == extracted
        assert result.ground_truth == "24"

    @pytest.mark.asyncio
    async def test_marker_prediction_matches_lighteval_exactly(self) -> None:
        # When the model emits ####, prediction extraction is byte-for-byte
        # lighteval (gsm8k_normalizer on both sides) and not flagged unparsed.
        result = await _grader().grade("Work...\n#### 1,024", "gold\n#### 1024")
        assert result.correct is True
        assert result.unparsed is False


class TestGradingIncorrect:
    @pytest.mark.asyncio
    async def test_wrong_number_is_incorrect(self) -> None:
        result = await _grader().grade("The answer is 25.", _RAW_GOLD)
        assert result.correct is False
        # No #### marker -> last-number fallback -> flagged unparsed.
        assert result.unparsed is True
        assert result.extracted_answer == "25"

    @pytest.mark.asyncio
    async def test_wrong_number_with_marker_is_not_unparsed(self) -> None:
        # A wrong answer that still follows the #### format is parseable.
        result = await _grader().grade("Work...\n#### 25", _RAW_GOLD)
        assert result.correct is False
        assert result.unparsed is False
        assert result.extracted_answer == "25"

    @pytest.mark.asyncio
    async def test_no_number_in_response_is_unparsed(self) -> None:
        result = await _grader().grade("I am not sure.", _RAW_GOLD)
        assert result.correct is False
        assert result.unparsed is True
        assert result.extracted_answer == "[invalid]"

    @pytest.mark.asyncio
    async def test_empty_response_is_unparsed(self) -> None:
        result = await _grader().grade("", _RAW_GOLD)
        assert result.correct is False
        assert result.unparsed is True

    @pytest.mark.asyncio
    async def test_invalid_gold_never_matches(self) -> None:
        # Gold without a #### marker normalizes to "[invalid]"; a valid
        # prediction must not be scored correct against it.
        result = await _grader().grade("The answer is 24.", "no marker")
        assert result.correct is False
        assert result.ground_truth == "[invalid]"


class TestNumbersMatch:
    @pytest.mark.parametrize(
        "gold,pred,expected",
        [
            param("24", "24", True, id="int-equal"),
            param("24", "24.0", True, id="int-vs-decimal"),
            param("24", "24.00", True, id="trailing-zeros"),
            param("24", "25", False, id="int-unequal"),
            param("-7", "-7.0", True, id="negative-equal"),
            # Non-numeric inputs fall back to exact string equality.
            param("[invalid]", "[invalid]", True, id="nonnumeric-equal"),
            param("[invalid]", "24", False, id="nonnumeric-vs-number"),
        ],
    )  # fmt: skip
    def test_numeric_then_string_fallback(
        self, gold: str, pred: str, expected: bool
    ) -> None:
        assert _numbers_match(gold, pred) is expected


class TestExtractAnswer:
    def test_prefers_marker_over_last_number(self) -> None:
        # Even with a later number, the #### marker wins (lighteval parity).
        assert _grader().extract_answer("foo 99 bar\n#### 24 then 7") == "24"

    def test_falls_back_to_last_number(self) -> None:
        assert _grader().extract_answer("steps 3, 4, then 12") == "12"

    def test_strips_commas(self) -> None:
        assert _grader().extract_answer("the result is 1,234,567") == "1234567"

    def test_no_number_returns_invalid(self) -> None:
        assert _grader().extract_answer("none") == "[invalid]"
