# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.accuracy.graders.multiple_choice import MultipleChoiceGrader
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType


def _make_grader() -> MultipleChoiceGrader:
    user_config = UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.COMPLETIONS,
            streaming=False,
        ),
        accuracy=AccuracyConfig(benchmark=AccuracyBenchmarkType.MMLU),
    )
    return MultipleChoiceGrader(user_config=user_config)


@pytest.mark.asyncio
class TestMultipleChoiceGraderGrade:
    async def test_correct_exact_match(self) -> None:
        result = await _make_grader().grade("A", "A")
        assert result.correct
        assert not result.unparsed
        assert result.confidence == 1.0
        assert result.extracted_answer == "A"
        assert result.ground_truth == "A"

    async def test_incorrect_wrong_answer(self) -> None:
        result = await _make_grader().grade("B", "A")
        assert not result.correct
        assert not result.unparsed
        assert result.confidence == 0.0

    async def test_strips_whitespace_from_prediction(self) -> None:
        result = await _make_grader().grade("  A  ", "A")
        assert result.correct
        assert not result.unparsed

    async def test_strips_whitespace_from_ground_truth(self) -> None:
        result = await _make_grader().grade("A", " A ")
        assert result.correct

    async def test_takes_only_first_line(self) -> None:
        result = await _make_grader().grade("A\nsome other text", "A")
        assert result.correct
        assert not result.unparsed
        assert result.extracted_answer == "A"

    async def test_empty_prediction_is_incorrect(self) -> None:
        result = await _make_grader().grade("", "A")
        assert not result.correct
        assert result.unparsed

    async def test_whitespace_only_prediction_is_incorrect(self) -> None:
        result = await _make_grader().grade("   ", "A")
        assert not result.correct
        assert result.unparsed

    async def test_case_sensitive_match(self) -> None:
        result = await _make_grader().grade("a", "A")
        assert not result.correct
        assert result.unparsed

    async def test_regex_fallback_sentence(self) -> None:
        result = await _make_grader().grade("The answer is B.", "B")
        assert result.correct
        assert result.unparsed
        assert result.extracted_answer == "B"

    async def test_regex_fallback_bold_markdown(self) -> None:
        result = await _make_grader().grade("**C**", "C")
        assert result.correct
        assert result.unparsed
        assert result.extracted_answer == "C"

    async def test_regex_fallback_parentheses(self) -> None:
        result = await _make_grader().grade("(D)", "D")
        assert result.correct
        assert result.unparsed
        assert result.extracted_answer == "D"

    async def test_regex_fallback_wrong_letter(self) -> None:
        result = await _make_grader().grade("The answer is B.", "A")
        assert not result.correct
        assert result.unparsed
        assert result.extracted_answer == "B"

    async def test_no_regex_match_is_unparsed(self) -> None:
        result = await _make_grader().grade("I don't know", "A")
        assert not result.correct
        assert result.unparsed


class TestMultipleChoiceGraderExtractAnswer:
    def test_plain_answer(self) -> None:
        assert _make_grader().extract_answer("B") == "B"

    def test_strips_surrounding_whitespace(self) -> None:
        assert _make_grader().extract_answer("  C  ") == "C"

    def test_takes_first_line_only(self) -> None:
        assert _make_grader().extract_answer("D\nQuestion: ...") == "D"

    def test_strips_after_newline_split(self) -> None:
        assert _make_grader().extract_answer("  A \nignored") == "A"

    def test_regex_fallback_sentence(self) -> None:
        assert _make_grader().extract_answer("The answer is B.") == "B"

    def test_regex_fallback_bold(self) -> None:
        assert _make_grader().extract_answer("**C**") == "C"

    def test_regex_fallback_parentheses(self) -> None:
        assert _make_grader().extract_answer("(D)") == "D"

    def test_no_match_returns_stripped_first_line(self) -> None:
        assert _make_grader().extract_answer("I don't know") == "I don't know"
