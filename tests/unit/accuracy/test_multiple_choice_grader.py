# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from aiperf.accuracy.graders.multiple_choice import MultipleChoiceGrader
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_grader() -> MultipleChoiceGrader:
    return MultipleChoiceGrader(
        run=make_benchmark_run(
            model_names=["test-model"],
            endpoint_type=EndpointType.COMPLETIONS,
            streaming=False,
            accuracy={"benchmark": AccuracyBenchmarkType.MMLU},
        )
    )


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

    async def test_answer_is_sentence_is_clean(self) -> None:
        # "The answer is (X)" is the requested CoT format -> clean parse.
        result = await _make_grader().grade("The answer is B.", "B")
        assert result.correct
        assert not result.unparsed
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

    async def test_answer_is_wrong_letter(self) -> None:
        result = await _make_grader().grade("The answer is B.", "A")
        assert not result.correct
        assert not result.unparsed
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


@pytest.mark.asyncio
class TestMultipleChoiceCoTFallback:
    async def test_cot_reasoning_then_answer_is_x(self) -> None:
        text = "We eliminate options one by one. The answer is (C)."
        result = await _make_grader().grade(text, "C")
        assert result.correct
        assert not result.unparsed  # "answer is (X)" is a clean parse
        assert result.extracted_answer == "C"

    async def test_reasoning_plus_content_first_line_not_letter(self) -> None:
        # Simulates reasoning+content concatenation defeating first-line-only.
        text = "We need to answer the question.\nThe answer is (A)."
        result = await _make_grader().grade(text, "A")
        assert result.correct
        assert result.extracted_answer == "A"

    async def test_bare_letter_first_line_still_wins_unchanged(self) -> None:
        # Parity: a clean first-line letter is parsed WITHOUT the fallback.
        result = await _make_grader().grade("B\n\nQuestion: unrelated D E F", "B")
        assert result.correct
        assert not result.unparsed
        assert result.extracted_answer == "B"

    async def test_answer_colon_beats_echoed_first_line_option(self) -> None:
        # Reasoning echoes the option list on the first line, then ends with
        # "Answer: B". Tier-2 must win over the echoed first-line "A".
        text = "Options: A. 0, B. 4, C. 2, D. 6. We compute.\nAnswer: B"
        result = await _make_grader().grade(text, "B")
        assert result.correct
        assert result.extracted_answer == "B"
