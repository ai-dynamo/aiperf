# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``MathGrader``.

Coverage targets the three layers separately:

1. Pure helpers (``_extract_last_boxed``, ``_extract_last_number``,
   ``_normalize``, ``_to_fraction``) — exhaustive parametrized cases.
2. ``MathGrader._extract_with_flag`` — extraction priority and the
   ``unparsed`` flag semantics.
3. ``MathGrader.grade`` — end-to-end correctness, including adversarial
   responses that try to fool the grader (decoy numbers, multiple boxes,
   "the answer is X — just kidding, it's Y" patterns, unicode, empty,
   whitespace-only).
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.accuracy.graders.math import (
    MathGrader,
    _extract_last_boxed,
    _extract_last_number,
    _normalize,
    _to_fraction,
)
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_run():
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.AIME},
    )


@pytest.fixture
def grader() -> MathGrader:
    return MathGrader(run=_make_run())


class TestExtractLastBoxed:
    """``_extract_last_boxed`` — balanced-brace LaTeX extraction."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            param("\\boxed{42}", "42", id="simple-int"),
            param("Some prose. \\boxed{42}", "42", id="prose-prefix"),
            param("\\boxed{42} and more", "42", id="prose-suffix"),
            param("\\boxed{1}\\boxed{2}\\boxed{3}", "3", id="multiple-takes-last"),
            param(
                "\\boxed{\\frac{1}{2}}",
                "\\frac{1}{2}",
                id="nested-frac",
            ),
            param(
                "\\boxed{a\\cdot b^{n+1}}",
                "a\\cdot b^{n+1}",
                id="nested-superscript",
            ),
            param("\\boxed{}", "", id="empty-box"),
            param("no box here", None, id="no-box"),
            param("", None, id="empty-input"),
        ],
    )  # fmt: skip
    def test_extract_last_boxed(self, text: str, expected: str | None) -> None:
        assert _extract_last_boxed(text) == expected

    def test_extract_last_boxed_unbalanced_returns_none(self) -> None:
        assert _extract_last_boxed("\\boxed{42") is None
        assert _extract_last_boxed("\\boxed{\\frac{1}{2}") is None

    def test_extract_last_boxed_takes_truly_last_with_dec(self) -> None:
        text = "First try \\boxed{99}, but actually the answer is \\boxed{42}."
        assert _extract_last_boxed(text) == "42"


class TestExtractLastNumber:
    """``_extract_last_number`` — last numeric literal in text."""

    @pytest.mark.parametrize(
        "text,expected",
        [
            param("42", "42", id="bare-int"),
            param("the answer is 42", "42", id="trailing-int"),
            param("3.14 then 2.71", "2.71", id="multiple-decimals"),
            param("1/2", "1/2", id="ratio"),
            param("-7 and -3", "-3", id="negatives"),
            param("price: $99.99 plus tax", "99.99", id="dollar-prefix"),
            param("answer is 42.", "42", id="trailing-period-not-decimal"),
            param("no numbers here", None, id="no-numbers"),
            param("", None, id="empty"),
        ],
    )  # fmt: skip
    def test_extract_last_number(self, text: str, expected: str | None) -> None:
        assert _extract_last_number(text) == expected


class TestNormalize:
    """``_normalize`` — pre-comparison string canonicalization."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            param("42", "42", id="int-passthrough"),
            param(" 42 ", "42", id="whitespace-stripped"),
            param("$42$", "42", id="dollar-strip"),
            param("$\\boxed{42}$", "\\boxed{42}", id="dollars-around-boxed"),
            param("\\dfrac{1}{2}", "(1)/(2)", id="dfrac-collapsed"),
            param("\\tfrac{1}{2}", "(1)/(2)", id="tfrac-collapsed"),
            param("\\frac{3}{4}", "(3)/(4)", id="frac-expanded"),
            param("\\left( 1 \\right)", "(1)", id="left-right-stripped"),
            param("\\text{42}", "42", id="text-wrapper-stripped"),
            param("\\mathrm{abc}", "abc", id="mathrm-wrapper-stripped"),
            param("42.", "42", id="trailing-period"),
            param("42,", "42", id="trailing-comma"),
            param("1 / 2", "1/2", id="interior-whitespace"),
            param("X", "X", id="case-preserved"),
            param("", "", id="empty"),
        ],
    )  # fmt: skip
    def test_normalize(self, raw: str, expected: str) -> None:
        assert _normalize(raw) == expected

    def test_normalize_idempotent(self) -> None:
        once = _normalize("$\\dfrac{1}{2}$.")
        twice = _normalize(once)
        assert once == twice


class TestToFraction:
    """``_to_fraction`` — numeric-literal parsing."""

    @pytest.mark.parametrize(
        "raw,expected_numerator,expected_denominator",
        [
            param("42", 42, 1, id="int"),
            param("-7", -7, 1, id="negative-int"),
            param("0", 0, 1, id="zero"),
            param("3.14", 157, 50, id="decimal"),  # 314/100 reduces to 157/50
            param("1/2", 1, 2, id="ratio"),
            param("-3/4", -3, 4, id="negative-ratio"),
            param("(1)/(2)", 1, 2, id="parenthesized-ratio"),
        ],
    )  # fmt: skip
    def test_to_fraction_parses(
        self, raw: str, expected_numerator: int, expected_denominator: int
    ) -> None:
        result = _to_fraction(raw)
        assert result is not None
        assert result.numerator == expected_numerator
        assert result.denominator == expected_denominator

    @pytest.mark.parametrize(
        "raw",
        [
            param("", id="empty"),
            param("abc", id="alpha"),
            param("x+1", id="expression"),
            param("1/0", id="div-by-zero"),
            param("\\sqrt{2}", id="latex-command"),
        ],
    )  # fmt: skip
    def test_to_fraction_rejects(self, raw: str) -> None:
        assert _to_fraction(raw) is None


class TestExtractWithFlag:
    """Extraction priority and ``unparsed`` semantics."""

    def test_boxed_is_primary(self, grader: MathGrader) -> None:
        answer, unparsed = grader._extract_with_flag("Reasoning... \\boxed{42}")
        assert answer == "42"
        assert unparsed is False

    def test_boxed_overrides_decoy_numbers(self, grader: MathGrader) -> None:
        answer, unparsed = grader._extract_with_flag(
            "I considered 99 and 11, but \\boxed{42}"
        )
        assert answer == "42"
        assert unparsed is False

    def test_phrase_fallback_when_no_box(self, grader: MathGrader) -> None:
        answer, unparsed = grader._extract_with_flag(
            "After careful thought, the answer is 42."
        )
        assert answer == "42"
        assert unparsed is True

    def test_phrase_recurses_into_boxed(self, grader: MathGrader) -> None:
        answer, unparsed = grader._extract_with_flag("the answer is \\boxed{42}")
        assert answer == "42"
        assert unparsed is False

    def test_last_number_fallback(self, grader: MathGrader) -> None:
        answer, unparsed = grader._extract_with_flag("I think it might be 99 or 42")
        assert answer == "42"
        assert unparsed is True

    def test_empty_response_marked_unparsed(self, grader: MathGrader) -> None:
        answer, unparsed = grader._extract_with_flag("")
        assert answer == ""
        assert unparsed is True

    def test_whitespace_only_response_marked_unparsed(self, grader: MathGrader) -> None:
        _, unparsed = grader._extract_with_flag("   \n\t  ")
        assert unparsed is True

    def test_no_numbers_falls_through_to_raw(self, grader: MathGrader) -> None:
        answer, unparsed = grader._extract_with_flag("I don't know.")
        assert unparsed is True
        assert "don't know" in answer

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            param("the answer is 3.14", "3.14", id="pi-like_decimal"),
            param("final answer: 0.5", "0.5", id="leading_zero_decimal"),
            param("Answer = 100.001", "100.001", id="multi_digit_decimal"),
            param("the answer is -2.5", "-2.5", id="negative_decimal"),
        ],
    )  # fmt: skip
    def test_phrase_fallback_preserves_decimal(
        self, grader: MathGrader, response: str, expected: str
    ) -> None:
        """The answer-phrase regex used to terminate on ``.`` and silently
        truncate decimals to their integer part. Pin that the full decimal
        survives so MATH-500-style numeric answers grade correctly."""
        answer, unparsed = grader._extract_with_flag(response)
        assert answer == expected
        assert unparsed is True

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            param(
                "the answer is 5. Wait, the answer is 12",
                "12",
                id="reflection_inline",
            ),
            param(
                "My first thought: the answer is 7. After reflection, the answer is 9.",
                "9",
                id="reflection_paragraph",
            ),
            param(
                "the answer is 5\nthe answer is 12",
                "12",
                id="reflection_newline",
            ),
            param(
                "Final answer: 5. Hmm, actually final answer: 12.",
                "12",
                id="self_correct_final_answer",
            ),
        ],
    )  # fmt: skip
    def test_phrase_fallback_takes_last_match(
        self, grader: MathGrader, response: str, expected: str
    ) -> None:
        """Reasoning models often self-correct mid-response. The
        answer-phrase regex used to take the first match and grade
        against the abandoned guess; pin that the LAST claim wins."""
        answer, _ = grader._extract_with_flag(response)
        assert answer == expected

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            param(
                "the answer is 3.14\nand more context follows",
                "3.14",
                id="decimal_then_newline",
            ),
            param(
                "the answer is X. The rest is unrelated.",
                "X",
                id="non_numeric_then_sentence",
            ),
            param(
                "the answer is 5.\nMore reasoning",
                "5",
                id="trailing_period_then_newline",
            ),
        ],
    )  # fmt: skip
    def test_phrase_fallback_terminator_handles_decimals_newlines_and_sentences(
        self, grader: MathGrader, response: str, expected: str
    ) -> None:
        """Pin the three-way terminator behaviour:

        * a period *not* surrounded by digits ends the tail (so a
          trailing sentence doesn't pollute the captured answer),
        * a ``\\n`` ends the tail (so the regex doesn't slurp the rest
          of a multi-line response),
        * end-of-string is the final fallback.
        """
        answer, _ = grader._extract_with_flag(response)
        assert answer == expected

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            param("The answer is \\frac{1}{2}.", "\\frac{1}{2}", id="frac_latex"),
            param("the answer is \\sqrt{2}", "\\sqrt{2}", id="sqrt_latex"),
            param("final answer: \\pi", "\\pi", id="bare_latex_command"),
            param("The answer is {1,2,3}", "{1,2,3}", id="braced_set"),
            param("the answer is \\binom{n}{2}", "\\binom{n}{2}", id="binom_latex"),
        ],
    )  # fmt: skip
    def test_phrase_fallback_preserves_latex_tail(
        self, grader: MathGrader, response: str, expected: str
    ) -> None:
        """A tail containing a backslash or curly brace must be preserved
        whole so ``strip_string`` + ``math_equal`` can normalize and
        compare it. The previous implementation ran the last-number
        extractor on the tail and silently graded ``"\\frac{1}{2}"`` as
        ``"2"`` — the opposite of what the equivalent ``\\boxed{...}``
        wrapping does."""
        answer, _ = grader._extract_with_flag(response)
        assert answer == expected

    @pytest.mark.asyncio
    async def test_unboxed_frac_grades_same_as_boxed_frac(
        self, grader: MathGrader
    ) -> None:
        """The reviewer's specific regression: ``"The answer is \\frac{1}{2}"``
        and ``"\\boxed{\\frac{1}{2}}"`` should grade identically against
        gold ``"1/2"``. Previously the unboxed form extracted ``"2"`` and
        graded as incorrect."""
        unboxed = await grader.grade("The answer is \\frac{1}{2}", "1/2")
        boxed = await grader.grade("\\boxed{\\frac{1}{2}}", "1/2")
        assert unboxed.correct is True
        assert boxed.correct is True
        assert unboxed.correct == boxed.correct


class TestGradeNumeric:
    """``MathGrader.grade`` — numeric equivalence."""

    @pytest.mark.asyncio
    async def test_boxed_int_match(self, grader: MathGrader) -> None:
        result = await grader.grade("\\boxed{42}", "42")
        assert result.correct is True
        assert result.unparsed is False
        assert result.confidence == 1.0
        assert result.extracted_answer == "42"

    @pytest.mark.asyncio
    async def test_boxed_int_mismatch(self, grader: MathGrader) -> None:
        result = await grader.grade("\\boxed{99}", "42")
        assert result.correct is False
        assert result.unparsed is False
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_decimal_matches_fraction(self, grader: MathGrader) -> None:
        result = await grader.grade("\\boxed{0.5}", "1/2")
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_dfrac_matches_decimal(self, grader: MathGrader) -> None:
        result = await grader.grade("\\boxed{\\dfrac{1}{2}}", "0.5")
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_negative_match(self, grader: MathGrader) -> None:
        result = await grader.grade("\\boxed{-7}", "-7")
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_zero_match(self, grader: MathGrader) -> None:
        result = await grader.grade("\\boxed{0}", "0")
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_dollar_wrapped_match(self, grader: MathGrader) -> None:
        result = await grader.grade("$\\boxed{42}$", "42")
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_left_right_stripped_match(self, grader: MathGrader) -> None:
        result = await grader.grade("\\boxed{\\left(\\dfrac{1}{2}\\right)}", "1/2")
        assert result.correct is True


class TestGradeAdversarial:
    """Pathological inputs that should NOT confuse the grader."""

    @pytest.mark.asyncio
    async def test_decoy_number_before_box(self, grader: MathGrader) -> None:
        """Earlier numbers must not override a later \\boxed{}."""
        response = "Initially I thought 99, but recomputing gives \\boxed{42}."
        result = await grader.grade(response, "42")
        assert result.correct is True
        assert result.unparsed is False

    @pytest.mark.asyncio
    async def test_multiple_boxes_takes_last(self, grader: MathGrader) -> None:
        """When the model emits multiple boxes, the last one is final."""
        response = "Try \\boxed{99}. Wait, actually \\boxed{42}."
        result = await grader.grade(response, "42")
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_correct_phrase_but_wrong_box(self, grader: MathGrader) -> None:
        """If both phrase and box are present, box wins (not phrase tail)."""
        response = "the answer is 42, oops I mean \\boxed{99}"
        result = await grader.grade(response, "42")
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_adjacent_text_does_not_pollute(self, grader: MathGrader) -> None:
        result = await grader.grade(
            "Therefore, after computing, \\boxed{42} is the answer.", "42"
        )
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_unicode_response_does_not_crash(self, grader: MathGrader) -> None:
        result = await grader.grade("∴ \\boxed{42} ✓", "42")
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_very_long_response(self, grader: MathGrader) -> None:
        """A long response should still extract the trailing box."""
        prefix = "Step. " * 5000
        result = await grader.grade(prefix + "\\boxed{42}", "42")
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_empty_response_is_incorrect_unparsed(
        self, grader: MathGrader
    ) -> None:
        result = await grader.grade("", "42")
        assert result.correct is False
        assert result.unparsed is True

    @pytest.mark.asyncio
    async def test_whitespace_only_response_is_incorrect(
        self, grader: MathGrader
    ) -> None:
        result = await grader.grade("   \n\n  \t  ", "42")
        assert result.correct is False
        assert result.unparsed is True

    @pytest.mark.asyncio
    async def test_empty_box_is_incorrect_but_parsed(self, grader: MathGrader) -> None:
        """The model followed format (boxed) but provided no content."""
        result = await grader.grade("\\boxed{}", "42")
        assert result.correct is False
        assert result.unparsed is False

    @pytest.mark.asyncio
    async def test_unbalanced_box_falls_through(self, grader: MathGrader) -> None:
        result = await grader.grade("\\boxed{42 trailing", "42")
        assert result.unparsed is True
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_phrase_with_negation(self, grader: MathGrader) -> None:
        """Adversarial: 'the answer is NOT 99' — we still grab 99 as a fallback,
        but unparsed=True flags this so the caller can audit."""
        response = "the answer is not 99"
        result = await grader.grade(response, "42")
        assert result.unparsed is True
        assert result.correct is False

    @pytest.mark.asyncio
    async def test_ground_truth_with_whitespace(self, grader: MathGrader) -> None:
        result = await grader.grade("\\boxed{42}", "  42  ")
        assert result.correct is True

    @pytest.mark.asyncio
    async def test_reasoning_field_is_populated(self, grader: MathGrader) -> None:
        result = await grader.grade("\\boxed{42}", "42")
        assert "extracted '42'" in result.reasoning
        assert "match=True" in result.reasoning

    @pytest.mark.asyncio
    async def test_reasoning_flags_regex_fallback(self, grader: MathGrader) -> None:
        result = await grader.grade("the answer is 42", "42")
        assert "regex fallback" in result.reasoning


class TestExtractAnswerInterface:
    """``extract_answer`` is the synchronous interface used by exporters."""

    def test_extract_answer_strips_outer_whitespace(self, grader: MathGrader) -> None:
        assert grader.extract_answer("  \\boxed{42}  ") == "42"

    def test_extract_answer_returns_raw_pre_normalization(
        self, grader: MathGrader
    ) -> None:
        """``extract_answer`` returns the raw extracted string, not the
        normalized form, so users can see what the model actually wrote."""
        out = grader.extract_answer("\\boxed{\\dfrac{1}{2}}")
        assert out == "\\dfrac{1}{2}"
