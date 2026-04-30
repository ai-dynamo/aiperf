# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Math grader for numeric and algebraic answer equivalence.

Ported in spirit from lighteval's MATH/AIME extraction and normalization
pipeline. The lighteval reference relies on the optional ``math_verify``
library for full symbolic equivalence; we deliberately limit ourselves to
stdlib-only normalization + fraction-based numeric comparison so that the
grader has zero extra dependencies and is fully deterministic.

Coverage:
- Integer answers (AIME, AIME24, AIME25)
- Simple fractions ``\\frac{a}{b}``, ``a/b``
- Decimal numbers
- LaTeX-wrapped expressions inside ``\\boxed{...}``
- Mixed-format strings like ``$\\boxed{42}$`` or ``the answer is 42``

Out of scope (these will fall through to normalized-string equality and
may grade strict-but-correct answers as incorrect; document as known):
- Symbolic equivalence (e.g. ``\\sqrt{2}`` vs ``2^{1/2}``)
- Algebraic simplification (e.g. ``x+1`` vs ``1+x``)
- Trig identities

Use ``--accuracy-grader exact_match`` to opt out of normalization when the
benchmark's gold answers are already canonicalized.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult
from aiperf.common.config import UserConfig

# Recognized "the answer is X" suffixes used by models that ignore the
# \\boxed{} instruction. Captures everything to end-of-line so we can re-run
# the boxed/numeric extractors on the captured tail.
_ANSWER_PHRASE_RE = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|answer\s*[:=]|answer\s+is)\s*[:=]?\s*(.+?)(?:\.|$)",
    re.IGNORECASE,
)

# Matches signed/unsigned int, decimal, or simple ratio (e.g. "1/2", "-3.14").
# Anchors avoid bleeding into LaTeX commands; we intentionally skip variables
# and only catch numeric literals.
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")

# LaTeX wrappers we strip before comparison. \\dfrac and \\tfrac collapse to
# \\frac because they are visual variants only.
_DFRAC_RE = re.compile(r"\\(?:dfrac|tfrac)")
_LEFT_RIGHT_RE = re.compile(r"\\(?:left|right)")
# \\frac{a}{b} → (a)/(b). Conservative: only matches single-brace contents
# without nested braces, which is the common case for AIME/MATH-500.
_SIMPLE_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_TEXT_WRAPPER_RE = re.compile(r"\\(?:text|mathrm|mathit|mathbf)\{([^{}]*)\}")
# Trailing punctuation that often appears after a final answer.
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?]+$")

_BOXED_TOKEN = "\\boxed{"


def _extract_last_boxed(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}`` in ``text``.

    Implements brace-balanced matching so that nested LaTeX like
    ``\\boxed{\\frac{1}{2}}`` is captured intact. Returns None when there
    is no balanced ``\\boxed{}`` in the input.
    """
    last_idx = text.rfind(_BOXED_TOKEN)
    if last_idx == -1:
        return None
    start = last_idx + len(_BOXED_TOKEN)
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return None


def _extract_last_number(text: str) -> str | None:
    """Return the last numeric literal in ``text`` (int, decimal, or a/b)."""
    matches = _NUMBER_RE.findall(text)
    return matches[-1] if matches else None


def _normalize(expr: str) -> str:
    """Normalize a math expression for string-equality comparison.

    Operations applied in order, mirroring lighteval's pre-comparison pass:

    1. Strip whitespace.
    2. Drop trailing punctuation (so ``$42$.`` doesn't keep the period
       and prevent the dollar-strip from firing).
    3. Strip surrounding ``$...$`` math delimiters.
    4. Remove ``\\left`` / ``\\right`` (visual sizing only).
    5. Collapse ``\\dfrac``/``\\tfrac`` to ``\\frac``.
    6. Expand simple ``\\frac{a}{b}`` to ``(a)/(b)``.
    7. Unwrap ``\\text{x}`` / ``\\mathrm{x}`` / similar to ``x``.
    8. Remove all interior whitespace (so ``1 / 2`` matches ``1/2``).

    Math is case-sensitive (variable names matter), so case is preserved.
    Idempotent: ``_normalize(_normalize(s)) == _normalize(s)``.
    """
    s = expr.strip()
    s = _TRAILING_PUNCT_RE.sub("", s)
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1]
    s = _LEFT_RIGHT_RE.sub("", s)
    s = _DFRAC_RE.sub(r"\\frac", s)
    s = _SIMPLE_FRAC_RE.sub(r"(\1)/(\2)", s)
    s = _TEXT_WRAPPER_RE.sub(r"\1", s)
    s = re.sub(r"\s+", "", s)
    return s


def _to_fraction(value: str) -> Fraction | None:
    """Parse ``value`` as a ``Fraction``, or return None if it isn't numeric.

    Accepts:
    - Integers: ``42``, ``-7``
    - Decimals: ``3.14``, ``-0.5``
    - Ratios: ``1/2``, ``-3/4``
    - Parenthesized ratios produced by ``_normalize``: ``(1)/(2)``

    Rejects anything else (variable names, expressions, empty string).
    """
    if not value:
        return None
    s = value.strip().replace("(", "").replace(")", "")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


class MathGrader(BaseGrader):
    """Grades numeric/algebraic responses by extracting then normalizing.

    Extraction priority (first hit wins):

    1. Last ``\\boxed{...}`` in the response (the canonical MATH/AIME format).
    2. Last "the answer is X" / "answer: X" phrase, recursively parsed.
    3. Last numeric literal in the response (int, decimal, or a/b).

    Comparison strategy:

    1. Normalize both predicted and gold answers (see ``_normalize``).
    2. Try to parse both as ``Fraction``. If both parse, compare numerically
       (so ``0.5 == 1/2 == \\frac{1}{2}``).
    3. Otherwise, fall back to exact string equality on the normalized form.

    The ``unparsed`` flag is set whenever extraction fell back past the
    boxed-answer step. A correct unparsed response is still scored correct,
    matching MultipleChoiceGrader's convention.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(user_config=user_config, **kwargs)

    def _extract_with_flag(self, response_text: str) -> tuple[str, bool]:
        """Return ``(answer, unparsed)``.

        ``unparsed`` is True when extraction had to fall back past the
        ``\\boxed{}`` step (i.e. the model didn't follow the boxed-answer
        instruction).
        """
        if not response_text:
            return "", True

        boxed = _extract_last_boxed(response_text)
        if boxed is not None:
            return boxed.strip(), False

        m = _ANSWER_PHRASE_RE.search(response_text)
        if m:
            tail = m.group(1).strip()
            tail_boxed = _extract_last_boxed(tail)
            if tail_boxed is not None:
                return tail_boxed.strip(), True
            tail_num = _extract_last_number(tail)
            if tail_num is not None:
                return tail_num, True
            return tail, True

        last_num = _extract_last_number(response_text)
        if last_num is not None:
            return last_num, True

        return response_text.strip(), True

    def extract_answer(self, response_text: str, **kwargs: Any) -> str:
        """Extract the answer from a model response (boxed > phrase > last number)."""
        answer, _ = self._extract_with_flag(response_text)
        return answer

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs: Any
    ) -> GradingResult:
        pred_raw, unparsed = self._extract_with_flag(response_text)
        pred_norm = _normalize(pred_raw)
        gold_norm = _normalize(ground_truth)

        pred_frac = _to_fraction(pred_norm)
        gold_frac = _to_fraction(gold_norm)
        if pred_frac is not None and gold_frac is not None:
            correct = pred_frac == gold_frac
            mode = "numeric"
        else:
            correct = pred_norm == gold_norm and pred_norm != ""
            mode = "string"

        return GradingResult(
            correct=correct,
            unparsed=unparsed,
            confidence=1.0 if correct else 0.0,
            reasoning=(
                f"extracted '{pred_raw}' (normalized '{pred_norm}'); "
                f"ground_truth '{ground_truth}' (normalized '{gold_norm}'); "
                f"compared via {mode}; match={correct}"
                + (" (regex fallback)" if unparsed else "")
            ),
            extracted_answer=pred_raw,
            ground_truth=ground_truth.strip(),
        )
