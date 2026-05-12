# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Math grader for numeric, algebraic, and LaTeX answer equivalence.

Ported faithfully from the trt-llm benchmark recipe's
``src/accuracy/aime/grader.py`` (``math_equal``), which itself descends
from Hendrycks' MATH release and the ToRA / DeepSeek-Math / OpenAI
PRM800K evaluation utilities. The grader pipeline is:

1. ``_extract_with_flag`` extracts the model's answer from the response
   (last ``\\boxed{...}`` first, then "the answer is X" / last-number
   fallbacks) — same as before.
2. ``_strip_string`` (ported in ``_math_strip``) normalizes the
   prediction and the gold using the recipe's exact pipeline.
3. ``_math_equal`` compares them: lowercase string equality, then
   numerical equality with a small tolerance, then symbolic equality
   via sympy + latex2sympy2-extended.

Optional dependencies:
    The full grader requires ``sympy`` and ``latex2sympy2-extended``.
    These ship in aiperf's ``[accuracy]`` optional-dependency group:
        ``uv pip install 'aiperf[accuracy]'``
    When they're missing, the grader falls back to a stdlib
    normalize+Fraction comparison and emits a single warning the first
    time it's invoked. Cases the stdlib path can't handle (symbolic
    equivalence like ``\\sqrt{2}`` ↔ ``2^{1/2}``, equation-form
    parsing) will then grade as incorrect.

Reference:
    trt-llm-benchmark-recipe/src/accuracy/aime/grader.py:math_equal
    trt-llm-benchmark-recipe/src/accuracy/aime/parser.py:strip_string
"""

from __future__ import annotations

import logging
import re
from fractions import Fraction
from math import isclose
from typing import Any

from aiperf.accuracy.graders._math_strip import strip_string
from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult
from aiperf.common.config import UserConfig

_log = logging.getLogger(__name__)

# Try-import the optional sympy stack. ``_HAS_SYMPY`` gates the
# symbolic-equality path; the stdlib fallback runs without it.
try:
    from sympy import N, simplify
    from sympy.parsing.sympy_parser import parse_expr

    _HAS_SYMPY = True
except ImportError:  # pragma: no cover - exercised only without optional dep
    _HAS_SYMPY = False
    parse_expr = None  # type: ignore[assignment]
    simplify = None  # type: ignore[assignment]
    N = None  # type: ignore[assignment]

try:
    from latex2sympy2_extended import latex2sympy

    _HAS_LATEX2SYMPY = True
except ImportError:  # pragma: no cover - exercised only without optional dep
    _HAS_LATEX2SYMPY = False
    latex2sympy = None  # type: ignore[assignment]

# One-time fallback warning so we don't spam logs per-grade.
_FALLBACK_WARNED = False


def _warn_fallback_once() -> None:
    """Emit a one-time warning when the stdlib fallback path is taken."""
    global _FALLBACK_WARNED
    if not _FALLBACK_WARNED:
        _FALLBACK_WARNED = True
        _log.warning(
            "MathGrader running in stdlib fallback mode (sympy/latex2sympy2-"
            "extended not installed). Symbolic equivalence (e.g. \\sqrt{2} "
            "vs 2^{1/2}) and LaTeX-form normalization will grade as "
            "incorrect. Install with: uv pip install 'aiperf[accuracy]'."
        )


# Recognized "the answer is X" suffixes used by models that ignore the
# \\boxed{} instruction. Captures everything to end-of-line so we can
# re-run the boxed/numeric extractors on the captured tail.
#
# Terminator combines three rules (decimals must survive, multi-line
# responses must terminate cleanly, sentence-ending periods should be
# stripped):
#
#   (?<!\d)\.(?!\d) — a period NOT surrounded by digits (sentence period)
#   \n              — end of line for multi-line responses
#   $               — end of string
#
# The previous implementation used just ``\.`` which silently truncated
# decimals like ``3.14`` to ``3``; the simpler ``\n|$`` alternative
# regresses to keeping trailing-sentence text in the tail. This hybrid
# handles all three concerns.
_ANSWER_PHRASE_RE = re.compile(
    r"(?:final\s+answer|the\s+answer\s+is|answer\s*[:=]|answer\s+is)\s*[:=]?\s*"
    r"(.+?)(?:(?<!\d)\.(?!\d)|\n|$)",
    re.IGNORECASE,
)

# Matches signed/unsigned int, decimal, or simple ratio (e.g. "1/2", "-3.14").
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+)?")

# Recognizes "expression-shaped" content in an answer-phrase tail —
# LaTeX commands (``\frac``, ``\sqrt``, ``\pi``, ...) or curly braces
# from any TeX construct. When the tail contains one of these, fall
# through to ``strip_string`` + ``math_equal`` whole, instead of
# running ``_extract_last_number`` which would silently grade
# ``"\frac{1}{2}"`` as ``"2"`` (the last digit it finds).
_LATEX_HINT_RE = re.compile(r"[\\{}]")

_BOXED_TOKEN = "\\boxed{"

# Numerical tolerance for ``isclose``. Mirrors the recipe's value.
_NUMERIC_ABS_TOL = 1e-4

# Recursion guard for ``math_equal``'s self-recursion (equation rewrites,
# choice-pattern unwraps). Mirrors the recipe.
_MAX_DEPTH = 5

# Single-choice-pattern preamble strippers from the recipe. If the
# prediction starts with one of these, we strip the prefix and recurse.
_SINGLE_CHOICE_PATTERNS: tuple[str, ...] = (
    r"^\(A\)",
    r"^\(B\)",
    r"^\(C\)",
    r"^\(D\)",
    r"^\(E\)",
    r"^A\.",
    r"^B\.",
    r"^C\.",
    r"^D\.",
    r"^E\.",
    r"^A\)",
    r"^B\)",
    r"^C\)",
    r"^D\)",
    r"^E\)",
    r"^\*\*A\*\*",
    r"^\*\*B\*\*",
    r"^\*\*C\*\*",
    r"^\*\*D\*\*",
    r"^\*\*E\*\*",
    r"^A:",
    r"^B:",
    r"^C:",
    r"^D:",
    r"^E:",
)


def _extract_last_boxed(text: str) -> str | None:
    """Return the contents of the last ``\\boxed{...}`` in ``text``.

    Brace-balanced match so nested LaTeX like ``\\boxed{\\frac{1}{2}}``
    is captured intact.
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


def _parse_digits(num: Any) -> float | None:
    """Mirror of recipe ``parser.parse_digits``: comma-strip → float, with
    ``%`` fallback dividing by 100. Returns None on parse failure."""
    s = str(num).replace(",", "")
    try:
        return float(s)
    except ValueError:
        if s.endswith("%"):
            inner = s[:-1]
            if inner.endswith("\\"):
                inner = inner[:-1]
            try:
                return float(inner) / 100
            except ValueError:
                pass
    return None


def _is_digit(num: Any) -> bool:
    """Whether ``num`` parses as a numeric literal via ``_parse_digits``."""
    return _parse_digits(num) is not None


def _to_fraction(value: str) -> Fraction | None:
    """Stdlib fallback: parse ``value`` as a Fraction (int / decimal / a/b)."""
    if not value:
        return None
    s = value.strip().replace("(", "").replace(")", "")
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def _sympy_parse(s: str) -> Any:
    """Try parsing ``s`` via sympy/latex2sympy parsers, in priority order.

    Returns the parsed sympy expression on first success, or the raw
    string on total failure (so the caller can still attempt direct
    string equality). Mirrors recipe's ``symbolic_equal._parse``.
    """
    parsers: list[Any] = []
    if _HAS_SYMPY:
        parsers.append(parse_expr)
    if _HAS_LATEX2SYMPY:
        parsers.append(latex2sympy)
    for f in parsers:
        # The recipe's ``s.replace("\\\\", "\\")`` first-pass.
        for arg in (s.replace("\\\\", "\\"), s):
            try:
                return f(arg)
            except Exception:  # noqa: BLE001, S112
                continue
    return s


def _symbolic_equal(a_raw: str, b_raw: str) -> bool:
    """Sympy-backed symbolic equivalence check, mirroring the recipe.

    Strategies tried in order: direct string-of-parsed-expr equality,
    ``equals`` / ``simplify(a - b) == 0``, and numerical equality at
    1e-4 absolute tolerance after ``N(.)`` evaluation.
    """
    if not _HAS_SYMPY:
        return False
    a = _sympy_parse(a_raw)
    b = _sympy_parse(b_raw)

    # Direct expression equality.
    try:
        if str(a) == str(b) or a == b:
            return True
    except Exception:  # noqa: BLE001, S110
        pass

    # Symbolic simplify.
    try:
        if a.equals(b) or simplify(a - b) == 0:
            return True
    except Exception:  # noqa: BLE001, S110
        pass

    # Numerical evaluation.
    try:
        if isclose(float(N(a)), float(N(b)), abs_tol=_NUMERIC_ABS_TOL):
            return True
    except Exception:  # noqa: BLE001, S110
        pass

    return False


def _handle_single_choice_prefix(prediction: str, reference: str) -> bool:
    """Match a single-letter A-E reference by extracting the last A-E letter
    from a cleaned prediction string."""
    if reference not in ("A", "B", "C", "D", "E"):
        return False
    cleaned = prediction.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")
    letters = re.findall(r"\b(A|B|C|D|E)\b", cleaned.upper())
    return bool(letters and letters[-1] == reference)


def _apply_single_choice_patterns(prediction: str, reference: str, depth: int) -> bool:
    """Strip a recognized choice-prefix pattern off the prediction and
    recurse into ``_math_equal`` on the remainder."""
    for pat in _SINGLE_CHOICE_PATTERNS:
        if re.match(pat, prediction):
            cleaned = re.sub(pat, "", prediction, count=1).strip()
            if _math_equal(cleaned, reference, depth + 1):
                return True
    return False


def _compare_comma_list(prediction: str, reference: str, depth: int) -> bool:
    """Sort comma-separated lists and compare element-wise via _math_equal."""
    if "," not in prediction or "," not in reference:
        return False
    pred_parts = sorted(p.strip() for p in prediction.split(","))
    ref_parts = sorted(p.strip() for p in reference.split(","))
    return len(pred_parts) == len(ref_parts) and all(
        _math_equal(pp, rp, depth + 1)
        for pp, rp in zip(pred_parts, ref_parts, strict=False)
    )


def _numeric_equality_with_percent(prediction: str, reference: str) -> bool | None:
    """Numeric comparison with percentage variants.

    Tri-state semantics — caller must respect them:

    * ``None``  → inputs are not both numeric; fall through to the next
      strategy.
    * ``True``  → numeric match (one of the percent-scaled candidates is
      within ``_NUMERIC_ABS_TOL``).
    * ``False`` → both inputs were numeric, but the parsed values are
      not close. **Terminal** — do not fall through to brace/paren,
      equation-rewrite, or symbolic checks. Mirrors the recipe's
      ``return any(...)`` short-circuit.
    """
    if not (_is_digit(prediction) and _is_digit(reference)):
        return None
    p_num = _parse_digits(prediction)
    r_num = _parse_digits(reference)
    if p_num is None or r_num is None:
        return False
    candidates = (r_num / 100, r_num, r_num * 100)
    return any(isclose(p_num, c, abs_tol=_NUMERIC_ABS_TOL) for c in candidates)


def _brace_paren_compare(prediction: str, reference: str) -> bool:
    """Strip mismatched outer brackets / braces / parens, then compare
    the residual strings case-insensitively."""
    pred_str = str(prediction).strip()
    ref_str = str(reference).strip()
    if (
        pred_str.startswith("[")
        and pred_str.endswith("]")
        and not ref_str.startswith("(")
    ) or (
        pred_str.startswith("(")
        and pred_str.endswith(")")
        and not ref_str.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ("{", "}", "(", ")"):
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    return pred_str.lower() == ref_str.lower()


def _equation_rewrites(prediction: str, reference: str, depth: int) -> bool:
    """Equation-form rewrites:

    * ``a = b`` vs ``c = d`` → compare ``a - (b)`` to ``c - (d)`` symbolically.
    * ``x = v`` (lhs ≤ 2 chars) on one side, plain value on the other →
      compare the rhs to the plain value via ``_math_equal``.
    """
    if prediction.count("=") == 1 and reference.count("=") == 1:
        p_lhs, p_rhs = (s.strip() for s in prediction.split("="))
        r_lhs, r_rhs = (s.strip() for s in reference.split("="))
        return _symbolic_equal(f"{p_lhs} - ({p_rhs})", f"{r_lhs} - ({r_rhs})")
    if (
        prediction.count("=") == 1
        and len(prediction.split("=")[0].strip()) <= 2
        and "=" not in reference
    ):
        return _math_equal(prediction.split("=")[1], reference, depth + 1)
    if (
        reference.count("=") == 1
        and len(reference.split("=")[0].strip()) <= 2
        and "=" not in prediction
    ):
        return _math_equal(prediction, reference.split("=")[1], depth + 1)
    return False


def _math_equal(prediction: str, reference: str, depth: int = 0) -> bool:
    """Hendrycks-style equality check, ported from the recipe.

    Strategy order (each falls through on no-match unless noted):

    1. Recursion-depth guard, then null-input guard.
    2. Direct lowercased string equality.
    3. ``_handle_single_choice_prefix`` — A-E letter answer matching.
    4. ``_apply_single_choice_patterns`` — strip a choice-prefix and recurse.
    5. ``_compare_comma_list`` — sorted comma-list element-wise compare.
    6. ``_numeric_equality_with_percent`` — **terminal** when both inputs
       look numeric: returns the verdict directly without falling through
       to symbolic.
    7. Empty-prediction shortcut (matches the recipe's exact placement
       AFTER the numeric branch).
    8. ``_brace_paren_compare`` — bracket-stripped lowercase compare.
    9. ``_equation_rewrites`` — equation-form normalizations.
    10. ``_symbolic_equal`` — sympy/latex2sympy symbolic equivalence
        (final fallback).

    ``depth`` is a recursion guard for the rewrites (capped at
    ``_MAX_DEPTH``).
    """
    if depth > _MAX_DEPTH:
        return False
    if prediction is None or reference is None:
        return False

    if str(prediction).strip().lower() == str(reference).strip().lower():
        return True

    if _handle_single_choice_prefix(prediction, reference):
        return True
    if _apply_single_choice_patterns(prediction, reference, depth):
        return True
    if _compare_comma_list(prediction, reference, depth):
        return True

    # Numeric equality is terminal — see the helper docstring for why.
    numeric_verdict = _numeric_equality_with_percent(prediction, reference)
    if numeric_verdict is not None:
        return numeric_verdict

    # Post-numeric empty-pred shortcut, kept inline because it sits
    # between two strategies in the recipe and excludes 0 / False to
    # avoid eating valid numeric-zero predictions that the numeric
    # branch above already handled.
    if not prediction and prediction not in (0, False):
        return False

    if _brace_paren_compare(prediction, reference):
        return True
    if _equation_rewrites(prediction, reference, depth):
        return True

    return _symbolic_equal(prediction, reference)


def _stdlib_fallback_equal(prediction: str, reference: str) -> bool:
    """Stdlib-only fallback used when sympy isn't installed.

    Same as the previous (pre-trt-llm-port) MathGrader behavior:
    normalize on both sides, attempt Fraction parse for numerics,
    otherwise compare the normalized strings.
    """
    pred_norm = _stdlib_normalize(prediction)
    gold_norm = _stdlib_normalize(reference)
    pred_frac = _to_fraction(pred_norm)
    gold_frac = _to_fraction(gold_norm)
    if pred_frac is not None and gold_frac is not None:
        return pred_frac == gold_frac
    return pred_norm == gold_norm and pred_norm != ""


_DFRAC_RE = re.compile(r"\\(?:dfrac|tfrac)")
_LEFT_RIGHT_RE = re.compile(r"\\(?:left|right)")
_SIMPLE_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
_TEXT_WRAPPER_RE = re.compile(r"\\(?:text|mathrm|mathit|mathbf)\{([^{}]*)\}")
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?]+$")


def _stdlib_normalize(expr: str) -> str:
    """Lightweight LaTeX-aware normalization for the stdlib fallback path.

    Strips $...$, \\left/\\right, expands \\frac{a}{b} to (a)/(b),
    unwraps \\text{...}, drops trailing punctuation, removes interior
    whitespace. Idempotent.
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


# Public alias retained for backward compatibility with the v1 tests
# that imported ``_normalize`` directly. The body is the stdlib path;
# in the new pipeline it's only used inside the fallback.
_normalize = _stdlib_normalize


class MathGrader(BaseGrader):
    """Grades math/AIME responses with the trt-llm reference algorithm.

    Extraction priority (mirrors recipe ``format_response``):

    1. Last ``\\boxed{...}`` in the response (canonical MATH/AIME format).
    2. Last "the answer is X" / "answer: X" phrase, recursively re-parsed.
    3. Last numeric literal in the response.

    Comparison pipeline (mirrors recipe ``check_is_correct`` →
    ``math_equal`` → ``symbolic_equal``):

    1. ``strip_string`` both sides (LaTeX/unit normalization).
    2. ``_math_equal``: lowercase string equality, choice-prefix unwrap,
       numerical isclose (abs_tol=1e-4) with percentage variants,
       brace/paren strip + compare, equation-form rewrite, finally
       symbolic equality via sympy + latex2sympy2-extended.

    The ``unparsed`` flag is set whenever extraction fell back past the
    boxed-answer step. A correct unparsed response is still scored
    correct, matching ``MultipleChoiceGrader``'s convention.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(user_config=user_config, **kwargs)

    def _extract_with_flag(self, response_text: str) -> tuple[str, bool]:
        """Return ``(answer, unparsed)``.

        ``unparsed`` is True when extraction had to fall back past the
        ``\\boxed{}`` step (i.e. the model didn't follow the boxed-answer
        instruction). Mirrors ``format_response`` from the recipe but
        also surfaces the unparsed flag for downstream auditing.
        """
        if not response_text:
            return "", True

        boxed = _extract_last_boxed(response_text)
        if boxed is not None:
            return _format_response_tail(boxed), False

        # Take the LAST answer-phrase match, not the first: reasoning
        # models often self-correct ("the answer is 5. Wait, actually
        # the answer is 12") and the final claim is what we want.
        phrase_matches = _ANSWER_PHRASE_RE.findall(response_text)
        if phrase_matches:
            tail = phrase_matches[-1].strip()
            tail_boxed = _extract_last_boxed(tail)
            if tail_boxed is not None:
                return _format_response_tail(tail_boxed), True
            # If the tail looks like a LaTeX expression (contains a
            # backslash or curly brace), preserve it whole so
            # ``strip_string`` + ``math_equal`` can normalize and
            # compare it. Falling through to the last-number extractor
            # would grade ``"\frac{1}{2}"`` against gold ``"1/2"`` as
            # ``"2"`` — exactly the asymmetry that ``\boxed{...}``
            # tails correctly avoid.
            if _LATEX_HINT_RE.search(tail):
                return tail, True
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

        pred_stripped = strip_string(pred_raw)
        gold_stripped = strip_string(ground_truth)

        # The full ``math_equal`` path needs BOTH sympy (for parse_expr
        # and ``_symbolic_equal``) and latex2sympy2-extended (for
        # LaTeX-shaped expressions inside ``_sympy_parse``). Requiring
        # both matches the contract the module docstring and the
        # warning text already promise — a "sympy alone" middle state
        # would silently regress LaTeX equivalence without telling
        # users to install the missing package.
        if _HAS_SYMPY and _HAS_LATEX2SYMPY:
            correct = _math_equal(pred_stripped, gold_stripped)
            mode = "math_equal"
        else:
            _warn_fallback_once()
            correct = _stdlib_fallback_equal(pred_stripped, gold_stripped)
            mode = "stdlib-fallback"

        return GradingResult(
            correct=correct,
            unparsed=unparsed,
            confidence=1.0 if correct else 0.0,
            reasoning=(
                f"extracted '{pred_raw}' (stripped '{pred_stripped}'); "
                f"ground_truth '{ground_truth}' (stripped '{gold_stripped}'); "
                f"compared via {mode}; match={correct}"
                + (" (regex fallback)" if unparsed else "")
            ),
            extracted_answer=pred_raw,
            ground_truth=ground_truth.strip(),
        )


def _format_response_tail(boxed_content: str) -> str:
    """Apply the recipe's ``format_response`` post-extraction cleanup.

    Handles the trailing-character trim that the recipe does AFTER
    extracting the boxed content: collapse newlines, drop a leading
    ``:``, drop a trailing ``.`` or ``/``.
    """
    pred = boxed_content
    pred = re.sub(r"\n\s*", "", pred)
    if pred and pred[0] == ":":
        pred = pred[1:]
    if pred and pred[-1] == ".":
        pred = pred[:-1]
    if pred and pred[-1] == "/":
        pred = pred[:-1]
    return pred
