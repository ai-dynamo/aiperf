# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSM8K grader implementing lighteval's ``quasi_exact_match_gsm8k``.

The recipe grades GSM8K with lighteval's ``quasi_exact_match_gsm8k``
(``trt-llm-benchmark-recipe/lighteval/src/lighteval/metrics/metrics.py``):

    quasi_exact_match_gsm8k = SampleLevelMetric(
        sample_level_fn=ExactMatches(
            strip_strings=True,
            normalize_pred=gsm8k_normalizer,
            normalize_gold=gsm8k_normalizer,
        ).compute,
        ...
    )

``ExactMatches`` with ``type_exact_match="full"`` reduces to::

    1 if gsm8k_normalizer(gold.strip()) == gsm8k_normalizer(pred.strip()) else 0

and ``gsm8k_normalizer`` pulls the number after ``####`` (comma-
stripped), returning ``"[invalid]"`` when there is none.

**Deliberate deviation for chat models.** Lighteval runs the *same*
``gsm8k_normalizer`` over the prediction. That only works for a base
model completing the few-shot ``#### <number>`` format; a chat model
answering ``"... so the answer is 24."`` produces no ``####`` and would
normalize to ``"[invalid]"`` — scoring essentially every chat response
wrong. So this grader keeps lighteval's gold extraction exactly, but
makes prediction extraction a superset: try ``gsm8k_normalizer`` first
(byte-for-byte lighteval parity when the model *does* emit ``####``),
and fall back to the last number in the response otherwise. Predictions
that need that fallback are flagged ``unparsed=True`` (the model didn't
follow the expected ``####`` format), mirroring ``MultipleChoiceGrader``
— but a correct fallback answer is still scored correct, since
``unparsed`` is a separate diagnostic counter from accuracy.

We also compare numerically rather than by string equality so ``"24"``,
``"24.0"`` and ``"24.00"`` all match — lighteval's string ``==`` would
reject those, but a chat model emitting ``"24.0"`` is plainly correct.
String equality is the final fallback when a side is non-numeric.

This grader is pure-regex and has no ``lighteval`` import, so it runs
without the ``[accuracy]`` extras.

Reference:
    trt-llm-benchmark-recipe/lighteval/src/lighteval/metrics/normalizations.py
        gsm8k_normalizer.
    trt-llm-benchmark-recipe/lighteval/src/lighteval/metrics/metrics_sample.py
        ExactMatches.compute_one_item.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

# lighteval's gsm8k_normalizer gold pattern: the number after "####".
_GSM8K_ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
_INVALID_ANS = "[invalid]"

# Fallback prediction pattern: a number (optionally signed, optional
# thousands commas, optional decimal part). Anchored on a leading digit
# so it never matches a bare "." or ",".
_NUMBER_RE = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?")

# Numbers within this absolute tolerance are treated as equal, so
# "24" == "24.0" == "24.00".
_NUMERIC_ABS_TOL = 1e-6


def gsm8k_normalizer(text: str) -> str:
    """Extract the number after ``####``, comma-stripped.

    Byte-for-byte port of lighteval's ``gsm8k_normalizer``
    (``normalizations.py``). Returns ``"[invalid]"`` when the text has
    no ``#### <number>`` marker.
    """
    match = _GSM8K_ANS_RE.search(text)
    if match:
        return match.group(1).strip().replace(",", "")
    return _INVALID_ANS


def _extract_prediction(response_text: str) -> tuple[str, bool]:
    """Extract the predicted answer number from a model response.

    Superset of lighteval's prediction normalization: prefer the
    ``#### <number>`` marker when present (exact lighteval parity), else
    fall back to the last standalone number in the response.

    Returns ``(answer, used_fallback)``. ``used_fallback`` is True when
    the response lacked the ``####`` marker and we resorted to the
    last-number heuristic — i.e. the model did not follow lighteval's
    expected format. This drives ``GradingResult.unparsed`` the same way
    ``MultipleChoiceGrader`` flags its regex fallback. ``answer`` is
    ``"[invalid]"`` when no number can be found at all.
    """
    normalized = gsm8k_normalizer(response_text)
    if normalized != _INVALID_ANS:
        return normalized, False
    matches = _NUMBER_RE.findall(response_text)
    if matches:
        return matches[-1].replace(",", ""), True
    return _INVALID_ANS, True


def _numbers_match(gold: str, pred: str) -> bool:
    """Compare two normalized answer strings.

    Numeric comparison with a small absolute tolerance when both parse
    as floats (so ``"24"`` matches ``"24.0"``); exact string equality
    otherwise (so non-numeric edge cases still grade deterministically).
    """
    try:
        return math.isclose(
            float(gold), float(pred), rel_tol=0.0, abs_tol=_NUMERIC_ABS_TOL
        )
    except ValueError:
        return gold == pred


class LightevalGSM8KGrader(BaseGrader):
    """Lighteval ``quasi_exact_match_gsm8k`` grader for GSM8K.

    Extracts the gold number with ``gsm8k_normalizer`` (the number after
    ``####``) and the predicted number from the model response (``####``
    marker if present, else the last number). Scores correct when the
    two match numerically. See the module docstring for why this
    deviates from lighteval's symmetric ``gsm8k_normalizer`` on the
    prediction side.
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(run=run, **kwargs)

    def extract_answer(self, response_text: str, **kwargs: Any) -> str:
        """Return the predicted answer number, or ``"[invalid]"``."""
        answer, _ = _extract_prediction(response_text or "")
        return answer

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs: Any
    ) -> GradingResult:
        gold = gsm8k_normalizer((ground_truth or "").strip())
        pred, used_fallback = _extract_prediction((response_text or "").strip())
        parseable = pred != _INVALID_ANS
        # A correct-but-fallback (no ``####``) response is still correct;
        # unparsed only flags that the expected format wasn't matched.
        unparsed = used_fallback or not parseable
        correct = parseable and gold != _INVALID_ANS and _numbers_match(gold, pred)
        if not parseable:
            note = " (no number in response)"
        elif used_fallback:
            note = " (no #### marker; used last-number fallback)"
        else:
            note = ""
        return GradingResult(
            correct=correct,
            unparsed=unparsed,
            confidence=1.0 if correct else 0.0,
            reasoning=(
                f"gsm8k quasi-exact-match: gold '{gold}' vs pred '{pred}'; "
                f"match={correct}{note}"
            ),
            extracted_answer=pred,
            ground_truth=gold,
        )
