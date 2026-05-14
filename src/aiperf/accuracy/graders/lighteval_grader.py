# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lighteval-backed graders for the AIME24/25, MATH-500, and GPQA-Diamond benchmarks.

These three benchmarks are graded by lighteval in the trt-llm benchmark
recipe (``trt-llm-benchmark-recipe/src/accuracy/acc_bench_lighteval.py``):

- ``aime24`` / ``aime25`` → ``expr_gold_metric`` (expression extraction)
- ``math_500`` → ``latex_gold_metric`` (LaTeX extraction with boxed
  priority)
- ``gpqa_diamond`` → ``gpqa_metric`` (multiple-choice index extraction)

Each is a different parameterization of lighteval's
``MultilingualExtractiveMatchMetric``. We expose three grader plugins
(``lighteval_expr``, ``lighteval_latex``, ``lighteval_gpqa``) so each
benchmark can pick the right one via its ``default_grader`` metadata.

The lighteval API moved between the recipe's vendored 0.6.0 and the
current 0.13.0 PyPI release: the factory function
``multilingual_extractive_match_metric`` is now a class
``MultilingualExtractiveMatchMetric``. The configuration is otherwise
identical, so the behavior is the same.

Call shape:
    lighteval expects a ``Doc`` (with ``query``/``choices``/
    ``gold_index``) and a ``ModelResponse`` (with ``text``). We bridge
    aiperf's ``(response_text, ground_truth)`` arguments to this shape
    inside ``_grade_via_lighteval``.

Reference:
    trt-llm-benchmark-recipe/src/accuracy/acc_bench_lighteval.py
        latex_gold_metric / expr_gold_metric / gpqa_metric definitions.
"""

from __future__ import annotations

import logging
from typing import Any

from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult
from aiperf.common.config import UserConfig

_log = logging.getLogger(__name__)

try:
    from lighteval.metrics.dynamic_metrics import (
        ExprExtractionConfig,
        LatexExtractionConfig,
        MultilingualExtractiveMatchMetric,
    )
    from lighteval.metrics.utils.extractive_match_utils import (
        IndicesExtractionConfig,
    )
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.requests import Doc
    from lighteval.utils.language import Language

    _HAS_LIGHTEVAL = True
except ImportError:  # pragma: no cover - exercised only without optional dep
    _HAS_LIGHTEVAL = False
    MultilingualExtractiveMatchMetric = None  # type: ignore[assignment]
    ExprExtractionConfig = None  # type: ignore[assignment]
    LatexExtractionConfig = None  # type: ignore[assignment]
    IndicesExtractionConfig = None  # type: ignore[assignment]
    ModelResponse = None  # type: ignore[assignment]
    Doc = None  # type: ignore[assignment]
    Language = None  # type: ignore[assignment]


_MISSING_LIGHTEVAL_HINT = (
    "lighteval is not installed; AIME24/AIME25/MATH-500/GPQA-Diamond "
    "graders cannot run. Install with: uv pip install 'aiperf[accuracy]'."
)


def _require_lighteval() -> None:
    """Raise a clear error if lighteval is missing.

    We don't fall back silently for these graders because there's no
    stdlib equivalent of lighteval's extractive-match pipeline that
    would produce comparable scores; running with a different grader
    silently would mislead users into thinking they're reproducing
    trt-llm reference numbers.
    """
    if not _HAS_LIGHTEVAL:
        raise RuntimeError(_MISSING_LIGHTEVAL_HINT)


class _LightevalBaseGrader(BaseGrader):
    """Shared infrastructure for lighteval-backed graders.

    Subclasses build a ``MultilingualExtractiveMatchMetric`` in
    ``__init__`` (matching one of the recipe's three configs) and the
    base class drives the call: build a fresh ``Doc`` per grade with
    the gold answer in ``choices[0]``, build a ``ModelResponse``
    wrapping the response text, call ``metric.compute(doc, response)``,
    return a ``GradingResult``.

    Lighteval's metric returns a ``float`` (1.0 correct / 0.0 wrong /
    sometimes a fraction for partial matches). We treat anything > 0.5
    as correct, mirroring how the recipe uses these metrics in a
    pass/fail context.
    """

    _CORRECTNESS_THRESHOLD = 0.5

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(user_config=user_config, **kwargs)
        _require_lighteval()
        self._metric = self._build_metric()

    def _build_metric(self) -> Any:
        """Subclasses return the configured ``MultilingualExtractiveMatchMetric``."""
        raise NotImplementedError

    def extract_answer(self, response_text: str, **kwargs: Any) -> str:
        """Return the raw response.

        lighteval's metric does its own extraction internally, so we
        don't surface a separate ``extract_answer`` step. The raw
        response is what the metric receives anyway, so returning it
        keeps ``GradingResult.extracted_answer`` informative.
        """
        return response_text.strip()

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs: Any
    ) -> GradingResult:
        score = self._safe_compute(response_text, ground_truth)
        correct = score is not None and score > self._CORRECTNESS_THRESHOLD
        unparsed = score is None
        return GradingResult(
            correct=correct,
            unparsed=unparsed,
            confidence=float(score) if score is not None else 0.0,
            reasoning=(
                f"lighteval {type(self).__name__}: score={score} "
                f"(threshold {self._CORRECTNESS_THRESHOLD})"
                + (" (raised exception)" if unparsed else "")
            ),
            extracted_answer=response_text.strip(),
            ground_truth=ground_truth.strip(),
        )

    def _safe_compute(self, response_text: str, ground_truth: str) -> float | None:
        """Run lighteval's metric.compute with crash-safety.

        lighteval can raise on malformed predictions (e.g. when its
        sympy-backed extractor times out or hits an unparsable LaTeX
        construct). We catch and report ``unparsed=True`` rather than
        crashing the record processor.
        """
        try:
            doc = self._build_doc(ground_truth)
            response = ModelResponse(text=[response_text])
            return float(self._metric.compute(doc, response))
        except Exception as exc:  # noqa: BLE001
            _log.debug("lighteval grader exception: %s", exc, exc_info=True)
            return None

    def _build_doc(self, ground_truth: str) -> Any:
        """Build the lighteval ``Doc`` for a single grade.

        Default implementation: gold goes in ``choices[0]``,
        ``gold_index=0``. ``query`` is unused by the extractive-match
        metric but lighteval requires it to be set, so we pass an
        empty string. Subclasses override for benchmark-specific
        shapes (e.g. GPQA-Diamond uses A/B/C/D choices and an integer
        gold index).
        """
        return Doc(
            task_name=type(self).__name__,
            query="",
            choices=[ground_truth],
            gold_index=0,
        )


class LightevalExprGrader(_LightevalBaseGrader):
    """Lighteval ``expr_gold_metric`` — used for AIME24, AIME25.

    Matches the recipe's:
        multilingual_extractive_match_metric(
            language=Language.ENGLISH,
            fallback_mode='first_match',
            precision=5,
            gold_extraction_target=(ExprExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(boxed_match_priority=0),
            ),
            aggregation_function=max,
        )
    """

    def _build_metric(self) -> Any:
        return MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(ExprExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(boxed_match_priority=0),
            ),
            aggregation_function=max,
        )


class LightevalLatexGrader(_LightevalBaseGrader):
    """Lighteval ``latex_gold_metric`` — used for MATH-500.

    Same structure as ``LightevalExprGrader`` but the gold extractor
    uses ``LatexExtractionConfig`` (gold answers in MATH-500 are LaTeX
    snippets like ``\\frac{1}{3}`` or ``\\sqrt{2}``).
    """

    def _build_metric(self) -> Any:
        return MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            fallback_mode="first_match",
            precision=5,
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(
                ExprExtractionConfig(),
                LatexExtractionConfig(boxed_match_priority=0),
            ),
            aggregation_function=max,
        )


class LightevalGPQAGrader(_LightevalBaseGrader):
    """Lighteval ``gpqa_metric`` — used for GPQA-Diamond.

    Looks for ``Answer: $LETTER`` (or any ``A``/``B``/``C``/``D``
    extraction) in both gold and prediction. The recipe's prompt
    template instructs the model to produce ``Answer: $LETTER``;
    aiperf's GPQA-Diamond loader uses the same template for parity.

    Because the metric extracts via NativeLetters (A/B/C/D), the
    ``ground_truth`` we pass should also be a single letter (``"A"``,
    ``"B"``, ``"C"``, or ``"D"``) — strip the leading-space
    convention of MultipleChoiceGrader before passing in.
    """

    def _build_metric(self) -> Any:
        return MultilingualExtractiveMatchMetric(
            language=Language.ENGLISH,
            gold_extraction_target=(
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters"),
            ),
            pred_extraction_target=(
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters"),
            ),
            precision=5,
        )

    def _build_doc(self, ground_truth: str) -> Any:
        """GPQA-Diamond gold is a single A/B/C/D letter.

        ``MultipleChoiceGrader``'s convention stores the gold as
        ``" A"`` (leading-space). lighteval expects a bare letter, so
        we strip here. We also build the four-choice shape lighteval
        uses for indices extraction.

        Raises:
            ValueError: when ``ground_truth`` does not normalize to a
                single A/B/C/D letter. Silently coercing invalid gold
                to choice A would treat every model picking "A" as
                correct on a malformed row, so we fail fast and let
                ``_safe_compute`` surface it as ``unparsed=True``.
        """
        letter = ground_truth.strip().upper()
        if letter not in {"A", "B", "C", "D"}:
            raise ValueError(
                "GPQA-Diamond ground_truth must normalize to one of "
                f"A/B/C/D; got {ground_truth!r} (cleaned: {letter!r})"
            )
        gold_index = "ABCD".index(letter)
        return Doc(
            task_name=type(self).__name__,
            query="",
            choices=["A", "B", "C", "D"],
            gold_index=gold_index,
        )
