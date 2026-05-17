# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict-equality grader matching DeepEval's ``Scorer.exact_match_score``.

The trt-llm benchmark recipe routes ``hellaswag`` and ``bigbench``
through ``deepeval.benchmarks`` (see
``trt-llm-benchmark-recipe/src/tools/acc_benchmark.py``). Both
benchmarks score with DeepEval's ``Scorer.exact_match_score``, which
is just::

    if not prediction:
        return 0
    return 1 if prediction.strip() == target.strip() else 0

Strict, case-sensitive, no normalization. We mirror this byte-for-
byte so aiperf's accuracy numbers reproduce the recipe's.

This is a deliberately conservative grader. Models that emit
``"The answer is A."`` instead of bare ``"A"`` will score 0 — as they
do in DeepEval. The escape hatch in DeepEval is structured generation
via ``MultipleChoiceSchema``; aiperf's equivalent is to enforce
``--accuracy-system-prompt`` constraints and request structured
outputs at the LLM-server level.

Reference:
    deepeval/scorer/scorer.py:Scorer.exact_match_score
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiperf.accuracy.graders.base import BaseGrader
from aiperf.accuracy.models import GradingResult

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class ExactMatchGrader(BaseGrader):
    """Strict ``pred.strip() == gold.strip()`` grader.

    Mirrors DeepEval's ``Scorer.exact_match_score``:

    - Empty / whitespace-only response → score 0 (``unparsed=True``).
    - Otherwise score 1 only when the stripped prediction equals the
      stripped gold byte-for-byte. Case-sensitive, no normalization.

    Used by HellaSwag and BigBench-Hard for trt-llm reference parity.
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(run=run, **kwargs)

    def extract_answer(self, response_text: str, **kwargs: Any) -> str:
        """Return the response stripped of outer whitespace, no other transforms."""
        return response_text.strip() if response_text else ""

    async def grade(
        self, response_text: str, ground_truth: str, **kwargs: Any
    ) -> GradingResult:
        pred = response_text.strip() if response_text else ""
        gold = ground_truth.strip() if ground_truth else ""
        unparsed = pred == "" and gold != ""
        correct = bool(pred) and pred == gold
        return GradingResult(
            correct=correct,
            unparsed=unparsed,
            confidence=1.0 if correct else 0.0,
            reasoning=(
                f"strict equality: stripped pred '{pred}' vs gold '{gold}'; "
                f"match={correct}" + (" (empty response)" if unparsed else "")
            ),
            extracted_answer=pred,
            ground_truth=gold,
        )
