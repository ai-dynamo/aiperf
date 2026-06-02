# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MATH-500 benchmark loader, aligned with the trt-llm lighteval reference.

Mirrors ``acc_bench_lighteval.py:math_500``:

    math_500 = LightevalTaskConfig(
        name="math_500",
        prompt_function=prompt_fn,        # query=line["problem"], choices=[line["solution"]]
        hf_repo="HuggingFaceH4/MATH-500",
        evaluation_splits=["test"],
        few_shots_split=None,
        generation_size=32768,
        metric=[latex_gold_metric],
    )

Two notable differences from the AIME24/AIME25 loaders:

1. ``ground_truth`` is the full ``solution`` text (which contains a
   ``\\boxed{answer}``), not a bare answer. ``LightevalLatexGrader``'s
   ``LatexExtractionConfig`` extracts the boxed answer from the
   solution at grade time. This matches the recipe's
   ``latex_gold_metric.gold_extraction_target=(LatexExtractionConfig(),)``.
2. Pair with ``LightevalLatexGrader`` (default), not
   ``LightevalExprGrader`` — gold answers are LaTeX expressions
   (fractions, square roots, etc.).

Reference:
    trt-llm-benchmark-recipe/src/accuracy/acc_bench_lighteval.py:156
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from datasets import Dataset, load_dataset

from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

DATASET_NAME = "HuggingFaceH4/MATH-500"
TASK_NAME = "math_500"

# lighteval's math_500 task config: ``generation_size=32768``.
DEFAULT_GENERATION_SIZE = 32768

# Schema field names in HuggingFaceH4/MATH-500.
PROBLEM_FIELD = "problem"
SOLUTION_FIELD = "solution"
SUBJECT_FIELD = "subject"
LEVEL_FIELD = "level"


class Math500Benchmark(AIPerfLoggerMixin):
    """MATH-500 lighteval-aligned benchmark loader.

    Loads ``HuggingFaceH4/MATH-500`` (test split) and emits one user
    message per problem containing the bare problem text — matching
    lighteval's ``prompt_fn``. Gold is the full ``solution`` text;
    ``LightevalLatexGrader`` extracts the boxed answer at grade time.
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load MATH-500 problems lighteval-style.

        Args:
            tasks: Ignored — lighteval's MATH-500 task has no subtask
                filtering (subjects are kept in metadata for reporting,
                but lighteval evaluates the full split). Use the
                aggregated CSV per-subject row to break results down
                after the run.
            n_shots: Ignored — the lighteval reference is zero-shot
                (``few_shots_split=None``).
            enable_cot: Ignored — lighteval's ``prompt_fn`` does not
                add a CoT trigger.

        Returns:
            One ``BenchmarkProblem`` per dataset row, in dataset order.
        """
        ds: Dataset = await asyncio.to_thread(load_dataset, DATASET_NAME, split="test")
        return await asyncio.to_thread(self._build_problems, ds)

    def _build_problems(self, ds: Dataset) -> list[BenchmarkProblem]:
        problems: list[BenchmarkProblem] = []
        for row in ds:
            problem = row[PROBLEM_FIELD]
            solution = row.get(SOLUTION_FIELD) or ""
            messages: list[AccuracyChatMessage] = [{"role": "user", "content": problem}]
            problems.append(
                BenchmarkProblem(
                    prompt=problem,
                    # Gold is the full solution containing \\boxed{answer};
                    # LightevalLatexGrader extracts the boxed expression.
                    ground_truth=solution,
                    # Use ``subject`` as the per-row task so the
                    # accuracy CSV breaks down by MATH subject.
                    task=row.get(SUBJECT_FIELD) or TASK_NAME,
                    metadata={
                        "subject": row.get(SUBJECT_FIELD, ""),
                        "level": row.get(LEVEL_FIELD),
                        "generation_size": DEFAULT_GENERATION_SIZE,
                    },
                    raw_messages=messages,
                )
            )
        return problems
