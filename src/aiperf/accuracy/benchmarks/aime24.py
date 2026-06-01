# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIME 2024 benchmark loader, aligned with the trt-llm lighteval reference.

Mirrors the recipe's ``acc_bench_lighteval.py`` configuration:

    aime24 = LightevalTaskConfig(
        name="aime24",
        prompt_function=aime_prompt_fn,
        hf_repo="HuggingFaceH4/aime_2024",
        hf_subset="default",
        evaluation_splits=["train"],
        few_shots_split=None,
        few_shots_select=None,
        generation_size=32768,
        metric=[expr_gold_metric],
    )

The recipe's ``aime_prompt_fn`` produces a ``Doc`` whose ``query`` is
the bare problem text — lighteval's prompt manager wraps it as a
single user message with no instruction prefix and no few-shot
priming (``few_shots_split=None``). We emit prompts the same way.
Pair with ``LightevalExprGrader`` for the recipe's ``expr_gold_metric``
extraction.

Reference:
    trt-llm-benchmark-recipe/src/accuracy/acc_bench_lighteval.py:128
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from datasets import Dataset, load_dataset

from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

DATASET_NAME = "HuggingFaceH4/aime_2024"
TASK_NAME = "aime24"

# lighteval's aime24 task config: ``generation_size=32768`` to give
# reasoning models room to think before emitting the boxed answer.
DEFAULT_GENERATION_SIZE = 32768

# Schema field names in HuggingFaceH4/aime_2024 (lowercase, lighteval
# canonical — distinct from the Maxwell-Jia mirror used by ``aime``).
PROBLEM_FIELD = "problem"
ANSWER_FIELD = "answer"


class AIME24Benchmark(AIPerfLoggerMixin):
    """AIME 2024 lighteval-aligned benchmark loader.

    Loads ``HuggingFaceH4/aime_2024`` (train split) and emits one user
    message per problem containing the bare problem text — the format
    lighteval's ``aime_prompt_fn`` + ``PromptManager`` produce when
    ``few_shots_split=None``. Pair with ``LightevalExprGrader`` for
    grading parity with the recipe.
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load AIME24 problems and format them lighteval-style.

        Args:
            tasks: Ignored — AIME24 has no subtasks.
            n_shots: Ignored — the lighteval reference is zero-shot
                (``few_shots_split=None``); accepting the parameter
                keeps the protocol uniform but emitting few-shots
                here would diverge from the reference.
            enable_cot: Ignored — lighteval's ``aime_prompt_fn`` does
                not add a CoT trigger; the model decides whether to
                reason based on the system prompt the user provides
                via ``--accuracy-system-prompt``.

        Returns:
            One ``BenchmarkProblem`` per dataset row, in dataset
            order.
        """
        ds: Dataset = await asyncio.to_thread(load_dataset, DATASET_NAME, split="train")
        return await asyncio.to_thread(self._build_problems, ds)

    def _build_problems(self, ds: Dataset) -> list[BenchmarkProblem]:
        problems: list[BenchmarkProblem] = []
        for row in ds:
            problem = row[PROBLEM_FIELD]
            messages: list[AccuracyChatMessage] = [{"role": "user", "content": problem}]
            problems.append(
                BenchmarkProblem(
                    prompt=problem,
                    ground_truth=str(row[ANSWER_FIELD]),
                    task=TASK_NAME,
                    metadata={"generation_size": DEFAULT_GENERATION_SIZE},
                    raw_messages=messages,
                )
            )
        return problems
