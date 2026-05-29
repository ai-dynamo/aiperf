# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIME 2025 benchmark loader, aligned with the trt-llm lighteval reference.

Mirrors ``acc_bench_lighteval.py:aime25``: same ``aime_prompt_fn``,
same zero-shot config, ``generation_size=32768``,
``hf_repo="yentinglin/aime_2025"``. See the AIME24 module for a fuller
explanation of the design.

Reference:
    trt-llm-benchmark-recipe/src/accuracy/acc_bench_lighteval.py:142
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from datasets import Dataset, load_dataset

from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

DATASET_NAME = "yentinglin/aime_2025"
TASK_NAME = "aime25"

# lighteval's aime25 task config: ``generation_size=32768``.
DEFAULT_GENERATION_SIZE = 32768

# Schema field names in yentinglin/aime_2025 (same lowercase shape as
# AIME24's HuggingFaceH4 mirror).
PROBLEM_FIELD = "problem"
ANSWER_FIELD = "answer"


class AIME25Benchmark(AIPerfLoggerMixin):
    """AIME 2025 lighteval-aligned benchmark loader.

    Loads ``yentinglin/aime_2025`` (train split) and emits one user
    message per problem containing the bare problem text — matching
    lighteval's zero-shot ``aime_prompt_fn`` rendering. Pair with
    ``LightevalExprGrader`` for grading parity with the recipe.
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load AIME25 problems and format them lighteval-style.

        Args:
            tasks: Ignored — AIME25 has no subtasks.
            n_shots: Ignored — the lighteval reference is zero-shot.
            enable_cot: Ignored — lighteval's ``aime_prompt_fn`` does
                not add a CoT trigger.

        Returns:
            One ``BenchmarkProblem`` per dataset row, in dataset order.
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
