# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GSM8K benchmark loader, aligned with the lighteval reference.

Mirrors lighteval's ``gsm8k_leaderboard`` task config
(``trt-llm-benchmark-recipe/lighteval/src/lighteval/tasks/default_tasks.py``):

    gsm8k_leaderboard = LightevalTaskConfig(
        name="gsm8k",
        prompt_function=prompt.gsm8k,      # "Question: {question}\\nAnswer:"
        hf_repo="gsm8k",
        hf_subset="main",
        evaluation_splits=["test"],
        few_shots_split=None,
        generation_size=256,
        metric=[Metrics.quasi_exact_match_gsm8k],
        stop_sequence=["Question=", "Question", "="],
        trust_dataset=True,
    )

lighteval's ``prompt.gsm8k`` (``default_prompts.py``) produces::

    Doc(query=f"Question: {line['question']}\\nAnswer:",
        choices=[f" {line['answer']}"],
        gold_index=0)

We emit the same prompt and store the raw ``answer`` field (which ends
with ``#### <number>``) as ``ground_truth``. ``LightevalGSM8KGrader``
runs ``gsm8k_normalizer`` over the gold at grade time to pull the
number after ``####`` — matching the recipe's
``quasi_exact_match_gsm8k`` ``normalize_gold``.

Unlike ``Math500Benchmark`` (subject-per-row task), GSM8K has a single
task name; there are no subtasks to break down by.

Reference:
    trt-llm-benchmark-recipe/lighteval/src/lighteval/tasks/default_tasks.py
        gsm8k_leaderboard task config.
    trt-llm-benchmark-recipe/lighteval/src/lighteval/tasks/default_prompts.py
        gsm8k prompt function.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from datasets import Dataset, load_dataset

from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
TASK_NAME = "gsm8k"

# lighteval's gsm8k_leaderboard task config: ``generation_size=256``.
DEFAULT_GENERATION_SIZE = 256

# Schema field names in the ``gsm8k`` (``main``) dataset.
QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"


class GSM8KBenchmark(AIPerfLoggerMixin):
    """GSM8K lighteval-aligned benchmark loader.

    Loads ``gsm8k`` (``main`` subset, test split) and emits one user
    message per problem formatted as ``"Question: {question}\\nAnswer:"``
    — matching lighteval's ``prompt.gsm8k``. Gold is the raw ``answer``
    field (which ends in ``#### <number>``); ``LightevalGSM8KGrader``
    extracts the number via ``gsm8k_normalizer`` at grade time.
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load GSM8K problems lighteval-style.

        Args:
            tasks: Ignored — GSM8K has no subtasks.
            n_shots: Ignored — the lighteval reference is zero-shot
                (``few_shots_split=None``).
            enable_cot: Ignored — lighteval's ``prompt.gsm8k`` does not
                add a CoT trigger; the model decides whether to reason
                based on the system prompt the user supplies via
                ``--accuracy-system-prompt``.

        Returns:
            One ``BenchmarkProblem`` per dataset row, in dataset order.
        """
        ds: Dataset = await asyncio.to_thread(
            load_dataset, DATASET_NAME, DATASET_CONFIG, split="test"
        )
        return await asyncio.to_thread(self._build_problems, ds)

    def _build_problems(self, ds: Dataset) -> list[BenchmarkProblem]:
        problems: list[BenchmarkProblem] = []
        for row in ds:
            question = row[QUESTION_FIELD]
            prompt = f"Question: {question}\nAnswer:"
            messages: list[AccuracyChatMessage] = [{"role": "user", "content": prompt}]
            problems.append(
                BenchmarkProblem(
                    prompt=prompt,
                    # Gold is the raw answer ending in ``#### <number>``;
                    # LightevalGSM8KGrader's gsm8k_normalizer extracts it.
                    ground_truth=str(row[ANSWER_FIELD]),
                    task=TASK_NAME,
                    metadata={"generation_size": DEFAULT_GENERATION_SIZE},
                    raw_messages=messages,
                )
            )
        return problems
