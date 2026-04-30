# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIME benchmark loader, ported from lighteval's competition-math task style.

Loads the Maxwell-Jia/AIME_2024 dataset (DeepEval's canonical AIME 2024
mirror, with capitalized ``Problem``/``Answer`` field names) and formats
each problem as a competition-math word problem instructing the model to
place its final integer answer inside ``\\boxed{...}``. Pair with
``MathGrader`` for numerical equivalence.

This benchmark is named ``aime`` (no year) so that downstream consumers
have a stable identifier for "AIME problems" without committing to a year.
For year-pinned variants see ``aime24``/``aime25``, which use lighteval's
``HuggingFaceH4/aime_2024`` and ``yentinglin/aime_2025`` mirrors with
lowercase schemas.

lighteval reference: lighteval/src/lighteval/tasks/extended/aime/main.py
"""

from __future__ import annotations

import asyncio
from typing import Any

from datasets import Dataset, load_dataset

from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.config import UserConfig
from aiperf.common.mixins import AIPerfLoggerMixin

DATASET_NAME = "Maxwell-Jia/AIME_2024"
TASK_NAME = "aime"

# AIME answers are integers in [0, 999]. We allow generous reasoning
# headroom so chain-of-thought solutions can complete; the per-request
# max_tokens is read from BenchmarkProblem.metadata["generation_size"]
# in AccuracyDatasetLoader. 4096 is a defensible default for non-reasoning
# models; reasoning-heavy models will need a higher cap via the user's
# server config or a future per-benchmark override.
DEFAULT_GENERATION_SIZE = 4096

# Instruction prefix shown to the model. Asks for \\boxed{} format so that
# MathGrader can extract the final answer reliably.
INSTRUCTION_PREFIX = (
    "Solve the following competition math problem. The answer is a "
    "non-negative integer. Place your final answer inside \\boxed{}.\n\n"
)

# Field names in the Maxwell-Jia/AIME_2024 schema. AIME24/AIME25 use
# different lowercase field names — see those loaders.
PROBLEM_FIELD = "Problem"
ANSWER_FIELD = "Answer"


class AIMEBenchmark(AIPerfLoggerMixin):
    """AIME (American Invitational Mathematics Examination) benchmark loader.

    Loads AIME competition problems from ``Maxwell-Jia/AIME_2024`` (train
    split) and produces ``BenchmarkProblem`` objects ready for both the
    completions endpoint (flat ``prompt``) and the chat endpoint
    (``raw_messages``). Few-shot examples are drawn sequentially from the
    beginning of the dataset.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.user_config = user_config

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load every AIME problem and format it for the LLM.

        Args:
            tasks: Ignored — AIME has no subtasks. Accepted for protocol
                parity with other benchmarks that support task filtering.
            n_shots: Number of few-shot examples to prepend (drawn from
                the start of the dataset). 0 disables few-shot prompting.
            enable_cot: When True, append ``Let's think step by step.`` to
                each query, encouraging chain-of-thought reasoning before
                the final ``\\boxed{}`` answer.

        Returns:
            One ``BenchmarkProblem`` per dataset row, in dataset order.
        """
        ds: Dataset = await asyncio.to_thread(load_dataset, DATASET_NAME, split="train")
        return await asyncio.to_thread(self._build_problems, ds, n_shots, enable_cot)

    def _build_problems(
        self, ds: Dataset, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        few_shots = self._build_few_shots(ds, n_shots)
        problems: list[BenchmarkProblem] = []
        for row in ds:
            prompt = self._format_prompt(row, few_shots, enable_cot)
            raw_messages = self._build_chat_messages(row, few_shots, enable_cot)
            problems.append(
                BenchmarkProblem(
                    prompt=prompt,
                    ground_truth=str(row[ANSWER_FIELD]),
                    task=TASK_NAME,
                    metadata={"generation_size": DEFAULT_GENERATION_SIZE},
                    raw_messages=raw_messages,
                )
            )
        return problems

    def _build_few_shots(self, ds: Dataset, n_shots: int) -> list[dict[str, str]]:
        """Build few-shot examples by sequentially sampling the dataset start.

        AIME has no separate dev/validation split, so we draw few-shot
        examples from the same train split. This means the first ``n_shots``
        problems will appear in their own prompts; lighteval's competition
        configs do the same when no held-out pool is available.
        """
        if n_shots <= 0:
            return []
        size = min(n_shots, len(ds))
        return [self._format_example(ds[i]) for i in range(size)]

    def _format_example(self, row: dict[str, Any]) -> dict[str, str]:
        """Format a dataset row as a few-shot example.

        Returns a ``{problem, answer, formatted}`` triple where ``formatted``
        is the flat completions form (used in the completions prompt) and
        ``problem``/``answer`` are reused when constructing chat messages.
        """
        answer = str(row[ANSWER_FIELD])
        problem = row[PROBLEM_FIELD]
        return {
            "problem": problem,
            "answer": answer,
            "formatted": f"Problem: {problem}\nAnswer: \\boxed{{{answer}}}",
        }

    def _format_prompt(
        self,
        row: dict[str, Any],
        few_shots: list[dict[str, str]],
        enable_cot: bool,
    ) -> str:
        """Build the flat completions prompt: instruction + shots + query."""
        few_shot_text = "\n\n".join(ex["formatted"] for ex in few_shots)
        if few_shot_text:
            few_shot_text += "\n\n"

        problem = row[PROBLEM_FIELD]
        if enable_cot:
            query = f"Problem: {problem}\nLet's think step by step.\nAnswer:"
        else:
            query = f"Problem: {problem}\nAnswer:"

        return INSTRUCTION_PREFIX + few_shot_text + query

    def _build_chat_messages(
        self,
        row: dict[str, Any],
        few_shots: list[dict[str, str]],
        enable_cot: bool,
    ) -> list[AccuracyChatMessage]:
        """Build multi-turn chat messages following lighteval's PromptManager.

        - First few-shot user message includes the instruction prefix.
        - Subsequent few-shot user messages contain only ``Problem: ... Answer:``.
        - Each few-shot assistant message holds ``\\boxed{answer}`` so the
          model is primed to emit the same boxed format.
        - Main query repeats the user-side format without re-instructing.
        - When there are no few-shots, the instruction is prepended to the
          single user message.
        """
        messages: list[AccuracyChatMessage] = []

        for ix, ex in enumerate(few_shots):
            q = f"Problem: {ex['problem']}\nAnswer:"
            if ix == 0:
                q = INSTRUCTION_PREFIX + q
            messages.append({"role": "user", "content": q})
            messages.append(
                {"role": "assistant", "content": f"\\boxed{{{ex['answer']}}}"}
            )

        problem = row[PROBLEM_FIELD]
        if enable_cot:
            main_q = f"Problem: {problem}\nLet's think step by step.\nAnswer:"
        else:
            main_q = f"Problem: {problem}\nAnswer:"

        if not few_shots:
            main_q = INSTRUCTION_PREFIX + main_q

        messages.append({"role": "user", "content": main_q})
        return messages
