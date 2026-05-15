# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AIME benchmark loader, aligned with the trt-llm benchmark recipe.

Loads ``Maxwell-Jia/AIME_2024`` and renders prompts character-for-
character identical to the recipe's ``AIMETemplate.generate_output``
(``trt-llm-benchmark-recipe/src/accuracy/aime/template.py``). This is
the DeepEval-backed AIME path in the trt-llm recipe — distinct from the
lighteval-backed ``aime24``/``aime25`` loaders that pin to specific
years and use lighteval's prompt manager.

Reference: trt-llm-benchmark-recipe/src/accuracy/aime/{aime,template}.py
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

# Recipe defaults: ``n_shots=8`` (capped), ``enable_cot=True``,
# ``n_problems=30``. We keep the cap available via the ``tasks``
# argument's "n_problems:N" semantics if needed in the future, but for
# now we mirror the recipe's evaluation: 8 shots, CoT on, all 30
# problems graded.
DEFAULT_N_SHOTS = 8
MAX_N_SHOTS = 8
DEFAULT_ENABLE_COT = True

# Recipe runs with the model's full reasoning budget. lighteval-aligned
# AIME tasks use 32768; we match that here so reasoning models have
# room to think before emitting the boxed answer.
DEFAULT_GENERATION_SIZE = 32768

# Recipe ``aime_test.json`` ships this as the system prompt; aiperf
# auto-injects it via the per-benchmark ``default_system_prompt``
# mechanism (plugins.yaml metadata) when the user doesn't override
# with ``--accuracy-system-prompt``.
DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Recipe ``AIMETemplate.generate_output`` instruction header — used
# only when there is at least one few-shot example. With zero shots,
# the recipe emits no header at all.
FEW_SHOT_HEADER = (
    "The following are problems from the American Invitational "
    "Mathematics Examination (AIME) 2024. AIME is a prestigious high "
    "school mathematics competition known for its challenging "
    "mathematical problems.\n\n"
)

# Recipe ``AIMETemplate.generate_output`` trailing CoT trigger. Note
# this lives AFTER the ``**Answer**:`` marker, not before.
COT_SUFFIX = "Let's think step-by-step."
NO_COT_SUFFIX = "No explanation needed. Just return a number."

# Schema field names in Maxwell-Jia/AIME_2024.
PROBLEM_FIELD = "Problem"
SOLUTION_FIELD = "Solution"
ANSWER_FIELD = "Answer"


class AIMEBenchmark(AIPerfLoggerMixin):
    """AIME benchmark loader matching the trt-llm DeepEval reference.

    Loads ``Maxwell-Jia/AIME_2024`` (train split) and produces
    ``BenchmarkProblem`` objects whose flat ``prompt`` is byte-equal to
    the recipe's ``AIMETemplate.generate_output`` output. Pair with
    ``MathGrader`` for the recipe's ``math_equal`` grading semantics.

    Default configuration mirrors ``aime_test.json``:
    - ``n_shots=8`` (the recipe asserts ``n_shots <= 8``; see
      ``MAX_N_SHOTS``)
    - ``enable_cot=True``
    - System prompt: ``"Please reason step by step, and put your final
      answer within \\boxed{}."`` (set via ``default_system_prompt``
      metadata in ``plugins.yaml`` so users see it documented)
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.user_config = user_config

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load all AIME problems and format prompts via the recipe template.

        Args:
            tasks: Ignored — AIME has no subtasks. Accepted for protocol
                parity with other benchmarks.
            n_shots: Number of few-shot examples to include
                (capped at :data:`MAX_N_SHOTS`, mirroring the recipe's
                ``assert n_shots <= 8``). 0 emits no header and no
                examples.
            enable_cot: When True, append ``Let's think step-by-step.``
                after the final ``**Answer**:`` marker; when False,
                append ``No explanation needed. Just return a number.``

        Returns:
            One ``BenchmarkProblem`` per dataset row, in dataset order.
        """
        if n_shots > MAX_N_SHOTS:
            raise ValueError(
                f"AIME supports at most {MAX_N_SHOTS} few-shot examples "
                f"(got {n_shots}). The trt-llm reference asserts "
                f"``n_shots <= 8``; raise that limit upstream first."
            )
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
        """Few-shot examples drawn sequentially from the start of the split.

        The recipe takes ``train_set[:n_shots]`` directly. AIME 2024 has
        no held-out pool, so the first ``n_shots`` problems appear in
        their own prompts as well — lighteval and the recipe both make
        this trade-off.
        """
        if n_shots <= 0:
            return []
        size = min(n_shots, len(ds))
        return [self._format_example(ds[i]) for i in range(size)]

    def _format_example(self, row: dict[str, Any]) -> dict[str, str]:
        """Bundle the per-row data the prompt builders need.

        Stores both the bare ``problem`` / ``solution`` / ``answer``
        fields and the recipe's ``formatted`` form. The actual final
        rendering is decided by ``_format_example_block`` (which honors
        ``enable_cot``: solutions only appear when CoT is on).
        """
        answer = str(row[ANSWER_FIELD])
        problem = row[PROBLEM_FIELD]
        solution = str(row.get(SOLUTION_FIELD, ""))
        return {
            "problem": problem,
            "solution": solution,
            "answer": answer,
        }

    @staticmethod
    def _format_example_block(example: dict[str, str], enable_cot: bool) -> str:
        """Render one few-shot example exactly as the recipe does.

        Mirrors ``AIMETemplate.format_example``:
        - ``**Problem**: <q>``
        - When CoT: ``**Solution**: <s>``
        - ``**Answer**: <a>``
        """
        block = "**Problem**: " + example["problem"] + "\n"
        if enable_cot:
            block += "**Solution**: " + example["solution"] + "\n"
        block += "**Answer**: " + example["answer"]
        return block

    def _format_prompt(
        self,
        row: dict[str, Any],
        few_shots: list[dict[str, str]],
        enable_cot: bool,
    ) -> str:
        """Render the flat completions prompt byte-equal to the recipe.

        Recipe ``AIMETemplate.generate_output`` structure:

            [FEW_SHOT_HEADER if n_shots > 0]
            <example_block>\\n\\n
            <example_block>\\n\\n
            ...
            **Problem**: <test_q>\\n**Answer**: \\n\\n
            <Let's think step-by-step. | No explanation needed. Just return a number.>
        """
        prompt = ""
        if few_shots:
            prompt += FEW_SHOT_HEADER
            for ex in few_shots:
                prompt += self._format_example_block(ex, enable_cot) + "\n\n"

        prompt += "**Problem**: " + row[PROBLEM_FIELD] + "\n**Answer**: \n\n"
        prompt += COT_SUFFIX if enable_cot else NO_COT_SUFFIX
        return prompt

    def _build_chat_messages(
        self,
        row: dict[str, Any],
        few_shots: list[dict[str, str]],
        enable_cot: bool,
    ) -> list[AccuracyChatMessage]:
        """Build a single user message that wraps the recipe-rendered prompt.

        The recipe doesn't use multi-turn chat for AIME — DeepEval sends
        the full prompt as a single string. We mirror that by emitting
        a single user message containing the same flat prompt that
        ``_format_prompt`` produces, so behavior is identical for both
        completions and chat endpoints.
        """
        return [
            {
                "role": "user",
                "content": self._format_prompt(row, few_shots, enable_cot),
            }
        ]
