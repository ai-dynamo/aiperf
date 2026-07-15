# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from TIGER-AI-Lab/MMLU-Pro
# (https://github.com/TIGER-AI-Lab/MMLU-Pro), licensed under Apache-2.0.

"""MMLU-Pro benchmark loader, ported from TIGER-AI-Lab MMLU-Pro.

Byte-equal to evaluate_from_api.py: dataset TIGER-Lab/MMLU-Pro, per-category
instruction, N/A-filtered A-J options, validation-split CoT few-shots, and the
"Let's think step by step." query primer. Pairs with MMLUProGrader.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from aiperf.accuracy.benchmarks._datasets_compat import load_dataset
from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

DATASET_NAME = "TIGER-Lab/MMLU-Pro"
CHOICE_MAP = "ABCDEFGHIJ"
GENERATION_SIZE = 4000

MMLU_PRO_CATEGORIES = [
    "biology",
    "business",
    "chemistry",
    "computer science",
    "economics",
    "engineering",
    "health",
    "history",
    "law",
    "math",
    "philosophy",
    "physics",
    "psychology",
    "other",
]


class MMLUProBenchmark(AIPerfLoggerMixin):
    """MMLU-Pro benchmark loader (14 categories, up to 10 options, CoT-native)."""

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        categories = self._resolve_categories(tasks)
        dd = await asyncio.to_thread(load_dataset, DATASET_NAME)
        return await asyncio.to_thread(self._build, dd, categories, n_shots, enable_cot)

    def _resolve_categories(self, tasks: list[str] | None) -> list[str]:
        if not tasks or "all" in tasks:
            return MMLU_PRO_CATEGORIES
        resolved: list[str] = []
        for t in tasks:
            if t not in MMLU_PRO_CATEGORIES:
                raise ValueError(
                    f"Unknown MMLU-Pro category '{t}'. Valid categories: "
                    f"{', '.join(MMLU_PRO_CATEGORIES)}."
                )
            resolved.append(t)
        return resolved

    def _build(
        self, dd: Any, categories: list[str], n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        cot_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in dd["validation"]:
            cot_by_cat[row["category"]].append(row)

        wanted = set(categories)
        problems: list[BenchmarkProblem] = []
        for row in dd["test"]:
            category = row["category"]
            if category not in wanted:
                continue
            few_shots = cot_by_cat.get(category, [])[:n_shots]
            prompt = self._build_prompt(row, category, few_shots, enable_cot)
            messages: list[AccuracyChatMessage] = [{"role": "user", "content": prompt}]
            problems.append(
                BenchmarkProblem(
                    prompt=prompt,
                    ground_truth=str(row["answer"]),
                    task=category,
                    metadata={
                        "category": category,
                        "generation_size": GENERATION_SIZE,
                    },
                    raw_messages=messages,
                )
            )
        return problems

    @staticmethod
    def _options(options: list[str]) -> list[str]:
        return [opt for opt in options if opt != "N/A"]

    def _format_example(
        self, question: str, options: list[str], cot_content: str
    ) -> str:
        if cot_content == "":
            cot_content = "Let's think step by step."
        if cot_content.startswith("A: "):
            cot_content = cot_content[3:]
        example = f"Question: {question}\nOptions: "
        for i, opt in enumerate(self._options(options)):
            example += f"{CHOICE_MAP[i]}. {opt}\n"
        example += f"Answer: {cot_content}\n\n"
        return example

    def _build_prompt(
        self,
        row: dict[str, Any],
        category: str,
        few_shots: list[dict[str, Any]],
        enable_cot: bool,
    ) -> str:
        if enable_cot:
            # Upstream-parity CoT instruction: reason, then "The answer is (X)".
            instruction = (
                "The following are multiple choice questions (with answers) about "
                f"{category}. Think step by step and then output the answer in the "
                'format of "The answer is (X)" at the end.\n\n'
            )
        else:
            # Non-CoT (AIPerf extension): request a bare letter, no CoT directive.
            instruction = (
                "The following are multiple choice questions (with answers) about "
                f"{category}.\n\n"
            )
        prompt = instruction
        for ex in few_shots:
            # CoT few-shots carry the reference reasoning; non-CoT few-shots show
            # only the bare gold letter so the model learns to answer directly.
            ex_answer = ex["cot_content"] if enable_cot else str(ex["answer"])
            prompt += self._format_example(ex["question"], ex["options"], ex_answer)

        if enable_cot:
            prompt += self._format_example(row["question"], row["options"], "")
        else:
            # Bare Answer: trailer, no "Let's think step by step." primer.
            example = f"Question: {row['question']}\nOptions: "
            for i, opt in enumerate(self._options(row["options"])):
                example += f"{CHOICE_MAP[i]}. {opt}\n"
            example += "Answer: \n\n"
            prompt += example
        return prompt
