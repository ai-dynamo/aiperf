# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MMLU benchmark loader, ported from lighteval's tasks/tasks/mmlu.py.

Faithfully reproduces lighteval's prompt format, dataset source, few-shot
strategy, and gold-answer representation to ensure scoring parity when
paired with MultipleChoiceGrader (which ports lighteval's ExactMatches).

lighteval reference: lighteval/src/lighteval/tasks/tasks/mmlu.py
"""

from __future__ import annotations

import asyncio
import random
from collections import defaultdict
from itertools import cycle
from string import ascii_uppercase
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.config import UserConfig
from aiperf.common.mixins import AIPerfLoggerMixin

DATASET_NAME = "lighteval/mmlu"

CHOICES = [f" {c}" for c in ascii_uppercase[:4]]

GENERATION_SIZE = 5
STOP_SEQUENCE = ["\n"]

MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


class MMLUBenchmark(AIPerfLoggerMixin):
    """MMLU (Massive Multitask Language Understanding) benchmark loader.

    Ported from lighteval's mmlu_prompt() and task configs. Matches lighteval
    on dataset source, prompt format, few-shot split, and gold representation.
    """

    def __init__(self, user_config: UserConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.user_config = user_config

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        subjects = self._resolve_subjects(tasks)
        problems: list[BenchmarkProblem] = []

        for subject in subjects:
            ds: DatasetDict = await asyncio.to_thread(
                load_dataset, DATASET_NAME, subject
            )
            subject_problems = await asyncio.to_thread(
                self._build_subject_problems, ds, subject, n_shots, enable_cot
            )
            problems.extend(subject_problems)

        return problems

    def _build_subject_problems(
        self,
        ds: DatasetDict,
        subject: str,
        n_shots: int,
        enable_cot: bool,
    ) -> list[BenchmarkProblem]:
        few_shots = self._build_few_shots(ds, n_shots)
        problems: list[BenchmarkProblem] = []
        for row in ds["test"]:
            prompt = self._format_prompt(row, subject, few_shots, enable_cot)
            raw_messages = self._build_chat_messages(
                row, subject, few_shots, enable_cot
            )
            gold_ix = (
                ascii_uppercase.index(row["answer"])
                if isinstance(row["answer"], str)
                else row["answer"]
            )
            problems.append(
                BenchmarkProblem(
                    prompt=prompt,
                    ground_truth=CHOICES[gold_ix],
                    task=subject,
                    metadata={
                        "subject": subject,
                        "generation_size": GENERATION_SIZE,
                        "stop_sequence": STOP_SEQUENCE,
                    },
                    raw_messages=raw_messages,
                )
            )
        return problems

    def _resolve_subjects(self, tasks: list[str] | None) -> list[str]:
        if not tasks or "all" in tasks:
            return MMLU_SUBJECTS
        resolved: list[str] = []
        for t in tasks:
            if t not in MMLU_SUBJECTS:
                raise ValueError(
                    f"Unknown MMLU subject '{t}'. "
                    f"See docs/accuracy/accuracy-benchmarking.md for the list of {len(MMLU_SUBJECTS)} valid subjects."
                )
            resolved.append(t)
        return resolved

    def _build_few_shots(self, ds: DatasetDict, n_shots: int) -> list[dict[str, str]]:
        if n_shots <= 0:
            return []
        if "dev" in ds:
            source = ds["dev"]
        elif "validation" in ds:
            source = ds["validation"]
        else:
            return []
        indices = self._balanced_sample_indices(source, n_shots)
        return [self._format_example(source[i]) for i in indices]

    @staticmethod
    def _balanced_sample_indices(source: Dataset, n_shots: int) -> list[int]:
        """Select few-shot indices using lighteval's balanced strategy.

        Groups examples by gold answer text, then round-robin samples across
        groups to ensure answer-label diversity. Matches lighteval's
        FewShotSampler with sorting="balanced" and variance_seed=0.
        """
        label_to_indices: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(source):
            gold_ix = (
                ascii_uppercase.index(row["answer"])
                if isinstance(row["answer"], str)
                else row["answer"]
            )
            label_to_indices[row["choices"][gold_ix]].append(i)

        counts_to_labels: dict[int, list[str]] = defaultdict(list)
        for label, indices in sorted(label_to_indices.items()):
            counts_to_labels[len(indices)].append(label)

        rng = random.Random(0)
        sorted_labels: list[str] = []
        for count in sorted(counts_to_labels, reverse=True):
            labels = counts_to_labels[count]
            rng.shuffle(labels)
            sorted_labels.extend(labels)

        result: list[int] = []
        remaining = min(len(source), n_shots + 1)
        labels_iter = cycle(sorted_labels)
        while remaining > 0:
            nl = next(labels_iter, None)
            if nl is None:
                break
            pool = label_to_indices[nl]
            if not pool:
                continue
            idx = rng.randrange(len(pool))
            result.append(pool.pop(idx))
            remaining -= 1

        return result[:n_shots]

    def _format_example(self, row: dict[str, Any]) -> dict[str, str]:
        """Format a single dataset row as a few-shot example.

        Mirrors lighteval's approach: query + choices[gold_index].
        """
        choices_str = "".join(
            f"\n{key}. {choice}"
            for key, choice in zip(ascii_uppercase, row["choices"], strict=False)
        )
        gold_ix = (
            ascii_uppercase.index(row["answer"])
            if isinstance(row["answer"], str)
            else row["answer"]
        )
        gold_choice = CHOICES[gold_ix]
        return {
            "question": row["question"],
            "choices": choices_str,
            "answer": gold_choice,
            "formatted": (
                f"Question: {row['question']}{choices_str}\nAnswer:{gold_choice}"
            ),
        }

    def _format_prompt(
        self,
        row: dict[str, Any],
        subject: str,
        few_shots: list[dict[str, str]],
        enable_cot: bool,
    ) -> str:
        """Build the full prompt, matching lighteval's mmlu_prompt() format.

        lighteval structure:
          instruction + few_shot_examples (query+gold each) + current query
        """
        instruction = (
            "The following are multiple choice questions (with answers) "
            f"about {subject.replace('_', ' ')}.\n\n"
        )

        few_shot_text = "\n\n".join(ex["formatted"] for ex in few_shots)
        if few_shot_text:
            few_shot_text += "\n\n"

        choices_str = "".join(
            f"\n{key}. {choice}"
            for key, choice in zip(ascii_uppercase, row["choices"], strict=False)
        )
        query = f"Question: {row['question']}{choices_str}\nAnswer:"

        if enable_cot:
            query = f"Question: {row['question']}{choices_str}\nLet's think step by step.\nAnswer:"

        return instruction + few_shot_text + query

    def _build_chat_messages(
        self,
        row: dict[str, Any],
        subject: str,
        few_shots: list[dict[str, str]],
        enable_cot: bool,
    ) -> list[AccuracyChatMessage]:
        """Build lighteval-style multi-turn chat messages.

        Replicates lighteval's PromptManager._prepare_chat_template() for MMLU:
        - First few-shot user message includes the instruction prefix
        - Subsequent few-shot user messages contain only "Question: ...Answer:"
        - Each few-shot answer is a separate assistant message with the gold choice
        - Main query follows the same stripped format

        This allows --endpoint-type chat to produce token sequences identical
        to lighteval when talking to the same sglang server.
        """
        instruction = (
            "The following are multiple choice questions (with answers) "
            f"about {subject.replace('_', ' ')}.\n\n"
        )

        messages: list[AccuracyChatMessage] = []

        for ix, ex in enumerate(few_shots):
            q = f"Question: {ex['question']}{ex['choices']}\nAnswer:"
            if ix == 0:
                q = instruction + q
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": ex["answer"]})

        choices_str = "".join(
            f"\n{key}. {choice}"
            for key, choice in zip(ascii_uppercase, row["choices"], strict=False)
        )

        if enable_cot:
            main_q = f"Question: {row['question']}{choices_str}\nLet's think step by step.\nAnswer:"
        else:
            main_q = f"Question: {row['question']}{choices_str}\nAnswer:"

        if not few_shots:
            main_q = instruction + main_q

        messages.append({"role": "user", "content": main_q})

        return messages
