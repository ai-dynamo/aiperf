# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPQA-Diamond benchmark loader, aligned with the trt-llm lighteval reference.

Mirrors ``acc_bench_lighteval.py:gpqa_diamond``:

    gpqa_diamond = LightevalTaskConfig(
        name="gpqa:diamond",
        prompt_function=gpqa_prompt_fn,
        hf_repo="Idavidrein/gpqa",
        hf_subset="gpqa_diamond",
        evaluation_splits=["train"],
        few_shots_split=None,
        generation_size=32768,
        metric=[gpqa_metric],
        stop_sequence=[],
        trust_dataset=True,
    )

The recipe's ``gpqa_prompt_fn`` builds the simple-evals template:

    Answer the following multiple choice question. The last line of
    your response should be of the following format: 'Answer: $LETTER'
    (without quotes) where LETTER is one of ABCD. Think step by step
    before answering.

    {Question}

    A) {A}
    B) {B}
    C) {C}
    D) {D}

The recipe's prompt_fn shuffles options with ``random.randint(0, 3)``
(stochastic, different per call). Aiperf instead uses **SHA-256-seeded
deterministic shuffling** (per the user direction during the alignment
review) so gold positions are reproducible across runs while still
distributed uniformly. This is the one intentional deviation from the
trt-llm reference, documented in
``docs/accuracy/accuracy-benchmarking.md``.

Pair with ``LightevalGPQAGrader`` (default), which extracts via
``IndicesExtractionConfig(prefix_for_extraction="NativeLetters")`` to
match the recipe's ``gpqa_metric``.

Reference:
    trt-llm-benchmark-recipe/src/accuracy/acc_bench_lighteval.py:108,170
"""

from __future__ import annotations

import asyncio
import hashlib
import random
from typing import TYPE_CHECKING, Any

from datasets import Dataset, load_dataset

from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.mixins import AIPerfLoggerMixin

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun

DATASET_NAME = "Idavidrein/gpqa"
DATASET_CONFIG = "gpqa_diamond"
TASK_NAME = "gpqa_diamond"

# lighteval's gpqa_diamond task config: ``generation_size=32768``.
DEFAULT_GENERATION_SIZE = 32768

# 4 choices per question (1 correct + 3 distractors).
NUM_CHOICES = 4

# Width of the SHA-256-derived seed when modded down to a 32-bit
# Python ``random.Random`` seed.
_SEED_MODULUS = 2**32

# Schema field names in the Idavidrein/gpqa dataset (Title Case with
# spaces — the upstream's choice).
QUESTION_FIELD = "Question"
CORRECT_ANSWER_FIELD = "Correct Answer"
INCORRECT_ANSWER_FIELDS = (
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
)
DOMAIN_FIELD = "High-level domain"
SUBDOMAIN_FIELD = "Subdomain"

# Recipe's ``gpqa_prompt_fn`` template. The model is told to emit
# ``Answer: $LETTER`` so ``LightevalGPQAGrader`` (with
# ``IndicesExtractionConfig(prefix_for_extraction="NativeLetters")``)
# can extract the letter cleanly.
_PROMPT_TEMPLATE = (
    "Answer the following multiple choice question. The last line of "
    "your response should be of the following format: 'Answer: $LETTER' "
    "(without quotes) where LETTER is one of ABCD. Think step by step "
    "before answering.\n\n"
    "{Question}\n\n"
    "A) {A}\n"
    "B) {B}\n"
    "C) {C}\n"
    "D) {D}"
)


def _seeded_shuffle_indices(key: str, n: int) -> list[int]:
    """Return a deterministic permutation of ``range(n)`` seeded by ``key``.

    Uses the leading 32 bits of SHA-256(key) as the seed for Python's
    ``random.Random``. This gives a stable, locale-independent,
    Python-version-independent permutation: regenerating prompts on a
    new machine produces identical letter orderings.

    The recipe shuffles via ``random.randint(0, 3)`` (stochastic per
    call) — see the module docstring for why aiperf chose
    determinism instead.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    seed = int(digest, 16) % _SEED_MODULUS
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    return indices


class GPQADiamondBenchmark(AIPerfLoggerMixin):
    """GPQA-Diamond lighteval-aligned benchmark loader.

    Loads ``Idavidrein/gpqa`` (config ``gpqa_diamond``, train split).
    Each row's correct + 3 incorrect answers are deterministically
    shuffled into A/B/C/D positions and rendered with the simple-evals
    template (matching ``gpqa_prompt_fn``). Pair with
    ``LightevalGPQAGrader`` for grading parity with the recipe.
    """

    def __init__(self, run: BenchmarkRun, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.run = run

    async def load_problems(
        self, tasks: list[str] | None, n_shots: int, enable_cot: bool
    ) -> list[BenchmarkProblem]:
        """Load GPQA-Diamond problems lighteval-style.

        Args:
            tasks: Ignored — lighteval's gpqa_diamond task has no
                subtask filtering (per-row High-level domain is in
                metadata for post-run reporting).
            n_shots: Ignored — the lighteval reference is zero-shot
                (``few_shots_split=None``).
            enable_cot: Ignored — the simple-evals template already
                includes "Think step by step before answering."

        Returns:
            One ``BenchmarkProblem`` per dataset row, in dataset order.
            ``ground_truth`` is the gold letter ("A", "B", "C", or
            "D") so ``LightevalGPQAGrader`` can pass it directly into
            its ``Doc.choices=["A","B","C","D"], gold_index=...``
            shape.
        """
        ds: Dataset = await asyncio.to_thread(
            load_dataset, DATASET_NAME, DATASET_CONFIG, split="train"
        )
        return await asyncio.to_thread(self._build_problems, ds)

    def _build_problems(self, ds: Dataset) -> list[BenchmarkProblem]:
        problems: list[BenchmarkProblem] = []
        for row in ds:
            choices, gold_letter = self._build_choices(row)
            prompt = self._format_prompt(row, choices)
            messages: list[AccuracyChatMessage] = [{"role": "user", "content": prompt}]
            problems.append(
                BenchmarkProblem(
                    prompt=prompt,
                    ground_truth=gold_letter,
                    task=TASK_NAME,
                    metadata={
                        "domain": row.get(DOMAIN_FIELD, ""),
                        "subdomain": row.get(SUBDOMAIN_FIELD, ""),
                        "generation_size": DEFAULT_GENERATION_SIZE,
                    },
                    raw_messages=messages,
                )
            )
        return problems

    @staticmethod
    def _build_choices(row: dict[str, Any]) -> tuple[list[str], str]:
        """Assemble 4 lettered choices and report the gold letter.

        Uses SHA-256-seeded permutation (see ``_seeded_shuffle_indices``)
        — deterministic per-question shuffle, distinct from the
        recipe's stochastic ``random.randint(0, 3)``.
        """
        raw = [
            row[CORRECT_ANSWER_FIELD],
            row[INCORRECT_ANSWER_FIELDS[0]],
            row[INCORRECT_ANSWER_FIELDS[1]],
            row[INCORRECT_ANSWER_FIELDS[2]],
        ]
        order = _seeded_shuffle_indices(row[QUESTION_FIELD], len(raw))
        ordered = [raw[i] for i in order]
        gold_index = order.index(0)
        gold_letter = "ABCD"[gold_index]
        return ordered, gold_letter

    def _format_prompt(self, row: dict[str, Any], choices: list[str]) -> str:
        """Render the simple-evals template byte-equal to the recipe."""
        return _PROMPT_TEMPLATE.format(
            Question=row[QUESTION_FIELD],
            A=choices[0],
            B=choices[1],
            C=choices[2],
            D=choices[3],
        )
