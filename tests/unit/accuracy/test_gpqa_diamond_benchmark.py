# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``GPQADiamondBenchmark`` after lighteval alignment.

Pins:
1. The simple-evals prompt template byte-equal to the recipe's
   ``gpqa_prompt_fn``.
2. Deterministic SHA-256-seeded shuffling (intentional deviation from
   the recipe's stochastic ``random.randint``).
3. ``ground_truth`` is the bare gold letter (``"A"``..``"D"``), the
   shape ``LightevalGPQAGrader`` expects.
"""

from __future__ import annotations

from collections import Counter
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.accuracy.benchmarks.gpqa_diamond import (
    DEFAULT_GENERATION_SIZE,
    NUM_CHOICES,
    TASK_NAME,
    GPQADiamondBenchmark,
    _seeded_shuffle_indices,
)
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run


def _make_run():
    return make_benchmark_run(
        model_names=["test-model"],
        endpoint_type=EndpointType.COMPLETIONS,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.GPQA_DIAMOND},
    )


def _make_row(
    question: str = "What is 2+2?",
    correct: str = "4",
    incorrect: tuple[str, str, str] = ("3", "5", "6"),
    domain: str = "Physics",
    subdomain: str = "Mechanics",
) -> dict[str, Any]:
    return {
        "Question": question,
        "Correct Answer": correct,
        "Incorrect Answer 1": incorrect[0],
        "Incorrect Answer 2": incorrect[1],
        "Incorrect Answer 3": incorrect[2],
        "High-level domain": domain,
        "Subdomain": subdomain,
    }


def _make_fake_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    ds = MagicMock()
    ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return ds


class TestPromptTemplateMatchesRecipe:
    """The flat prompt is byte-equal to the recipe's
    ``gpqa_prompt_fn`` simple-evals template."""

    @pytest.mark.asyncio
    async def test_prompt_uses_simple_evals_template(self) -> None:
        rows = [_make_row(question="Q?", correct="W", incorrect=("X", "Y", "Z"))]
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        prompt = problems[0].prompt
        assert prompt.startswith(
            "Answer the following multiple choice question. The last "
            "line of your response should be of the following format: "
            "'Answer: $LETTER'"
        )
        assert "Think step by step before answering." in prompt
        assert "Q?" in prompt
        # The four-letter format uses ``A) `` / ``B) `` etc — NOT
        # ``A. ``. The grader's ``Answer: $LETTER`` extractor matches
        # against ``ABCD`` regardless, but the prompt format itself
        # should match the recipe.
        assert "A) " in prompt
        assert "B) " in prompt
        assert "C) " in prompt
        assert "D) " in prompt

    @pytest.mark.asyncio
    async def test_all_four_choices_present(self) -> None:
        rows = [_make_row(correct="GOLD", incorrect=("DECOY1", "DECOY2", "DECOY3"))]
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        prompt = problems[0].prompt
        assert "GOLD" in prompt
        assert "DECOY1" in prompt
        assert "DECOY2" in prompt
        assert "DECOY3" in prompt


class TestGroundTruthIsBareLetter:
    """``LightevalGPQAGrader`` expects the gold as a bare letter
    (``"A"``..``"D"``). The previous SHA-seeded grader stored
    ``" A"`` (leading-space CHOICES convention) — we no longer use
    that since lighteval's ``IndicesExtractionConfig`` doesn't need
    it."""

    @pytest.mark.asyncio
    async def test_ground_truth_is_letter(self) -> None:
        rows = [_make_row(question=f"Q{i}", correct="GOLD") for i in range(3)]
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        for p in problems:
            assert p.ground_truth in ("A", "B", "C", "D")

    @pytest.mark.asyncio
    async def test_ground_truth_letter_indexes_into_correct_text(self) -> None:
        rows = [_make_row(question="Q1", correct="GOLD")]
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        gold_letter = problems[0].ground_truth
        # The line ``A) GOLD`` (or B) C) D)) for the gold_letter slot
        # must appear in the prompt.
        assert f"\n{gold_letter}) GOLD" in problems[0].prompt


class TestSeededShuffleIsDeterministic:
    """SHA-256-seeded permutation: same key → same permutation,
    distinct keys → distinct permutations, distribution roughly
    uniform across many keys."""

    def test_same_key_same_permutation(self) -> None:
        a = _seeded_shuffle_indices("hello", NUM_CHOICES)
        b = _seeded_shuffle_indices("hello", NUM_CHOICES)
        assert a == b

    def test_different_keys_different_permutations(self) -> None:
        a = _seeded_shuffle_indices("alpha", NUM_CHOICES)
        b = _seeded_shuffle_indices("beta", NUM_CHOICES)
        assert a != b

    def test_distribution_across_many_keys(self) -> None:
        positions = Counter()
        for i in range(1000):
            order = _seeded_shuffle_indices(f"q-{i}", NUM_CHOICES)
            positions[order.index(0)] += 1
        # Each of the 4 slots should land ~250 times; allow ±20%.
        for slot in range(NUM_CHOICES):
            assert 200 <= positions[slot] <= 300, (
                f"slot {slot} got {positions[slot]} (expected ~250)"
            )


class TestNShotsAndCoTAreIgnored:
    @pytest.mark.asyncio
    async def test_n_shots_argument_does_not_affect_prompt(self) -> None:
        rows = [_make_row()]
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            zero_shot = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
            five_shot = await bench.load_problems(
                tasks=None, n_shots=5, enable_cot=False
            )
        assert zero_shot[0].prompt == five_shot[0].prompt


class TestLoadProblemsCore:
    @pytest.mark.asyncio
    async def test_returns_one_problem_per_row(self) -> None:
        rows = [_make_row(question=f"Q{i}") for i in range(3)]
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert len(problems) == 3
        assert all(isinstance(p, BenchmarkProblem) for p in problems)

    @pytest.mark.asyncio
    async def test_metadata_carries_domain_and_gen_size(self) -> None:
        rows = [_make_row(domain="Chemistry", subdomain="Organic")]
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        meta = problems[0].metadata
        assert meta["domain"] == "Chemistry"
        assert meta["subdomain"] == "Organic"
        assert meta["generation_size"] == DEFAULT_GENERATION_SIZE
        assert DEFAULT_GENERATION_SIZE == 32768

    @pytest.mark.asyncio
    async def test_task_name_is_constant(self) -> None:
        rows = [_make_row(domain="Physics"), _make_row(domain="Biology")]
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert all(p.task == TASK_NAME for p in problems)


class TestPathologicalDatasetRows:
    @pytest.mark.asyncio
    async def test_empty_dataset_returns_empty_list(self) -> None:
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset([]),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems == []

    @pytest.mark.asyncio
    async def test_optional_subdomain_field_missing(self) -> None:
        rows = [
            {
                "Question": "Q?",
                "Correct Answer": "yes",
                "Incorrect Answer 1": "no",
                "Incorrect Answer 2": "maybe",
                "Incorrect Answer 3": "perhaps",
                "High-level domain": "Physics",
                # Subdomain absent.
            }
        ]
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].metadata["subdomain"] == ""

    @pytest.mark.asyncio
    async def test_unicode_question_text_preserved(self) -> None:
        rows = [_make_row(question="∮ E·dl = ?")]
        with patch(
            "aiperf.accuracy.benchmarks.gpqa_diamond.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = GPQADiamondBenchmark(run=_make_run())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert "∮ E·dl" in problems[0].prompt
