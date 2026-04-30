# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``AIMEBenchmark``.

The HuggingFace dataset is mocked end-to-end so the suite is fully offline
and deterministic. Coverage targets:

1. Prompt construction (completions form): instruction, few-shots, CoT.
2. Chat-message construction: lighteval-style multi-turn structure with
   the instruction on the first user message only and ``\\boxed{}``
   assistant primers.
3. Few-shot sampling: order-stable, capped at dataset size, empty when
   ``n_shots <= 0``.
4. ``load_problems`` end-to-end: returns one ``BenchmarkProblem`` per row
   with the right field shape (``raw_messages``, ``ground_truth``,
   ``metadata.generation_size``, ``task``), and the ``tasks`` parameter
   is correctly ignored.
5. Pathological dataset rows: empty splits, very long problems, integer
   answers stringified, unicode in problem text.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aiperf.accuracy.benchmarks.aime import (
    DEFAULT_GENERATION_SIZE,
    INSTRUCTION_PREFIX,
    TASK_NAME,
    AIMEBenchmark,
)
from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import EndpointConfig, UserConfig
from aiperf.common.config.accuracy_config import AccuracyConfig
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType


def _make_user_config() -> UserConfig:
    return UserConfig(
        endpoint=EndpointConfig(
            model_names=["test-model"],
            type=EndpointType.COMPLETIONS,
            streaming=False,
        ),
        accuracy=AccuracyConfig(benchmark=AccuracyBenchmarkType.AIME),
    )


def _make_row(problem: str = "What is 1+1?", answer: int = 2) -> dict[str, Any]:
    return {"Problem": problem, "Answer": answer}


def _make_fake_dataset(rows: list[dict[str, Any]]) -> MagicMock:
    """Return a MagicMock that quacks like a HuggingFace ``Dataset``.

    Supports iteration (``for row in ds``), length (``len(ds)``), and
    integer indexing (``ds[i]``) — the three operations the AIME loader
    actually uses.
    """
    ds = MagicMock()
    ds.__iter__ = MagicMock(side_effect=lambda: iter(rows))
    ds.__len__ = MagicMock(return_value=len(rows))
    ds.__getitem__ = MagicMock(side_effect=lambda i: rows[i])
    return ds


@pytest.fixture
def bench() -> AIMEBenchmark:
    return AIMEBenchmark(user_config=_make_user_config())


class TestFormatPrompt:
    """Flat-completions prompt construction."""

    def test_instruction_prefix_is_present(self, bench: AIMEBenchmark) -> None:
        prompt = bench._format_prompt(
            _make_row("What is 5+5?", 10), few_shots=[], enable_cot=False
        )
        assert prompt.startswith(INSTRUCTION_PREFIX)

    def test_problem_text_appears(self, bench: AIMEBenchmark) -> None:
        prompt = bench._format_prompt(
            _make_row("Compute 2^10.", 1024),
            few_shots=[],
            enable_cot=False,
        )
        assert "Compute 2^10." in prompt

    def test_zero_shot_no_cot_ends_with_answer_marker(
        self, bench: AIMEBenchmark
    ) -> None:
        prompt = bench._format_prompt(
            _make_row("trivial", 1), few_shots=[], enable_cot=False
        )
        assert prompt.endswith("Answer:")
        assert "step by step" not in prompt

    def test_cot_inserts_step_by_step_before_answer(self, bench: AIMEBenchmark) -> None:
        prompt = bench._format_prompt(
            _make_row("trivial", 1), few_shots=[], enable_cot=True
        )
        assert "Let's think step by step" in prompt
        assert prompt.endswith("Answer:")

    def test_few_shot_examples_precede_test_problem(self, bench: AIMEBenchmark) -> None:
        shot = bench._format_example(_make_row("FIRST", 1))
        prompt = bench._format_prompt(
            _make_row("SECOND", 2), few_shots=[shot], enable_cot=False
        )
        assert prompt.index("FIRST") < prompt.index("SECOND")

    def test_few_shot_uses_boxed_answer(self, bench: AIMEBenchmark) -> None:
        """Few-shot examples must show \\boxed{} so the model is primed
        to emit the same format."""
        shot = bench._format_example(_make_row("ex", 7))
        prompt = bench._format_prompt(
            _make_row("test", 8), few_shots=[shot], enable_cot=False
        )
        assert "\\boxed{7}" in prompt

    def test_few_shot_does_not_use_boxed_for_test_query(
        self, bench: AIMEBenchmark
    ) -> None:
        """The trailing test query has no boxed answer (the model must
        produce one)."""
        shot = bench._format_example(_make_row("ex", 7))
        prompt = bench._format_prompt(
            _make_row("test", 8), few_shots=[shot], enable_cot=False
        )
        # The "8" answer is the gold for the test query and must NOT appear
        # in the prompt (would be cheating).
        assert "\\boxed{8}" not in prompt


class TestBuildChatMessages:
    """lighteval-style multi-turn chat construction."""

    def test_zero_shot_zero_cot_is_single_user_message(
        self, bench: AIMEBenchmark
    ) -> None:
        msgs = bench._build_chat_messages(
            _make_row("Q?", 1), few_shots=[], enable_cot=False
        )
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_zero_shot_user_message_includes_instruction(
        self, bench: AIMEBenchmark
    ) -> None:
        msgs = bench._build_chat_messages(
            _make_row("Q?", 1), few_shots=[], enable_cot=False
        )
        assert "non-negative integer" in msgs[0]["content"]
        assert "\\boxed" in msgs[0]["content"]

    def test_cot_appends_step_by_step(self, bench: AIMEBenchmark) -> None:
        msgs = bench._build_chat_messages(
            _make_row("Q?", 1), few_shots=[], enable_cot=True
        )
        assert "step by step" in msgs[-1]["content"]

    def test_few_shot_pairs_are_user_then_assistant(self, bench: AIMEBenchmark) -> None:
        shots = [
            bench._format_example(_make_row("A", 1)),
            bench._format_example(_make_row("B", 2)),
        ]
        msgs = bench._build_chat_messages(
            _make_row("C", 3), few_shots=shots, enable_cot=False
        )
        # Expected: user/assistant/user/assistant/user
        assert [m["role"] for m in msgs] == [
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
        ]

    def test_only_first_user_message_has_instruction(
        self, bench: AIMEBenchmark
    ) -> None:
        shots = [
            bench._format_example(_make_row("A", 1)),
            bench._format_example(_make_row("B", 2)),
        ]
        msgs = bench._build_chat_messages(
            _make_row("C", 3), few_shots=shots, enable_cot=False
        )
        user_messages = [m["content"] for m in msgs if m["role"] == "user"]
        assert "non-negative integer" in user_messages[0]
        for content in user_messages[1:]:
            assert "non-negative integer" not in content

    def test_assistant_messages_use_boxed_format(self, bench: AIMEBenchmark) -> None:
        shots = [bench._format_example(_make_row("A", 7))]
        msgs = bench._build_chat_messages(
            _make_row("B", 8), few_shots=shots, enable_cot=False
        )
        assistant = [m for m in msgs if m["role"] == "assistant"]
        assert assistant[0]["content"] == "\\boxed{7}"


class TestFormatExample:
    """Few-shot example formatting helper."""

    def test_answer_is_stringified(self, bench: AIMEBenchmark) -> None:
        ex = bench._format_example(_make_row("Q?", 42))
        assert ex["answer"] == "42"
        assert isinstance(ex["answer"], str)

    def test_formatted_uses_boxed_answer(self, bench: AIMEBenchmark) -> None:
        ex = bench._format_example(_make_row("Q?", 42))
        assert "\\boxed{42}" in ex["formatted"]

    def test_formatted_contains_problem_text(self, bench: AIMEBenchmark) -> None:
        ex = bench._format_example(_make_row("MyProblem", 1))
        assert "MyProblem" in ex["formatted"]


class TestBuildFewShots:
    """Few-shot sampling: order-stable, bounded, sequential."""

    def test_zero_shots_returns_empty(self, bench: AIMEBenchmark) -> None:
        ds = _make_fake_dataset([_make_row("a", 1), _make_row("b", 2)])
        assert bench._build_few_shots(ds, n_shots=0) == []

    def test_negative_shots_returns_empty(self, bench: AIMEBenchmark) -> None:
        ds = _make_fake_dataset([_make_row("a", 1)])
        assert bench._build_few_shots(ds, n_shots=-3) == []

    def test_n_shots_clamped_to_dataset_size(self, bench: AIMEBenchmark) -> None:
        ds = _make_fake_dataset([_make_row("only", 1)])
        shots = bench._build_few_shots(ds, n_shots=5)
        assert len(shots) == 1

    def test_shots_drawn_from_start_in_order(self, bench: AIMEBenchmark) -> None:
        rows = [_make_row(f"problem-{i}", i) for i in range(10)]
        ds = _make_fake_dataset(rows)
        shots = bench._build_few_shots(ds, n_shots=3)
        assert [s["problem"] for s in shots] == [
            "problem-0",
            "problem-1",
            "problem-2",
        ]


class TestLoadProblems:
    """End-to-end ``load_problems`` with a mocked HuggingFace dataset."""

    @pytest.mark.asyncio
    async def test_returns_one_problem_per_row(self) -> None:
        rows = [_make_row(f"q{i}", i) for i in range(5)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert len(problems) == 5
        assert all(isinstance(p, BenchmarkProblem) for p in problems)

    @pytest.mark.asyncio
    async def test_ground_truth_is_string_form_of_integer_answer(self) -> None:
        rows = [_make_row("q", 42)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].ground_truth == "42"
        assert isinstance(problems[0].ground_truth, str)

    @pytest.mark.asyncio
    async def test_task_name_is_aime(self) -> None:
        rows = [_make_row("q", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].task == TASK_NAME

    @pytest.mark.asyncio
    async def test_metadata_carries_default_generation_size(self) -> None:
        rows = [_make_row("q", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].metadata["generation_size"] == DEFAULT_GENERATION_SIZE

    @pytest.mark.asyncio
    async def test_raw_messages_populated_for_chat_endpoint(self) -> None:
        rows = [_make_row("q", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].raw_messages is not None
        assert len(problems[0].raw_messages) >= 1
        assert problems[0].raw_messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_few_shot_count_matches_n_shots(self) -> None:
        rows = [_make_row(f"q{i}", i) for i in range(10)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=3, enable_cot=False
            )
        # With 3 few-shots, every problem's chat messages should have:
        # 3 user + 3 assistant + 1 user = 7 messages.
        assert len(problems[0].raw_messages) == 7

    @pytest.mark.asyncio
    async def test_tasks_argument_is_ignored(self) -> None:
        """AIME has no subtasks; ``tasks=["foo"]`` must not filter or error."""
        rows = [_make_row("a", 1), _make_row("b", 2)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            none_problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
            named_problems = await bench.load_problems(
                tasks=["aime"], n_shots=0, enable_cot=False
            )
            unknown_problems = await bench.load_problems(
                tasks=["does-not-exist"], n_shots=0, enable_cot=False
            )
        assert len(none_problems) == len(named_problems) == len(unknown_problems) == 2

    @pytest.mark.asyncio
    async def test_cot_propagates_to_every_problem(self) -> None:
        rows = [_make_row(f"q{i}", i) for i in range(3)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(tasks=None, n_shots=0, enable_cot=True)
        assert all("step by step" in p.prompt for p in problems)


class TestPathologicalDatasetRows:
    """Adversarial / boundary inputs that must not crash the loader."""

    @pytest.mark.asyncio
    async def test_empty_dataset_returns_empty_list(self) -> None:
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset([]),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems == []

    @pytest.mark.asyncio
    async def test_unicode_problem_text_preserved(self) -> None:
        rows = [_make_row("Solve ∑₁ⁿ k² for n=10. ✓", 385)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert "∑₁ⁿ" in problems[0].prompt

    @pytest.mark.asyncio
    async def test_very_long_problem_text_does_not_crash(self) -> None:
        long_problem = "Q. " + ("blah " * 50_000) + "Find x."
        rows = [_make_row(long_problem, 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert len(problems) == 1
        assert long_problem in problems[0].prompt

    @pytest.mark.asyncio
    async def test_zero_padded_three_digit_answer_stringifies_cleanly(
        self,
    ) -> None:
        """AIME answers can be 0-999; some sources zero-pad. Our schema
        uses Python ints, so str(7) == '7' (no padding). Document via test."""
        rows = [_make_row("q", 7)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=0, enable_cot=False
            )
        assert problems[0].ground_truth == "7"

    @pytest.mark.asyncio
    async def test_n_shots_larger_than_dataset_clamps(self) -> None:
        rows = [_make_row("only-one", 1)]
        with patch(
            "aiperf.accuracy.benchmarks.aime.load_dataset",
            return_value=_make_fake_dataset(rows),
        ):
            bench = AIMEBenchmark(user_config=_make_user_config())
            problems = await bench.load_problems(
                tasks=None, n_shots=999, enable_cot=False
            )
        # 1 few-shot pair (user + assistant) + 1 main user = 3 messages.
        assert len(problems[0].raw_messages) == 3
