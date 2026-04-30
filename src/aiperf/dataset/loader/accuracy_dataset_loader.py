# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy benchmark dataset loader.

Converts BenchmarkProblem objects from accuracy benchmarks (e.g., MMLU)
into Conversation/Turn objects for aiperf's DatasetManager pipeline.
Each BenchmarkProblem becomes a single-turn Conversation with pre-formatted
OpenAI-compatible messages in Turn.raw_messages.

The problem ordering is deterministic: Conversation i corresponds to
BenchmarkProblem i. AccuracyRecordProcessor maps each response back to its
problem via session_num % len(problems), which handles both single-pass and
multi-pass (num_requests > dataset size) runs correctly. This mapping is only
valid when the dataset is sampled sequentially; DatasetManager enforces that
invariant and rejects non-sequential strategies in accuracy mode.
"""

from __future__ import annotations

from aiperf.accuracy.benchmark_loader import load_benchmark_problems
from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.config import UserConfig
from aiperf.common.models.dataset_models import Conversation, Text, Turn
from aiperf.common.session_id_generator import SessionIDGenerator

# Default max_tokens when a benchmark omits generation_size from metadata.
# MMLU sets 5 (single-letter answer); long-form benchmarks should set
# their own value in BenchmarkProblem.metadata["generation_size"].
DEFAULT_GENERATION_SIZE = 100


class AccuracyDatasetLoader:
    """Loads accuracy benchmark problems and converts them to Conversations.

    Invoked by DatasetManager when accuracy mode is enabled, bypassing the
    normal file-based or synthetic dataset pipelines.
    """

    def __init__(self, *, user_config: UserConfig) -> None:
        self.user_config = user_config

    async def load(self) -> list[Conversation]:
        """Load benchmark problems and convert to Conversations."""
        problems = await self._load_problems()
        return self._convert_to_conversations(problems)

    async def _load_problems(self) -> list[BenchmarkProblem]:
        return await load_benchmark_problems(self.user_config)

    def _convert_to_conversations(
        self, problems: list[BenchmarkProblem]
    ) -> list[Conversation]:
        session_gen = SessionIDGenerator(seed=self.user_config.input.random_seed)
        system_prompt = self.user_config.accuracy.system_prompt
        conversations: list[Conversation] = []

        for problem in problems:
            session_id = session_gen.next()

            if problem.raw_messages is not None:
                messages: list[AccuracyChatMessage] = list(problem.raw_messages)
            else:
                messages = [{"role": "user", "content": problem.prompt}]

            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            gen_size = (
                problem.metadata.get("generation_size", DEFAULT_GENERATION_SIZE)
                if problem.metadata
                else DEFAULT_GENERATION_SIZE
            )

            prompt_text = (
                f"{system_prompt}\n\n{problem.prompt}"
                if system_prompt
                else problem.prompt
            )

            turn = Turn(
                role="user",
                raw_messages=messages,
                max_tokens=gen_size,
                texts=[Text(contents=[prompt_text])],
            )

            conversations.append(Conversation(session_id=session_id, turns=[turn]))

        return conversations
