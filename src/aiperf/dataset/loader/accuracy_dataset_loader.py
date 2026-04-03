# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy benchmark dataset loader.

Converts BenchmarkProblem objects from accuracy benchmarks (e.g., MMLU)
into Conversation/Turn objects for aiperf's DatasetManager pipeline.
Each BenchmarkProblem becomes a single-turn Conversation with pre-formatted
OpenAI-compatible messages in Turn.raw_messages.

The problem ordering is deterministic: Conversation i corresponds to
BenchmarkProblem i, ensuring AccuracyRecordProcessor (which also loads
problems in the same order) can index by session_num.
"""

from __future__ import annotations

from aiperf.accuracy.models import BenchmarkProblem
from aiperf.common.config import UserConfig
from aiperf.common.models.dataset_models import Conversation, Text, Turn
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType


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
        acc_cfg = self.user_config.accuracy
        benchmark_cls = plugins.get_class(
            PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark
        )
        benchmark = benchmark_cls(user_config=self.user_config)

        n_shots = acc_cfg.n_shots
        if n_shots == 0:
            meta = plugins.get_metadata(
                PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark
            )
            default_n = meta.get("default_n_shots")
            if default_n is not None:
                n_shots = default_n

        return await benchmark.load_problems(
            tasks=acc_cfg.tasks,
            n_shots=n_shots,
            enable_cot=acc_cfg.enable_cot,
        )

    def _convert_to_conversations(
        self, problems: list[BenchmarkProblem]
    ) -> list[Conversation]:
        session_gen = SessionIDGenerator(seed=self.user_config.input.random_seed)
        system_prompt = self.user_config.accuracy.system_prompt
        conversations: list[Conversation] = []

        for problem in problems:
            session_id = session_gen.next()

            if problem.chat_messages is not None:
                messages: list[dict[str, str]] = list(problem.chat_messages)
            else:
                messages = [{"role": "user", "content": problem.prompt}]

            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})

            gen_size = (
                problem.metadata.get("generation_size", 100)
                if problem.metadata
                else 100
            )

            turn = Turn(
                role="user",
                raw_messages=messages,
                max_tokens=gen_size,
                texts=[Text(contents=[problem.prompt])],
            )

            conversations.append(Conversation(session_id=session_id, turns=[turn]))

        return conversations
