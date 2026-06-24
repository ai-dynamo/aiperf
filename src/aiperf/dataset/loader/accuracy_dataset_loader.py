# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Accuracy benchmark dataset loader.

Converts BenchmarkProblem objects from accuracy benchmarks (e.g., MMLU)
into Conversation/Turn objects for aiperf's DatasetManager pipeline.
Each BenchmarkProblem becomes a single-turn Conversation with pre-formatted
OpenAI-compatible messages in Turn.raw_messages.

The problem ordering is deterministic: Conversation i corresponds to
BenchmarkProblem i. Each Conversation carries accuracy_ground_truth and
accuracy_task so that DatasetManager can propagate them through
ConversationMetadata inside DatasetConfiguredNotification. Processors
(AccuracyRecordProcessor, AccuracyResultsProcessor) receive these values
from the notification instead of independently re-loading the benchmark.
The session_num % len(conversations) mapping handles both single-pass and
multi-pass (num_requests > dataset size) runs and is only valid when the
dataset is sampled sequentially; DatasetManager enforces that invariant and
rejects non-sequential strategies in accuracy mode.
"""

from __future__ import annotations

from aiperf.accuracy.benchmark_loader import load_benchmark_problems
from aiperf.accuracy.models import AccuracyChatMessage, BenchmarkProblem
from aiperf.common.models.dataset_models import Conversation, Text, Turn
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.config import BenchmarkRun
from aiperf.plugin import plugins

# Default max_tokens when a benchmark omits generation_size from metadata.
# MMLU sets 5 (single-letter answer); long-form benchmarks should set
# their own value in BenchmarkProblem.metadata["generation_size"].
DEFAULT_GENERATION_SIZE = 100


class AccuracyDatasetLoader:
    """Loads accuracy benchmark problems and converts them to Conversations.

    Invoked by DatasetManager when accuracy mode is enabled, bypassing the
    normal file-based or synthetic dataset pipelines.
    """

    def __init__(self, *, run: BenchmarkRun) -> None:
        self.run = run

    async def load(self) -> list[Conversation]:
        """Load benchmark problems and convert to Conversations.

        Raises:
            ValueError: if the benchmark returns 0 problems (e.g. bad --accuracy-tasks).
        """
        problems = await load_benchmark_problems(self.run)
        if not problems:
            acc_cfg = self.run.cfg.accuracy
            raise ValueError(
                f"Benchmark '{acc_cfg.benchmark}' returned 0 problems "
                f"(tasks={acc_cfg.tasks}, n_shots={acc_cfg.n_shots}). "
                f"Check that --accuracy-tasks names a valid subtask "
                f"(see docs/accuracy/accuracy-benchmarking.md) or omit "
                f"the flag to evaluate all tasks."
            )
        return self._convert_to_conversations(problems)

    def _convert_to_conversations(
        self, problems: list[BenchmarkProblem]
    ) -> list[Conversation]:
        dataset_config = self.run.cfg.get_default_dataset()
        seed = dataset_config.random_seed or self.run.random_seed
        session_gen = SessionIDGenerator(seed=seed)
        acc_cfg = self.run.cfg.accuracy
        system_prompt = acc_cfg.system_prompt if acc_cfg is not None else None
        # When the user hasn't supplied --accuracy-system-prompt, fall back to
        # the benchmark plugin's ``default_system_prompt`` metadata (so MMLU
        # /humaneval/etc. get their idiomatic instruction injected). Empty
        # strings count as "no default" so a plugin can opt out explicitly.
        if not system_prompt and acc_cfg is not None:
            from aiperf.plugin.enums import PluginType

            try:
                meta = plugins.get_metadata(
                    PluginType.ACCURACY_BENCHMARK, acc_cfg.benchmark
                )
            except Exception:
                meta = None
            if meta:
                default = (
                    meta.get("default_system_prompt")
                    if isinstance(meta, dict)
                    else None
                )
                if isinstance(default, str) and default:
                    system_prompt = default
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

            conversations.append(
                Conversation(
                    session_id=session_id,
                    turns=[turn],
                    accuracy_ground_truth=problem.ground_truth,
                    accuracy_task=problem.task,
                )
            )

        return conversations
