# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader


class SpeedBenchLoader(BaseHFDatasetLoader):
    """HuggingFace dataset loader for nvidia/SPEED-Bench.

    SPEED-Bench (SPEculative Evaluation Dataset) provides prompts for
    benchmarking speculative decoding across diverse semantic domains and
    input sequence lengths. Each row contains a ``turns`` column with a
    list of plain strings and a ``category`` column identifying the
    semantic domain or entropy tier. Only the first turn is used as the
    benchmark prompt; subsequent turns are discarded.

    When ``category`` is set in plugin metadata, only rows matching that
    category are loaded. This enables per-category acceptance rate
    measurement by running one category at a time against a
    speculative-decoding-enabled server.

    **Qualitative subset categories** (80 samples each):
    coding, humanities, math, multilingual, qa, rag, reasoning, roleplay,
    stem, summarization, writing

    **Throughput subset categories** (512 samples each per ISL bucket):
    low_entropy, mixed, high_entropy

    Example plugins.yaml entries::

        speed_bench_qualitative:
          class: aiperf.dataset.loader.speed_bench:SpeedBenchLoader
          metadata:
            hf_dataset_name: nvidia/SPEED-Bench
            hf_split: test
            hf_subset: qualitative

        speed_bench_coding:
          class: aiperf.dataset.loader.speed_bench:SpeedBenchLoader
          metadata:
            hf_dataset_name: nvidia/SPEED-Bench
            hf_split: test
            hf_subset: qualitative
            category: coding
    """

    def __init__(
        self,
        user_config: UserConfig,
        category: str | None = None,
        **kwargs,
    ) -> None:
        self.category = category
        super().__init__(user_config=user_config, **kwargs)

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each dataset row into a single-turn Conversation."""
        dataset = data["dataset"]
        conversations: list[Conversation] = []
        skipped = 0
        max_conversations = self._max_conversations()

        for row in dataset:
            if (
                max_conversations is not None
                and len(conversations) >= max_conversations
            ):
                break

            if self.category and row.get("category") != self.category:
                continue

            turns = row.get("turns")
            if not turns or not isinstance(turns, list) or not turns[0]:
                skipped += 1
                continue

            prompt = str(turns[0]).strip()
            if not prompt:
                skipped += 1
                continue

            conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[
                        Turn(texts=[Text(contents=[prompt])]),
                    ],
                )
            )

        self.debug(
            lambda: (
                f"Converted {len(conversations)} rows"
                f" (skipped {skipped} empty"
                f"{f', filtered to category={self.category!r}' if self.category else ''})"
            )
        )
        return conversations
