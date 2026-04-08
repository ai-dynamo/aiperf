# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import orjson

from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy


class SpecBenchLoader(BasePublicDatasetLoader):
    """SpecBench dataset loader for speculative decoding benchmarks.

    Downloads the SpecBench JSONL file from GitHub and converts each entry
    into a single-turn AIPerf Conversation using the first turn of each question.
    """

    tag = "SpecBench"
    url = "https://raw.githubusercontent.com/hemingkx/Spec-Bench/fd2c1cd7d2201ef71db4c5f4e455008f017967bf/data/spec_bench/question.jsonl"
    filename = "spec_bench.jsonl"

    async def load_dataset(self) -> dict[str, Any]:
        """Load the SpecBench JSONL file from cache or download it."""
        raw = await self._load_dataset(headers={})
        rows = [orjson.loads(line) for line in raw.splitlines() if line.strip()]
        return {"dataset": rows}

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each SpecBench entry into a single-turn Conversation."""
        dataset = data["dataset"]
        conversations = []
        skipped = 0

        for row in dataset:
            turns = row.get("turns", [])
            if not turns or not str(turns[0]).strip():
                skipped += 1
                continue

            conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[
                        Turn(
                            texts=[Text(contents=[str(turns[0])])],
                        )
                    ],
                )
            )

        self.debug(
            lambda: f"Converted {len(conversations)} rows (skipped {skipped} empty)"
        )
        return conversations

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred sampling strategy for this dataset."""
        return DatasetSamplingStrategy.SEQUENTIAL
