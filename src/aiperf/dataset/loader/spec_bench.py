# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import orjson

from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader
from aiperf.plugin.enums import DatasetSamplingStrategy

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


class SpecBenchLoader(BasePublicDatasetLoader):
    """SpecBench dataset loader for speculative decoding benchmarks.

    Downloads the SpecBench JSONL file from GitHub and converts each entry
    into a single-turn AIPerf Conversation using the first turn of each question.
    With ``multi_turn=True``, all turns in each entry are used.
    """

    tag = "SpecBench"
    url = "https://raw.githubusercontent.com/hemingkx/Spec-Bench/fd2c1cd7d2201ef71db4c5f4e455008f017967bf/data/spec_bench/question.jsonl"
    filename = "spec_bench.jsonl"

    def __init__(
        self,
        run: BenchmarkRun | None = None,
        *,
        multi_turn: bool = False,
        **kwargs,
    ) -> None:
        self.multi_turn = multi_turn
        super().__init__(run=run, **kwargs)

    async def load_dataset(self) -> dict[str, Any]:
        """Load the SpecBench JSONL file from cache or download it."""
        raw = await self._load_dataset(headers={})
        rows = [orjson.loads(line) for line in raw.splitlines() if line.strip()]
        return {"dataset": rows}

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each SpecBench entry into a Conversation (single- or multi-turn)."""
        dataset = data["dataset"]
        conversations = []
        skipped = 0

        for row in dataset:
            turns_raw = row.get("turns")
            if not isinstance(turns_raw, list) or not turns_raw:
                skipped += 1
                continue

            if self.multi_turn:
                conv_turns: list[Turn] = []
                for t in turns_raw:
                    text = str(t).strip() if t else ""
                    if text:
                        conv_turns.append(Turn(texts=[Text(contents=[text])]))
                if not conv_turns:
                    skipped += 1
                    continue
                conversations.append(
                    Conversation(
                        session_id=self.session_id_generator.next(),
                        turns=conv_turns,
                    )
                )
            else:
                prompt = str(turns_raw[0]).strip() if turns_raw[0] else ""
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
            lambda: f"Converted {len(conversations)} rows (skipped {skipped} empty)"
        )
        return conversations

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred sampling strategy for this dataset."""
        return DatasetSamplingStrategy.SEQUENTIAL
