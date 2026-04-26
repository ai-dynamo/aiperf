# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any

from aiperf.common.config.user_config import UserConfig
from aiperf.common.utils import load_json_str
from aiperf.dataset.loader.models import MultiTurn, SingleTurn
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader


class SpeedBenchLoader(MultiTurnDatasetLoader):
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
        filename: str,
        user_config: UserConfig,
        category: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.category = category
        super().__init__(filename=filename, user_config=user_config, **kwargs)

    def load_dataset(self) -> dict[str, list[MultiTurn]]:
        """Load multi-turn data from a JSONL file.

        Each line represents a complete multi-turn conversation with its own
        session_id and multiple turns.

        Returns:
            A dictionary mapping session_id to list of MultiTurn objects.
        """
        data: dict[str, list[MultiTurn]] = defaultdict(list)

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                loaded_line = load_json_str(line)

                if self.category and loaded_line.get("category") != self.category:
                    continue

                multi_turn_data = MultiTurn(
                    session_id=loaded_line["question_id"],
                    turns=[
                        SingleTurn(text=message["content"], role=message["role"])
                        for message in loaded_line["messages"]
                    ],
                )

                data[multi_turn_data.session_id].append(multi_turn_data)

        return data
