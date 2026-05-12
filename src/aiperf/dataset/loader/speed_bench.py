# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any

from aiperf.common.config.user_config import UserConfig
from aiperf.common.utils import load_json_str
from aiperf.dataset.loader.models import MultiTurn, SingleTurn
from aiperf.dataset.loader.multi_turn import MultiTurnDatasetLoader


class SpeedBenchLoader(MultiTurnDatasetLoader):
    """HuggingFace dataset loader for nvidia/SPEED-Bench.

    SPEED-Bench (SPEculative Evaluation Dataset) provides prompts for
    benchmarking speculative decoding across diverse semantic domains and
    input sequence lengths. Each JSONL row contains a ``question_id``, a
    ``category`` identifying the semantic domain or entropy tier, and a
    ``messages`` array of OpenAI-style ``role``/``content`` dictionaries.

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

    TURNS_PLACEHOLDER = (
        "FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH"
    )

    def __init__(
        self,
        filename: str,
        user_config: UserConfig,
        category: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.category = category
        super().__init__(filename=filename, user_config=user_config, **kwargs)

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Return whether a JSON object matches the SPEED-Bench JSONL shape."""
        if data is None or not isinstance(data, dict):
            return False

        question_id = data.get("question_id")
        category = data.get("category")
        messages = data.get("messages")

        if not isinstance(question_id, str) or not question_id.strip():
            return False
        if not isinstance(category, str) or not category.strip():
            return False
        if not isinstance(messages, list) or not messages:
            return False

        return all(
            isinstance(message, dict)
            and isinstance(message.get("role"), str)
            and bool(message["role"].strip())
            and isinstance(message.get("content"), str)
            and bool(message["content"].strip())
            and message["content"] != cls.TURNS_PLACEHOLDER
            for message in messages
        )

    def load_dataset(self) -> dict[str, list[MultiTurn]]:
        """Load SPEED-Bench multi-turn data from a JSONL file.

        Each line is mapped to a ``MultiTurn`` where ``session_id`` is taken
        from the line's ``question_id``, and ``turns`` is built from the
        ``messages`` array by converting each ``{"role", "content"}`` entry
        into a ``SingleTurn(role=..., text=...)``.

        When ``self.category`` is set, lines whose ``category`` field does not
        match are skipped. If the filter eliminates every row, a warning is
        emitted to surface a likely category/file mismatch rather than
        silently returning an empty dataset.

        Returns:
            A dictionary mapping session_id (the SPEED-Bench question_id) to
            a list of MultiTurn objects.
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

        if self.category and not data:
            self.warning(
                lambda: (
                    f"SPEED-Bench category filter {self.category!r} matched no rows "
                    f"in {self.filename}. Verify the configured category exists in "
                    f"this dataset."
                )
            )

        return data
