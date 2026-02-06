# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.models import AARWLTEntry
from aiperf.plugin.enums import DatasetSamplingStrategy


class AARWLTDatasetLoader(BaseFileLoader):
    """AA-RWLT agentic benchmark dataset loader.

    Loads pre-recorded multi-turn trajectories with cumulative message history.
    Each trajectory is processed sequentially with zero inter-turn delay,
    and LLM responses are discarded since the dataset contains pre-recorded
    responses for subsequent turns.
    """

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format."""
        if data is None:
            return False

        try:
            AARWLTEntry.model_validate(data)
            return True
        except ValidationError:
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Trajectories are consumed in order."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[AARWLTEntry]]:
        """Load AA-RWLT entries from a JSONL file, grouped by conversation_id.

        Returns:
            A dictionary mapping conversation_id to sorted list of AARWLTEntry objects.

        Raises:
            ValueError: If entries have non-sequential indexing (gaps or duplicates).
        """
        data: dict[str, list[AARWLTEntry]] = defaultdict(list)

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue
                entry = AARWLTEntry.model_validate_json(line)
                data[entry.conversation_id].append(entry)

        # Sort each group by conversation_idx and validate sequential indexing
        for conv_id, entries in data.items():
            entries.sort(key=lambda e: e.conversation_idx)
            for i, entry in enumerate(entries):
                if entry.conversation_idx != i:
                    raise ValueError(
                        f"Trajectory {conv_id!r} has non-sequential indexing: "
                        f"expected idx {i}, got {entry.conversation_idx}."
                    )

        return data

    def convert_to_conversations(
        self, data: dict[str, list[AARWLTEntry]]
    ) -> list[Conversation]:
        """Convert AA-RWLT entries to Conversation objects.

        Each trajectory becomes a Conversation with:
        - session_id from conversation_id
        - discard_assistant_response=True (pre-recorded cumulative history)
        - tools from the first entry (consistent across trajectory)
        - One Turn per entry with raw_messages and zero delay
        """
        conversations = []
        for conv_id, entries in data.items():
            tools = entries[0].tools if entries else None
            turns = [Turn(raw_messages=entry.messages, delay=0) for entry in entries]
            conversations.append(
                Conversation(
                    session_id=conv_id,
                    turns=turns,
                    tools=tools,
                    discard_assistant_response=True,
                )
            )
        return conversations
