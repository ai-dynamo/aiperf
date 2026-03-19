# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader


class HFInstructionResponseDatasetLoader(BaseHFDatasetLoader):
    """HuggingFace dataset loader for flat instruction/response datasets.

    Converts datasets with a prompt column into single-turn AIPerf Conversations.

    Example plugins.yaml entry:

        aimo:
          class: aiperf.dataset.loader.hf_instruction_response:HFInstructionResponseDatasetLoader
          metadata:
            hf_dataset_name: AI-MO/NuminaMath-TIR
            prompt_column: problem
    """

    def __init__(
        self,
        user_config: UserConfig,
        prompt_column: str,
        **kwargs,
    ) -> None:
        self.prompt_column = prompt_column
        super().__init__(user_config=user_config, **kwargs)

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each dataset row into a single-turn Conversation."""
        dataset = data["dataset"]
        conversations = []
        skipped = 0

        for row in dataset:
            prompt = row.get(self.prompt_column)
            if not prompt or not str(prompt).strip():
                skipped += 1
                continue

            conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[
                        Turn(
                            texts=[Text(contents=[str(prompt)])],
                        )
                    ],
                )
            )

        self.debug(
            lambda: f"Converted {len(conversations)} rows (skipped {skipped} empty)"
        )
        return conversations
