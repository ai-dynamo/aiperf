# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader


class HFInstructionResponseDatasetLoader(BaseHFDatasetLoader):
    """HuggingFace dataset loader for flat instruction/response datasets.

    Converts datasets with a flat prompt column into single-turn AIPerf Conversations.
    Optionally attaches an image per row when image_column is configured.

    Example plugins.yaml entry::

        aimo:
          class: aiperf.dataset.loader.hf_instruction_response:HFInstructionResponseDatasetLoader
          metadata:
            hf_dataset_name: AI-MO/NuminaMath-TIR
            prompt_column: problem

        mmstar:
          class: aiperf.dataset.loader.hf_instruction_response:HFInstructionResponseDatasetLoader
          metadata:
            hf_dataset_name: Lin-Chen/MMStar
            hf_split: val
            prompt_column: question
            image_column: image
    """

    def __init__(
        self,
        user_config: UserConfig,
        prompt_column: str,
        image_column: str | None = None,
        **kwargs,
    ) -> None:
        self.prompt_column = prompt_column
        self.image_column = image_column
        super().__init__(user_config=user_config, **kwargs)

    async def convert_to_conversations(
        self, data: dict[str, Any]
    ) -> list[Conversation]:
        """Convert each dataset row into a single-turn Conversation."""
        dataset = data["dataset"]
        conversations = []
        skipped = 0
        max_conversations = self._max_conversations()

        for row in dataset:
            if (
                max_conversations is not None
                and len(conversations) >= max_conversations
            ):
                break

            prompt = row.get(self.prompt_column)
            if not prompt or not str(prompt).strip():
                skipped += 1
                continue

            images = (
                self._extract_images(row, self.image_column)
                if self.image_column
                else []
            )

            conversations.append(
                Conversation(
                    session_id=self.session_id_generator.next(),
                    turns=[
                        Turn(
                            texts=[Text(contents=[str(prompt)])],
                            images=images,
                        )
                    ],
                )
            )

        self.debug(
            lambda: f"Converted {len(conversations)} rows (skipped {skipped} empty)"
        )
        return conversations
