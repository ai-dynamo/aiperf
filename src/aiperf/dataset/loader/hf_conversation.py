# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from aiperf.common.config.user_config import UserConfig
from aiperf.common.models import Conversation, Text, Turn
from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader


class HFConversationDatasetLoader(BaseHFDatasetLoader):
    """HuggingFace dataset loader for conversation-array datasets.

    Handles datasets where each row stores messages as a list of dicts.
    Extracts the first message as the prompt, producing single-turn Conversations.
    Optionally attaches an image per row when image_column is configured.

    Normalises two common dataset quirks automatically:
    - List-of-lists wrapping (VisionArena): each turn is wrapped in its own list
    - Image placeholder tokens (LLaVA): ``<image>`` tokens are stripped from text

    Example plugins.yaml entry::

        vision_arena:
          class: aiperf.dataset.loader.hf_conversation:HFConversationDatasetLoader
          metadata:
            hf_dataset_name: lmarena-ai/VisionArena-Chat
            hf_split: train
            conversation_column: conversation
            message_content_key: content
            image_column: images

        llava_onevision:
          class: aiperf.dataset.loader.hf_conversation:HFConversationDatasetLoader
          metadata:
            hf_dataset_name: lmms-lab/LLaVA-OneVision-Data
            hf_split: train
            hf_subset: sharegpt4o
            conversation_column: conversations
            message_content_key: value
            image_column: image
    """

    def __init__(
        self,
        user_config: UserConfig,
        conversation_column: str,
        message_content_key: str = "content",
        image_column: str | None = None,
        **kwargs,
    ) -> None:
        self.conversation_column = conversation_column
        self.message_content_key = message_content_key
        self.image_column = image_column
        super().__init__(user_config=user_config, **kwargs)

    def _extract_first_message(self, messages: list[Any]) -> str | None:
        """Extract the text of the first message, handling dataset-specific quirks.

        Unwraps list-of-lists turns (VisionArena) and strips ``<image>``
        placeholder tokens (LLaVA) that are meaningless for benchmarking.
        """
        if not messages:
            return None
        message = messages[0]
        if isinstance(message, list):
            message = message[0] if message else None
        if not message or not isinstance(message, dict):
            return None
        value = message.get(self.message_content_key)
        if not isinstance(value, str):
            return None
        return value.replace("<image>", "").strip() or None

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

            messages = row.get(self.conversation_column) or []
            prompt = self._extract_first_message(messages)
            if not prompt:
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
                            texts=[Text(contents=[prompt])],
                            images=images,
                        )
                    ],
                )
            )

        self.debug(
            lambda: f"Converted {len(conversations)} rows (skipped {skipped} empty)"
        )
        return conversations
