# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aiperf.common.enums import CustomDatasetType, DatasetSamplingStrategy, MediaType
from aiperf.common.factories import CustomDatasetFactory
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.mixins import MediaConversionMixin
from aiperf.dataset.loader.models import MultiTurn


@CustomDatasetFactory.register(CustomDatasetType.MULTI_TURN)
class MultiTurnDatasetLoader(BaseFileLoader, MediaConversionMixin):
    """A dataset loader that loads multi-turn data from a file.

    The multi-turn type
      - supports multi-modal data (e.g. text, image, audio)
      - supports multi-turn features (e.g. delay, sessions, etc.)
      - supports client-side batching for each data (e.g. batch_size > 1)

    NOTE: If the user specifies multiple multi-turn entries with same session ID,
    the loader will group them together. If the timestamps are specified, they will
    be sorted in ascending order later in the timing manager.

    Examples:
    1. Simple version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"text": "Hello", "image": "url", "delay": 0},
            {"text": "Hi there", "delay": 1000}
        ]
    }
    ```

    2. Batched version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"texts": ["Who are you?", "Hello world"], "images": ["/path/1.png", "/path/2.png"]},
            {"texts": ["What is in the image?", "What is AI?"], "images": ["/path/3.png", "/path/4.png"]}
        ]
    }
    ```

    3. Fixed schedule version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"timestamp": 0, "text": "What is deep learning?"},
            {"timestamp": 1000, "text": "Who are you?"}
        ]
    }
    ```

    4. Time delayed version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"delay": 0, "text": "What is deep learning?"},
            {"delay": 1000, "text": "Who are you?"}
        ]
    }
    ```

    5. full-featured version (multi-batch, multi-modal, multi-fielded, session-based, etc.)
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {
                "timestamp": 1234,
                "texts": [
                    {"name": "text_field_a", "contents": ["hello", "world"]},
                    {"name": "text_field_b", "contents": ["hi there"]}
                ],
                "images": [
                    {"name": "image_field_a", "contents": ["/path/1.png", "/path/2.png"]},
                    {"name": "image_field_b", "contents": ["/path/3.png"]}
                ]
            }
        ]
    }
    ```
    """

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        For multi-turn data, simply validate the data against the MultiTurn model.
        This will handle all of the validation logic for the different input combinations.
        """
        if data is None:
            return False

        try:
            MultiTurn.model_validate(data)
            return True
        except ValidationError:
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for MultiTurn."""
        return DatasetSamplingStrategy.SEQUENTIAL

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

                multi_turn_data = MultiTurn.model_validate_json(line)
                session_id = (
                    multi_turn_data.session_id or self.session_id_generator.next()
                )
                data[session_id].append(multi_turn_data)

        return data

    def convert_to_conversations(
        self, data: dict[str, list[MultiTurn]]
    ) -> list[Conversation]:
        """Convert multi-turn data to conversation objects.

        Args:
            data: A dictionary mapping session_id to list of MultiTurn objects.

        Returns:
            A list of conversations.
        """
        conversations = []
        for session_id, multi_turns in data.items():
            conversation = Conversation(session_id=session_id)

            # Process all MultiTurn objects for this session
            for multi_turn in multi_turns:
                for single_turn in multi_turn.turns:
                    media = self.convert_to_media_objects(single_turn)
                    conversation.turns.append(
                        Turn(
                            texts=media[MediaType.TEXT],
                            images=media[MediaType.IMAGE],
                            audios=media[MediaType.AUDIO],
                            videos=media[MediaType.VIDEO],
                            timestamp=single_turn.timestamp,
                            delay=single_turn.delay,
                            role=single_turn.role,
                        )
                    )
            conversations.append(conversation)
        return conversations
