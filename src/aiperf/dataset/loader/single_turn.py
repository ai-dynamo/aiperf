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
from aiperf.dataset.loader.models import SingleTurn


@CustomDatasetFactory.register(CustomDatasetType.SINGLE_TURN)
class SingleTurnDatasetLoader(BaseFileLoader, MediaConversionMixin):
    """A dataset loader that loads single turn data from a file.

    The single turn type
      - supports multi-modal data (e.g. text, image, audio)
      - supports client-side batching for each data (e.g. batch_size > 1)
      - DOES NOT support multi-turn features (e.g. delay, sessions, etc.)

    Examples:
    1. Single-batch, text only
    ```json
    {"text": "What is deep learning?"}
    ```

    2. Single-batch, multi-modal
    ```json
    {"text": "What is in the image?", "image": "/path/to/image.png"}
    ```

    3. Multi-batch, multi-modal
    ```json
    {"texts": ["Who are you?", "Hello world"], "images": ["/path/to/image.png", "/path/to/image2.png"]}
    ```

    4. Fixed schedule version
    ```json
    {"timestamp": 0, "text": "What is deep learning?"},
    {"timestamp": 1000, "text": "Who are you?"},
    {"timestamp": 2000, "text": "What is AI?"}
    ```

    5. Time delayed version
    ```json
    {"delay": 0, "text": "What is deep learning?"},
    {"delay": 1234, "text": "Who are you?"}
    ```

    6. Full-featured version (Multi-batch, multi-modal, multi-fielded)
    ```json
    {
        "texts": [
            {"name": "text_field_A", "contents": ["Hello", "World"]},
            {"name": "text_field_B", "contents": ["Hi there"]}
        ],
        "images": [
            {"name": "image_field_A", "contents": ["/path/1.png", "/path/2.png"]},
            {"name": "image_field_B", "contents": ["/path/3.png"]}
        ]
    }
    ```
    """

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: str | Path | None = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        SingleTurn format has modality fields (text/texts, image/images, etc.)
        but does NOT have a "turns" field. Use the SingleTurn model to validate the data.
        """
        if data is None:
            return False

        try:
            SingleTurn.model_validate(data)
            return True
        except ValidationError:
            return False

    @classmethod
    def get_preferred_sampling_strategy(cls) -> DatasetSamplingStrategy:
        """Get the preferred dataset sampling strategy for SingleTurn."""
        return DatasetSamplingStrategy.SEQUENTIAL

    def load_dataset(self) -> dict[str, list[SingleTurn]]:
        """Load single-turn data from a JSONL file.

        Each line represents a single turn conversation. Multiple turns with
        the same session_id (or generated UUID) are grouped together.

        Returns:
            A dictionary mapping session_id to list of CustomData.
        """
        data: dict[str, list[SingleTurn]] = defaultdict(list)

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                single_turn_data = SingleTurn.model_validate_json(line)
                session_id = self.session_id_generator.next()
                data[session_id].append(single_turn_data)

        return data

    def convert_to_conversations(
        self, data: dict[str, list[SingleTurn]]
    ) -> list[Conversation]:
        """Convert single turn data to conversation objects.

        Args:
            data: A dictionary mapping session_id to list of SingleTurn objects.

        Returns:
            A list of conversations.
        """
        conversations = []
        for session_id, single_turns in data.items():
            conversation = Conversation(session_id=session_id)
            for single_turn in single_turns:
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
