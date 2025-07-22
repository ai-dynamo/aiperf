# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections import defaultdict

from aiperf.common.enums import CustomDatasetType
from aiperf.common.factories import CustomDatasetFactory
from aiperf.common.models import Audio, Conversation, Image, Text, Turn
from aiperf.services.dataset.loader.models import SingleTurn


@CustomDatasetFactory.register(CustomDatasetType.SINGLE_TURN)
class SingleTurnDatasetLoader:
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

    def __init__(self, filename: str):
        self.filename = filename

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
                session_id = str(uuid.uuid4())
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
                turn = self._convert_single_turn_to_turn(single_turn)
                conversation.turns.append(turn)
            conversations.append(conversation)
        return conversations

    def _convert_single_turn_to_turn(self, single_turn: SingleTurn) -> Turn:
        """Convert a SingleTurn object to a Turn object.

        Args:
            single_turn: The SingleTurn object to convert.

        Returns:
            A Turn object.
        """
        turn = Turn(
            timestamp=single_turn.timestamp,
            delay=single_turn.delay,
            role=single_turn.role,
        )

        # Convert text fields
        if single_turn.text:
            turn.texts.append(Text(name="text", contents=[single_turn.text]))
        elif single_turn.texts:
            turn.texts.extend(self._convert_to_text_objects(single_turn.texts))

        # Convert image fields
        if single_turn.image:
            turn.images.append(Image(name="image_url", contents=[single_turn.image]))
        elif single_turn.images:
            turn.images.extend(self._convert_to_image_objects(single_turn.images))

        # Convert audio fields
        if single_turn.audio:
            turn.audios.append(Audio(name="input_audio", contents=[single_turn.audio]))
        elif single_turn.audios:
            turn.audios.extend(self._convert_to_audio_objects(single_turn.audios))

        return turn

    def _convert_to_text_objects(self, texts: list[str] | list[Text]) -> list[Text]:
        """Convert text data to Text objects."""
        if not texts:
            return []

        # If already Text objects, return as is
        if isinstance(texts[0], Text):
            return texts  # type: ignore

        # Convert list of strings to single Text object
        return [Text(name="text", contents=texts)]  # type: ignore

    def _convert_to_image_objects(self, images: list[str] | list[Image]) -> list[Image]:
        """Convert image data to Image objects."""
        if not images:
            return []

        # If already Image objects, return as is
        if isinstance(images[0], Image):
            return images  # type: ignore

        # Convert list of strings to single Image object
        return [Image(name="image_url", contents=images)]  # type: ignore

    def _convert_to_audio_objects(self, audios: list[str] | list[Audio]) -> list[Audio]:
        """Convert audio data to Audio objects."""
        if not audios:
            return []

        # If already Audio objects, return as is
        if isinstance(audios[0], Audio):
            return audios  # type: ignore

        # Convert list of strings to single Audio object
        return [Audio(name="input_audio", contents=audios)]  # type: ignore
