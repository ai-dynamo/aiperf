# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections import defaultdict

from aiperf.common.enums import CustomDatasetType
from aiperf.common.factories import CustomDatasetFactory
from aiperf.common.models import Audio, Conversation, Image, Text, Turn
from aiperf.services.dataset.loader.models import MultiTurn, SingleTurn


@CustomDatasetFactory.register(CustomDatasetType.MULTI_TURN)
class MultiTurnDatasetLoader:
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

    def __init__(self, filename: str):
        self.filename = filename

    def load_dataset(self) -> dict[str, list[MultiTurn]]:
        """Load multi-turn data from a JSONL file.

        Each line represents a complete multi-turn conversation with its own
        session_id and multiple turns.

        Returns:
            A dictionary mapping session_id to list of CustomData (containing the MultiTurn).
        """
        data: dict[str, list[MultiTurn]] = defaultdict(list)

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                multi_turn_data = MultiTurn.model_validate_json(line)
                session_id = multi_turn_data.session_id or str(uuid.uuid4())
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
