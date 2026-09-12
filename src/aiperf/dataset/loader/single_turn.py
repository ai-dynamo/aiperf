# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aiperf.common.enums import ConversationContextMode, MediaType
from aiperf.common.models import Conversation, Turn
from aiperf.dataset.loader.base_loader import BaseFileLoader
from aiperf.dataset.loader.mixins import MediaConversionMixin
from aiperf.dataset.loader.models import SingleTurn
from aiperf.plugin.enums import DatasetSamplingStrategy


class SingleTurnDatasetLoader(BaseFileLoader, MediaConversionMixin):
    """A dataset loader that loads single turn data from a file.

    The single turn type
      - supports multi-modal data (e.g. text, image, audio)
      - supports client-side batching for each data (e.g. batch_size > 1)
      - supports optional session_id for causal ordering across entries

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
        """Load single-turn data from a file or inline records.

        Each record represents a single turn conversation. Multiple records with
        the same session_id (or generated UUID) are grouped together.
        """
        data: dict[str, list[SingleTurn]] = defaultdict(list)
        for record_dict in self._iter_record_dicts():
            single_turn_data = SingleTurn.model_validate(record_dict)
            session_id = single_turn_data.session_id or self.session_id_generator.next()
            data[session_id].append(single_turn_data)
        return data

    def convert_to_conversations(
        self, data: dict[str, list[SingleTurn]]
    ) -> list[Conversation]:
        """Convert single turn data to conversation objects.

        When `endpoint.uuid_and_strip` is set, images repeated in later turns
        of a conversation keep their UUID but drop their payload.

        Args:
            data: A dictionary mapping session_id to list of SingleTurn objects.

        Returns:
            A list of conversations.
        """
        uuid_and_strip = self.run.cfg.endpoint.uuid_and_strip
        conversations = []
        for session_id, single_turns in data.items():
            conversation = Conversation(
                session_id=session_id,
                context_mode=(
                    ConversationContextMode.MESSAGE_ARRAY_WITH_RESPONSES
                    if len(single_turns) > 1
                    else None
                ),
            )
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
                        max_tokens=single_turn.output_length,
                        extra_body=single_turn.extra,
                        raw_messages=single_turn.raw_messages,
                        raw_tools=single_turn.raw_tools,
                    )
                )
            if uuid_and_strip:
                self._dedup_repeated_images_inplace(conversation)
            conversations.append(conversation)
        return conversations

    @staticmethod
    def _dedup_repeated_images_inplace(conversation: Conversation) -> None:
        """Drop image bytes for repeated UUIDs within one conversation.

        Images repeated within one turn retain their payload because the server
        resolves that request's cache misses before populating its cache. Only
        UUIDs whose content AIPerf observed in an earlier turn are stripped.
        Explicit cache-only references pass through regardless of local history.
        """
        seen: set[str] = set()
        for turn in conversation.turns:
            new_uuids: set[str] = set()
            for image in turn.images:
                if not image.uuids:
                    continue
                for i, uuid in enumerate(image.uuids):
                    if uuid in seen:
                        image.contents[i] = ""
                    elif image.contents[i]:
                        new_uuids.add(uuid)
            seen.update(new_uuids)
