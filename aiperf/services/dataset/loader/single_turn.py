# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid
from collections import defaultdict

from aiperf.common.enums import CustomDatasetType
from aiperf.common.factories import CustomDatasetFactory
from aiperf.services.dataset.loader.models import CustomData, SingleTurnCustomData


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
    {"text": ["Who are you?", "Hello world"], "image": ["/path/to/image.png", "/path/to/image2.png"]}
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
        "text": [
            {"name": "text_field_A", "content": ["Hello", "World"]},
            {"name": "text_field_B", "content": ["Hi there"]}
        ],
        "image": [
            {"name": "image_field_A", "content": ["/path/1.png", "/path/2.png"]},
            {"name": "image_field_B", "content": ["/path/3.png"]}
        ]
    }
    ``
    """

    def __init__(self, filename: str):
        self.filename = filename

    def load_dataset(self) -> dict[str, list[CustomData]]:
        """Load single-turn data from a JSONL file.

        Each line represents a single turn conversation. Multiple turns with
        the same session_id (or generated UUID) are grouped together.

        Returns:
            A dictionary mapping session_id to list of CustomData.
        """
        data: dict[str, list[CustomData]] = defaultdict(list)

        with open(self.filename) as f:
            for line in f:
                if (line := line.strip()) == "":
                    continue  # Skip empty lines

                single_turn_data = SingleTurnCustomData.model_validate_json(line)
                session_id = str(uuid.uuid4())
                data[session_id].append(single_turn_data)

        return data
