# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CustomDatasetType
from aiperf.services.dataset.loader.factory import CustomDatasetFactory
from aiperf.services.dataset.loader.models import CustomData


@CustomDatasetFactory.register(CustomDatasetType.MULTI_TURN)
class MultiTurnDatasetLoader:
    """Multi-turn custom dataset.

    User can use this format to provide a custom multi-turn dataset.
    Each line in the file will be treated as a single turn conversation.

    The multi-turn type
      - supports multi-modal data (e.g. text, image, audio)
      - supports multi-turn features (e.g. delay, sessions, etc.)
      - supports client-side batching for each data (e.g. batch_size > 1)

    Example:
    1. Simple version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"text": "Hello", "image": "url", "delay": 0},
            {"text": "Hi there", "delay": 1000},
        ],
    }
    ```

    2. Advanced version
    ```json
    {
        "session_id": "session_123",
        "turns": [
            {"text": [{"name": "nameA", "content": "..."}, ...], "image": [...], "delay": 0},
            {"text": [{"name": "nameA", "content": "..."}, ...], "delay": 1000},
        ]
    }
    ```
    """

    def __init__(self, filename: str):
        self.filename = filename

    def load_dataset(self) -> dict[str, list[CustomData]]:
        raise NotImplementedError
