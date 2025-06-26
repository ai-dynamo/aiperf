# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CustomDatasetType
from aiperf.services.dataset.loader.factory import CustomDatasetFactory
from aiperf.services.dataset.loader.models import CustomData


@CustomDatasetFactory.register(CustomDatasetType.SINGLE_TURN)
class SingleTurnDatasetLoader:
    """Minimal single turn custom dataset.

    User can use this format to quickly provide a custom single turn dataset.
    Each line in the file will be treated as a single turn conversation.

    The single turn type
      - supports multi-modal data (e.g. text, image, audio)
      - DOES NOT support multi-turn features (e.g. delay, sessions, etc.)
      - supports client-side batching for each data (e.g. batch_size > 1)

    Example:
    1. Single-batch, multi-modal
    ```json
    {"text": "What is in the image?", "image": "/path/to/image.png"}
    {"text": "What is deep learning?"}
    ```

    2. Multi-batch, multi-modal
    ```json
    {
        "text": ["What is the weather today?", "What is deep learning?"],
        "image": ["/path/to/image.png", "/path/to/image2.png"],
    }
    ```
    """

    def __init__(self, filename: str):
        self.filename = filename

    def load_dataset(self) -> dict[str, list[CustomData]]:
        raise NotImplementedError
