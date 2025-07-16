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

    See `SingleTurnCustomData` for the schema of the data and more details.

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
