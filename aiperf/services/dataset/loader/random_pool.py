# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums import CustomDatasetType
from aiperf.services.dataset.loader.factory import CustomDatasetFactory
from aiperf.services.dataset.loader.models import CustomData


@CustomDatasetFactory.register(CustomDatasetType.RANDOM_POOL)
class RandomPoolDatasetLoader:
    """A random pool of conversations.

    A random pool of conversations is a pool of conversations that are randomly selected from a file.

    Example:
    ```json
    {"text": "Hello world"}
    ```
    """

    def __init__(self, filename: str):
        self.filename = filename

    def load_dataset(self) -> dict[str, list[CustomData]]:
        raise NotImplementedError
