# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod

from aiperf.common.dataset_models import Conversation
from aiperf.services.dataset.config import DatasetConfig
from aiperf.services.dataset.generator import (
    AudioGenerator,
    ImageGenerator,
    PromptGenerator,
)


class BaseDatasetComposer(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.prompt_generator = PromptGenerator(config.prompt, config.tokenizer)
        self.image_generator = ImageGenerator(config.image)
        self.audio_generator = AudioGenerator(config.audio)

    @abstractmethod
    def create_dataset(self) -> list[Conversation]:
        """
        Create a set of conversation objects from the given configuration.

        Returns:
            list[Conversation]: A list of conversation objects.
        """
        ...

    @property
    def prefix_prompt_enabled(self) -> bool:
        return self.config.prompt.prefix_prompt.length > 0
