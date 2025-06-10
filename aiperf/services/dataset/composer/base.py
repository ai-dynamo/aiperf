# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod

from aiperf.services.dataset.config import CustomDataConfig
from aiperf.services.dataset.conversation import Conversation
from aiperf.services.dataset.generator import (
    AudioGenerator,
    ImageGenerator,
    PromptGenerator,
)


class BaseConversationComposer(ABC):
    def __init__(self, config: CustomDataConfig):
        self.config = config
        # self.tokenizer = config.tokenizer
        self.logger = logging.getLogger(__name__)

        # TODO: make these singletons?
        self._prompt_generator = PromptGenerator(config.prompt)
        self._image_generator = ImageGenerator(config.image)
        self._audio_generator = AudioGenerator(config.audio)

    @abstractmethod
    def create_conversation(self) -> Conversation:
        """
        Create a conversation dataset from the given configuration.

        This method is responsible for creating a conversation dataset from the given configuration.
        It should return a Conversation object, which is a dataset representation of a full conversation.

        Returns:
            Conversation: A dataset representation of a full conversation.
        """
        ...

    @property
    def add_prefix_prompt(self) -> bool:
        return self.config.prompt.prefix_prompt.length > 0
