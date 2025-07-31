# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import random
from abc import ABC, abstractmethod

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums.model_enums import ModelSelectionStrategy
from aiperf.common.models import Conversation
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator import (
    AudioGenerator,
    ImageGenerator,
    PromptGenerator,
)


class BaseDatasetComposer(ABC):
    def __init__(self, config: UserConfig, tokenizer: Tokenizer):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.prompt_generator = PromptGenerator(config.input.prompt, tokenizer)
        self.image_generator = ImageGenerator(config.input.image)
        self.audio_generator = AudioGenerator(config.input.audio)
        self.turn_count = 0

    @abstractmethod
    def create_dataset(self) -> list[Conversation]:
        """
        Create a set of conversation objects from the given configuration.

        Returns:
            list[Conversation]: A list of conversation objects.
        """
        ...

    def _select_model_name(self) -> str:
        if (
            self.config.endpoint.model_selection_strategy
            == ModelSelectionStrategy.RANDOM
        ):
            return random.choice(self.config.endpoint.model_names)
        elif (
            self.config.endpoint.model_selection_strategy
            == ModelSelectionStrategy.ROUND_ROBIN
        ):
            model_name = self.config.endpoint.model_names[
                self.turn_count % len(self.config.endpoint.model_names)
            ]
            self.turn_count += 1
            return model_name
        else:
            raise ValueError(
                f"Invalid model selection strategy: {self.config.endpoint.model_selection_strategy}."
            )

    @property
    def prefix_prompt_enabled(self) -> bool:
        return self.config.input.prompt.prefix_prompt.length > 0
