# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from abc import ABC, abstractmethod

from aiperf.common.config.user_config import UserConfig
from aiperf.common.enums.model_enums import ModelSelectionStrategy
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset import utils
from aiperf.dataset.generator import (
    AudioGenerator,
    ImageGenerator,
    PromptGenerator,
)


class BaseDatasetComposer(AIPerfLoggerMixin, ABC):
    def __init__(self, config: UserConfig, tokenizer: Tokenizer, **kwargs):
        self.config = config
        super().__init__(config=config, tokenizer=tokenizer, **kwargs)
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

    def _set_max_tokens(self, turn: Turn) -> None:
        """Set max_tokens for the turn based on the output configuration.

        Args:
            turn: The turn object to finalize.
        """
        output_config = self.config.input.prompt.output_tokens
        if output_config.mean is not None:
            stddev = output_config.stddev or 0  # Default to 0 if stddev is None
            turn.max_tokens = utils.sample_positive_normal_integer(
                output_config.mean, stddev
            )

    def _finalize_turn(self, turn: Turn) -> None:
        """Finalize a turn by populating all required metadata fields.

        This method handles:
        - Model name selection
        - Max tokens sampling based on output configuration
        - Any other turn-level metadata that needs to be set

        Args:
            turn: The turn object to finalize.
        """
        turn.model = self._select_model_name()
        self._set_max_tokens(turn)

    @property
    def prefix_prompt_enabled(self) -> bool:
        return self.config.input.prompt.prefix_prompt.length > 0
