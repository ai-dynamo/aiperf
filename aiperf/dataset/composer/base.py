# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from abc import ABC, abstractmethod

from aiperf.common.config import UserConfig
from aiperf.common.enums import ModelSelectionStrategy
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

        # Initialize sequence distribution
        self._seq_distribution = config.input.prompt.get_sequence_distribution()

        # Cache for sampled sequence lengths per turn
        self._turn_sequence_cache = {}

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

    def _sample_sequence_lengths(self, turn_id: str) -> tuple[int, int]:
        """Sample ISL/OSL pair for a turn, caching the result.

        Args:
            turn_id: Unique identifier for the turn

        Returns:
            Tuple of (input_seq_len, output_seq_len)
        """
        if turn_id not in self._turn_sequence_cache:
            # Use random seed from config if available for reproducible results
            random_seed = getattr(self.config.input, "random_seed", None)
            self._turn_sequence_cache[turn_id] = self._seq_distribution.sample(
                random_state=random_seed
            )
        return self._turn_sequence_cache[turn_id]

    def _set_max_tokens(self, turn: Turn) -> None:
        """Set max_tokens for the turn based on the sequence distribution or output configuration.

        Args:
            turn: The turn object to finalize.
        """
        # Check if we have a cached OSL from sequence distribution
        turn_id = getattr(turn, "_turn_id", None)
        if turn_id and turn_id in self._turn_sequence_cache:
            _, osl = self._turn_sequence_cache[turn_id]
            turn.max_tokens = osl
        else:
            # For custom datasets or when no sequence distribution is used
            # Check if we should sample from distribution anyway
            if self.config.input.prompt.sequence_distribution is not None:
                # Generate a new sample for this turn (custom datasets)
                turn_id = f"custom_turn_{id(turn)}"
                _, osl = self._sample_sequence_lengths(turn_id)
                turn.max_tokens = osl
            else:
                # Fallback to legacy behavior
                output_tokens_config = self.config.input.prompt.output_tokens
                if output_tokens_config.mean is not None:
                    stddev = output_tokens_config.stddev
                    turn.max_tokens = utils.sample_positive_normal_integer(
                        output_tokens_config.mean, stddev
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
