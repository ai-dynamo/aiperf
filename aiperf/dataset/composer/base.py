# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random
from abc import ABC, abstractmethod

import numpy as np

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

        # Initialize RNG for sequence distribution sampling (avoid reseeding on each sample)
        seed = getattr(self.config.input, "random_seed", None)
        self._seq_rng = np.random.default_rng(seed) if seed is not None else None

        # Cache for turn-level sequence lengths to ensure ISL/OSL pairing consistency
        self._turn_sequence_cache: dict[int, tuple[int, int]] = {}

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

    def _get_turn_sequence_lengths(self, turn_id: int) -> tuple[int, int]:
        """Get or sample ISL/OSL pair for a specific turn, ensuring consistency.

        This method caches the sequence lengths per turn to ensure that the same
        ISL/OSL pair is used for both prompt generation and max_tokens setting.

        Args:
            turn_id: Unique identifier for the turn

        Returns:
            Tuple of (input_seq_len, output_seq_len)
        """
        if turn_id in self._turn_sequence_cache:
            return self._turn_sequence_cache[turn_id]

        if self._seq_distribution is None:
            seq_lengths = (
                self.config.input.prompt.input_tokens.mean,
                self.config.input.prompt.output_tokens.mean
                or max(128, self.config.input.prompt.input_tokens.mean // 2),
            )
        else:
            seq_lengths = self._seq_distribution.sample(random_state=self._seq_rng)

        self._turn_sequence_cache[turn_id] = seq_lengths
        return seq_lengths

    def _clear_turn_cache(self, turn_id: int) -> None:
        """Clear cached sequence lengths for a specific turn.

        Args:
            turn_id: Turn identifier to remove from cache
        """
        self._turn_sequence_cache.pop(turn_id, None)

    def _set_max_tokens(self, turn: Turn) -> None:
        """Set max_tokens for the turn based on the sequence distribution or output configuration.

        Args:
            turn: The turn object to finalize.
        """
        if self._seq_distribution is not None:
            # Use cached sequence distribution to get OSL (ensures ISL/OSL pairing consistency)
            turn_id = id(turn)
            _, osl = self._get_turn_sequence_lengths(turn_id)
            turn.max_tokens = osl
        else:
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

        # Clear cached sequence lengths for this turn to free memory
        turn_id = id(turn)
        self._clear_turn_cache(turn_id)

    @property
    def prefix_prompt_enabled(self) -> bool:
        return self.config.input.prompt.prefix_prompt.length > 0
