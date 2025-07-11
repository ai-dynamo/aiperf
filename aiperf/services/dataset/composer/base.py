# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from abc import ABC, abstractmethod

from aiperf.common.config import InputConfig
from aiperf.common.dataset_models import Conversation
from aiperf.common.exceptions import NotInitializedError
from aiperf.common.tokenizer import Tokenizer
from aiperf.services.dataset.generator import (
    AudioGenerator,
    BaseGenerator,
    ImageGenerator,
    PromptGenerator,
)


class BaseDatasetComposer(ABC):
    def __init__(self, config: InputConfig, tokenizer: Tokenizer):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = config
        self.tokenizer = tokenizer
        self._generators: dict[str, BaseGenerator] = {}
        self.composer_initialized: asyncio.Event = asyncio.Event()

    async def initialize(self) -> None:
        """Initialize the dataset composer."""
        self.logger.debug("Initializing dataset composer: %s", self.__class__.__name__)

        # Initialize all generators
        self._generators.update(
            {
                "prompt": PromptGenerator(self.config.prompt, self.tokenizer),
                "image": ImageGenerator(self.config.image),
                "audio": AudioGenerator(self.config.audio),
            }
        )

        await asyncio.gather(
            *[g.initialize() for g in self._generators.values()],
            return_exceptions=True,
        )
        self.composer_initialized.set()

    @abstractmethod
    async def create_dataset(self) -> list[Conversation]:
        """
        Create a set of conversation objects from the given configuration.

        Returns:
            list[Conversation]: A list of conversation objects.
        """

    async def wait_for_composer_initialized(self) -> None:
        """Wait for the dataset composer to be initialized."""
        await self.composer_initialized.wait()

    @property
    def prefix_prompt_enabled(self) -> bool:
        """Check if prefix prompts are enabled."""
        if "prompt" not in self._generators:
            raise NotInitializedError("Prompt generator is not initialized.")
        return self._generators["prompt"].prefix_prompt_enabled

    @property
    def prompt_generator(self) -> PromptGenerator | None:
        """Get the prompt generator."""
        return self._generators.get("prompt", None)

    @property
    def image_generator(self) -> ImageGenerator | None:
        """Get the image generator."""
        return self._generators.get("image", None)

    @property
    def audio_generator(self) -> AudioGenerator | None:
        """Get the audio generator."""
        return self._generators.get("audio", None)
