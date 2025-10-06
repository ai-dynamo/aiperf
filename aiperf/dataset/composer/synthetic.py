# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

from aiperf.common.config import UserConfig
from aiperf.common.enums import ComposerType
from aiperf.common.factories import ComposerFactory
from aiperf.common.models import Audio, Conversation, Image, Text, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset import utils
from aiperf.dataset.composer.base import BaseDatasetComposer


@ComposerFactory.register(ComposerType.SYNTHETIC)
class SyntheticDatasetComposer(BaseDatasetComposer):
    def __init__(self, config: UserConfig, tokenizer: Tokenizer):
        super().__init__(config, tokenizer)

        if (
            not self.include_prompt
            and not self.include_image
            and not self.include_audio
        ):
            raise ValueError(
                "All synthetic data are disabled. "
                "Please enable at least one of prompt, image, or audio by "
                "setting the mean to a positive value."
            )

    def create_dataset(self) -> list[Conversation]:
        """Create a synthetic conversation dataset from the given configuration.

        It generates a set of conversations with a varying number of turns,
        where each turn contains synthetic text, image, and audio payloads.

        Returns:
            list[Conversation]: A list of conversation objects.
        """
        conversations = []
        for _ in range(self.config.input.conversation.num):
            conversation = Conversation(session_id=str(uuid.uuid4()))

            num_turns = utils.sample_positive_normal_integer(
                self.config.input.conversation.turn.mean,
                self.config.input.conversation.turn.stddev,
            )
            self.logger.debug("Creating conversation with %d turns", num_turns)

            for turn_idx in range(num_turns):
                turn = self._create_turn(is_first=(turn_idx == 0))
                conversation.turns.append(turn)
            conversations.append(conversation)
        return conversations

    def _create_turn(self, is_first: bool) -> Turn:
        """Create a turn object that contains synthetic payloads to send.

        It generates multi-modal data (e.g. text, image, audio) using synthetic
        generators and also the delay between turns.

        Args:
            is_first: Whether the turn is the first turn in the conversation.

        Returns:
            Turn: A dataset representation of a single turn.
        """
        turn = Turn()

        if self.include_prompt:
            turn.texts.append(self._generate_text_payloads(turn, is_first))
        if self.include_image:
            turn.images.append(self._generate_image_payloads())
        if self.include_audio:
            turn.audios.append(self._generate_audio_payloads())

        if not is_first:
            delay = utils.sample_positive_normal_integer(
                self.config.input.conversation.turn.delay.mean,
                self.config.input.conversation.turn.delay.stddev,
            )
            turn.delay = delay * self.config.input.conversation.turn.delay.ratio

        if not turn.texts and not turn.images and not turn.audios:
            self.logger.warning(
                "There were no synthetic payloads generated. "
                "Please enable at least one of prompt, image, or audio by "
                "setting the mean to a positive value."
            )

        self._finalize_turn(turn)

        return turn

    def _generate_text_payloads(self, turn: Turn, is_first: bool) -> Text:
        """Generate text payloads for a single turn.

        Args:
            turn: The turn object (used for caching sequence lengths)
            is_first: Whether the turn is the first turn in the conversation.

        Returns:
            Text: A text payload object.
        """
        text = Text(name="text")

        # Sample ISL/OSL pair for this request (cached for consistency)
        turn_id = id(turn)
        isl, _ = self._get_turn_sequence_lengths(turn_id)

        # Preserve original variance unless sequence distribution is active
        stddev = (
            0
            if self._seq_distribution is not None
            else self.config.input.prompt.input_tokens.stddev
        )

        for _ in range(self.config.input.prompt.batch_size):
            # Generate prompt content using the sampled input sequence length
            content = self.prompt_generator.generate(mean=isl, stddev=stddev)

            # Add prefix prompt if this is the first turn and prefix is enabled
            if is_first and self.prefix_prompt_enabled:
                prefix = self.prompt_generator.get_random_prefix_prompt()
                content = f"{prefix} {content}"

            text.contents.append(content)

        return text

    def _generate_image_payloads(self) -> Image:
        """
        Generate synthetic images if the image width and height are specified.

        Returns:
            Image: An image payload object.
        """
        image = Image(name="image_url")
        for _ in range(self.config.input.image.batch_size):
            data = self.image_generator.generate()
            image.contents.append(data)
        return image

    def _generate_audio_payloads(self) -> Audio:
        """
        Generate synthetic audios if the audio length is specified.

        Returns:
            Audio: An audio payload object.
        """
        audio = Audio(name="input_audio")
        for _ in range(self.config.input.audio.batch_size):
            data = self.audio_generator.generate()
            audio.contents.append(data)
        return audio

    @property
    def include_prompt(self) -> bool:
        return self.config.input.prompt.input_tokens.mean > 0

    @property
    def include_image(self) -> bool:
        return (
            self.config.input.image.width.mean > 0
            and self.config.input.image.height.mean > 0
        )

    @property
    def include_audio(self) -> bool:
        return self.config.input.audio.length.mean > 0
