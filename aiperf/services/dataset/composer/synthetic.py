# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

from aiperf.common.dataset_models import Audio, Conversation, Image, Text, Turn
from aiperf.services.dataset import utils
from aiperf.services.dataset.composer.base import BaseDatasetComposer
from aiperf.services.dataset.config import DatasetConfig


class SyntheticDatasetComposer(BaseDatasetComposer):
    def __init__(self, config: DatasetConfig):
        super().__init__(config)

    def create_dataset(self) -> list[Conversation]:
        """Create a synthetic conversation dataset from the given configuration.

        It generates a set of conversations with a varying number of turns,
        where each turn contains synthetic text, image, and audio payloads.

        Returns:
            list[Conversation]: A list of conversation objects.
        """
        conversations = []
        for _ in range(self.config.num_conversations):
            conversation = Conversation(session_id=str(uuid.uuid4()))

            num_turns = utils.sample_positive_normal_integer(
                self.config.turn.mean,
                self.config.turn.stddev,
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
        self._generate_text_payloads(turn, is_first)

        if self.include_image:
            self._generate_image_payloads(turn)
        if self.include_audio:
            self._generate_audio_payloads(turn)

        # Add randomized delays between each turn. Skip if first turn.
        if not is_first:
            turn.delay = utils.sample_positive_normal_integer(
                self.config.turn.delay.mean,
                self.config.turn.delay.stddev,
            )

        return turn

    def _generate_text_payloads(self, turn: Turn, is_first: bool) -> None:
        """Generate synthetic text payloads.

        If the turn is the first turn in the conversation, it could add a prefix prompt
        to the prompt.

        Args:
            turn: The turn object to add the text payloads to.
            is_first: Whether the turn is the first turn in the conversation.
        """
        text = Text(name="text")
        for _ in range(self.config.prompt.batch_size):
            prompt = self.prompt_generator.generate(
                mean=self.config.prompt.mean,
                stddev=self.config.prompt.stddev,
            )

            if self.add_prefix_prompt and is_first:
                # TODO: Rename
                prefix_prompt = self.prompt_generator.get_random_prefix_prompt()
                prompt = f"{prefix_prompt} {prompt}"

            text.content.append(prompt)
        turn.text.append(text)

    def _generate_image_payloads(self, turn: Turn) -> None:
        """
        Generate synthetic images if the image width and height are specified.
        """
        image = Image(name="image_url")
        for _ in range(self.config.image.batch_size):
            data = self.image_generator.generate()
            image.content.append(data)
        turn.image.append(image)

    def _generate_audio_payloads(self, turn: Turn) -> None:
        """
        Generate synthetic audios if the audio length is specified.
        """
        audio = Audio(name="input_audio")
        for _ in range(self.config.audio.batch_size):
            data = self.audio_generator.generate()
            audio.content.append(data)
        turn.audio.append(audio)

    @property
    def include_image(self) -> bool:
        return self.config.image.width_mean > 0 and self.config.image.height_mean > 0

    @property
    def include_audio(self) -> bool:
        return self.config.audio.length_mean > 0
