# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import uuid

from aiperf.services.dataset import utils
from aiperf.services.dataset.composer.base import BaseConversationComposer
from aiperf.services.dataset.config import CustomDataConfig
from aiperf.services.dataset.conversation import Conversation, Image, Text, Turn


class SyntheticConversationComposer(BaseConversationComposer):
    # TODO: only take necessary configs
    def __init__(self, config: CustomDataConfig):
        super().__init__(config)

    def create_conversation(self) -> Conversation:
        """Create a synthetic conversation dataset from the given configuration.

        It generates a conversation with a varying number of turns, where each turn
        contains synthetic text, image, and audio payloads.

        Returns:
            Conversation: A dataset representation of a full conversation.
        """

        conversation = Conversation()
        conversation.session_id = str(uuid.uuid4())

        num_turns = utils.sample_positive_normal_integer(
            self.config.turn.mean,
            self.config.turn.stddev,
        )
        self.logger.debug("Creating conversation with %d turns", num_turns)

        for turn_idx in range(num_turns):
            turn = self._create_turn(is_first=(turn_idx == 0))
            conversation.turns.append(turn)

        return conversation

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

        # TODO: async generation
        self._generate_text_payloads(turn, is_first)
        self._generate_image_payloads(turn)
        self._generate_audio_payloads(turn)

        # TODO: Is this still valid? (Was only for PA)
        if not is_first:
            session_delay = utils.sample_positive_normal_integer(
                self.config.turn.delay.mean,
                self.config.turn.delay.stddev,
            )
            turn.delay = session_delay

        return turn

    def _generate_text_payloads(self, turn: Turn, is_first: bool) -> None:
        """Generate synthetic text payloads.

        If the turn is the first turn in the conversation, it could add a prefix prompt
        to the prompt.

        Args:
            turn: The turn object to add the text payloads to.
            is_first: Whether the turn is the first turn in the conversation.
        """
        for _ in range(self.config.prompt.batch_size):
            text = Text(name="text")
            prompt = self._prompt_generator.generate()

            # XXX: Should we take this outside of this class? (since this is shared with custom composer)
            if self.add_prefix_prompt and is_first:
                # TODO: Rename
                prefix_prompt = self._prompt_generator.get_random_prefix_prompt()
                prompt = f"{prefix_prompt} {prompt}"

            text.content = prompt
            turn.text.append(text)

    def _generate_image_payloads(self, turn: Turn) -> None:
        """
        Generate synthetic images if the image width and height are specified.
        """
        if self.include_image:
            for _ in range(self.config.image.batch_size):
                image = Image(name="image_url")
                image.content = self._image_generator.generate()
                turn.image.append(image)

    def _generate_audio_payloads(self, turn: Turn) -> None:
        """
        Generate synthetic audios if the audio length is specified.
        """
        if self.include_audio:
            for _ in range(self.config.audio.batch_size):
                turn.audio.append(self._audio_generator.generate())

    @property
    def include_image(self) -> bool:
        return self.config.image.width_mean > 0 and self.config.image.height_mean > 0

    @property
    def include_audio(self) -> bool:
        return self.config.audio.length_mean > 0


if __name__ == "__main__":
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.services.dataset.config import (
        PrefixPromptConfig,
        PromptConfig,
    )

    tokenizer = Tokenizer.from_pretrained("gpt2")
    config = CustomDataConfig(
        prompt=PromptConfig(
            tokenizer=tokenizer,
            mean=10,
            stddev=0,
            prefix_prompt=PrefixPromptConfig(
                pool_size=1,
                length=4,
            ),
            block_size=512,
        ),
        # image=ImageConfig(
        #    width_mean=5,
        #    height_mean=5,
        # ),
        # audio=AudioConfig(
        #    length_mean=1,
        # ),
        num_dataset_entries=2,
        tokenizer=tokenizer,
    )
    composer = SyntheticConversationComposer(config)
    conversation = composer.create_conversation()

    from rich import print

    print(conversation)
