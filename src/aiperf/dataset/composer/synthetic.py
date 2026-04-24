# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common import random_generator as rng
from aiperf.common.models import Audio, Conversation, Image, Text, Turn, Video
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.types import NormalDistribution
from aiperf.dataset.composer.base import BaseDatasetComposer

if TYPE_CHECKING:
    from aiperf.config import BenchmarkRun


class SyntheticDatasetComposer(BaseDatasetComposer):
    def __init__(self, run: BenchmarkRun, tokenizer: Tokenizer | None):
        super().__init__(run, tokenizer)
        # Use dataset-specific random seed or fall back to global
        seed = self.dataset_config.random_seed or run.cfg.random_seed
        self.session_id_generator = SessionIDGenerator(seed=seed)

        self._turn_sampler_rng = rng.derive("composer.conversation.turn_count")
        self._delay_sampler_rng = rng.derive("composer.conversation.turn_delay")

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
        num_entries = self.dataset_config.entries

        turns_config = self.dataset_config.turns

        for _ in range(num_entries):
            conversation = Conversation(session_id=self.session_id_generator.next())

            num_turns = (
                turns_config.sample_int(self._turn_sampler_rng) if turns_config else 1
            )
            self.logger.debug("Creating conversation with %d turns", num_turns)

            for turn_idx in range(num_turns):
                turn = self._create_turn(is_first=(turn_idx == 0))
                conversation.turns.append(turn)
            conversations.append(conversation)

        # Finalize all conversations (turn metadata + context prompts)
        self._finalize_conversations(conversations)
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
        if self.include_video:
            turn.videos.append(self._generate_video_payloads())

        turn_delay_config = self.dataset_config.turn_delay
        if not is_first and turn_delay_config and turn_delay_config.mean > 0:
            delay = turn_delay_config.sample_int(self._delay_sampler_rng)
            turn.delay = delay * self.dataset_config.turn_delay_ratio

        if not turn.texts and not turn.images and not turn.audios and not turn.videos:
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

        Raises:
            ValueError: If prompt_generator is not available (tokenizer was not configured).
        """
        if self.prompt_generator is None:
            raise ValueError(
                "Text prompt generation requires a tokenizer. Either provide a "
                "--tokenizer or use an endpoint that supports tokenization."
            )

        text = Text(name="text")

        # Sample ISL/OSL pair for this request (cached for consistency)
        turn_id = id(turn)
        isl, _ = self._get_turn_sequence_lengths(turn_id)

        # Get prompts config
        prompts_config = getattr(self.dataset_config, "prompts", None)

        isl_stddev = 0
        if self._seq_distribution is None and prompts_config and prompts_config.isl:
            isl_stddev = (
                prompts_config.isl.stddev
                if isinstance(prompts_config.isl, NormalDistribution)
                else 0
            )

        batch_size = prompts_config.batch_size if prompts_config else 1

        for _ in range(batch_size):
            # Generate prompt content using the sampled input sequence length
            content = self.prompt_generator.generate(mean=isl, stddev=isl_stddev)

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
        images_config = getattr(self.dataset_config, "images", None)
        batch_size = images_config.batch_size if images_config else 1
        for _ in range(batch_size):
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
        audio_config = getattr(self.dataset_config, "audio", None)
        batch_size = audio_config.batch_size if audio_config else 1
        for _ in range(batch_size):
            data = self.audio_generator.generate()
            audio.contents.append(data)
        return audio

    def _generate_video_payloads(self) -> Video:
        """
        Generate synthetic videos if the video width and height are specified.

        Returns:
            Video: A video payload object.
        """
        video = Video(name="video_url")
        video_config = getattr(self.dataset_config, "video", None)
        batch_size = video_config.batch_size if video_config else 1
        for _ in range(batch_size):
            data = self.video_generator.generate()
            if data:  # Only append if video was actually generated
                video.contents.append(data)
        return video

    @property
    def include_prompt(self) -> bool:
        prompts_config = getattr(self.dataset_config, "prompts", None)
        if not prompts_config or not prompts_config.isl:
            return False
        return prompts_config.isl.mean > 0

    @property
    def include_image(self) -> bool:
        images_config = getattr(self.dataset_config, "images", None)
        if not images_config:
            return False
        width_mean = images_config.width.mean
        height_mean = images_config.height.mean
        return width_mean > 0 and height_mean > 0

    @property
    def include_audio(self) -> bool:
        audio_config = getattr(self.dataset_config, "audio", None)
        if not audio_config:
            return False
        return audio_config.length.mean > 0

    @property
    def include_video(self) -> bool:
        video_config = getattr(self.dataset_config, "video", None)
        if not video_config:
            return False
        return bool(video_config.width and video_config.height)
