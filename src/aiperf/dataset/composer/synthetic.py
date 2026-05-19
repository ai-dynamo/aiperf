# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.common import random_generator as rng
from aiperf.common.models import Audio, Conversation, Image, Text, Turn, Video
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset import SyntheticDataset
from aiperf.dataset.composer.base import BaseDatasetComposer

if TYPE_CHECKING:
    from aiperf.config.resolution.plan import BenchmarkRun


def _expected(distribution: object | None) -> float:
    """Mean of a ``SamplingDistribution`` (or 0)."""
    if distribution is None:
        return 0.0
    return float(getattr(distribution, "expected_value", 0.0))


def _stddev_int(distribution: object | None) -> int:
    """Integer stddev of a distribution (Normal only) or 0."""
    return int(getattr(distribution, "stddev", 0) or 0)


class SyntheticDatasetComposer(BaseDatasetComposer):
    def __init__(self, *, run: BenchmarkRun, tokenizer: Tokenizer | None, **kwargs):
        super().__init__(run=run, tokenizer=tokenizer, **kwargs)

        dataset = run.cfg.get_default_dataset()
        if not isinstance(dataset, SyntheticDataset):
            raise ValueError("SyntheticDatasetComposer requires a synthetic dataset.")

        self.session_id_generator = SessionIDGenerator(
            seed=dataset.random_seed
            if dataset.random_seed is not None
            else run.random_seed
        )

        self._turn_sampler_rng = rng.derive("composer.conversation.turn_count")
        self._delay_sampler_rng = rng.derive("composer.conversation.turn_delay")

        # Cache the dataset shape and derived counts. Dataset sub-shapes
        # (prompts/images/audio/video/turns/turn_delay) may be None when the
        # user didn't configure them — apply the canonical defaults
        # (turn=1, batch=1, delay=0) explicitly here.
        self._num_entries = dataset.entries
        self._turn_mean = max(1, int(_expected(dataset.turns)))
        self._turn_stddev = _stddev_int(dataset.turns)
        self._turn_delay_mean = _expected(dataset.turn_delay)
        self._turn_delay_stddev = _stddev_int(dataset.turn_delay)
        self._turn_delay_ratio = dataset.turn_delay_ratio
        self._prompt_batch_size = (
            dataset.prompts.batch_size if dataset.prompts is not None else 1
        )
        self._image_batch_size = (
            dataset.images.batch_size if dataset.images is not None else 1
        )
        self._audio_batch_size = (
            dataset.audio.batch_size if dataset.audio is not None else 1
        )
        self._video_batch_size = (
            dataset.video.batch_size if dataset.video is not None else 1
        )
        self._isl_stddev = _stddev_int(
            dataset.prompts.isl if dataset.prompts is not None else None
        )

        # Inclusion flags (computed once at init).
        self._include_prompt = (
            _expected(dataset.prompts.isl if dataset.prompts is not None else None) > 0
        )
        self._include_image = (
            dataset.images is not None
            and _expected(dataset.images.width) > 0
            and _expected(dataset.images.height) > 0
        )
        self._include_audio = (
            dataset.audio is not None and _expected(dataset.audio.length) > 0
        )
        self._include_video = bool(
            dataset.video is not None and dataset.video.width and dataset.video.height
        )

        if (
            not self._include_prompt
            and not self._include_image
            and not self._include_audio
            and not self._include_video
        ):
            raise ValueError(
                "All synthetic data are disabled. "
                "Please enable at least one of prompt, image, audio, or video by "
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
        for _ in range(self._num_entries):
            conversation = Conversation(session_id=self.session_id_generator.next())

            num_turns = self._turn_sampler_rng.sample_positive_normal_integer(
                self._turn_mean,
                self._turn_stddev,
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

        if not is_first and self._turn_delay_mean > 0:
            delay = self._delay_sampler_rng.sample_positive_normal_integer(
                int(self._turn_delay_mean),
                self._turn_delay_stddev,
            )
            turn.delay = delay * self._turn_delay_ratio

        if not turn.texts and not turn.images and not turn.audios and not turn.videos:
            self.logger.warning(
                "There were no synthetic payloads generated. "
                "Please enable at least one of prompt, image, audio, or video by "
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

        # Preserve original variance unless sequence distribution is active
        stddev = 0 if self._seq_distribution is not None else self._isl_stddev

        for _ in range(self._prompt_batch_size):
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
        for _ in range(self._image_batch_size):
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
        for _ in range(self._audio_batch_size):
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
        for _ in range(self._video_batch_size):
            data = self.video_generator.generate()
            if data:  # Only append if video was actually generated
                video.contents.append(data)
        return video

    @property
    def include_prompt(self) -> bool:
        return self._include_prompt

    @property
    def include_image(self) -> bool:
        return self._include_image

    @property
    def include_audio(self) -> bool:
        return self._include_audio

    @property
    def include_video(self) -> bool:
        return self._include_video
