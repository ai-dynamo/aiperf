# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.config.config_defaults import InputDefaults
from aiperf.common.models import Audio, Conversation, Image, Text, Turn, Video
from aiperf.common.session_id_generator import SessionIDGenerator
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.composer.base import BaseDatasetComposer
from aiperf.dataset.composer.media_mix_resolver import MediaMixResolver, ResolvedTurn


class SyntheticDatasetComposer(BaseDatasetComposer):
    def __init__(self, config: UserConfig, tokenizer: Tokenizer | None):
        super().__init__(config, tokenizer)
        self.session_id_generator = SessionIDGenerator(seed=config.input.random_seed)

        self._turn_sampler_rng = rng.derive("composer.conversation.turn_count")
        self._delay_sampler_rng = rng.derive("composer.conversation.turn_delay")

        # Set default sampling strategy for synthetic datasets if not explicitly set
        if self.config.input.dataset_sampling_strategy is None:
            self.config.input.dataset_sampling_strategy = (
                InputDefaults.DATASET_SAMPLING_STRATEGY
            )
            self.info(
                f"Using default sampling strategy for synthetic dataset: {InputDefaults.DATASET_SAMPLING_STRATEGY}"
            )

        # Initialize media mix resolver if configured
        self._media_mix_resolver: MediaMixResolver | None = None
        if config.input.media_mix:
            self._media_mix_resolver = MediaMixResolver(config.input.media_mix)

        # Validate that at least one data source is enabled (skip when media mix is active)
        if self._media_mix_resolver is None and (
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
        for _ in range(self.config.input.conversation.num_dataset_entries):
            conversation = Conversation(session_id=self.session_id_generator.next())

            num_turns = self._turn_sampler_rng.sample_positive_normal_integer(
                self.config.input.conversation.turn.mean,
                self.config.input.conversation.turn.stddev,
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
        if self._media_mix_resolver is not None:
            return self._create_media_mix_turn(is_first)

        turn = Turn()

        if self.include_prompt:
            turn.texts.append(self._generate_text_payloads(turn, is_first))
        if self.include_image:
            turn.images.append(self._generate_image_payloads())
        if self.include_audio:
            turn.audios.append(self._generate_audio_payloads())
        if self.include_video:
            turn.videos.append(self._generate_video_payloads())

        if not is_first and self.config.input.conversation.turn.delay.mean > 0:
            delay = self._delay_sampler_rng.sample_positive_normal_integer(
                self.config.input.conversation.turn.delay.mean,
                self.config.input.conversation.turn.delay.stddev,
            )
            turn.delay = delay * self.config.input.conversation.turn.delay.ratio

        if not turn.texts and not turn.images and not turn.audios and not turn.videos:
            self.logger.warning(
                "There were no synthetic payloads generated. "
                "Please enable at least one of prompt, image, or audio by "
                "setting the mean to a positive value."
            )

        self._finalize_turn(turn)

        return turn

    def _create_media_mix_turn(self, is_first: bool) -> Turn:
        """Create a turn using media mix archetype sampling.

        Args:
            is_first: Whether the turn is the first turn in the conversation.

        Returns:
            Turn: A dataset turn with modalities determined by the sampled archetype.
        """
        resolved = self._media_mix_resolver.resolve_turn()
        turn = Turn(archetype_name=resolved.archetype_name)

        if resolved.include_text and self.prompt_generator is not None:
            turn.texts.append(
                self._generate_text_payloads_with_override(turn, is_first, resolved)
            )

        self._populate_image_payloads(turn, resolved.image_generators)
        self._populate_audio_payloads(turn, resolved.audio_generators)
        self._populate_video_payloads(turn, resolved.video_generators)

        self._apply_turn_delay(turn, is_first)
        self._cache_resolved_sequence_lengths(turn, resolved)

        self._finalize_turn(turn)
        return turn

    def _populate_image_payloads(self, turn: Turn, items: list[tuple]) -> None:
        """Generate and attach image payloads to the turn for each (generator, batch_size)."""
        for generator, batch_size in items:
            image = Image(name="image_url")
            for _ in range(batch_size):
                image.contents.append(generator.generate())
            turn.images.append(image)

    def _populate_audio_payloads(self, turn: Turn, items: list[tuple]) -> None:
        """Generate and attach audio payloads to the turn for each (generator, batch_size)."""
        for generator, batch_size in items:
            audio = Audio(name="input_audio")
            for _ in range(batch_size):
                audio.contents.append(generator.generate())
            turn.audios.append(audio)

    def _populate_video_payloads(self, turn: Turn, items: list[tuple]) -> None:
        """Generate and attach video payloads to the turn for each (generator, batch_size)."""
        for generator, batch_size in items:
            video = Video(name="video_url")
            for _ in range(batch_size):
                data = generator.generate()
                if data:
                    video.contents.append(data)
            turn.videos.append(video)

    def _apply_turn_delay(self, turn: Turn, is_first: bool) -> None:
        """Apply inter-turn delay if configured and this isn't the first turn."""
        if is_first or self.config.input.conversation.turn.delay.mean <= 0:
            return
        delay = self._delay_sampler_rng.sample_positive_normal_integer(
            self.config.input.conversation.turn.delay.mean,
            self.config.input.conversation.turn.delay.stddev,
        )
        turn.delay = delay * self.config.input.conversation.turn.delay.ratio

    def _cache_resolved_sequence_lengths(
        self, turn: Turn, resolved: ResolvedTurn
    ) -> None:
        """Cache overridden ISL/OSL so _finalize_turn picks them up via _set_max_tokens."""
        if resolved.output_tokens_mean is None:
            return
        isl = (
            resolved.input_tokens_mean
            if resolved.input_tokens_mean is not None
            else self.config.input.prompt.input_tokens.mean
        )
        self._turn_sequence_cache[id(turn)] = (isl, resolved.output_tokens_mean)

    def _generate_text_payloads_with_override(
        self, turn: Turn, is_first: bool, resolved: ResolvedTurn
    ) -> Text:
        """Generate text payloads with per-archetype ISL/OSL overrides.

        Args:
            turn: The turn object (used for caching sequence lengths).
            is_first: Whether the turn is the first turn in the conversation.
            resolved: Resolved turn with optional text overrides.

        Returns:
            Text: A text payload object.
        """
        if self.prompt_generator is None:
            raise ValueError(
                "Text prompt generation requires a tokenizer. Either provide a "
                "--tokenizer or use an endpoint that supports tokenization."
            )

        text = Text(name="text")

        # Use override ISL or fall back to global config
        isl = (
            resolved.input_tokens_mean
            if resolved.input_tokens_mean is not None
            else self.config.input.prompt.input_tokens.mean
        )
        stddev = (
            resolved.input_tokens_stddev
            if resolved.input_tokens_stddev is not None
            else (
                0
                if self._seq_distribution is not None
                else self.config.input.prompt.input_tokens.stddev
            )
        )

        for _ in range(self.config.input.prompt.batch_size):
            content = self.prompt_generator.generate(mean=isl, stddev=stddev)

            if is_first and self.prefix_prompt_enabled:
                prefix = self.prompt_generator.get_random_prefix_prompt()
                content = f"{prefix} {content}"

            text.contents.append(content)

        return text

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

    def _generate_video_payloads(self) -> Video:
        """
        Generate synthetic videos if the video width and height are specified.

        Returns:
            Video: A video payload object.
        """
        video = Video(name="video_url")
        for _ in range(self.config.input.video.batch_size):
            data = self.video_generator.generate()
            if data:  # Only append if video was actually generated
                video.contents.append(data)
        return video

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

    @property
    def include_video(self) -> bool:
        return bool(self.config.input.video.width and self.config.input.video.height)
