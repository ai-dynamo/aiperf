# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aiperf.common import random_generator as rng
from aiperf.common.enums import ConversationContextMode, ModelSelectionStrategy
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Turn
from aiperf.common.models.sequence_distribution import (
    SequenceLengthDistribution,
    SequenceLengthPair,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset import FileDataset, SyntheticDataset
from aiperf.dataset.generator.audio import AudioGenerator
from aiperf.dataset.generator.image import ImageGenerator
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.generator.video import VideoGenerator

if TYPE_CHECKING:
    from aiperf.config.dataset import VideoConfig
    from aiperf.config.dataset.content import (
        AudioConfig,
        ImageConfig,
        PrefixPromptConfig,
        PromptConfig,
    )
    from aiperf.config.distributions import SamplingDistribution
    from aiperf.config.resolution.plan import BenchmarkRun


class BaseDatasetComposer(AIPerfLoggerMixin, ABC):
    def __init__(
        self,
        *,
        run: BenchmarkRun,
        tokenizer: Tokenizer | None,
        **kwargs,
    ):
        self.run = run
        self.tokenizer = tokenizer
        super().__init__(run=run, tokenizer=tokenizer, **kwargs)

        # Cache the dataset shape and the synthetic-only sub-shapes once
        # so per-call accessors don't re-narrow.
        dataset = run.cfg.get_default_dataset()
        self._dataset = dataset
        synthetic = dataset if isinstance(dataset, SyntheticDataset) else None
        self._synthetic_prompts: PromptConfig | None = (
            synthetic.prompts if synthetic is not None else None
        )
        self._synthetic_prefix_prompts: PrefixPromptConfig | None = (
            synthetic.prefix_prompts if synthetic is not None else None
        )
        self._synthetic_images: ImageConfig | None = (
            synthetic.images if synthetic is not None else None
        )
        self._synthetic_audio: AudioConfig | None = (
            synthetic.audio if synthetic is not None else None
        )
        self._synthetic_video: VideoConfig | None = (
            synthetic.video if synthetic is not None else None
        )

        # Create generators (prompt generator requires a tokenizer)
        self.prompt_generator: PromptGenerator | None = (
            PromptGenerator(
                prompts=self._synthetic_prompts,
                prefix_prompts=self._synthetic_prefix_prompts,
                tokenizer=tokenizer,
            )
            if tokenizer
            else None
        )
        self.image_generator = ImageGenerator(self._synthetic_images)
        self.audio_generator = AudioGenerator(self._synthetic_audio)
        self.video_generator = VideoGenerator(self._synthetic_video)

        self._model_selector_rng = rng.derive("composer.turn.model_selection")
        self._max_tokens_rng = rng.derive("composer.turn.max_tokens")

        self.turn_count = 0

        # ``PromptConfig.sequence_distribution`` is a
        # ``list[SequenceDistributionEntry]`` of typed ``SamplingDistribution``
        # objects. Convert each entry directly to a ``SequenceLengthPair``
        # (extracting mean + stddev from the underlying distribution) and
        # build the runtime distribution without re-serializing through
        # ``DistributionParser.parse``, which only accepts strings.
        self._seq_distribution = self._build_sequence_distribution()

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

    def _build_sequence_distribution(self) -> SequenceLengthDistribution | None:
        """Build a runtime sequence-length distribution from config entries.

        ``PromptConfig.sequence_distribution`` is a list of
        ``SequenceDistributionEntry`` carrying typed ``SamplingDistribution``
        ISL/OSL fields (Fixed/Normal/LogNormal/...). Pull the mean and the
        normal-distribution stddev (0 for non-normal types) off each entry to
        construct ``SequenceLengthPair`` directly. ``DistributionParser.parse``
        only accepts strings and would reject this list shape.
        """
        if self._synthetic_prompts is None:
            return None
        entries = self._synthetic_prompts.sequence_distribution
        if not entries:
            return None

        pairs = [
            SequenceLengthPair(
                input_seq_len=int(entry.isl.expected_value),
                output_seq_len=int(entry.osl.expected_value),
                probability=float(entry.probability),
                input_seq_len_stddev=float(getattr(entry.isl, "stddev", 0.0) or 0.0),
                output_seq_len_stddev=float(getattr(entry.osl, "stddev", 0.0) or 0.0),
            )
            for entry in entries
        ]
        return SequenceLengthDistribution(pairs)

    def _osl_distribution(self) -> SamplingDistribution | None:
        """Resolve the OSL distribution to use as a fallback for max_tokens.

        Synthetic datasets carry OSL on ``PromptConfig.osl``; file datasets
        carry it on ``FileDataset.osl`` (routed there from ``--osl`` by the
        CLI converter). Per-line ``output_length`` on a turn always wins
        over either of these.
        """
        if self._synthetic_prompts is not None and self._synthetic_prompts.osl:
            return self._synthetic_prompts.osl
        if isinstance(self._dataset, FileDataset):
            return self._dataset.osl
        return None

    def get_default_context_mode(self) -> ConversationContextMode | None:
        """Dataset-level default context mode inferred by the composer or its loader.

        Override in subclasses that delegate to a loader with format-specific defaults.
        Returns None to fall through to the global DELTAS_WITHOUT_RESPONSES default.
        """
        return None

    # TODO: This can be refactored to be similar to the DatasetSamplingStrategyProtocol in order
    # to allow for more flexible model selection strategies in the future.
    def _select_model_name(self) -> str:
        strategy = self.run.cfg.models.strategy
        model_names = self.run.cfg.get_model_names()
        if strategy == ModelSelectionStrategy.RANDOM:
            return self._model_selector_rng.choice(model_names)
        elif strategy == ModelSelectionStrategy.ROUND_ROBIN:
            model_name = model_names[self.turn_count % len(model_names)]
            self.turn_count += 1
            return model_name
        else:
            raise ValueError(f"Invalid model selection strategy: {strategy}.")

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
            isl_mean = (
                int(self._synthetic_prompts.isl.expected_value)
                if self._synthetic_prompts is not None
                and self._synthetic_prompts.isl is not None
                else 0
            )
            osl_mean = (
                int(self._synthetic_prompts.osl.expected_value)
                if self._synthetic_prompts is not None
                and self._synthetic_prompts.osl is not None
                else None
            )
            seq_lengths = (
                isl_mean,
                osl_mean or max(128, isl_mean // 2),
            )
        else:
            seq_lengths = self._seq_distribution.sample()

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

        If the turn already has max_tokens set (e.g., from per-line input data),
        the existing value is preserved. Per-line values take precedence over
        global --osl and --seq-dist settings.

        Args:
            turn: The turn object to finalize.
        """
        if turn.max_tokens is not None:
            return

        if self._seq_distribution is not None:
            # Use cached sequence distribution to get OSL (ensures ISL/OSL pairing consistency)
            turn_id = id(turn)
            _, osl = self._get_turn_sequence_lengths(turn_id)
            if osl > 0:
                turn.max_tokens = osl
        else:
            osl_dist = self._osl_distribution()
            if osl_dist is not None:
                osl_mean = int(osl_dist.expected_value)
                if osl_mean <= 0:
                    return
                osl_stddev = int(getattr(osl_dist, "stddev", 0.0) or 0.0)
                turn.max_tokens = self._max_tokens_rng.sample_positive_normal_integer(
                    osl_mean, osl_stddev
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
        if turn.model is None:
            turn.model = self._select_model_name()
        self._set_max_tokens(turn)

        # Clear cached sequence lengths for this turn to free memory
        turn_id = id(turn)
        self._clear_turn_cache(turn_id)

    @property
    def prefix_prompt_enabled(self) -> bool:
        prefix_length = (
            self._synthetic_prefix_prompts.length
            if self._synthetic_prefix_prompts is not None
            else None
        )
        return (
            self.prompt_generator is not None
            and prefix_length is not None
            and prefix_length > 0
        )

    def _finalize_conversations(self, conversations: list[Conversation]) -> None:
        """Finalize conversations by adding conversation-level context prompts.

        Injects shared system prompts and per-conversation user context prompts.
        Note: Turn-level finalization (_finalize_turn) is handled by each composer
        according to its needs (eager in synthetic, lazy in custom).

        Args:
            conversations: List of conversations to finalize
        """
        self._inject_context_prompts(conversations)

    def _inject_context_prompts(self, conversations: list[Conversation]) -> None:
        """Inject shared system and user context prompts into conversations.

        Sets the system_message and context_message fields on Conversation objects,
        which endpoint formatters will prepend to the first turn when creating payloads.

        Args:
            conversations: List of conversations to inject prompts into
        """
        if self.prompt_generator is None:
            return

        prefix_prompts = self._synthetic_prefix_prompts
        has_shared_system = (
            prefix_prompts is not None
            and prefix_prompts.shared_system_length is not None
        )
        has_user_context = (
            prefix_prompts is not None
            and prefix_prompts.user_context_length is not None
        )

        if not (has_shared_system or has_user_context):
            return

        self.debug(
            lambda: f"Injecting context prompts into {len(conversations)} conversations"
        )

        # Get shared system prompt once (same for all sessions)
        shared_system_prompt = None
        if has_shared_system:
            shared_system_prompt = self.prompt_generator.get_shared_system_prompt()

        # Iterate through conversations and set conversation-level fields
        for session_index, conversation in enumerate(conversations):
            # Set shared system prompt
            if shared_system_prompt:
                conversation.system_message = shared_system_prompt
                self.trace(
                    lambda conv=conversation: f"Set system_message on conversation {conv.session_id}"
                )

            # Set user context prompt (unique per session)
            if has_user_context:
                user_context = self.prompt_generator.generate_user_context_prompt(
                    session_index
                )
                conversation.user_context_message = user_context
                self.trace(
                    lambda idx=session_index,
                    conv=conversation: f"Set user_context_message for session {idx} "
                    f"(conversation {conv.session_id})"
                )
