# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import bisect
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aiperf.common import random_generator as rng
from aiperf.common.enums import ConversationContextMode, ModelSelectionStrategy
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset import FileDataset, SyntheticDataset
from aiperf.dataset.generator.audio import AudioGenerator
from aiperf.dataset.generator.image import ImageGenerator
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.generator.video import VideoGenerator

if TYPE_CHECKING:
    from aiperf.common.random_generator import RandomGenerator
    from aiperf.config.dataset import VideoConfig
    from aiperf.config.dataset.content import (
        AudioConfig,
        ImageConfig,
        PrefixPromptConfig,
        PromptConfig,
    )
    from aiperf.config.distributions import SamplingDistribution
    from aiperf.config.resolution.plan import BenchmarkRun
    from aiperf.config.types import SequenceDistributionEntry


class _TypedSequenceDistribution:
    """Weighted ISL/OSL buckets whose per-bucket typed distributions sample
    their full shape (the legacy SequenceLengthDistribution only supported
    fixed-or-normal buckets).

    Weights are relative and normalized by their total here; config-level
    validation only guarantees a positive total (so this division is safe).
    """

    def __init__(
        self, entries: list[SequenceDistributionEntry], rng_instance: RandomGenerator
    ) -> None:
        # Zero-weight buckets are valid config (disabled buckets in sweep
        # templates); drop them so they never sample and never widen a
        # cumulative interval. Config-level validation normally guarantees a
        # positive-weight entry survives; the guard below is defensive.
        self._entries = [e for e in entries if e.probability > 0]
        if not self._entries:
            raise ValueError(
                "sequence_distribution requires at least one positive-weight entry"
            )
        self._rng = rng_instance
        total = sum(e.probability for e in self._entries)
        cumulative = 0.0
        self._cumulative: list[float] = []
        for entry in self._entries:
            cumulative += entry.probability / total
            self._cumulative.append(cumulative)

    def sample_bucket(self) -> SequenceDistributionEntry:
        """Draw one weighted bucket. Called once per conversation: the bucket
        is the conversation's workload class and stays fixed for its life."""
        r = self._rng.random()
        idx = bisect.bisect_right(self._cumulative, r)
        return self._entries[min(idx, len(self._entries) - 1)]

    def sample_lengths(
        self, bucket: SequenceDistributionEntry, *, is_first: bool = False
    ) -> tuple[int, int]:
        """Draw one (ISL, OSL) pair from the held bucket. The first turn's ISL
        comes from the bucket's first_turn_isl (seed context) when set."""
        isl_dist = (
            bucket.first_turn_isl
            if is_first and bucket.first_turn_isl is not None
            else bucket.isl
        )
        return (
            isl_dist.sample_int(self._rng),
            bucket.osl.sample_int(self._rng),
        )


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
        self._seq_len_rng = rng.derive("composer.turn.sequence_lengths")

        self.turn_count = 0

        # ``PromptConfig.sequence_distribution`` is a
        # ``list[SequenceDistributionEntry]`` of typed ``SamplingDistribution``
        # objects. Each conversation draws ONE bucket at creation and keeps it
        # for every turn (sticky per-conversation workload class); the runtime
        # sampler draws that bucket's ISL/OSL from their full distribution shape
        # (lognormal/multimodal/empirical/percentile), not a flattened
        # mean+stddev.
        self._seq_distribution: _TypedSequenceDistribution | None = (
            self._build_sequence_distribution()
        )

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

    def _build_sequence_distribution(self) -> _TypedSequenceDistribution | None:
        """Build the runtime sequence-length sampler from config entries.

        ``PromptConfig.sequence_distribution`` is a list of
        ``SequenceDistributionEntry`` carrying typed ``SamplingDistribution``
        ISL/OSL fields. Each bucket samples its full distribution shape —
        lognormal/multimodal/empirical/percentile buckets are NOT flattened
        to mean+stddev.
        """
        if self._synthetic_prompts is None:
            return None
        entries = self._synthetic_prompts.sequence_distribution
        if not entries:
            return None
        return _TypedSequenceDistribution(
            entries, rng.derive("composer.sequence.distribution")
        )

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

    def _get_turn_sequence_lengths(
        self,
        turn_id: int,
        *,
        is_first: bool = False,
        bucket: SequenceDistributionEntry | None = None,
    ) -> tuple[int, int]:
        """Sample (or return the cached) ISL/OSL pair for a specific turn.

        Both lengths are drawn from their full typed distributions
        (Fixed/Normal/LogNormal/Multimodal/Empirical/Percentile) exactly once
        per turn; the cache guarantees prompt generation and max_tokens see
        the same pair.

        Args:
            turn_id: Unique identifier for the turn
            is_first: When True the ISL is drawn from the first-turn starting-
                context distribution instead of ``isl`` (``prompts.first_turn_isl``
                on the plain path, ``bucket.first_turn_isl`` on the
                sequence_distribution path).
            bucket: The conversation's sticky sequence_distribution bucket.
                Required on the sequence_distribution path when the cache is
                cold; None falls back to a fresh bucket draw (non-synthetic
                composers that never threaded one, e.g. rankings max_tokens).

        Returns:
            Tuple of (input_seq_len, output_seq_len)
        """
        if turn_id in self._turn_sequence_cache:
            return self._turn_sequence_cache[turn_id]

        if self._seq_distribution is None:
            prompts = self._synthetic_prompts
            isl_dist = None
            if prompts is not None:
                isl_dist = (
                    prompts.first_turn_isl
                    if is_first and prompts.first_turn_isl is not None
                    else prompts.isl
                )
            isl = (
                isl_dist.sample_int(self._seq_len_rng)
                if isl_dist is not None and isl_dist.expected_value > 0
                else 0
            )
            osl_dist = self._osl_distribution()
            osl = (
                osl_dist.sample_int(self._seq_len_rng)
                if osl_dist is not None and osl_dist.expected_value > 0
                else None
            )
            seq_lengths = (isl, osl if osl is not None else max(128, isl // 2))
        else:
            if bucket is None:
                bucket = self._seq_distribution.sample_bucket()
            seq_lengths = self._seq_distribution.sample_lengths(
                bucket, is_first=is_first
            )

        self._turn_sequence_cache[turn_id] = seq_lengths
        return seq_lengths

    def _clear_turn_cache(self, turn_id: int) -> None:
        """Clear cached sequence lengths for a specific turn.

        Args:
            turn_id: Turn identifier to remove from cache
        """
        self._turn_sequence_cache.pop(turn_id, None)

    def _set_max_tokens(
        self, turn: Turn, bucket: SequenceDistributionEntry | None = None
    ) -> None:
        """Set max_tokens for the turn based on the sequence distribution or output configuration.

        If the turn already has max_tokens set (e.g., from per-line input data),
        the existing value is preserved. Per-line values take precedence over
        global --osl and --seq-dist settings.

        Args:
            turn: The turn object to finalize.
            bucket: The conversation's sticky sequence_distribution bucket, so the
                cached OSL comes from the same class the turn's ISL was drawn from.
        """
        if turn.max_tokens is not None:
            return

        if self._seq_distribution is not None:
            # Use cached sequence distribution to get OSL (ensures ISL/OSL pairing consistency)
            turn_id = id(turn)
            _, osl = self._get_turn_sequence_lengths(turn_id, bucket=bucket)
            if osl > 0:
                turn.max_tokens = osl
        else:
            osl_dist = self._osl_distribution()
            if osl_dist is not None and osl_dist.expected_value > 0:
                _, osl = self._get_turn_sequence_lengths(id(turn))
                turn.max_tokens = osl

    def _finalize_turn(
        self, turn: Turn, bucket: SequenceDistributionEntry | None = None
    ) -> None:
        """Finalize a turn by populating all required metadata fields.

        This method handles:
        - Model name selection
        - Max tokens sampling based on output configuration
        - Any other turn-level metadata that needs to be set

        Args:
            turn: The turn object to finalize.
            bucket: The conversation's sticky sequence_distribution bucket,
                forwarded so max_tokens is drawn from the same class.
        """
        if turn.model is None:
            turn.model = self._select_model_name()
        self._set_max_tokens(turn, bucket)

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

    @staticmethod
    def _loader_accepts_kwarg(loader_class: type, name: str) -> bool:
        """Return True when ``name`` is an explicitly declared parameter of
        ``loader_class.__init__`` (or any class in its MRO before ``BaseMixin``).

        ``**kwargs`` does not count as acceptance — the silent-swallow chain
        through ``BaseMixin.__init__`` is exactly what this check exists to
        catch.
        """
        for klass in loader_class.__mro__:
            if klass.__name__ == "BaseMixin":
                break
            try:
                params = inspect.signature(klass.__init__).parameters
            except (TypeError, ValueError):
                continue
            param = params.get(name)
            if param is not None and param.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return True
        return False
