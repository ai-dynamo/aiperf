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
from aiperf.config.dataset import FileDataset
from aiperf.dataset.generator.audio import AudioGenerator
from aiperf.dataset.generator.image import ImageGenerator
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.generator.video import VideoGenerator

if TYPE_CHECKING:
    from aiperf.common.random_generator import RandomGenerator
    from aiperf.config import BenchmarkRun
    from aiperf.config.distributions import SamplingDistribution
    from aiperf.config.types import SequenceDistributionEntry


class _SequenceDistributionSampler:
    """Draw (ISL, OSL) pairs from a configured ``sequence_distribution``.

    Each entry carries a ``probability`` weight plus full ``isl``/``osl``
    :class:`SamplingDistribution` objects. ``sample`` picks a bucket by weight,
    then draws ISL and OSL from that bucket's distributions via
    ``sample_int(rng)`` so non-normal shapes (uniform/lognormal/multimodal/
    empirical) are reproduced faithfully instead of being collapsed to their
    means. The RNG is the composer's own ``self._max_tokens_rng`` so a fixed
    seed yields identical samples.

    Example:
        >>> sampler = _SequenceDistributionSampler(entries, max_tokens_rng)
        >>> isl, osl = sampler.sample()  # e.g. (2041, 64) from a bimodal ISL
    """

    def __init__(
        self, entries: list[SequenceDistributionEntry], rng: RandomGenerator
    ) -> None:
        if not entries:
            raise ValueError(
                "sequence_distribution sampler requires at least one entry; got an "
                "empty list. Check that prompts.sequence_distribution is populated."
            )
        self._entries = entries
        self._rng = rng
        total = sum(entry.probability for entry in entries)
        running = 0.0
        cumulative: list[float] = []
        for entry in entries:
            running += entry.probability / total
            cumulative.append(running)
        self._cumulative = cumulative

    def sample(self) -> tuple[int, int]:
        """Sample one (ISL, OSL) pair, preserving each bucket's distribution shape."""
        idx = bisect.bisect_right(self._cumulative, self._rng.random())
        idx = min(idx, len(self._entries) - 1)
        entry = self._entries[idx]
        return (entry.isl.sample_int(self._rng), entry.osl.sample_int(self._rng))


class BaseDatasetComposer(AIPerfLoggerMixin, ABC):
    def __init__(self, run: BenchmarkRun, tokenizer: Tokenizer | None, **kwargs):
        self.run = run
        self.dataset_config = run.cfg.get_default_dataset()
        super().__init__(run=run, tokenizer=tokenizer, **kwargs)

        # Create generators (prompt generator requires a tokenizer)
        self.prompt_generator: PromptGenerator | None = (
            PromptGenerator(run=run, tokenizer=tokenizer) if tokenizer else None
        )
        self.image_generator = ImageGenerator(run=run)
        self.audio_generator = AudioGenerator(run=run)
        self.video_generator = VideoGenerator(run=run)

        self._model_selector_rng = rng.derive("composer.turn.model_selection")
        self._max_tokens_rng = rng.derive("composer.turn.max_tokens")

        self.turn_count = 0

        # Initialize sequence distribution from prompts config if available.
        # Carry the full isl/osl SamplingDistribution objects so non-normal
        # shapes are sampled faithfully rather than reduced to (mean, stddev).
        self._seq_distribution: _SequenceDistributionSampler | None = None
        prompts_config = getattr(self.dataset_config, "prompts", None)
        if prompts_config is not None and prompts_config.sequence_distribution:
            self._seq_distribution = _SequenceDistributionSampler(
                list(prompts_config.sequence_distribution), self._max_tokens_rng
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

    def get_default_context_mode(self) -> ConversationContextMode | None:
        """Dataset-level default context mode inferred by the composer or its loader.

        Override in subclasses that delegate to a loader with format-specific defaults.
        Returns None to fall through to the global DELTAS_WITHOUT_RESPONSES default.
        """
        return None

    # TODO: This can be refactored to be similar to the DatasetSamplingStrategyProtocol in order
    # to allow for more flexible model selection strategies in the future.
    def _select_model_name(self) -> str:
        model_names = self.run.cfg.get_model_names()
        model_selection_strategy = self.run.cfg.models.strategy

        if model_selection_strategy == ModelSelectionStrategy.RANDOM:
            return self._model_selector_rng.choice(model_names)
        elif model_selection_strategy == ModelSelectionStrategy.ROUND_ROBIN:
            model_name = model_names[self.turn_count % len(model_names)]
            self.turn_count += 1
            return model_name
        else:
            raise ValueError(
                f"Invalid model selection strategy: {model_selection_strategy}."
            )

    def _get_turn_sequence_lengths(self, turn_id: int) -> tuple[int, int | None]:
        """Get or sample ISL/OSL pair for a specific turn, ensuring consistency.

        This method caches the sequence lengths per turn to ensure that the same
        ISL/OSL pair is used for both prompt generation and max_tokens setting.

        Args:
            turn_id: Unique identifier for the turn

        Returns:
            Tuple of (input_seq_len, output_seq_len). output_seq_len may be None
            if OSL is not configured.
        """
        if turn_id in self._turn_sequence_cache:
            return self._turn_sequence_cache[turn_id]

        if self._seq_distribution is None:
            prompts_config = getattr(self.dataset_config, "prompts", None)
            if prompts_config and prompts_config.isl:
                isl_val = prompts_config.isl.sample_int(self._max_tokens_rng)
            else:
                isl_val = 128  # Default

            if prompts_config and prompts_config.osl:
                osl_val = prompts_config.osl.sample_int(self._max_tokens_rng)
            else:
                osl_val = None

            seq_lengths = (isl_val, osl_val)
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

        # Use cached sequence distribution to get OSL (ensures ISL/OSL pairing consistency)
        turn_id = id(turn)
        _, osl = self._get_turn_sequence_lengths(turn_id)
        if osl is not None:
            turn.max_tokens = int(osl)
            return

        osl_dist = self._file_osl_distribution()
        if osl_dist is None:
            return
        # Sample directly from the configured OSL distribution (same object the
        # primary path uses) so an empirical/lognormal/multimodal fallback keeps
        # its shape instead of collapsing to a normal around expected_value.
        turn.max_tokens = self._max_tokens_rng and osl_dist.sample_int(
            self._max_tokens_rng
        )

    def _file_osl_distribution(self) -> SamplingDistribution | None:
        """OSL fallback distribution for file datasets.

        File datasets carry OSL on ``FileDataset.osl`` (routed there from
        ``--osl`` by the CLI converter) as a per-record fallback. Per-line
        ``output_length`` values always win — ``_set_max_tokens`` returns
        before consulting this when the turn already has ``max_tokens``.
        """
        if isinstance(self.dataset_config, FileDataset):
            return self.dataset_config.osl
        return None

    def _finalize_turn(self, turn: Turn) -> None:
        """Finalize a turn by populating all required metadata fields.

        This method handles:
        - Model name selection (skipped when the turn carries a per-turn
          ``model`` override, e.g. from a dag_jsonl record)
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
        if self.prompt_generator is None:
            return False
        prefix_config = getattr(self.dataset_config, "prefix_prompts", None)
        if prefix_config is None:
            return False
        return (prefix_config.length or 0) > 0

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

        prefix_config = getattr(self.dataset_config, "prefix_prompts", None)
        if prefix_config is None:
            return

        has_shared_system = prefix_config.shared_system_length is not None
        has_user_context = prefix_config.user_context_length is not None

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
