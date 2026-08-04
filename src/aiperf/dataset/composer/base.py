# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from aiperf.common import random_generator as rng
from aiperf.common.enums import ConversationContextMode, ModelSelectionStrategy
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Turn
from aiperf.common.models.sequence_distribution import SequenceLengthSampler
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset import SyntheticDataset
from aiperf.dataset.generator.audio import AudioGenerator
from aiperf.dataset.generator.corpus import resolve_prompt_generator
from aiperf.dataset.generator.image import ImageGenerator
from aiperf.dataset.generator.video import VideoGenerator
from aiperf.dataset.protocols import CorpusGeneratorProtocol

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


_CHAT_TEMPLATE_PROBE_SAMPLES: tuple[str, ...] = (
    "Hello, how are you today?",
    "Could you write a Python function to reverse a string?",
    "What's the difference between TCP and UDP in networking?",
)


def _estimate_chat_template_overheads(
    tokenizer: Tokenizer | None,
) -> tuple[int, int]:
    """Decompose chat-template overhead into (per_request_fixed, per_msg_wrap).

    The chat template renders the entire ``messages`` array in one pass at
    request time. Total wrapping is::

        wire_tokens =  per_request_fixed
                     + Σ_{m in messages} (per_msg_wrap + content_tokens(m))

    where ``per_request_fixed`` ≈ generation-prompt suffix (BOS is excluded
    because ``RangeRatioDistribution`` already subtracts it via
    ``num_special_tokens``) and ``per_msg_wrap`` ≈ role header + end-of-turn
    marker (averaged over user/assistant). Measuring the two separately lets
    callers apply the fixed cost only to the first user turn and the
    per-message wrap to every turn.

    Returns ``(0, 0)`` when the tokenizer is ``None``/has no underlying HF
    tokenizer, has no ``apply_chat_template`` (e.g. tiktoken), the model has
    no chat template configured, or a probe yields an implausible (negative)
    result (defensive: skipping compensation beats over-correcting).
    """
    if tokenizer is None:
        return 0, 0
    inner = getattr(tokenizer, "_tokenizer", None)
    apply = getattr(inner, "apply_chat_template", None)
    if apply is None:
        return 0, 0

    bos_tokens = tokenizer.num_prompt_special_tokens()
    fixed_costs: list[float] = []
    wrap_costs: list[float] = []
    for sample in _CHAT_TEMPLATE_PROBE_SAMPLES:
        try:
            single_text = apply(
                [{"role": "user", "content": sample}],
                tokenize=False,
                add_generation_prompt=True,
            )
            triple_text = apply(
                [
                    {"role": "user", "content": sample},
                    {"role": "assistant", "content": sample},
                    {"role": "user", "content": sample},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return 0, 0
        if not isinstance(single_text, str) or not isinstance(triple_text, str):
            return 0, 0
        single_len = len(tokenizer.encode(single_text))
        triple_len = len(tokenizer.encode(triple_text))
        bare_len = len(tokenizer.encode(sample))
        avg_wrap = (triple_len - single_len - 2 * bare_len) / 2
        per_request_fixed = single_len - bare_len - avg_wrap - bos_tokens
        if avg_wrap < 0 or per_request_fixed < 0:
            return 0, 0
        wrap_costs.append(avg_wrap)
        fixed_costs.append(per_request_fixed)

    return (
        round(sum(fixed_costs) / len(fixed_costs)),
        round(sum(wrap_costs) / len(wrap_costs)),
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

        # ISL budget compensation — keeps synthetic wire ISL matching the
        # user's ``--isl`` / ``--shared-system-prompt-length`` once cache-bust
        # markers and chat-template wrapping are added at request time. All
        # components resolve to 0 for non-synthetic datasets (weka et al. carry
        # no ``prompts``/``prefix_prompts``), so this is byte-neutral for trace
        # replay and only affects the SyntheticComposer's generated text.
        compensated_prefix_prompts = self._init_isl_compensation(tokenizer)

        # Create generators (prompt generator requires a tokenizer)
        self.prompt_generator: CorpusGeneratorProtocol | None = (
            resolve_prompt_generator(
                corpus=self.run.cfg.get_prompt_corpus(),
                default_corpus=None,
                tokenizer=tokenizer,
                prompts=self._synthetic_prompts,
                prefix_prompts=compensated_prefix_prompts,
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

        num_special_tokens = (
            tokenizer.num_prompt_special_tokens() if tokenizer is not None else 0
        )
        self._seq_distribution: SequenceLengthSampler | None = (
            self._build_sequence_distribution(num_special_tokens)
        )

        # Cache for turn-level sequence lengths to ensure ISL/OSL pairing consistency
        self._turn_sequence_cache: dict[int, tuple[int, int]] = {}

    def _init_isl_compensation(
        self, tokenizer: Tokenizer | None
    ) -> PrefixPromptConfig | None:
        """Compute synthetic ISL compensation budgets and return prefix prompts.

        Decomposes three components subtracted at request-build time so the
        wire ISL matches the user's ``--isl`` / ``--shared-system-prompt-length``:

        (a) cache-bust marker token cost (0 for NONE);
        (b) chat-template wrapping (per-request fixed + per-message wrap), only
            active under ``--apply-chat-template`` and a real chat template;
        (c) shared-system-prompt shrink when a SYSTEM_* marker lands on the
            synthetic system prompt.

        Returns a (possibly ``model_copy``-compensated) ``PrefixPromptConfig``
        for the PromptGenerator. Non-synthetic datasets have no synthetic
        prompts, so every budget is 0 and the original prefix prompts pass
        through unchanged.
        """
        from aiperf.common.enums import CacheBustTarget
        from aiperf.timing.strategies.cache_bust import estimate_marker_token_cost

        prompts = self._synthetic_prompts
        prefix_prompts = self._synthetic_prefix_prompts

        cache_bust = getattr(prompts, "cache_bust", None) if prompts else None
        cache_bust_target = (
            cache_bust.target if cache_bust is not None else CacheBustTarget.NONE
        )
        configured_shared_sys_len = (
            getattr(prefix_prompts, "shared_system_length", None)
            if prefix_prompts is not None
            else None
        )
        has_synthetic_system_prompt = configured_shared_sys_len is not None
        is_system_target = cache_bust_target in (
            CacheBustTarget.SYSTEM_PREFIX,
            CacheBustTarget.SYSTEM_SUFFIX,
        )

        # Component (a): cache-bust marker token cost.
        self._cache_bust_marker_tokens = (
            estimate_marker_token_cost(cache_bust_target, tokenizer)
            if cache_bust_target != CacheBustTarget.NONE and tokenizer is not None
            else 0
        )

        # Component (a) routing: mirrors ``worker._apply_cache_bust``'s fallback:
        # SYSTEM_* + system prompt -> marker on system msg; otherwise (incl.
        # SYSTEM_* with no system message, and FIRST_TURN_*) -> first user turn.
        marker_on_shared_system_prompt = (
            is_system_target and has_synthetic_system_prompt
        )
        marker_on_first_user_turn = (
            cache_bust_target != CacheBustTarget.NONE
            and not marker_on_shared_system_prompt
        )
        self._first_turn_cache_bust_marker_tokens = (
            self._cache_bust_marker_tokens if marker_on_first_user_turn else 0
        )

        # Component (b): chat-template wrapping. Both 0 when no chat template,
        # and both 0 when ``--apply-chat-template`` is not set (opt-out).
        if (
            self.run.cfg.tokenizer is not None
            and self.run.cfg.tokenizer.apply_chat_template
        ):
            (
                self._chat_template_per_request_fixed_tokens,
                self._chat_template_per_msg_wrap_tokens,
            ) = _estimate_chat_template_overheads(tokenizer)
        else:
            self._chat_template_per_request_fixed_tokens = 0
            self._chat_template_per_msg_wrap_tokens = 0

        # Component (c): shrink the synthetic shared system prompt by the marker
        # cost so its wire length still matches ``--shared-system-prompt-length``.
        # Compensate via a model_copy passed to PromptGenerator — never mutate
        # the user-facing config in place.
        if marker_on_shared_system_prompt and configured_shared_sys_len is not None:
            compensated_shared_sys_len = max(
                1, configured_shared_sys_len - self._cache_bust_marker_tokens
            )
            self._prefix_prompt_tokens = 0
            return prefix_prompts.model_copy(
                update={"shared_system_length": compensated_shared_sys_len}
            )

        # Component (d): prefix prompt tokens consumed from the first-turn ISL
        # budget. The prefix is prepended to the body at request time, so the body
        # must be shorter by the prefix length to keep total wire ISL on target.
        # Only active when pool_size > 0 — without a pool there are no prefixes
        # to prepend.
        self._prefix_prompt_tokens = (
            prefix_prompts.length
            if prefix_prompts is not None
            and prefix_prompts.length is not None
            and prefix_prompts.pool_size is not None
            and prefix_prompts.pool_size > 0
            else 0
        )

        return prefix_prompts

    @property
    def first_turn_isl_adjustment(self) -> int:
        """Total tokens to subtract from the FIRST user turn's synthetic ISL.

        Composed of the per-request chat-template fixed cost (gen-prompt
        suffix), the per-message chat-template wrap (role header + EOT), the
        cache-bust marker when it lands on the first user turn, and the prefix
        prompt length so that prefix + body stays within the configured ISL.
        """
        return (
            self._chat_template_per_request_fixed_tokens
            + self._chat_template_per_msg_wrap_tokens
            + self._first_turn_cache_bust_marker_tokens
            + self._prefix_prompt_tokens
        )

    @property
    def subsequent_turn_isl_adjustment(self) -> int:
        """Tokens to subtract from each non-first user turn's synthetic ISL.

        Just the per-message chat-template wrap; the per-request fixed cost and
        the cache-bust marker apply only to the first turn (later turns'
        raw_messages are not mutated; BOS / gen-prompt suffix emit once per
        request, not per message).
        """
        return self._chat_template_per_msg_wrap_tokens

    @abstractmethod
    def create_dataset(self) -> list[Conversation]:
        """
        Create a set of conversation objects from the given configuration.

        Returns:
            list[Conversation]: A list of conversation objects.
        """
        ...

    def _build_sequence_distribution(
        self, num_special_tokens: int = 0
    ) -> SequenceLengthSampler | None:
        """Build a runtime sequence-length sampler from config.

        Delegates to ``PromptConfig.get_sequence_distribution()`` which handles
        both ``sequence_distribution`` (probabilistic ISL/OSL pairs) and
        ``random_range_ratio`` (uniform window sampling).

        For ``RandomCorpusStyle.SGLANG`` with ``--apply-chat-template``, passes
        the measured template overhead so ``SGLangRangeRatioDistribution`` can
        subtract it from ``isl_mean`` before computing bounds — matching
        SGLang's ``use_chat_template`` bound adjustment.
        """
        if self._synthetic_prompts is None:
            return None
        from aiperf.common.enums import RandomCorpusStyle

        chat_template_len = 0
        if (
            self._synthetic_prompts.random_corpus_style == RandomCorpusStyle.SGLANG
            and self.run.cfg.tokenizer is not None
            and self.run.cfg.tokenizer.apply_chat_template
        ):
            chat_template_len = (
                self._chat_template_per_request_fixed_tokens
                + self._chat_template_per_msg_wrap_tokens
            )
        return self._synthetic_prompts.get_sequence_distribution(
            num_special_tokens=num_special_tokens,
            chat_template_len=chat_template_len,
        )

    def _osl_distribution(self) -> SamplingDistribution | None:
        """Resolve the OSL distribution to use as a fallback for max_tokens.

        Synthetic datasets carry OSL on ``PromptConfig.osl``; file AND public
        (HF-backed weka) trace datasets carry it on the flat ``osl`` field
        (routed there from ``--osl`` by the CLI converter). Per-line
        ``output_length`` on a turn always wins over either of these.
        """
        if self._synthetic_prompts is not None and self._synthetic_prompts.osl:
            return self._synthetic_prompts.osl
        return getattr(self._dataset, "osl", None)

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

        ``max_tokens`` is clamped to a minimum of 1: the OpenAI-compatible
        chat-completions API rejects ``max_completion_tokens=0`` outright on
        most servers (and silently produces empty completions on others),
        which surfaces as opaque request failures during a benchmark.

        Args:
            turn: The turn object to finalize.
        """
        if turn.max_tokens is not None:
            if turn.max_tokens <= 0:
                self.warning(
                    f"max_tokens={turn.max_tokens} on turn is invalid (must be > 0); "
                    "clamping to 1"
                )
                turn.max_tokens = 1
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

        if turn.max_tokens is not None and turn.max_tokens <= 0:
            self.warning(
                f"Sampled max_tokens={turn.max_tokens} is invalid (must be > 0); "
                "clamping to 1"
            )
            turn.max_tokens = 1

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
