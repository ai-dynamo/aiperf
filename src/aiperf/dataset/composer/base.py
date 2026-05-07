# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod

from aiperf.common import random_generator as rng
from aiperf.common.config import UserConfig
from aiperf.common.enums import (
    CacheBustTarget,
    ConversationContextMode,
    ModelSelectionStrategy,
)
from aiperf.common.mixins import AIPerfLoggerMixin
from aiperf.common.models import Conversation, Turn
from aiperf.common.tokenizer import Tokenizer
from aiperf.dataset.generator.audio import AudioGenerator
from aiperf.dataset.generator.image import ImageGenerator
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.generator.video import VideoGenerator
from aiperf.timing.strategies.cache_bust import estimate_marker_token_cost

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

    where:
      - ``per_request_fixed`` ≈ BOS + generation-prompt suffix
      - ``per_msg_wrap`` ≈ role-header + end-of-turn marker (averaged over
        user/assistant; templates with materially different per-role wraps
        would need a richer probe).

    We measure the two quantities separately so callers can apply the
    fixed cost only to the first user turn (where it actually lands) and
    the per-message wrap to every turn.

    Probe construction
    ------------------
    For each sample S, we render two templated prompts and tokenize each
    with the bare encoder for the content::

        single = template([user(S)]                    , add_gen_prompt=True)
              ≈ per_request_fixed + 1·per_msg_wrap + bare(S)

        triple = template([user(S), asst(S), user(S)], add_gen_prompt=True)
              ≈ per_request_fixed + 3·per_msg_wrap + 3·bare(S)
                                  + asst_wrap_correction      [≈ 0 if symmetric]

    Solving::

        avg_wrap            ≈ (triple - single - 2·bare(S)) / 2
        per_request_fixed   ≈ single - bare(S) - avg_wrap

    The ``[user, assistant, user]`` shape is chosen because every chat
    template we care about (Llama-3, Qwen, Mistral, DeepSeek, GPT family)
    accepts that pattern. Pure same-role probes get rejected by some
    templates that enforce alternation; the first message must commonly
    be ``user``.

    Returns ``(0, 0)`` when:
      - tokenizer is ``None`` or has no underlying HF tokenizer.
      - underlying tokenizer has no ``apply_chat_template`` (e.g. tiktoken).
      - the model has no chat template configured (``apply_chat_template``
        raises) — un-templated requests have no wrapping to compensate.
      - the probe produces a negative or implausible result for any
        sample (defensive: better to skip compensation than over-correct).
    """
    if tokenizer is None:
        return 0, 0
    inner = getattr(tokenizer, "_tokenizer", None)
    apply = getattr(inner, "apply_chat_template", None)
    if apply is None:
        return 0, 0

    fixed_costs: list[float] = []
    wrap_costs: list[float] = []
    for sample in _CHAT_TEMPLATE_PROBE_SAMPLES:
        try:
            single = apply(
                [{"role": "user", "content": sample}],
                tokenize=True,
                add_generation_prompt=True,
            )
            triple = apply(
                [
                    {"role": "user", "content": sample},
                    {"role": "assistant", "content": sample},
                    {"role": "user", "content": sample},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception:
            return 0, 0
        bare_len = len(tokenizer.encode(sample))
        avg_wrap = (len(triple) - len(single) - 2 * bare_len) / 2
        per_request_fixed = len(single) - bare_len - avg_wrap
        if avg_wrap < 0 or per_request_fixed < 0:
            return 0, 0
        wrap_costs.append(avg_wrap)
        fixed_costs.append(per_request_fixed)

    return (
        round(sum(fixed_costs) / len(fixed_costs)),
        round(sum(wrap_costs) / len(wrap_costs)),
    )


class BaseDatasetComposer(AIPerfLoggerMixin, ABC):
    def __init__(self, config: UserConfig, tokenizer: Tokenizer | None, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        super().__init__(config=config, tokenizer=tokenizer, **kwargs)

        # ISL budget compensation budget — see
        # ``docs/reference/isl-budget-compensation.md`` for the full model.
        # Three components, each subtracted at a specific point in the
        # synthetic-prompt pipeline so that wire ISL matches the user's
        # ``--isl`` (and ``--shared-system-prompt-length``) values.
        cache_bust_target = config.input.prompt.cache_bust.target
        configured_shared_sys_len = (
            config.input.prompt.prefix_prompt.shared_system_prompt_length
        )
        has_synthetic_system_prompt = configured_shared_sys_len is not None
        is_system_target = cache_bust_target in (
            CacheBustTarget.SYSTEM_PREFIX,
            CacheBustTarget.SYSTEM_SUFFIX,
        )

        # Component (a): cache-bust marker token cost. Always 0 for NONE;
        # otherwise the deterministic average from
        # ``estimate_marker_token_cost`` over a handful of distinct markers.
        self._cache_bust_marker_tokens = (
            estimate_marker_token_cost(cache_bust_target, tokenizer)
            if cache_bust_target != CacheBustTarget.NONE and tokenizer is not None
            else 0
        )

        # Component (a) routing: where does the marker land at request time?
        # Mirrors ``worker._apply_cache_bust``'s fallback rule:
        #   - SYSTEM_* + shared system prompt configured -> marker on system msg
        #   - SYSTEM_* + no system message              -> falls back to first user turn
        #   - FIRST_TURN_*                              -> first user turn
        #   - NONE                                      -> nowhere
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

        # Component (b): chat-template wrapping. Decomposed into per-request
        # fixed (BOS + generation prompt) and per-message wrap (role header
        # + EOT). Both 0 when the tokenizer has no chat template, AND both
        # 0 when ``--apply-chat-template`` is not set: the user has opted
        # out of chat-template-aware ISL accounting, so synthetic prompts
        # pass through at their bare-text token count.
        if config.tokenizer.apply_chat_template:
            (
                self._chat_template_per_request_fixed_tokens,
                self._chat_template_per_msg_wrap_tokens,
            ) = _estimate_chat_template_overheads(tokenizer)
        else:
            self._chat_template_per_request_fixed_tokens = 0
            self._chat_template_per_msg_wrap_tokens = 0

        # Component (c): shared system prompt compensation for SYSTEM_*.
        # When the marker lands on the system prompt, reduce the synthetic
        # system prompt length by the marker cost so wire system message
        # length still matches the user's ``--shared-system-prompt-length``.
        # We do this by passing a ``model_copy``-d prompt config to
        # PromptGenerator (which generates the system prompt eagerly during
        # __init__) — never mutating the user-facing config in place.
        prompt_config = config.input.prompt
        if marker_on_shared_system_prompt and configured_shared_sys_len is not None:
            compensated_shared_sys_len = max(
                1, configured_shared_sys_len - self._cache_bust_marker_tokens
            )
            compensated_prefix = prompt_config.prefix_prompt.model_copy(
                update={"shared_system_prompt_length": compensated_shared_sys_len}
            )
            prompt_config = prompt_config.model_copy(
                update={"prefix_prompt": compensated_prefix}
            )

        # Create generators (prompt generator requires a tokenizer)
        self.prompt_generator: PromptGenerator | None = (
            PromptGenerator(prompt_config, tokenizer) if tokenizer else None
        )
        self.image_generator = ImageGenerator(config.input.image)
        self.audio_generator = AudioGenerator(config.input.audio)
        self.video_generator = VideoGenerator(config.input.video)

        self._model_selector_rng = rng.derive("composer.turn.model_selection")
        self._max_tokens_rng = rng.derive("composer.turn.max_tokens")

        self.turn_count = 0

        # Initialize sequence distribution
        self._seq_distribution = config.input.prompt.get_sequence_distribution()

        # Cache for turn-level sequence lengths to ensure ISL/OSL pairing consistency
        self._turn_sequence_cache: dict[int, tuple[int, int]] = {}

    @property
    def first_turn_isl_adjustment(self) -> int:
        """Total tokens to subtract from the FIRST user turn's synthetic ISL.

        Composed of:
          - per-request chat-template fixed cost (BOS + gen-prompt suffix)
          - per-message chat-template wrap (role header + EOT)
          - cache-bust marker (when it lands on the first user turn)
        """
        return (
            self._chat_template_per_request_fixed_tokens
            + self._chat_template_per_msg_wrap_tokens
            + self._first_turn_cache_bust_marker_tokens
        )

    @property
    def subsequent_turn_isl_adjustment(self) -> int:
        """Tokens to subtract from each non-first user turn's synthetic ISL.

        Just the per-message chat-template wrap; per-request fixed cost
        and the cache-bust marker only apply to the first turn (the marker
        because later turns' raw_messages are not mutated; the fixed cost
        because BOS / generation-prompt suffix are emitted once per request,
        not per message).
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

    def get_default_context_mode(self) -> ConversationContextMode | None:
        """Dataset-level default context mode inferred by the composer or its loader.

        Override in subclasses that delegate to a loader with format-specific defaults.
        Returns None to fall through to the global DELTAS_WITHOUT_RESPONSES default.
        """
        return None

    # TODO: This can be refactored to be similar to the DatasetSamplingStrategyProtocol in order
    # to allow for more flexible model selection strategies in the future.
    def _select_model_name(self) -> str:
        if (
            self.config.endpoint.model_selection_strategy
            == ModelSelectionStrategy.RANDOM
        ):
            return self._model_selector_rng.choice(self.config.endpoint.model_names)
        elif (
            self.config.endpoint.model_selection_strategy
            == ModelSelectionStrategy.ROUND_ROBIN
        ):
            model_name = self.config.endpoint.model_names[
                self.turn_count % len(self.config.endpoint.model_names)
            ]
            self.turn_count += 1
            return model_name
        else:
            raise ValueError(
                f"Invalid model selection strategy: {self.config.endpoint.model_selection_strategy}."
            )

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
            seq_lengths = (
                self.config.input.prompt.input_tokens.mean,
                self.config.input.prompt.output_tokens.mean
                or max(128, self.config.input.prompt.input_tokens.mean // 2),
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
            turn.max_tokens = osl
        else:
            output_tokens_config = self.config.input.prompt.output_tokens
            if output_tokens_config.mean is not None:
                stddev = output_tokens_config.stddev
                turn.max_tokens = self._max_tokens_rng.sample_positive_normal_integer(
                    output_tokens_config.mean, stddev
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
        - Model name selection (only when the turn doesn't already carry an
          explicit per-turn model override from the loader — e.g., ``dag_jsonl``
          and ``mooncake_trace`` both support per-turn ``model`` fields that
          must win over the CLI-level ``--model`` default).
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
        return (
            self.prompt_generator is not None
            and self.config.input.prompt.prefix_prompt.length > 0
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

        config = self.config.input.prompt.prefix_prompt
        has_shared_system = config.shared_system_prompt_length is not None
        has_user_context = config.user_context_prompt_length is not None

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
