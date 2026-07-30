# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ISL budget compensation tests covering cache-bust marker cost, chat-template wrapping (per-request fixed + per-message wrap), and shared-system-prompt regeneration (see docs/reference/isl-budget-compensation.md)."""

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.common.models import Turn
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer
from tests.unit.conftest import make_run_from_cli

# v2 imports ``estimate_marker_token_cost`` lazily inside
# ``BaseDatasetComposer._init_isl_compensation`` (rather than at module scope),
# so it must be patched at its definition site, not on the composer base module.
_MARKER_COST_PATCH = "aiperf.timing.strategies.cache_bust.estimate_marker_token_cost"


def _make_config(
    *,
    cache_bust_target: CacheBustTarget = CacheBustTarget.NONE,
    shared_system_prompt_length: int | None = None,
    prefix_pool_size: int | None = None,
    prefix_length: int | None = None,
    isl_mean: int = 100,
    apply_chat_template: bool = True,
):
    """Build a real v2 BenchmarkRun (via the resolver) for budget tests, defaulting apply_chat_template=True since this module exercises chat-template-aware ISL accounting."""
    overrides: dict = {
        "model_names": ["test-model"],
        "conversation_num_dataset_entries": 1,
        "prompt_input_tokens_mean": isl_mean,
        "prompt_input_tokens_stddev": 0,
        "cache_bust": cache_bust_target,
        "apply_chat_template": apply_chat_template,
    }
    if shared_system_prompt_length is not None:
        overrides["prompt_prefix_shared_system_length"] = shared_system_prompt_length
    if prefix_pool_size is not None:
        overrides["prompt_prefix_pool_size"] = prefix_pool_size
    if prefix_length is not None:
        overrides["prompt_prefix_length"] = prefix_length
    return make_run_from_cli(CLIConfig(**overrides))


def _make_tokenizer_no_chat_template():
    """A tokenizer mock with no apply_chat_template — overheads collapse to 0."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=list(range(10)))
    tokenizer._tokenizer = MagicMock(spec=[])  # spec=[] -> no attributes
    return tokenizer


def _build_composer(
    run,
    *,
    marker_cost: int = 0,
    chat_fixed: int = 0,
    chat_wrap: int = 0,
):
    """Build SyntheticDatasetComposer with deterministic budget components."""
    tokenizer = _make_tokenizer_no_chat_template()
    with (
        patch(
            _MARKER_COST_PATCH,
            return_value=marker_cost,
        ),
        patch(
            "aiperf.dataset.composer.base._estimate_chat_template_overheads",
            return_value=(chat_fixed, chat_wrap),
        ),
        patch("aiperf.dataset.generator.prompt.PromptGenerator"),
    ):
        return SyntheticDatasetComposer(run=run, tokenizer=tokenizer)


class TestCacheBustMarkerRouting:
    """Marker-cost compensation must mirror worker._apply_cache_bust fallback."""

    def test_first_turn_target_compensates_first_user_turn(self):
        config = _make_config(cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX)
        composer = _build_composer(config, marker_cost=10)
        assert composer._first_turn_cache_bust_marker_tokens == 10

    def test_first_turn_suffix_compensates_first_user_turn(self):
        config = _make_config(cache_bust_target=CacheBustTarget.FIRST_TURN_SUFFIX)
        composer = _build_composer(config, marker_cost=10)
        assert composer._first_turn_cache_bust_marker_tokens == 10

    def test_system_prefix_with_shared_system_does_not_compensate_user_turn(self):
        """Marker stays on system prompt -> user-turn comp would double-debit."""
        config = _make_config(
            cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
            shared_system_prompt_length=200,
        )
        composer = _build_composer(config, marker_cost=10)
        assert composer._first_turn_cache_bust_marker_tokens == 0

    def test_system_suffix_with_shared_system_does_not_compensate_user_turn(self):
        config = _make_config(
            cache_bust_target=CacheBustTarget.SYSTEM_SUFFIX,
            shared_system_prompt_length=200,
        )
        composer = _build_composer(config, marker_cost=10)
        assert composer._first_turn_cache_bust_marker_tokens == 0

    def test_system_prefix_without_shared_system_compensates_first_user_turn(self):
        """SYSTEM_* with no system message falls back to first user turn."""
        config = _make_config(cache_bust_target=CacheBustTarget.SYSTEM_PREFIX)
        composer = _build_composer(config, marker_cost=10)
        assert composer._first_turn_cache_bust_marker_tokens == 10

    def test_system_suffix_without_shared_system_compensates_first_user_turn(self):
        config = _make_config(cache_bust_target=CacheBustTarget.SYSTEM_SUFFIX)
        composer = _build_composer(config, marker_cost=10)
        assert composer._first_turn_cache_bust_marker_tokens == 10

    def test_none_target_compensates_nothing(self):
        config = _make_config(cache_bust_target=CacheBustTarget.NONE)
        composer = _build_composer(config, marker_cost=10)
        assert composer._first_turn_cache_bust_marker_tokens == 0
        assert composer._cache_bust_marker_tokens == 0


class TestSharedSystemPromptCompensation:
    """SYSTEM_* + shared system prompt: regenerate at length - marker_cost."""

    def test_shared_system_prompt_length_reduced_for_system_prefix(self):
        config = _make_config(
            cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
            shared_system_prompt_length=200,
        )
        tokenizer = _make_tokenizer_no_chat_template()
        with (
            patch(
                _MARKER_COST_PATCH,
                return_value=15,
            ),
            patch(
                "aiperf.dataset.composer.base._estimate_chat_template_overheads",
                return_value=(0, 0),
            ),
            patch(
                "aiperf.dataset.generator.corpus.PromptGenerator"
            ) as mock_prompt_gen_cls,
        ):
            SyntheticDatasetComposer(run=config, tokenizer=tokenizer)

        # PromptGenerator was constructed with a config whose
        # shared_system_prompt_length is 200 - 15 = 185.
        passed_prefix = mock_prompt_gen_cls.call_args.kwargs["prefix_prompts"]
        assert passed_prefix.shared_system_length == 200 - 15

    def test_first_turn_target_does_not_touch_shared_system_prompt_length(self):
        """FIRST_TURN_* marker doesn't land on system prompt -> length unchanged."""
        config = _make_config(
            cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
            shared_system_prompt_length=200,
        )
        tokenizer = _make_tokenizer_no_chat_template()
        with (
            patch(
                _MARKER_COST_PATCH,
                return_value=15,
            ),
            patch(
                "aiperf.dataset.composer.base._estimate_chat_template_overheads",
                return_value=(0, 0),
            ),
            patch(
                "aiperf.dataset.generator.corpus.PromptGenerator"
            ) as mock_prompt_gen_cls,
        ):
            SyntheticDatasetComposer(run=config, tokenizer=tokenizer)

        passed_prefix = mock_prompt_gen_cls.call_args.kwargs["prefix_prompts"]
        assert passed_prefix.shared_system_length == 200

    def test_marker_larger_than_shared_system_floors_at_one(self):
        """Pathological case: shared system length < marker cost."""
        config = _make_config(
            cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
            shared_system_prompt_length=5,
        )
        tokenizer = _make_tokenizer_no_chat_template()
        with (
            patch(
                _MARKER_COST_PATCH,
                return_value=20,
            ),
            patch(
                "aiperf.dataset.composer.base._estimate_chat_template_overheads",
                return_value=(0, 0),
            ),
            patch(
                "aiperf.dataset.generator.corpus.PromptGenerator"
            ) as mock_prompt_gen_cls,
        ):
            SyntheticDatasetComposer(run=config, tokenizer=tokenizer)

        passed_prefix = mock_prompt_gen_cls.call_args.kwargs["prefix_prompts"]
        assert passed_prefix.shared_system_length == 1

    def test_user_facing_config_is_not_mutated(self):
        """We must use model_copy, not mutate the user's config in place."""
        config = _make_config(
            cache_bust_target=CacheBustTarget.SYSTEM_PREFIX,
            shared_system_prompt_length=200,
        )
        tokenizer = _make_tokenizer_no_chat_template()
        with (
            patch(
                _MARKER_COST_PATCH,
                return_value=15,
            ),
            patch(
                "aiperf.dataset.composer.base._estimate_chat_template_overheads",
                return_value=(0, 0),
            ),
            patch("aiperf.dataset.generator.corpus.PromptGenerator"),
        ):
            SyntheticDatasetComposer(run=config, tokenizer=tokenizer)

        # Original config is untouched.
        prefix_prompts = config.cfg.get_default_dataset().prefix_prompts
        assert prefix_prompts.shared_system_length == 200


class TestChatTemplateOverheadProbe:
    """Two-shot probe must isolate per-request fixed cost from per-msg wrap."""

    def test_returns_zeros_when_no_apply_chat_template(self):
        from aiperf.dataset.composer.base import _estimate_chat_template_overheads

        tokenizer = MagicMock()
        tokenizer._tokenizer = MagicMock(spec=[])
        assert _estimate_chat_template_overheads(tokenizer) == (0, 0)

    def test_returns_zeros_when_tokenizer_is_none(self):
        from aiperf.dataset.composer.base import _estimate_chat_template_overheads

        assert _estimate_chat_template_overheads(None) == (0, 0)

    def test_returns_zeros_when_apply_chat_template_raises(self):
        from aiperf.dataset.composer.base import _estimate_chat_template_overheads

        inner = MagicMock()
        inner.apply_chat_template = MagicMock(
            side_effect=ValueError("no chat template")
        )
        tokenizer = MagicMock()
        tokenizer._tokenizer = inner
        tokenizer.encode = MagicMock(return_value=list(range(5)))
        assert _estimate_chat_template_overheads(tokenizer) == (0, 0)

    def test_decomposes_fixed_and_wrap(self):
        """Synthetic Llama-3-like template: BOS=1, gen_prompt=3, wrap=5/msg.

        BOS is excluded from per_request_fixed because RangeRatioDistribution
        already accounts for it via num_special_tokens. The function returns
        only the template-structural overhead (gen_prompt suffix, not BOS).
        """
        from aiperf.dataset.composer.base import (
            _CHAT_TEMPLATE_PROBE_SAMPLES,
            _estimate_chat_template_overheads,
        )

        per_msg_wrap = 5
        bos = 1
        gen_prompt = 3
        template_total_fixed = bos + gen_prompt  # what apply_chat_template adds

        def fake_apply(messages, tokenize=True, add_generation_prompt=True, **_kwargs):
            # tokenizer.encode counts by whitespace-delimited words, so return a
            # string with exactly the right number of space-separated words.
            content_tokens = sum(len(m["content"].split()) for m in messages)
            wrapping = per_msg_wrap * len(messages)
            total = template_total_fixed + wrapping + content_tokens
            return " ".join(["x"] * total)

        inner = MagicMock()
        inner.apply_chat_template = MagicMock(side_effect=fake_apply)
        tokenizer = MagicMock()
        tokenizer._tokenizer = inner
        tokenizer.encode = MagicMock(
            side_effect=lambda text: list(range(len(text.split())))
        )
        tokenizer.num_prompt_special_tokens = MagicMock(return_value=bos)

        fixed, wrap = _estimate_chat_template_overheads(tokenizer)
        assert fixed == gen_prompt  # BOS excluded — handled by RangeRatioDistribution
        assert wrap == per_msg_wrap
        # 2 templates per sample -> 2 * len(samples) apply calls.
        assert inner.apply_chat_template.call_count == 2 * len(
            _CHAT_TEMPLATE_PROBE_SAMPLES
        )

    def test_returns_zeros_on_implausible_negative_wrap(self):
        """Defensive: never trust a probe that gives negative numbers."""
        from aiperf.dataset.composer.base import _estimate_chat_template_overheads

        # Templated < 2*bare + single -> avg_wrap negative.
        def fake_apply(messages, **_kwargs):
            return list(range(1))  # tiny

        inner = MagicMock()
        inner.apply_chat_template = MagicMock(side_effect=fake_apply)
        tokenizer = MagicMock()
        tokenizer._tokenizer = inner
        tokenizer.encode = MagicMock(return_value=list(range(50)))

        assert _estimate_chat_template_overheads(tokenizer) == (0, 0)


class TestAdjustmentProperties:
    """The two public properties must compose the components correctly."""

    def test_first_turn_adjustment_composes_all_four(self):
        config = _make_config(
            cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX,
            prefix_pool_size=3,
            prefix_length=20,
        )
        composer = _build_composer(config, marker_cost=10, chat_fixed=4, chat_wrap=5)
        # 4 (fixed) + 5 (wrap) + 10 (marker) + 20 (prefix) = 39
        assert composer.first_turn_isl_adjustment == 39

    def test_subsequent_turn_adjustment_only_per_msg_wrap(self):
        config = _make_config(cache_bust_target=CacheBustTarget.FIRST_TURN_PREFIX)
        composer = _build_composer(config, marker_cost=10, chat_fixed=4, chat_wrap=5)
        # Only 5 (wrap): fixed, marker, and prefix all apply only to the first turn.
        assert composer.subsequent_turn_isl_adjustment == 5

    def test_no_adjustment_when_everything_zero(self):
        config = _make_config(cache_bust_target=CacheBustTarget.NONE)
        composer = _build_composer(config)
        assert composer.first_turn_isl_adjustment == 0
        assert composer.subsequent_turn_isl_adjustment == 0

    def test_prefix_length_included_in_first_turn_adjustment(self):
        """Prefix tokens are subtracted from the first-turn ISL so body + prefix = target."""
        config = _make_config(prefix_pool_size=3, prefix_length=30)
        composer = _build_composer(config)
        assert composer.first_turn_isl_adjustment == 30
        assert composer.subsequent_turn_isl_adjustment == 0

    def test_no_prefix_length_when_pool_size_absent(self):
        """prefix_length without pool_size has no effect on adjustment."""
        config = _make_config(prefix_length=30)
        composer = _build_composer(config)
        assert composer.first_turn_isl_adjustment == 0


class TestSyntheticPromptBudgetSubtraction:
    """End-to-end: synthetic composer reduces ISL passed to prompt generator."""

    def _build(
        self,
        *,
        marker_cost: int = 0,
        chat_fixed: int = 0,
        chat_wrap: int = 0,
        cache_bust_target: CacheBustTarget = CacheBustTarget.FIRST_TURN_PREFIX,
        isl_mean: int = 100,
    ):
        config = _make_config(cache_bust_target=cache_bust_target, isl_mean=isl_mean)
        composer = _build_composer(
            config,
            marker_cost=marker_cost,
            chat_fixed=chat_fixed,
            chat_wrap=chat_wrap,
        )
        composer.prompt_generator = MagicMock()
        composer.prompt_generator.generate = MagicMock(return_value="prompt-text")
        return composer

    def test_first_turn_subtracts_fixed_plus_wrap_plus_marker(self):
        composer = self._build(marker_cost=10, chat_fixed=4, chat_wrap=5, isl_mean=100)
        composer._generate_text_payloads(Turn(), is_first=True)
        # 100 - 10 - 4 - 5 = 81
        assert composer.prompt_generator.generate.call_args.kwargs["mean"] == 81

    def test_subsequent_turn_subtracts_only_per_msg_wrap(self):
        composer = self._build(marker_cost=10, chat_fixed=4, chat_wrap=5, isl_mean=100)
        composer._generate_text_payloads(Turn(), is_first=False)
        # 100 - 5 = 95 (no marker, no fixed)
        assert composer.prompt_generator.generate.call_args.kwargs["mean"] == 95

    def test_compensation_floors_at_one_for_tiny_isl(self):
        """ISL=5 with 19-token first-turn compensation must not become 0 or negative."""
        composer = self._build(marker_cost=10, chat_fixed=4, chat_wrap=5, isl_mean=5)
        composer._generate_text_payloads(Turn(), is_first=True)
        assert composer.prompt_generator.generate.call_args.kwargs["mean"] == 1

    def test_no_compensation_passes_isl_through(self):
        composer = self._build(
            marker_cost=0,
            chat_fixed=0,
            chat_wrap=0,
            cache_bust_target=CacheBustTarget.NONE,
            isl_mean=100,
        )
        composer._generate_text_payloads(Turn(), is_first=True)
        assert composer.prompt_generator.generate.call_args.kwargs["mean"] == 100

    def test_chat_template_only_no_cache_bust(self):
        """Tokenizer has chat template but cache-bust off: still compensate."""
        composer = self._build(
            marker_cost=0,
            chat_fixed=4,
            chat_wrap=5,
            cache_bust_target=CacheBustTarget.NONE,
            isl_mean=100,
        )
        composer._generate_text_payloads(Turn(), is_first=True)
        # 100 - 4 - 5 = 91 on first turn
        assert composer.prompt_generator.generate.call_args.kwargs["mean"] == 91

        composer._generate_text_payloads(Turn(), is_first=False)
        # 100 - 5 = 95 on subsequent turns
        assert composer.prompt_generator.generate.call_args.kwargs["mean"] == 95


class TestApplyChatTemplateOptOut:
    """Without --apply-chat-template (the default), the composer skips the overhead probe so synthetic ISL passes through at the bare-text token count."""

    def test_overhead_probe_not_invoked_when_flag_off(self):
        """The expensive probe (multiple template renders + encodes) must not fire when the user opted out."""
        config = _make_config(apply_chat_template=False)
        tokenizer = _make_tokenizer_no_chat_template()
        with (
            patch(
                _MARKER_COST_PATCH,
                return_value=0,
            ),
            patch(
                "aiperf.dataset.composer.base._estimate_chat_template_overheads",
                return_value=(99, 99),
            ) as mock_probe,
            patch("aiperf.dataset.generator.prompt.PromptGenerator"),
        ):
            composer = SyntheticDatasetComposer(run=config, tokenizer=tokenizer)

        mock_probe.assert_not_called()
        assert composer._chat_template_per_request_fixed_tokens == 0
        assert composer._chat_template_per_msg_wrap_tokens == 0
        assert composer.first_turn_isl_adjustment == 0
        assert composer.subsequent_turn_isl_adjustment == 0

    def test_synthetic_isl_passes_through_when_flag_off(self):
        """End-to-end: the prompt generator receives the user's --isl verbatim (no template-wrapping subtraction)."""
        config = _make_config(
            apply_chat_template=False,
            cache_bust_target=CacheBustTarget.NONE,
            isl_mean=100,
        )
        tokenizer = _make_tokenizer_no_chat_template()
        with (
            patch(
                _MARKER_COST_PATCH,
                return_value=0,
            ),
            patch(
                "aiperf.dataset.composer.base._estimate_chat_template_overheads",
                return_value=(4, 5),
            ),
            patch("aiperf.dataset.generator.prompt.PromptGenerator"),
        ):
            composer = SyntheticDatasetComposer(run=config, tokenizer=tokenizer)
        composer.prompt_generator = MagicMock()
        composer.prompt_generator.generate = MagicMock(return_value="prompt-text")

        composer._generate_text_payloads(Turn(), is_first=True)
        assert composer.prompt_generator.generate.call_args.kwargs["mean"] == 100

        composer._generate_text_payloads(Turn(), is_first=False)
        assert composer.prompt_generator.generate.call_args.kwargs["mean"] == 100


@pytest.mark.parametrize(
    "target,has_shared_system,expected_marker_estimator_calls",
    [
        # marker on first user turn -> estimator runs once
        (CacheBustTarget.FIRST_TURN_PREFIX, False, 1),
        (CacheBustTarget.FIRST_TURN_SUFFIX, False, 1),
        (CacheBustTarget.FIRST_TURN_PREFIX, True, 1),
        (CacheBustTarget.SYSTEM_PREFIX, False, 1),
        (CacheBustTarget.SYSTEM_SUFFIX, False, 1),
        # marker on shared system prompt -> estimator runs to compensate it
        (CacheBustTarget.SYSTEM_PREFIX, True, 1),
        (CacheBustTarget.SYSTEM_SUFFIX, True, 1),
        # NONE -> never invoked
        (CacheBustTarget.NONE, False, 0),
        (CacheBustTarget.NONE, True, 0),
    ],
)
def test_marker_estimator_is_invoked_when_compensation_is_needed(
    target, has_shared_system, expected_marker_estimator_calls
):
    """Under NONE the encode round-trip is skipped entirely (cheap)."""
    config = _make_config(
        cache_bust_target=target,
        shared_system_prompt_length=200 if has_shared_system else None,
    )
    tokenizer = _make_tokenizer_no_chat_template()
    with (
        patch(
            _MARKER_COST_PATCH,
            return_value=10,
        ) as mock_estimate,
        patch(
            "aiperf.dataset.composer.base._estimate_chat_template_overheads",
            return_value=(0, 0),
        ),
        patch("aiperf.dataset.generator.corpus.PromptGenerator"),
    ):
        SyntheticDatasetComposer(run=config, tokenizer=tokenizer)

    assert mock_estimate.call_count == expected_marker_estimator_calls
