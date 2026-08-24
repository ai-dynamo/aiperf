# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Degenerate range-ratio configs must be rejected, mirroring vLLM's guard.

vLLM's ``RandomDataset.sample`` raises when the minimum possible input is
unusable::

    min_total_input = prefix_len + floor(max(0, isl - num_special) * (1 - r))
    if min_total_input < 1: raise ValueError(...)

Without the equivalent guard, ``--isl 2 --random-range-ratio 0.9`` yields
bounds ``(0, 4)`` and draws of 0 are silently clamped to 1, so the run
completes and reports numbers for one-token prompts nobody asked for.
"""

from unittest.mock import MagicMock, patch

import pytest

from aiperf.common.enums import RandomCorpusStyle
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.dataset.composer.synthetic import SyntheticDatasetComposer
from tests.unit.conftest import make_run_from_cli

_MARKER_COST_PATCH = "aiperf.timing.strategies.cache_bust.estimate_marker_token_cost"


def _tokenizer(num_special: int = 0):
    tok = MagicMock()
    tok.encode = MagicMock(return_value=list(range(10)))
    tok._tokenizer = MagicMock(spec=[])
    tok.num_prompt_special_tokens = MagicMock(return_value=num_special)
    return tok


def _build(
    *,
    isl: int,
    ratio: float,
    prefix_len: int | None = None,
    style: RandomCorpusStyle = RandomCorpusStyle.VLLM,
    num_special: int = 0,
):
    overrides: dict = {
        "model_names": ["test-model"],
        "conversation_num_dataset_entries": 1,
        "prompt_input_tokens_mean": isl,
        "prompt_output_tokens_mean": 16,
        "prompt_random_range_ratio": str(ratio),
        "prompt_random_corpus_style": style,
        "prompt_corpus": "random",
    }
    if prefix_len is not None:
        overrides["prompt_prefix_length"] = prefix_len
        overrides["prompt_prefix_pool_size"] = 1
    run = make_run_from_cli(CLIConfig(**overrides))
    # Patch the generator factory rather than PromptGenerator itself: the real
    # factory builds a real generator, and prefix-pool construction then samples
    # against the mock tokenizer's vocab and dies on unrelated plumbing. The
    # guard runs before any prompt is generated, so stubbing the factory keeps
    # this test on the guard alone. Prefix composition against a real tokenizer
    # is covered end-to-end via the CLI.
    with (
        patch(_MARKER_COST_PATCH, return_value=0),
        patch(
            "aiperf.dataset.composer.base._estimate_chat_template_overheads",
            return_value=(0, 0),
        ),
        patch("aiperf.dataset.composer.base.resolve_prompt_generator"),
    ):
        return SyntheticDatasetComposer(run=run, tokenizer=_tokenizer(num_special))


class TestDegenerateRangeRatioGuard:
    def test_degenerate_window_is_rejected(self):
        """isl=2, r=0.9 -> bounds (0, 4); vLLM raises, so we must too."""
        with pytest.raises(ValueError, match="minimum input of"):
            _build(isl=2, ratio=0.9)

    def test_prefix_rescues_a_tiny_body(self):
        """Mirrors vLLM: min_total_input counts prefix_len, so a large prefix
        makes an otherwise-degenerate body acceptable."""
        composer = _build(isl=2, ratio=0.9, prefix_len=20)
        assert composer is not None

    def test_normal_config_unaffected(self):
        composer = _build(isl=128, ratio=0.3)
        assert composer is not None

    def test_sglang_style_cannot_bottom_out(self):
        """SGLANG's window is [max(1, int(mean*r)), mean] -- lower bound >= 1 by
        construction, so the guard is VLLM-specific and must not fire here."""
        composer = _build(isl=2, ratio=0.9, style=RandomCorpusStyle.SGLANG)
        assert composer is not None

    def test_special_tokens_counted_against_the_budget(self):
        """num_special is subtracted before the window is computed, so a
        tokenizer with a BOS can push a marginal config over the edge."""
        with pytest.raises(ValueError, match="special token"):
            _build(isl=1, ratio=0.5, num_special=1)

    def test_error_names_all_three_levers(self):
        with pytest.raises(ValueError) as exc:
            _build(isl=2, ratio=0.9)
        message = str(exc.value)
        assert "--isl" in message
        assert "--prompt-prefix-length" in message
        assert "--random-range-ratio" in message
