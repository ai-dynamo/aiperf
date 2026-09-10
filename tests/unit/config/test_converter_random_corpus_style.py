# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`--random-corpus-style` must survive the CLI converter without a range ratio.

The style write used to sit after `_apply_random_range_ratio`'s early return,
so `--random-corpus-style sglang` alone was silently discarded and the user got
vLLM's token pool. Only the sglang request was observably lost -- vllm is the
default, so a dropped write lands there anyway -- which is why one of the four
cells below is the entire bug.

With no ratio, the pool is the *only* thing the style selects, so this is
precisely the case where someone reaches for the flag on its own.
"""

import pytest
from pytest import param

from aiperf.common.enums import RandomCorpusStyle
from aiperf.config.flags.cli_config import CLIConfig
from tests.unit.conftest import make_run_from_cli

_BASE = dict(
    model_names=["m"],
    conversation_num_dataset_entries=2,
    prompt_corpus="random",
    prompt_input_tokens_mean=128,
    prompt_output_tokens_mean=16,
)


def _resolved_style(**overrides) -> RandomCorpusStyle:
    run = make_run_from_cli(CLIConfig(**{**_BASE, **overrides}))
    return run.cfg.get_default_dataset().prompts.random_corpus_style


@pytest.mark.parametrize(
    "style",
    [
        param(RandomCorpusStyle.VLLM, id="vllm"),
        param(RandomCorpusStyle.SGLANG, id="sglang"),
    ],
)  # fmt: skip
@pytest.mark.parametrize(
    "ratio",
    [param(None, id="no-ratio"), param("0.3", id="with-ratio")],
)  # fmt: skip
def test_style_survives_with_and_without_a_ratio(style, ratio):
    overrides = {"prompt_random_corpus_style": style}
    if ratio is not None:
        overrides["prompt_random_range_ratio"] = ratio
    assert _resolved_style(**overrides) == style


def test_sglang_without_ratio_is_the_regression_case():
    """Pinned separately: this is the one cell that was broken, and it is the
    one a user hits when they want SGLang's full-vocab pool and nothing else."""
    assert (
        _resolved_style(prompt_random_corpus_style=RandomCorpusStyle.SGLANG)
        == RandomCorpusStyle.SGLANG
    )


def test_default_is_vllm_when_flag_is_absent():
    """Guard against over-correcting: an unset flag must not start writing."""
    assert _resolved_style() == RandomCorpusStyle.VLLM
