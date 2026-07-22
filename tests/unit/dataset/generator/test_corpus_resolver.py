# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pytest import param

from aiperf.common.enums import PromptCorpus
from aiperf.config.dataset.content import PromptConfig
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.corpus import resolve_prompt_generator
from aiperf.dataset.generator.prompt import PromptGenerator


@pytest.fixture
def mock_tokenizer(mock_tokenizer_cls):
    """Tokenizer instance for corpus resolver tests."""
    return mock_tokenizer_cls.from_pretrained("gpt2")


@pytest.mark.parametrize(
    ("corpus", "default_corpus", "expected_type"),
    [
        param(PromptCorpus.CODING, None, CodingContentGenerator, id="explicit-coding"),
        param(PromptCorpus.SONNET, PromptCorpus.CODING, PromptGenerator, id="explicit-sonnet-wins"),
        param(None, PromptCorpus.CODING, CodingContentGenerator, id="default-coding"),
        param(None, None, PromptGenerator, id="fallback-sonnet"),
        param(None, "coding", CodingContentGenerator, id="default-str-coding"),
    ],
)  # fmt: skip
def test_resolve_prompt_generator_selection(
    mock_tokenizer, corpus, default_corpus, expected_type
):
    gen = resolve_prompt_generator(
        corpus=corpus,
        default_corpus=default_corpus,
        tokenizer=mock_tokenizer,
        prompts=PromptConfig(),
    )
    assert isinstance(gen, expected_type)


def test_resolve_prompt_generator_passes_prefix_prompts_to_coding(mock_tokenizer):
    from aiperf.config.dataset.content import PrefixPromptConfig

    prefix = PrefixPromptConfig(pool_size=2, length=5)
    gen = resolve_prompt_generator(
        corpus=PromptCorpus.CODING,
        default_corpus=None,
        tokenizer=mock_tokenizer,
        prompts=PromptConfig(),
        prefix_prompts=prefix,
    )
    assert isinstance(gen, CodingContentGenerator)
    assert gen.prefix_prompts is prefix
    assert len(gen._prefix_prompts) == 2
