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
