# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DM5: ``CodingContentGenerator.generate(hash_ids=...)`` with a default config.

``PromptConfig.block_size`` defaults to ``None``; the hash-id path must fall
back to ``InputTokensDefaults.BLOCK_SIZE`` exactly like
``PromptGenerator.generate`` does, instead of raising ``TypeError`` on
``(len(hash_ids) - 1) * None``.
"""

from __future__ import annotations

from aiperf.config import PromptConfig
from aiperf.config.dataset.defaults import InputTokensDefaults
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from tests.harness.fake_tokenizer import FakeTokenizer


def test_generate_hash_ids_default_config_falls_back_to_default_block_size() -> None:
    generator = CodingContentGenerator(config=PromptConfig(), tokenizer=FakeTokenizer())

    result = generator.generate(mean=InputTokensDefaults.BLOCK_SIZE, hash_ids=[1])

    assert isinstance(result, str) and result
    assert len(generator._cache[1]) == InputTokensDefaults.BLOCK_SIZE


def test_generate_hash_ids_explicit_block_size_still_wins() -> None:
    generator = CodingContentGenerator(config=PromptConfig(), tokenizer=FakeTokenizer())

    generator.generate(mean=64, hash_ids=[7], block_size=64)

    assert len(generator._cache[7]) == 64
