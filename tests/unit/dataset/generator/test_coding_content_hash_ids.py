# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DM5: the hash-id block size resolves from the explicit argument, else the ``InputTokensDefaults`` fallback."""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config import PromptConfig
from aiperf.config.dataset.defaults import InputTokensDefaults
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from tests.harness.fake_tokenizer import FakeTokenizer


@pytest.mark.parametrize(
    "block_size,hash_id,expected_block_size",
    [
        param(None, 1, InputTokensDefaults.BLOCK_SIZE, id="default_config_falls_back"),
        param(64, 7, 64, id="explicit_block_size_wins"),
    ],
)  # fmt: skip
def test_generate_hash_ids_resolves_block_size(
    block_size: int | None, hash_id: int, expected_block_size: int
) -> None:
    """A default ``PromptConfig`` leaves ``block_size`` as ``None``, so the hash-id path must fall back to ``InputTokensDefaults.BLOCK_SIZE`` rather than raising on ``(len(hash_ids) - 1) * None``."""
    generator = CodingContentGenerator(config=PromptConfig(), tokenizer=FakeTokenizer())
    kwargs = {} if block_size is None else {"block_size": block_size}

    result = generator.generate(mean=expected_block_size, hash_ids=[hash_id], **kwargs)

    assert isinstance(result, str) and result
    assert len(generator._cache[hash_id]) == expected_block_size
