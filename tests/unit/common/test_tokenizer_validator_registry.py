# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest
from rich.console import Console

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry
from aiperf.common.tokenizer_validator import (
    _prefetch_tokenizers,
    set_default_registry,
)


@pytest.mark.asyncio
async def test_prefetch_tokenizers_populates_default_registry() -> None:
    reg = TokenizerBundleRegistry()
    set_default_registry(reg)
    try:
        _prefetch_tokenizers(
            {"gpt2"},
            trust_remote_code=False,
            revision="main",
            logger=AIPerfLogger("test"),
            console=Console(),
        )
    finally:
        set_default_registry(None)

    entry = reg.get("gpt2")
    assert entry is not None
    snapshot_dir, event = entry
    assert event.is_set()
    assert snapshot_dir is not None
    assert snapshot_dir.is_absolute()
    # gpt2 has tokenizer.json in modern HF cache; older revisions only ship vocab.json + merges.txt
    assert (snapshot_dir / "tokenizer.json").exists() or (
        snapshot_dir / "vocab.json"
    ).exists()


@pytest.mark.asyncio
async def test_prefetch_tokenizers_no_op_when_no_default_registry() -> None:
    set_default_registry(None)
    _prefetch_tokenizers(
        {"gpt2"},
        trust_remote_code=False,
        revision="main",
        logger=AIPerfLogger("test"),
        console=Console(),
    )
