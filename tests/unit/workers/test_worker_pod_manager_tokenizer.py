# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke test: the rewritten _prefetch_tokenizers calls download_tokenizer
once per unique tokenizer and publishes GroupTokenizerReady on success."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from aiperf.common.pod_lifecycle_structs import GroupTokenizerReady


@pytest.mark.asyncio
async def test_prefetch_publishes_group_tokenizer_ready(
    monkeypatch, tmp_path: Path
) -> None:
    from aiperf.workers import worker_pod_manager as wpm

    fake_download = AsyncMock(
        side_effect=lambda *, name, dest_root, **_: dest_root / name
    )
    monkeypatch.setattr(wpm, "download_tokenizer", fake_download)

    published: list[GroupTokenizerReady] = []

    mgr = MagicMock()
    mgr._publish_group_message = AsyncMock(side_effect=published.append)
    mgr._unique_tokenizer_names = MagicMock(return_value=["gpt2", "bert-base-uncased"])
    mgr._tokenizer_dest_root = MagicMock(return_value=tmp_path)
    mgr.run.cfg.runtime.dataset_api_base_url = "http://api"
    mgr.service_id = "wgm-0"

    await wpm.WorkerGroupManagerBase._prefetch_tokenizers(mgr)

    assert fake_download.await_count == 2
    assert len(published) == 1
    assert isinstance(published[0], GroupTokenizerReady)
    assert published[0].success
    assert set(published[0].bundles) == {"gpt2", "bert-base-uncased"}
