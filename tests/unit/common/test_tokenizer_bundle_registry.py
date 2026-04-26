# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry


@pytest.mark.asyncio
async def test_register_pending_creates_unready_entry(tmp_path: Path) -> None:
    reg = TokenizerBundleRegistry()
    reg.register_pending("gpt2")
    snapshot, event = reg.get("gpt2")
    assert snapshot is None
    assert not event.is_set()


@pytest.mark.asyncio
async def test_mark_ready_sets_event_and_snapshot(tmp_path: Path) -> None:
    reg = TokenizerBundleRegistry()
    reg.register_pending("gpt2")
    reg.mark_ready("gpt2", tmp_path / "snap")
    snapshot, event = reg.get("gpt2")
    assert snapshot == tmp_path / "snap"
    assert event.is_set()


@pytest.mark.asyncio
async def test_get_unknown_returns_none() -> None:
    reg = TokenizerBundleRegistry()
    assert reg.get("never-registered") is None


@pytest.mark.asyncio
async def test_register_pending_idempotent() -> None:
    reg = TokenizerBundleRegistry()
    reg.register_pending("gpt2")
    _, event_first = reg.get("gpt2")
    reg.register_pending("gpt2")
    _, event_second = reg.get("gpt2")
    assert event_first is event_second  # same event reused


@pytest.mark.asyncio
async def test_mark_ready_unblocks_waiter(tmp_path: Path) -> None:
    reg = TokenizerBundleRegistry()
    reg.register_pending("gpt2")

    async def wait_then_get() -> Path:
        _, event = reg.get("gpt2")
        await event.wait()
        snapshot, _ = reg.get("gpt2")
        return snapshot

    waiter = asyncio.create_task(wait_then_get())
    await asyncio.sleep(0)  # let waiter park
    reg.mark_ready("gpt2", tmp_path / "snap")
    result = await asyncio.wait_for(waiter, timeout=1.0)
    assert result == tmp_path / "snap"
