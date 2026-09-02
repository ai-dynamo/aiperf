# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loader forkserver context must not silently ignore later preload args."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import aiperf.dataset._mp_context as mpc


@pytest.fixture(autouse=True)
def _reset_loader_ctx() -> None:
    """Isolate the process-global loader context cache between tests."""
    mpc._loader_ctx = None
    mpc._loader_ctx_key = None
    yield
    mpc._loader_ctx = None
    mpc._loader_ctx_key = None


@pytest.fixture
def _stub_forkserver(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Avoid booting a real forkserver helper in unit tests."""
    fake_ctx = MagicMock(name="loader-mp-ctx")
    monkeypatch.setattr(mpc.multiprocessing, "get_context", lambda method: fake_ctx)
    monkeypatch.setattr(mpc, "_eagerly_start_forkserver", lambda: None)
    # Force the Linux forkserver path regardless of host OS.
    monkeypatch.setattr(mpc, "IS_LINUX", True)
    return fake_ctx


def test_same_preload_tokenizer_reuses_cached_context(
    _stub_forkserver: MagicMock,
) -> None:
    ctx1 = mpc.get_loader_mp_context(preload_tokenizer="tok-a")
    ctx2 = mpc.get_loader_mp_context(preload_tokenizer="tok-a")
    assert ctx1 is ctx2 is _stub_forkserver


def test_different_preload_tokenizer_fails_loud(
    _stub_forkserver: MagicMock,
) -> None:
    """First-wins cache must not silently ignore a later tokenizer identity."""
    mpc.get_loader_mp_context(preload_tokenizer="tok-a")

    with pytest.raises(ValueError, match=r"tok-a|tok-b|preload"):
        mpc.get_loader_mp_context(preload_tokenizer="tok-b")


def test_different_trust_or_revision_fails_loud(
    _stub_forkserver: MagicMock,
) -> None:
    mpc.get_loader_mp_context(
        preload_tokenizer="tok-a",
        trust_remote_code=False,
        revision="main",
    )

    with pytest.raises(ValueError, match=r"preload|tok-a"):
        mpc.get_loader_mp_context(
            preload_tokenizer="tok-a",
            trust_remote_code=True,
            revision="main",
        )

    with pytest.raises(ValueError, match=r"preload|tok-a"):
        mpc.get_loader_mp_context(
            preload_tokenizer="tok-a",
            trust_remote_code=False,
            revision="v2",
        )


def test_later_none_preload_reuses_existing_context(
    _stub_forkserver: MagicMock,
) -> None:
    """Callers that omit preload still share the already-built context."""
    ctx1 = mpc.get_loader_mp_context(preload_tokenizer="tok-a")
    ctx2 = mpc.get_loader_mp_context()
    assert ctx1 is ctx2
