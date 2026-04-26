# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TokenizerRouter -- tar+zstd snapshot bundle streaming."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import zstandard
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from aiperf.api.routers.tokenizer import build_tokenizer_router
from aiperf.common.tokenizer_bundle_registry import TokenizerBundleRegistry


def _make_snapshot(tmp_path: Path) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "tokenizer.json").write_text('{"version":"1.0"}')
    (snap / "tokenizer_config.json").write_text("{}")
    return snap


@pytest.fixture
def app_and_registry(tmp_path: Path) -> tuple[FastAPI, TokenizerBundleRegistry, Path]:
    reg = TokenizerBundleRegistry()
    snap = _make_snapshot(tmp_path)
    app = FastAPI()
    app.include_router(build_tokenizer_router(reg))
    return app, reg, snap


@pytest.mark.asyncio
async def test_404_when_not_registered(app_and_registry) -> None:
    app, _, _ = app_and_registry
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/unknown/bundle")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_503_when_pending(app_and_registry) -> None:
    app, reg, _ = app_and_registry
    reg.register_pending("gpt2")
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/gpt2/bundle")
    assert resp.status_code == 503
    assert resp.headers.get("retry-after") == "1"


@pytest.mark.asyncio
async def test_200_streams_tar_zstd_round_trip(app_and_registry) -> None:
    app, reg, snap = app_and_registry
    reg.register_pending("gpt2")
    reg.mark_ready("gpt2", snap)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/gpt2/bundle")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zstd"

    # Decompress + untar; assert files round-trip. Use stream_reader because the
    # server emits a streaming zstd frame without a known content-size header.
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(io.BytesIO(resp.content)) as reader:
        tar_bytes = reader.read()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        names = sorted(m.name for m in tf.getmembers() if m.isfile())
    assert names == ["tokenizer.json", "tokenizer_config.json"]


@pytest.mark.asyncio
async def test_path_with_slash_routes_correctly(
    app_and_registry, tmp_path: Path
) -> None:
    """Verify `:path` converter handles `org/model` style names."""
    app, reg, _ = app_and_registry
    snap = tmp_path / "ll"
    snap.mkdir()
    (snap / "tokenizer.json").write_text("{}")
    reg.register_pending("meta-llama/Llama-3.1-8B")
    reg.mark_ready("meta-llama/Llama-3.1-8B", snap)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/meta-llama/Llama-3.1-8B/bundle")
    assert resp.status_code == 200
