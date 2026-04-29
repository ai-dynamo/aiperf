# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TokenizerRouter -- tar+zstd snapshot bundle streaming.

Patches ``_resolve_snapshot_dir`` so the router is tested in isolation from
the HuggingFace Hub. Live HF round-trip is covered by the
component-integration test.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
import zstandard
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from aiperf.api.routers import tokenizer as tokenizer_router_mod
from aiperf.api.routers.tokenizer import build_tokenizer_router


def _make_snapshot(tmp_path: Path, files: dict[str, str]) -> Path:
    snap = tmp_path / "snap"
    snap.mkdir()
    for name, body in files.items():
        (snap / name).write_text(body)
    return snap


def _patch_resolver(monkeypatch, snap: Path) -> None:
    async def _resolver(name: str, registry=None) -> Path:
        if name == "unknown":
            raise HTTPException(status_code=404, detail=f"tokenizer '{name}' not found")
        return snap

    monkeypatch.setattr(tokenizer_router_mod, "_resolve_snapshot_dir", _resolver)


@pytest.fixture
def app_with_mock_hf(monkeypatch, tmp_path: Path) -> tuple[FastAPI, Path]:
    snap = _make_snapshot(
        tmp_path, {"tokenizer.json": '{"version":"1.0"}', "tokenizer_config.json": "{}"}
    )
    _patch_resolver(monkeypatch, snap)
    app = FastAPI()
    app.include_router(build_tokenizer_router())
    return app, snap


@pytest.mark.asyncio
async def test_404_when_repo_unknown(app_with_mock_hf) -> None:
    app, _ = app_with_mock_hf
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/unknown/bundle")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_200_streams_tar_zstd_round_trip(app_with_mock_hf) -> None:
    app, _ = app_with_mock_hf
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
async def test_path_with_slash_routes_correctly(app_with_mock_hf) -> None:
    """Verify ``:path`` converter handles ``org/model`` style names."""
    app, _ = app_with_mock_hf
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/meta-llama/Llama-3.1-8B/bundle")
    assert resp.status_code == 200
