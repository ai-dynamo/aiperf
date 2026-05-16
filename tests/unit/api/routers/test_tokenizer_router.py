# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TokenizerRouter -- tar+zstd snapshot bundle streaming.

Patches ``_resolve_snapshot_dir`` so the router is tested in isolation from
the HuggingFace Hub. Live HF round-trip is covered by the
component-integration test.
"""

from __future__ import annotations

import asyncio
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
    async def _resolver(name: str) -> Path:
        if name == "unknown":
            raise HTTPException(status_code=404, detail=f"tokenizer '{name}' not found")
        return snap

    monkeypatch.setattr(tokenizer_router_mod, "_resolve_snapshot_dir", _resolver)


@pytest.fixture(autouse=True)
def _reset_module_cache() -> None:
    """Drop the module-level bundle cache between tests.

    The cache is intentionally process-wide in production (one api container
    per controller pod) but that wrecks test isolation -- a payload cached
    by one test would shadow the next test's resolver mock.
    """
    tokenizer_router_mod.reset_cache_for_tests()
    yield
    tokenizer_router_mod.reset_cache_for_tests()


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


@pytest.mark.asyncio
async def test_prewarm_serves_request_without_resolving(
    monkeypatch, tmp_path: Path
) -> None:
    """After ``prewarm_bundle``, the request path must NOT call the resolver.

    Regression for the production hang on ``nvidia/NVIDIA-Nemotron-3-Super-...``:
    when materialisation happens on the request path under a single global
    cache lock, an aiohttp client timeout (5 min) cancels the server-side
    task before tar+zstd finishes. The cache stays empty and every retry
    pays the same cost. With prewarm, the bytes are already cached when
    the first request arrives, and the resolver is never invoked.
    """
    snap = _make_snapshot(tmp_path, {"tokenizer.json": "{}"})

    resolve_calls: list[str] = []

    async def _counting_resolver(name: str) -> Path:
        resolve_calls.append(name)
        return snap

    monkeypatch.setattr(
        tokenizer_router_mod, "_resolve_snapshot_dir", _counting_resolver
    )

    # Prewarm BEFORE building the router. Production calls prewarm_bundle()
    # from api_service._prewarm_tokenizers, which runs before uvicorn binds.
    await tokenizer_router_mod.prewarm_bundle("nvidia/Nemotron-3-Super-120B")
    assert len(resolve_calls) == 1

    app = FastAPI()
    app.include_router(build_tokenizer_router())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/nvidia/Nemotron-3-Super-120B/bundle")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zstd"
    # Request path served from cache; resolver wasn't called a second time.
    assert len(resolve_calls) == 1, (
        f"resolver re-invoked on request path despite prewarm: {resolve_calls}"
    )


@pytest.mark.asyncio
async def test_concurrent_cold_requests_serialise_per_name(
    monkeypatch, tmp_path: Path
) -> None:
    """Two concurrent requests for the same un-prewarmed name must build once.

    Per-name lock prevents a thundering herd from tar+zstd-ing the same
    snapshot N times in parallel. Verified by counting how many times the
    materialisation work runs.
    """
    snap = _make_snapshot(tmp_path, {"tokenizer.json": "{}"})

    materialise_calls = 0

    async def _slow_resolver(name: str) -> Path:
        nonlocal materialise_calls
        # Yield the loop so the second concurrent caller can race in
        # before this one populates the cache.
        await asyncio.sleep(0)
        materialise_calls += 1
        return snap

    monkeypatch.setattr(tokenizer_router_mod, "_resolve_snapshot_dir", _slow_resolver)

    app = FastAPI()
    app.include_router(build_tokenizer_router())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        # Two concurrent requests for the same name.
        r1, r2 = await asyncio.gather(
            c.get("/api/tokenizer/some-org/some-model/bundle"),
            c.get("/api/tokenizer/some-org/some-model/bundle"),
        )
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Per-name lock + double-checked cache means one materialisation.
    assert materialise_calls == 1


@pytest.mark.asyncio
async def test_prewarm_failure_does_not_poison_request_path(
    monkeypatch, tmp_path: Path
) -> None:
    """A failed prewarm must leave the cache empty so the request path can retry.

    Production guarantee: prewarm_bundle catches its own exceptions in the
    caller (api_service._prewarm_tokenizers) so server startup is never
    blocked. The next request rebuilds via the on-demand fallback.
    """
    snap = _make_snapshot(tmp_path, {"tokenizer.json": "{}"})

    call = {"n": 0}

    async def _flaky_resolver(name: str) -> Path:
        call["n"] += 1
        if call["n"] == 1:
            raise HTTPException(status_code=503, detail="cold cache")
        return snap

    monkeypatch.setattr(tokenizer_router_mod, "_resolve_snapshot_dir", _flaky_resolver)

    # First prewarm fails. Cache stays empty; lock released.
    with pytest.raises(HTTPException):
        await tokenizer_router_mod.prewarm_bundle("flaky")

    # Request path retries via the on-demand fallback and succeeds.
    app = FastAPI()
    app.include_router(build_tokenizer_router())
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        resp = await c.get("/api/tokenizer/flaky/bundle")
    assert resp.status_code == 200


def _untar_names_and_sizes(payload: bytes) -> dict[str, int]:
    """Decompress + untar a bundle payload and return ``{name: size_bytes}``."""
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(io.BytesIO(payload)) as reader:
        tar_bytes = reader.read()
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:") as tf:
        return {m.name: m.size for m in tf.getmembers() if m.isfile()}


def test_materialize_bundle_excludes_safetensors(tmp_path: Path) -> None:
    """Weight shards (``*.safetensors``) must not be included in the bundle.

    Regression for the production OOM on Nemotron-3-Super-120B: the api
    container's HF_HOME mounts a shared PVC where the snapshot dir contains
    ``model-XXXXX-of-NNNNN.safetensors`` symlinks pointing at multi-GB blobs.
    A naive ``tar(..., dereference=True)`` over the whole snapshot dir would
    have inflated the bundle to ~120 GB. The allowlist must keep weights out.
    """
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "tokenizer.json").write_text('{"version":"1.0"}')
    (snap / "tokenizer_config.json").write_text("{}")
    (snap / "config.json").write_text("{}")
    # 50 MB of zeros -- representative of a (small) safetensors shard.
    (snap / "model-00001-of-00001.safetensors").write_bytes(
        b"\x00" * (50 * 1024 * 1024)
    )
    # Other non-tokenizer artefacts that must also be excluded.
    (snap / "pytorch_model.bin").write_bytes(b"\x00" * 1024)
    (snap / "README.md").write_text("# model card")

    payload = tokenizer_router_mod._materialize_bundle(snap)

    members = _untar_names_and_sizes(payload)
    assert "model-00001-of-00001.safetensors" not in members
    assert "pytorch_model.bin" not in members
    assert "README.md" not in members
    assert set(members) == {"tokenizer.json", "tokenizer_config.json", "config.json"}
    # Sanity: bundle stays small even though the snapshot dir has a 50 MB shard.
    assert len(payload) < 1 * 1024 * 1024, (
        f"bundle is {len(payload)} bytes; allowlist failed to drop weight shards"
    )


def test_materialize_bundle_includes_custom_python(tmp_path: Path) -> None:
    """Custom-tokenizer ``*.py`` modules must round-trip.

    Models like ``nvidia/NVIDIA-Nemotron-...`` ship custom tokenizer code
    referenced via ``auto_map`` in ``tokenizer_config.json``; AutoTokenizer
    under ``trust_remote_code=True`` imports those modules at load time, so
    they must be present in the worker-side snapshot extracted from the bundle.
    """
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "tokenizer.json").write_text("{}")
    (snap / "tokenizer_config.json").write_text(
        '{"auto_map": {"AutoTokenizer": ["tokenization_nemotron_h.NemotronHTokenizer", null]}}'
    )
    (snap / "configuration_nemotron_h.py").write_text("# config module")
    (snap / "modeling_nemotron_h.py").write_text("# modeling module")
    (snap / "tokenization_nemotron_h.py").write_text("# tokenizer module")

    payload = tokenizer_router_mod._materialize_bundle(snap)

    members = _untar_names_and_sizes(payload)
    assert "configuration_nemotron_h.py" in members
    assert "modeling_nemotron_h.py" in members
    assert "tokenization_nemotron_h.py" in members
    assert "tokenizer.json" in members
    assert "tokenizer_config.json" in members


def test_materialize_bundle_size_cap_raises(tmp_path: Path) -> None:
    """An allowlisted file that exceeds the size cap must raise.

    Safety net: if a future HF release introduces an allowlisted filename
    that legitimately holds large bytes, we want a loud failure naming the
    cap rather than a silent OOM at startup. Test forges an oversized
    ``tokenizer.json`` (allowlisted) past the 50 MiB cap.
    """
    # Override the cap to keep the test fast; we just need to exceed whatever
    # value the module exposes.
    snap = tmp_path / "snap"
    snap.mkdir()
    cap = tokenizer_router_mod._TOKENIZER_BUNDLE_MAX_BYTES
    # tokenizer.json is on the allowlist, so the allowlist gate would let it
    # through; only the size cap should reject it.
    (snap / "tokenizer.json").write_bytes(b"\x00" * (cap + 1024))

    with pytest.raises(ValueError, match="exceeds cap"):
        tokenizer_router_mod._materialize_bundle(snap)
