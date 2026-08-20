# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``download_tokenizer``.

Spins up a local ``aiohttp.web`` server via ``AppRunner`` + ``TCPSite``
because this project does not depend on ``pytest-aiohttp`` (so the
``aiohttp_server`` fixture from the plan is not available).
"""

from __future__ import annotations

import io
import logging
import socket
import sys
import tarfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import zstandard
from aiohttp import web

from aiperf.common.constants import IS_WINDOWS
from aiperf.workers import worker_pod_tokenizer_download
from aiperf.workers.worker_pod_tokenizer_download import (
    _bundle_lock,
    _extract_bundle,
    download_tokenizer,
)


def _make_bundle(payload_files: dict[str, bytes]) -> bytes:
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w") as tf:
        for name, data in payload_files.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return zstandard.ZstdCompressor().compress(tar_buf.getvalue())


@asynccontextmanager
async def _stub_server() -> AsyncIterator[tuple[str, dict]]:
    """Start a local aiohttp server and yield (base_url, mutable state).

    Adapted from the plan's ``aiohttp_server`` fixture (pytest-aiohttp is not
    a project dependency); behaviour is identical.
    """
    state: dict = {"requests": 0, "fail_first_n": 0, "bundle": b"", "tokenizer": "gpt2"}

    async def handler(request: web.Request) -> web.Response:
        state["requests"] += 1
        name = request.match_info["name"]
        if name != state["tokenizer"]:
            return web.Response(status=404)
        if state["requests"] <= state["fail_first_n"]:
            return web.Response(status=503, headers={"Retry-After": "1"})
        return web.Response(body=state["bundle"], content_type="application/zstd")

    app = web.Application()
    app.router.add_get("/api/tokenizer/{name:.+}/bundle", handler)

    runner = web.AppRunner(app)
    await runner.setup()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    try:
        yield f"http://127.0.0.1:{port}", state
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
async def test_happy_path(tmp_path: Path) -> None:
    async with _stub_server() as (base_url, state):
        state["bundle"] = _make_bundle({"tokenizer.json": b'{"v":1}'})
        out = await download_tokenizer(
            api_base_url=base_url,
            name="gpt2",
            dest_root=tmp_path,
            max_retries=3,
            logger=logging.getLogger("test"),
        )
    assert (out / "tokenizer.json").read_text() == '{"v":1}'


@pytest.mark.asyncio
async def test_503_then_success(tmp_path: Path) -> None:
    async with _stub_server() as (base_url, state):
        state["bundle"] = _make_bundle({"tokenizer.json": b"{}"})
        state["fail_first_n"] = 2
        out = await download_tokenizer(
            api_base_url=base_url,
            name="gpt2",
            dest_root=tmp_path,
            max_retries=5,
            logger=logging.getLogger("test"),
        )
        assert state["requests"] == 3
    assert (out / "tokenizer.json").exists()


@pytest.mark.asyncio
async def test_404_raises(tmp_path: Path) -> None:
    async with _stub_server() as (base_url, _):
        with pytest.raises(RuntimeError, match="404"):
            await download_tokenizer(
                api_base_url=base_url,
                name="not-registered",
                dest_root=tmp_path,
                max_retries=3,
                logger=logging.getLogger("test"),
            )


@pytest.mark.asyncio
async def test_url_encoded_org_slash_model(tmp_path: Path) -> None:
    async with _stub_server() as (base_url, state):
        state["tokenizer"] = "meta-llama/Llama-3.1-8B"
        state["bundle"] = _make_bundle({"tokenizer.json": b"{}"})
        out = await download_tokenizer(
            api_base_url=base_url,
            name="meta-llama/Llama-3.1-8B",
            dest_root=tmp_path,
            max_retries=3,
            logger=logging.getLogger("test"),
        )
    # Slug uses URL-quoted form so the on-disk dir is unambiguous.
    assert out.name == "meta-llama%2FLlama-3.1-8B"
    assert (out / "tokenizer.json").exists()


@pytest.mark.asyncio
async def test_extract_crash_then_retry_succeeds(tmp_path: Path, monkeypatch) -> None:
    """A crash during extraction must not leave a partial bundle dir."""
    from aiperf.workers import worker_pod_tokenizer_download as wptd

    async with _stub_server() as (base_url, state):
        state["bundle"] = _make_bundle(
            {"tokenizer.json": b'{"v":1}', "vocab.json": b"{}"}
        )

        real_extract = wptd._extract_bundle
        calls = {"n": 0}

        def crashing_extract(compressed: bytes, dest: Path) -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate a partial extract: write one file, then raise.
                (dest / "tokenizer.json").write_bytes(b'{"v":1}')
                raise RuntimeError("simulated extract crash")
            real_extract(compressed, dest)

        monkeypatch.setattr(wptd, "_extract_bundle", crashing_extract)

        # First attempt crashes mid-extract; the helper raises.
        with pytest.raises(RuntimeError, match="simulated"):
            await wptd.download_tokenizer(
                api_base_url=base_url,
                name="gpt2",
                dest_root=tmp_path,
                max_retries=1,
                logger=logging.getLogger("test"),
            )

        # No half-state left at the final dest.
        final = tmp_path / wptd.slug_for_tokenizer("gpt2")
        assert not final.exists() or not any(final.iterdir()), (
            f"extract crash left partial files at {final}"
        )

        # Second attempt (real extractor) succeeds.
        out = await wptd.download_tokenizer(
            api_base_url=base_url,
            name="gpt2",
            dest_root=tmp_path,
            max_retries=2,
            logger=logging.getLogger("test"),
        )
    assert (out / "tokenizer.json").read_bytes() == b'{"v":1}'
    assert (out / "vocab.json").exists()
    assert (out / ".ready").exists()


@pytest.mark.asyncio
async def test_bundle_lock_windows_is_noop_without_fcntl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Windows the lock degrades to a no-op and must never import fcntl."""
    monkeypatch.setattr(worker_pod_tokenizer_download, "IS_WINDOWS", True)
    # Simulate a platform without the module: any ``import fcntl`` raises.
    monkeypatch.setitem(sys.modules, "fcntl", None)

    lock_path = tmp_path / "gpt2.lock"
    lock = _bundle_lock(lock_path)
    async with lock:
        assert lock._fd is None
    assert not lock_path.exists()


@pytest.mark.asyncio
@pytest.mark.skipif(IS_WINDOWS, reason="fcntl.flock only exists on POSIX")
async def test_bundle_lock_posix_acquires_and_releases(tmp_path: Path) -> None:
    lock_path = tmp_path / "gpt2.lock"
    lock = _bundle_lock(lock_path)
    async with lock:
        assert lock._fd is not None
        assert lock_path.exists()
    assert lock._fd is None


def test_extract_rejects_path_traversal(tmp_path: Path) -> None:
    """A ``../`` tar member (tar-slip / CWE-22) must not write outside dest.

    Runs through the real ``_extract_bundle`` path (which uses PEP 706's
    ``filter="data"`` on 3.11.4+/3.12+): extraction must reject the escaping
    member and leave nothing in the parent of the destination dir.
    """
    dest = tmp_path / "dest"
    dest.mkdir()
    bundle = _make_bundle({"../escaped.txt": b"pwned"})

    with pytest.raises(tarfile.TarError):
        _extract_bundle(bundle, dest)

    assert not (tmp_path / "escaped.txt").exists()


def test_extract_rejects_path_traversal_pre_pep706_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the 3.11.0-3.11.3 (pre-PEP-706) branch and assert it still blocks.

    Deleting ``tarfile.data_filter`` makes ``hasattr`` return False, exercising
    the manual per-member resolved-path validation fallback that guards the
    handful of patch releases predating the ``data`` filter backport.
    """
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    dest = tmp_path / "dest"
    dest.mkdir()
    bundle = _make_bundle({"../escaped.txt": b"pwned"})

    with pytest.raises(tarfile.TarError, match="escapes extraction dir"):
        _extract_bundle(bundle, dest)

    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_extract_fallback_extracts_safe_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The manual fallback must still extract legitimate (non-escaping) files.

    Forcing the pre-PEP-706 branch on a modern interpreter routes through the
    unfiltered ``extractall`` (guarded by the manual validation above it),
    which emits the 3.14 filter DeprecationWarning; that warning is irrelevant
    to real 3.11.0-3.11.3 targets, so it is suppressed here.
    """
    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    dest = tmp_path / "dest"
    dest.mkdir()
    bundle = _make_bundle({"tokenizer.json": b'{"v":1}', "nested/vocab.json": b"{}"})

    _extract_bundle(bundle, dest)

    assert (dest / "tokenizer.json").read_bytes() == b'{"v":1}'
    assert (dest / "nested" / "vocab.json").exists()
