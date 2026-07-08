# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encoding-correctness tests for the public-dataset local cache.

Regression guard for PR #826 (commit ef3a11460): the local-cache reads and
writes must use UTF-8 explicitly so non-ASCII dataset payloads round-trip on a
non-UTF-8 default locale (LANG=C, Windows cp1252). A wholesale port dropped the
``encoding="utf-8"`` on both ``open()`` calls; this file locks it back in.
"""

from pathlib import Path
from typing import Any

import pytest

from aiperf.dataset.loader.base_public_dataset import BasePublicDatasetLoader
from tests.unit.dataset.loader.conftest import _make_run

# A payload that cannot be encoded by the ASCII/latin-1 default codecs a
# non-UTF-8 locale would select: accented Latin, CJK, and an emoji.
NON_ASCII_PAYLOAD = 'café — naïve — 日本語 — \U0001f600 — {"k": "über"}'


class _StubPublicLoader(BasePublicDatasetLoader):
    """Minimal concrete loader exercising only the local-cache read/write path."""

    tag = "StubPublic"
    url = "https://example.invalid/dataset.json"
    filename = "stub_public_dataset.json"

    async def load_dataset(self) -> dict[str, Any]:
        raise NotImplementedError

    async def convert_to_conversations(self, dataset: dict[str, Any]) -> list:
        raise NotImplementedError


@pytest.fixture
async def stub_loader(default_user_config, tmp_path: Path) -> _StubPublicLoader:
    """A stub public loader whose cache lives under a per-test tmp dir.

    Async because ``AioHttpClient`` (built in the base ``__init__``) creates a
    ``TCPConnector`` that requires a running event loop.
    """
    loader = _StubPublicLoader(run=_make_run(default_user_config))
    loader.cache_filepath = tmp_path / _StubPublicLoader.filename
    yield loader
    await loader.http_client.close()


@pytest.mark.asyncio
class TestPublicDatasetCacheEncoding:
    async def test_save_then_load_round_trips_non_ascii(
        self, stub_loader: _StubPublicLoader
    ) -> None:
        """_save_to_local + _load_from_local preserve non-ASCII content exactly."""
        stub_loader._save_to_local(NON_ASCII_PAYLOAD)
        assert stub_loader._load_from_local() == NON_ASCII_PAYLOAD

    async def test_cache_file_is_written_as_utf8_bytes(
        self, stub_loader: _StubPublicLoader
    ) -> None:
        """The bytes on disk decode as UTF-8, regardless of the process locale."""
        stub_loader._save_to_local(NON_ASCII_PAYLOAD)
        raw = stub_loader.cache_filepath.read_bytes()
        assert raw.decode("utf-8") == NON_ASCII_PAYLOAD

    async def test_load_reads_utf8_bytes_written_out_of_band(
        self, stub_loader: _StubPublicLoader
    ) -> None:
        """_load_from_local decodes UTF-8 even when the file was written elsewhere."""
        stub_loader.cache_filepath.write_bytes(NON_ASCII_PAYLOAD.encode("utf-8"))
        assert stub_loader._load_from_local() == NON_ASCII_PAYLOAD

    async def test_cache_open_calls_pin_utf8_encoding(
        self, stub_loader: _StubPublicLoader, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both the write and read open() on the cache file pass encoding='utf-8'.

        Locale-independent guard: a bare open() would pass ``encoding=None`` and
        silently fall back to the platform default (ASCII/cp1252 under LANG=C /
        Windows), which is exactly the #826 regression. Asserting the kwarg
        catches the bug even on a UTF-8 CI host where a round-trip would pass.
        """
        import builtins

        real_open = builtins.open
        cache_open_encodings: list[str | None] = []

        def spy_open(file, mode="r", *args, **kwargs):
            if file == stub_loader.cache_filepath:
                cache_open_encodings.append(kwargs.get("encoding"))
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", spy_open)

        stub_loader._save_to_local(NON_ASCII_PAYLOAD)
        assert stub_loader._load_from_local() == NON_ASCII_PAYLOAD

        # One open for the write, one for the read; both must pin UTF-8.
        assert len(cache_open_encodings) == 2
        assert all(enc == "utf-8" for enc in cache_open_encodings), cache_open_encodings
