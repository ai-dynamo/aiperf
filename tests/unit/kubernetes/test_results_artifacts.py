# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.results_artifacts low-level helpers.

Focuses on behavior not exercised by tests/unit/kubernetes/test_results.py
(which covers the top-level retrieve_all_artifacts flow). Focus:
- _download_artifact unsafe-filename rejection (traversal, dotfile)
- _download_artifact X-Filename header basename stripping (defense-in-depth)
- _download_artifact content-length mismatch retry loop
- _download_artifact connection-error retries and final-failure None
- _download_artifact 404 short-circuit
- _list_available_artifacts malformed payload paths
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
import orjson
import pytest
from pytest import param

from aiperf.kubernetes.results_artifacts import (
    API_RESULTS_FILES_PATH,
    API_RESULTS_LIST_PATH,
    _download_artifact,
    _list_available_artifacts,
)

# ============================================================
# Fakes
# ============================================================


class FakeResponse:
    """Minimal fake aiohttp response with async context-manager support."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        json_data: dict | None = None,
        content_length: int | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self.headers = headers or {}
        self._json_data = json_data
        self._content_length = content_length

    @property
    def content_length(self) -> int | None:  # type: ignore[override]
        return self._content_length

    async def read(self) -> bytes:
        return self._body

    async def json(self) -> dict:
        if self._json_data is not None:
            return self._json_data
        return orjson.loads(self._body)

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=self.status,
                message="error",
            )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class FakeSession:
    """Fake aiohttp.ClientSession with per-URL response queues.

    Each queue is popped in order so retries receive different responses.
    """

    def __init__(self, queues: dict[str, list]) -> None:
        self._queues = {url: list(items) for url, items in queues.items()}
        self.get_calls: list[str] = []

    def get(self, url: str, **_kwargs):
        self.get_calls.append(url)
        items = self._queues.get(url)
        if not items:
            return FakeResponse(status=404)
        item = items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ============================================================
# _download_artifact — unsafe filename rejection
# ============================================================


class TestDownloadArtifactUnsafeFilename:
    """Verify server-provided filenames are sanitized before use."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "filename",
        [
            param(".hidden.json", id="dotfile"),
            param("", id="empty"),
            param("/", id="slash-only"),
        ],
    )  # fmt: skip
    async def test_unsafe_names_return_none_without_network(
        self, tmp_path: Path, filename: str
    ) -> None:
        session = FakeSession({})
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            filename,
            tmp_path,
        )
        assert result is None
        assert session.get_calls == []  # no HTTP attempt

    @pytest.mark.asyncio
    async def test_traversal_filename_downloads_basename_only(
        self, tmp_path: Path
    ) -> None:
        # ``Path("../../etc/passwd").name`` is ``passwd``; downloads as ``passwd``.
        session = FakeSession(
            {
                "http://localhost/api/passwd": [FakeResponse(body=b"stripped")],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "../../etc/passwd",
            tmp_path,
        )
        assert result == ("passwd", len(b"stripped"))
        assert (tmp_path / "passwd").read_bytes() == b"stripped"
        # Never attempted any path-traversal URL
        assert all("../" not in u for u in session.get_calls)


# ============================================================
# _download_artifact — X-Filename header
# ============================================================


class TestDownloadArtifactXFilenameHeader:
    """Verify X-Filename header is basename-stripped (defense-in-depth)."""

    @pytest.mark.asyncio
    async def test_x_filename_traversal_is_stripped(self, tmp_path: Path) -> None:
        # Malicious / buggy server returns an X-Filename with traversal.
        session = FakeSession(
            {
                "http://localhost/api/legit.json": [
                    FakeResponse(
                        body=b"data",
                        headers={"x-filename": "../../etc/passwd"},
                    )
                ],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "legit.json",
            tmp_path,
        )
        # Should fall back to basename ``passwd`` under output_dir, not escape.
        assert result is not None
        dest_name, _ = result
        assert dest_name == "passwd"
        # Written under tmp_path, not outside it.
        assert (tmp_path / "passwd").exists()

    @pytest.mark.asyncio
    async def test_x_filename_overrides_listing_name(self, tmp_path: Path) -> None:
        session = FakeSession(
            {
                "http://localhost/api/src": [
                    FakeResponse(
                        body=b"data",
                        headers={"x-filename": "renamed.json"},
                    )
                ],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "src",
            tmp_path,
        )
        assert result == ("renamed.json", 4)
        assert (tmp_path / "renamed.json").read_bytes() == b"data"

    @pytest.mark.asyncio
    async def test_x_filename_empty_basename_falls_back(self, tmp_path: Path) -> None:
        # X-Filename that basename-reduces to empty string falls back to source name.
        session = FakeSession(
            {
                "http://localhost/api/src.json": [
                    FakeResponse(
                        body=b"data",
                        headers={"x-filename": "/"},
                    )
                ],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "src.json",
            tmp_path,
        )
        assert result == ("src.json", 4)


# ============================================================
# _download_artifact — 404 short-circuit
# ============================================================


class TestDownloadArtifact404:
    """Verify 404 responses short-circuit without retries."""

    @pytest.mark.asyncio
    async def test_404_returns_none_without_retry(self, tmp_path: Path) -> None:
        session = FakeSession(
            {
                "http://localhost/api/gone.json": [FakeResponse(status=404)],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "gone.json",
            tmp_path,
            max_retries=2,
        )
        assert result is None
        # Exactly one attempt.
        assert session.get_calls == ["http://localhost/api/gone.json"]


# ============================================================
# _download_artifact — content-length mismatch retries
# ============================================================


class TestDownloadArtifactContentLengthMismatch:
    """Verify a short body is retried then dropped after exhaustion."""

    @pytest.mark.asyncio
    async def test_mismatch_then_success(self, tmp_path: Path) -> None:
        session = FakeSession(
            {
                "http://localhost/api/a.json": [
                    FakeResponse(body=b"short", content_length=10),
                    FakeResponse(body=b"0123456789", content_length=10),
                ],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "a.json",
            tmp_path,
            max_retries=2,
        )
        assert result == ("a.json", 10)
        assert (tmp_path / "a.json").read_bytes() == b"0123456789"
        assert len(session.get_calls) == 2

    @pytest.mark.asyncio
    async def test_mismatch_exhausts_retries_returns_none(self, tmp_path: Path) -> None:
        session = FakeSession(
            {
                "http://localhost/api/b.json": [
                    FakeResponse(body=b"a", content_length=10),
                    FakeResponse(body=b"b", content_length=10),
                    FakeResponse(body=b"c", content_length=10),
                ],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "b.json",
            tmp_path,
            max_retries=2,
        )
        assert result is None
        assert not (tmp_path / "b.json").exists()
        assert len(session.get_calls) == 3  # 1 + 2 retries

    @pytest.mark.asyncio
    async def test_no_content_length_skips_check(self, tmp_path: Path) -> None:
        # content_length=None must not trigger the mismatch path.
        session = FakeSession(
            {
                "http://localhost/api/c.json": [FakeResponse(body=b"abc")],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "c.json",
            tmp_path,
        )
        assert result == ("c.json", 3)


# ============================================================
# _download_artifact — connection / response errors
# ============================================================


class TestDownloadArtifactConnectionErrors:
    """Verify connection and response errors surface correctly."""

    @pytest.mark.asyncio
    async def test_connection_error_retries_then_none(self, tmp_path: Path) -> None:
        exc = aiohttp.ClientConnectionError("boom")
        session = FakeSession(
            {
                "http://localhost/api/a.json": [exc, exc, exc],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "a.json",
            tmp_path,
            max_retries=2,
        )
        assert result is None
        assert len(session.get_calls) == 3

    @pytest.mark.asyncio
    async def test_connection_error_then_success(self, tmp_path: Path) -> None:
        session = FakeSession(
            {
                "http://localhost/api/a.json": [
                    aiohttp.ClientConnectionError("boom"),
                    FakeResponse(body=b"ok"),
                ],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "a.json",
            tmp_path,
            max_retries=2,
        )
        assert result == ("a.json", 2)

    @pytest.mark.asyncio
    async def test_response_error_not_retried(self, tmp_path: Path) -> None:
        session = FakeSession(
            {
                "http://localhost/api/a.json": [FakeResponse(status=500)],
            }
        )
        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "a.json",
            tmp_path,
            max_retries=3,
        )
        assert result is None
        # 500 currently is caught at raise_for_status: one attempt only.
        assert len(session.get_calls) == 1


# ============================================================
# _list_available_artifacts
# ============================================================


class TestListAvailableArtifacts:
    """Verify JSON listing parsing and error handling."""

    @pytest.mark.asyncio
    async def test_returns_names_from_listing(self) -> None:
        list_url = f"http://localhost/api{API_RESULTS_LIST_PATH}"
        session = FakeSession(
            {
                list_url: [
                    FakeResponse(
                        json_data={
                            "files": [
                                {"name": "a.json"},
                                {"name": "b.txt"},
                            ]
                        }
                    )
                ]
            }
        )
        result = await _list_available_artifacts(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "job-1",
        )
        assert result == ["a.json", "b.txt"]

    @pytest.mark.asyncio
    async def test_empty_files_returns_empty_list(self) -> None:
        list_url = f"http://localhost/api{API_RESULTS_LIST_PATH}"
        session = FakeSession({list_url: [FakeResponse(json_data={"files": []})]})
        result = await _list_available_artifacts(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "job-1",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_files_key_returns_empty_list(self) -> None:
        list_url = f"http://localhost/api{API_RESULTS_LIST_PATH}"
        session = FakeSession({list_url: [FakeResponse(json_data={})]})
        result = await _list_available_artifacts(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "job-1",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_entry_missing_name_raises_keyerror_caught(self) -> None:
        # Entry without 'name' key triggers KeyError in listcomp;
        # helper catches KeyError and returns None.
        list_url = f"http://localhost/api{API_RESULTS_LIST_PATH}"
        session = FakeSession(
            {list_url: [FakeResponse(json_data={"files": [{"not_name": "x"}]})]}
        )
        result = await _list_available_artifacts(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "job-1",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_client_error_returns_none(self) -> None:
        list_url = f"http://localhost/api{API_RESULTS_LIST_PATH}"
        session = FakeSession({list_url: [aiohttp.ClientError("broken")]})
        result = await _list_available_artifacts(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "job-1",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_500_status_returns_none(self) -> None:
        list_url = f"http://localhost/api{API_RESULTS_LIST_PATH}"
        session = FakeSession({list_url: [FakeResponse(status=500)]})
        result = await _list_available_artifacts(
            session,  # type: ignore[arg-type]
            "http://localhost/api",
            "job-1",
        )
        assert result is None


# ============================================================
# Module constants
# ============================================================


class TestModuleConstants:
    """Verify the module exposes the expected API path constants."""

    def test_api_paths_are_prefixed(self) -> None:
        assert API_RESULTS_FILES_PATH.startswith("/api/")
        assert API_RESULTS_LIST_PATH.startswith("/api/")
        assert API_RESULTS_FILES_PATH != API_RESULTS_LIST_PATH
