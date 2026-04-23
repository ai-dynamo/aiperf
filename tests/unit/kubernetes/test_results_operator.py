# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for aiperf.kubernetes.results_operator low-level helpers.

Focuses on behavior not exercised by tests/unit/kubernetes/test_results.py.
Focus:
- _download_operator_file unsafe-filename rejection
- _download_operator_file 404 returns None
- _download_operator_file client error is swallowed and returns None
- _verify_operator_health status code + connection-error branches
- _list_operator_files empty/missing payload handling
- RESULTS_SERVER_PORT default
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import aiohttp
import orjson
import pytest
from pytest import param

from aiperf.kubernetes.results_operator import (
    RESULTS_SERVER_PORT,
    _download_operator_file,
    _list_operator_files,
    _verify_operator_health,
)

# ============================================================
# Fakes
# ============================================================


class _Chunks:
    """Fake response content with iter_chunked support."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, _chunk_size: int):
        for chunk in self._chunks:
            yield chunk


class FakeResponse:
    """Minimal fake aiohttp response (async context manager)."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b"",
        json_data: dict | None = None,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self._json_data = json_data
        self.headers = headers or {}
        # Mirror results.py iter_chunked contract
        self.content = _Chunks(chunks if chunks is not None else [body])

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
    """Fake aiohttp.ClientSession with per-URL response queues."""

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

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ============================================================
# _download_operator_file — unsafe filename rejection
# ============================================================


class TestDownloadOperatorFileUnsafe:
    """Verify server-provided filenames are sanitized before use."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "display_name",
        [
            param(".hidden", id="dotfile"),
            param("", id="empty"),
            param("/", id="slash-only"),
        ],
    )  # fmt: skip
    async def test_unsafe_names_rejected_without_network(
        self, tmp_path: Path, display_name: str
    ) -> None:
        session = FakeSession({})
        result = await _download_operator_file(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
            file_info={"name": display_name},
            output_dir=tmp_path,
        )
        assert result is None
        assert session.get_calls == []

    @pytest.mark.asyncio
    async def test_traversal_reduces_to_basename(self, tmp_path: Path) -> None:
        # ``Path("../../etc/passwd").name`` is ``passwd``; downloaded safely.
        session = FakeSession(
            {
                "http://localhost/api/v1/results/ns/job-1/passwd": [
                    FakeResponse(
                        body=b"data",
                        chunks=[b"data"],
                        headers={"Content-Encoding": "identity"},
                    )
                ],
            }
        )
        result = await _download_operator_file(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
            file_info={"name": "../../etc/passwd"},
            output_dir=tmp_path,
        )
        assert result == ("passwd", 4)
        assert (tmp_path / "passwd").read_bytes() == b"data"
        # Never attempted traversal URL
        assert all(".." not in u for u in session.get_calls)


# ============================================================
# _download_operator_file — HTTP status handling
# ============================================================


class TestDownloadOperatorFileStatus:
    """Verify HTTP status code handling."""

    @pytest.mark.asyncio
    async def test_404_returns_none(self, tmp_path: Path) -> None:
        session = FakeSession(
            {
                "http://localhost/api/v1/results/ns/job-1/a.json": [
                    FakeResponse(status=404)
                ],
            }
        )
        result = await _download_operator_file(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
            file_info={"name": "a.json"},
            output_dir=tmp_path,
        )
        assert result is None
        assert not (tmp_path / "a.json").exists()

    @pytest.mark.asyncio
    async def test_500_returns_none(self, tmp_path: Path) -> None:
        session = FakeSession(
            {
                "http://localhost/api/v1/results/ns/job-1/a.json": [
                    FakeResponse(status=500)
                ],
            }
        )
        result = await _download_operator_file(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
            file_info={"name": "a.json"},
            output_dir=tmp_path,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_client_error_returns_none(self, tmp_path: Path) -> None:
        session = FakeSession(
            {
                "http://localhost/api/v1/results/ns/job-1/a.json": [
                    aiohttp.ClientError("broken")
                ],
            }
        )
        result = await _download_operator_file(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
            file_info={"name": "a.json"},
            output_dir=tmp_path,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_success_identity_encoding(self, tmp_path: Path) -> None:
        content = b'{"m": 1}'
        session = FakeSession(
            {
                "http://localhost/api/v1/results/ns/job-1/a.json": [
                    FakeResponse(
                        body=content,
                        chunks=[content],
                        headers={"Content-Encoding": "identity"},
                    )
                ],
            }
        )
        result = await _download_operator_file(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
            file_info={"name": "a.json"},
            output_dir=tmp_path,
        )
        assert result == ("a.json", len(content))
        assert (tmp_path / "a.json").read_bytes() == content

    @pytest.mark.asyncio
    async def test_success_no_encoding_header_defaults_to_identity(
        self, tmp_path: Path
    ) -> None:
        # When the server does not set Content-Encoding, default is 'identity'.
        content = b"plain"
        session = FakeSession(
            {
                "http://localhost/api/v1/results/ns/job-1/a.json": [
                    FakeResponse(body=content, chunks=[content])
                ],
            }
        )
        result = await _download_operator_file(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
            file_info={"name": "a.json"},
            output_dir=tmp_path,
        )
        assert result == ("a.json", len(content))
        assert (tmp_path / "a.json").read_bytes() == content


# ============================================================
# _verify_operator_health
# ============================================================


class TestVerifyOperatorHealth:
    """Verify operator health-check behavior."""

    @pytest.mark.asyncio
    async def test_healthy_returns_true(self) -> None:
        from unittest.mock import patch

        session = FakeSession({"http://localhost/healthz": [FakeResponse(status=200)]})
        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            ok = await _verify_operator_health("http://localhost")
        assert ok is True

    @pytest.mark.asyncio
    async def test_non_200_returns_false(self) -> None:
        from unittest.mock import patch

        session = FakeSession({"http://localhost/healthz": [FakeResponse(status=503)]})
        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            ok = await _verify_operator_health("http://localhost")
        assert ok is False

    @pytest.mark.asyncio
    async def test_client_error_returns_false(self) -> None:
        from unittest.mock import patch

        session = FakeSession(
            {"http://localhost/healthz": [aiohttp.ClientError("down")]}
        )
        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            ok = await _verify_operator_health("http://localhost")
        assert ok is False


# ============================================================
# _list_operator_files
# ============================================================


class TestListOperatorFiles:
    """Verify listing helper handles empty / error payloads."""

    @pytest.mark.asyncio
    async def test_returns_file_dicts(self) -> None:
        list_url = "http://localhost/api/v1/results/ns/job-1"
        session = FakeSession(
            {
                list_url: [
                    FakeResponse(
                        json_data={
                            "namespace": "ns",
                            "job_id": "job-1",
                            "files": [{"name": "a.json"}, {"name": "b.json"}],
                        }
                    )
                ],
            }
        )
        result = await _list_operator_files(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
        )
        assert result == [{"name": "a.json"}, {"name": "b.json"}]

    @pytest.mark.asyncio
    async def test_404_returns_none(self) -> None:
        list_url = "http://localhost/api/v1/results/ns/job-1"
        session = FakeSession({list_url: [FakeResponse(status=404)]})
        result = await _list_operator_files(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_500_returns_none(self) -> None:
        list_url = "http://localhost/api/v1/results/ns/job-1"
        session = FakeSession({list_url: [FakeResponse(status=500)]})
        result = await _list_operator_files(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_client_error_returns_none(self) -> None:
        list_url = "http://localhost/api/v1/results/ns/job-1"
        session = FakeSession({list_url: [aiohttp.ClientError("boom")]})
        result = await _list_operator_files(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_files_list_returns_none(self) -> None:
        list_url = "http://localhost/api/v1/results/ns/job-1"
        session = FakeSession({list_url: [FakeResponse(json_data={"files": []})]})
        result = await _list_operator_files(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_missing_files_key_returns_none(self) -> None:
        list_url = "http://localhost/api/v1/results/ns/job-1"
        session = FakeSession({list_url: [FakeResponse(json_data={})]})
        result = await _list_operator_files(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
        )
        assert result is None


# ============================================================
# Module constants
# ============================================================


class TestModuleConstants:
    """Verify exported module constants."""

    def test_results_server_port_default(self) -> None:
        # The sidecar container port shipped in the Helm chart.
        assert RESULTS_SERVER_PORT == 8081
