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
    _download_all_operator_files,
    _download_operator_file,
    _list_operator_files,
    _resolve_operator_run,
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
        self.json_loads = None
        # Mirror results.py iter_chunked contract
        self.content = _Chunks(chunks if chunks is not None else [body])

    async def read(self) -> bytes:
        return self._body

    async def json(self, *, loads=None) -> dict:
        self.json_loads = loads
        if self._json_data is not None:
            return self._json_data
        parser = loads or orjson.loads
        return parser(self._body)

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
    async def test_traversal_refuses_without_network(self, tmp_path: Path) -> None:
        # A traversal name must be rejected, never collapsed to its basename.
        session = FakeSession({})
        result = await _download_operator_file(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
            file_info={"name": "../../etc/passwd"},
            output_dir=tmp_path,
        )
        assert result is None
        assert session.get_calls == []
        assert list(tmp_path.rglob("*")) == []

    @pytest.mark.asyncio
    async def test_nested_checkpoint_preserves_relative_path(
        self, tmp_path: Path
    ) -> None:
        # Checkpoint files arrive with nested names and must be requested and
        # written with the full API-visible relative path preserved.
        session = FakeSession(
            {
                "http://localhost/api/v1/results/ns/job-1/checkpoints/records-0.parquet": [
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
            file_info={"name": "checkpoints/records-0.parquet"},
            output_dir=tmp_path,
        )
        assert result == ("checkpoints/records-0.parquet", 4)
        assert (tmp_path / "checkpoints" / "records-0.parquet").read_bytes() == b"data"


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
        response = FakeResponse(
            body=b'{"namespace":"ns","job_id":"job-1","files":[{"name":"a.json"},{"name":"b.json"}]}'
        )
        session = FakeSession({list_url: [response]})
        result = await _list_operator_files(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
        )
        assert result == [{"name": "a.json"}, {"name": "b.json"}]
        assert response.json_loads is orjson.loads

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
    async def test_not_ready_empty_files_warns_missing_ready_marker(self) -> None:
        from unittest.mock import patch

        list_url = "http://localhost/api/v1/results/ns/job-1/runs/1714150923"
        session = FakeSession(
            {list_url: [FakeResponse(json_data={"ready": False, "files": []})]}
        )

        with patch("aiperf.kubernetes.results_operator.print_warning") as warning:
            result = await _list_operator_files(
                session,  # type: ignore[arg-type]
                api_base="http://localhost",
                namespace="ns",
                job_id="job-1",
                run="1714150923",
            )

        assert result is None
        warning.assert_called_once()
        assert ".aiperf_results_ready.json is missing" in warning.call_args.args[0]

    @pytest.mark.asyncio
    async def test_not_ready_checkpoint_files_warns_missing_ready_marker(self) -> None:
        from unittest.mock import patch

        list_url = "http://localhost/api/v1/results/ns/job-1/runs/1714150923"
        session = FakeSession(
            {
                list_url: [
                    FakeResponse(
                        json_data={
                            "ready": False,
                            "files": [{"name": "checkpoints/phase-1.json"}],
                        }
                    )
                ]
            }
        )

        with patch("aiperf.kubernetes.results_operator.print_warning") as warning:
            result = await _list_operator_files(
                session,  # type: ignore[arg-type]
                api_base="http://localhost",
                namespace="ns",
                job_id="job-1",
                run="1714150923",
            )

        assert result is None
        warning.assert_called_once()
        assert ".aiperf_results_ready.json is missing" in warning.call_args.args[0]

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


class TestDownloadAllOperatorFiles:
    """Verify end-to-end operator file listing/download URL selection."""

    @pytest.mark.asyncio
    async def test_resolve_operator_run_uses_orjson_parser(self) -> None:
        response = FakeResponse(
            body=b'{"latest_epoch":"1714150923"}',
        )
        session = FakeSession(
            {"http://localhost/api/v1/results/ns/job-1/runs": [response]}
        )

        latest = await _resolve_operator_run(
            session,  # type: ignore[arg-type]
            api_base="http://localhost",
            namespace="ns",
            job_id="job-1",
            run=None,
        )

        assert latest == "1714150923"
        assert response.json_loads is orjson.loads

    @pytest.mark.asyncio
    async def test_default_download_resolves_latest_epoch_before_listing(
        self, tmp_path: Path
    ) -> None:
        session = FakeSession(
            {
                "http://localhost/api/v1/results/ns/job-1/runs": [
                    FakeResponse(json_data={"latest_epoch": "1714150923", "runs": []})
                ],
                "http://localhost/api/v1/results/ns/job-1/runs/1714150923": [
                    FakeResponse(json_data={"files": [{"name": "a.json"}]})
                ],
                "http://localhost/api/v1/results/ns/job-1/runs/1714150923/a.json": [
                    FakeResponse(
                        body=b"{}",
                        chunks=[b"{}"],
                        headers={"Content-Encoding": "identity"},
                    )
                ],
            }
        )

        from unittest.mock import patch

        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=None,
            ),
        ):
            downloaded = await _download_all_operator_files(
                api_base="http://localhost",
                namespace="ns",
                job_id="job-1",
                output_dir=tmp_path,
            )

        assert downloaded == [("a.json", 2)]
        assert "http://localhost/api/v1/results/ns/job-1" not in session.get_calls
        assert all(
            "/runs/1714150923" in url or url.endswith("/runs")
            for url in session.get_calls
        )


class TestModuleConstants:
    """Verify exported module constants."""

    def test_results_server_port_default(self) -> None:
        # The sidecar container port shipped in the Helm chart.
        assert RESULTS_SERVER_PORT == 8081

    def test_results_server_port_env_override(self, monkeypatch) -> None:
        """``AIPERF_RESULTS_SERVER_PORT`` env override propagates at module-load.

        Same env var the operator's results-server reads
        (``aiperf.operator.results_server:SERVER_PORT``) — single source of
        truth so a non-default chart install (e.g. ``--set
        resultsServer.port=9001``) plus ``export
        AIPERF_RESULTS_SERVER_PORT=9001`` in the user's shell makes both
        the server bind and the CLI port-forward target match.
        """
        import importlib

        monkeypatch.setenv("AIPERF_RESULTS_SERVER_PORT", "9001")
        from aiperf.kubernetes import results_operator

        importlib.reload(results_operator)
        try:
            assert results_operator.RESULTS_SERVER_PORT == 9001
        finally:
            monkeypatch.delenv("AIPERF_RESULTS_SERVER_PORT", raising=False)
            importlib.reload(results_operator)
