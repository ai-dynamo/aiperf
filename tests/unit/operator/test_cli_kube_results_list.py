# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``aiperf kube results list-runs``.

The command opens an operator pod port-forward and hits
``/api/v1/results/<ns>/<name>/runs``. Tests here mock the table renderer
directly with fixture payloads, and mock the HTTP + port-forward chain for
end-to-end command coverage.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest

from aiperf.cli_commands.kube.results import _print_runs_table, list_runs
from aiperf.config.kube import KubeManageOptions


@pytest.fixture
def sample_payload() -> dict:
    """Response payload shape from ``/api/v1/results/<ns>/<name>/runs``."""
    return {
        "namespace": "default",
        "job_id": "foo",
        "latest_epoch": "1714150923",
        "runs": [
            {
                "epoch": "1714150923",
                "mtime_epoch": 1714150925,
                "file_count": 7,
                "total_size_bytes": 4823912,
                "is_latest": True,
            },
            {
                "epoch": "1714064523",
                "mtime_epoch": 1714064525,
                "file_count": 7,
                "total_size_bytes": 4823912,
                "is_latest": False,
            },
        ],
    }


# =============================================================================
# _print_runs_table: renderer-in-isolation tests
# =============================================================================


class TestPrintRunsTable:
    """Text-formatting helper can be unit-tested without any k8s/HTTP mocks."""

    def test_renders_all_rows(self, sample_payload: dict, capsys) -> None:
        from aiperf.kubernetes.console import console as _console

        _console.width = 200
        try:
            _print_runs_table(sample_payload)
        finally:
            _console.width = None

        out = capsys.readouterr().out
        assert "EPOCH" in out
        assert "TIMESTAMP" in out
        assert "FILES" in out
        assert "SIZE" in out
        assert "LATEST" in out
        assert "1714150923" in out
        assert "1714064523" in out
        # Human-readable size and UTC-formatted timestamp
        assert "4.6 MiB" in out
        assert "2024-04-26" in out

    def test_empty_runs_prints_info_message(self, capsys) -> None:
        _print_runs_table({"namespace": "default", "job_id": "bar", "runs": []})
        out = capsys.readouterr().out
        assert "No runs found for default/bar" in out

    def test_marks_only_latest_row(self, sample_payload: dict, capsys) -> None:
        from aiperf.kubernetes.console import console as _console

        _console.width = 200
        try:
            _print_runs_table(sample_payload)
        finally:
            _console.width = None

        out = capsys.readouterr().out
        # Checkmark rendered for exactly one row
        assert out.count("✓") == 1


# =============================================================================
# list_runs: full-command tests (HTTP + port-forward mocked)
# =============================================================================


def _mock_http_response(*, status: int = 200, json_payload: dict | None = None):
    """Return an ``aiohttp.ClientResponse``-shaped async-context mock."""
    resp = MagicMock()
    resp.status = status
    resp.raise_for_status = MagicMock()
    if status >= 400:
        resp.raise_for_status.side_effect = RuntimeError(f"HTTP {status}")
    resp.json = AsyncMock(return_value=json_payload or {})

    @asynccontextmanager
    async def _as_ctx():
        yield resp

    return _as_ctx


def _mock_session_cm(get_cm):
    session = MagicMock()
    session.get = MagicMock(return_value=get_cm())

    @asynccontextmanager
    async def _as_ctx(*_args, **_kwargs):
        yield session

    return _as_ctx


@asynccontextmanager
async def _mock_port_forward(*_args, **_kwargs):
    yield 12345


@pytest.fixture
def mock_resolve_and_pod():
    """Mock ``resolve_job`` + ``find_operator_pod`` for the list_runs flow."""
    resolved = MagicMock()
    resolved.job_id = "foo"
    resolved.namespace = "default"
    resolved.api = MagicMock()

    with (
        patch(
            "aiperf.kubernetes.cli_helpers.resolve_job",
            new=AsyncMock(return_value=resolved),
        ),
        patch(
            "aiperf.kubernetes.client.find_operator_pod",
            new=AsyncMock(return_value=("operator-pod-x", "Running")),
        ),
        patch(
            "aiperf.kubernetes.port_forward.port_forward_with_status",
            new=_mock_port_forward,
        ),
    ):
        yield resolved


@pytest.mark.asyncio
async def test_list_runs_text_output_formats_table(
    mock_resolve_and_pod, sample_payload: dict, capsys
) -> None:
    get_cm = _mock_http_response(status=200, json_payload=sample_payload)
    session_cm = _mock_session_cm(get_cm)

    from aiperf.kubernetes.console import console as _console

    _console.width = 200
    try:
        with patch("aiohttp.ClientSession", new=session_cm):
            await list_runs(
                job_id="foo",
                manage_options=KubeManageOptions(),
                output="text",
            )
    finally:
        _console.width = None

    out = capsys.readouterr().out
    assert "EPOCH" in out
    assert "1714150923" in out
    assert "4.6 MiB" in out


@pytest.mark.asyncio
async def test_list_runs_json_output_parseable(
    mock_resolve_and_pod, sample_payload: dict, capsys
) -> None:
    get_cm = _mock_http_response(status=200, json_payload=sample_payload)
    session_cm = _mock_session_cm(get_cm)

    from aiperf.kubernetes.console import console as _console

    _console.width = 200
    try:
        with patch("aiohttp.ClientSession", new=session_cm):
            await list_runs(
                job_id="foo",
                manage_options=KubeManageOptions(),
                output="json",
            )
    finally:
        _console.width = None

    out = capsys.readouterr().out.strip()
    # Rich prints the JSON; strip leading/trailing whitespace, then parse
    parsed = orjson.loads(out)
    assert parsed["namespace"] == "default"
    assert parsed["job_id"] == "foo"
    assert len(parsed["runs"]) == 2


@pytest.mark.asyncio
async def test_list_runs_404_raises_informative_error(
    mock_resolve_and_pod, capsys
) -> None:
    get_cm = _mock_http_response(status=404, json_payload={})
    session_cm = _mock_session_cm(get_cm)

    with (
        patch("aiohttp.ClientSession", new=session_cm),
        pytest.raises(SystemExit),
    ):
        await list_runs(
            job_id="ghost",
            manage_options=KubeManageOptions(),
            output="text",
        )

    out = capsys.readouterr().out
    assert "No runs found" in out
