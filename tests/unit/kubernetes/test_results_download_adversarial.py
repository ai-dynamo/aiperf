# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes results download workflows.

Focuses on:
- path traversal and reserved ready-marker names supplied by result servers;
- corrupt compressed response bodies and cleanup/idempotence on failed writes;
- URL path segment encoding for namespace/job/run/file identifiers;
- sweep aggregate artifact nested-path allowlisting.

Out of scope: live Kubernetes API discovery and pod port-forward lifecycles; those
are covered by ``tests/unit/kubernetes/test_results.py`` and CLI target-resolution
regression tests.
"""

from __future__ import annotations

import gzip
import zlib
from pathlib import Path
from typing import Self
from urllib.parse import quote

import aiohttp
import pytest
from pytest import param

from aiperf.common.results_markers import READY_MARKER_NAME
from aiperf.kubernetes.results_artifacts import _download_artifact
from aiperf.kubernetes.results_operator import (
    _download_and_decompress,
    _download_operator_file,
    _download_sweep_operator_file,
)

# ============================================================
# Helpers
# ============================================================


class _FakeStreamContent:
    """Fake aiohttp stream content for chunked download tests."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, _chunk_size: int):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    """Minimal async-context response surface used by download helpers."""

    def __init__(
        self,
        *,
        body: bytes = b"",
        status: int = 200,
        headers: dict[str, str] | None = None,
        content_length: int | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body
        self._content_length = content_length
        self.content = _FakeStreamContent([body])

    @property
    def content_length(self) -> int | None:
        return self._content_length

    async def read(self) -> bytes:
        return self._body

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=None,
                history=(),
                status=self.status,
                message="operator results server rejected request",
            )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _RecordingSession:
    """Fake ClientSession that records requested URLs and request headers."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.urls: list[str] = []
        self.headers: list[dict[str, str] | None] = []

    def get(self, url: str, headers: dict[str, str] | None = None) -> _FakeResponse:
        self.urls.append(url)
        self.headers.append(headers)
        return self._response


async def _collect_downloaded_sweep_artifact(
    tmp_path: Path, display_name: str
) -> tuple[tuple[str, int] | None, _RecordingSession]:
    session = _RecordingSession(_FakeResponse(body=b"aggregate"))
    result = await _download_sweep_operator_file(
        session,
        api_base="http://localhost:31081",
        namespace="bench-prod",
        sweep_name="llama3-sweep",
        run="1770000000",
        file_info={"name": display_name},
        output_dir=tmp_path,
    )
    return result, session


# ============================================================
# Path traversal and ready-marker trust boundaries
# ============================================================


class TestOperatorDownloadPathTraversal:
    """Operator-backed downloads treat server-provided names as untrusted input."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "display_name",
        [
            param("../metrics.json", id="parent-traversal"),
            param("../../tenant-b/secret.json", id="multi-parent-traversal"),
            param("checkpoints/../../secret.json", id="nested-parent-traversal"),
            param("/etc/passwd", id="absolute-path"),
            param("checkpoints/.hidden.parquet", id="hidden-nested-leaf"),
        ],
    )  # fmt: skip
    async def test_download_operator_file_traversal_display_name_refuses_request(
        self, tmp_path: Path, display_name: str
    ) -> None:
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        session = _RecordingSession(_FakeResponse(body=b"{}"))

        result = await _download_operator_file(
            session,
            api_base="http://localhost:31081",
            namespace="bench-prod",
            job_id="latency-bench-7f2a",
            file_info={"name": display_name},
            output_dir=output_dir,
            run="1770000000",
        )

        assert result is None
        assert session.urls == []
        assert list(output_dir.rglob("*")) == []

    @pytest.mark.asyncio
    async def test_download_operator_file_nested_checkpoint_writes_under_output_dir(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        session = _RecordingSession(_FakeResponse(body=b"{}"))

        result = await _download_operator_file(
            session,
            api_base="http://localhost:31081",
            namespace="bench-prod",
            job_id="latency-bench-7f2a",
            file_info={"name": "checkpoints/records-0.parquet"},
            output_dir=output_dir,
            run="1770000000",
        )

        assert result == ("checkpoints/records-0.parquet", 2)
        assert (output_dir / "checkpoints" / "records-0.parquet").read_bytes() == b"{}"
        assert session.urls == [
            "http://localhost:31081/api/v1/results/bench-prod/latency-bench-7f2a/runs/1770000000/checkpoints/records-0.parquet"
        ]

    @pytest.mark.asyncio
    async def test_download_operator_file_ready_marker_name_refuses_write(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        session = _RecordingSession(_FakeResponse(body=b"{}"))

        result = await _download_operator_file(
            session,
            api_base="http://localhost:31081",
            namespace="bench-prod",
            job_id="latency-bench-7f2a",
            file_info={"name": READY_MARKER_NAME},
            output_dir=output_dir,
            run="1770000000",
        )

        assert result is None
        assert session.urls == []
        assert (output_dir / READY_MARKER_NAME).exists() is False

    @pytest.mark.asyncio
    async def test_download_artifact_x_filename_traversal_collapses_to_basename(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        session = _RecordingSession(
            _FakeResponse(body=b"safe", headers={"x-filename": "../../outside.txt"})
        )

        result = await _download_artifact(
            session,
            "http://localhost:31090/api/results/files",
            "metrics.json",
            output_dir,
        )

        assert result == ("outside.txt", 4)
        assert (output_dir / "outside.txt").read_bytes() == b"safe"
        assert (tmp_path / "outside.txt").exists() is False

    @pytest.mark.asyncio
    async def test_download_artifact_x_filename_ready_marker_refuses_write(
        self, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "downloads"
        output_dir.mkdir()
        session = _RecordingSession(
            _FakeResponse(body=b"{}", headers={"x-filename": READY_MARKER_NAME})
        )

        result = await _download_artifact(
            session,
            "http://localhost:31090/api/results/files",
            "metrics.json",
            output_dir,
        )

        assert result is None
        assert (output_dir / READY_MARKER_NAME).exists() is False


# ============================================================
# URL encoding contract
# ============================================================


class TestOperatorDownloadUrlEncoding:
    """Every untrusted URL path segment is percent-encoded before HTTP GET."""

    @pytest.mark.asyncio
    async def test_download_operator_file_namespace_job_run_and_file_are_encoded(
        self, tmp_path: Path
    ) -> None:
        session = _RecordingSession(_FakeResponse(body=b"{}"))
        namespace = "bench-prod"
        job_id = "llama.3-throughput"
        run = "1770000000"
        filename = "profile export.json"

        await _download_operator_file(
            session,
            api_base="http://localhost:31081",
            namespace=namespace,
            job_id=job_id,
            file_info={"name": filename},
            output_dir=tmp_path,
            run=run,
        )

        expected = "/".join(
            [
                "http://localhost:31081/api/v1/results",
                quote(namespace, safe=""),
                quote(job_id, safe=""),
                "runs",
                quote(run, safe=""),
                quote(filename, safe=""),
            ]
        )
        assert session.urls == [expected]
        assert session.headers == [{"Accept-Encoding": "zstd, gzip, identity"}]


# ============================================================
# Corrupt compressed body and idempotence contracts
# ============================================================


class TestDownloadDecompressionFailure:
    """Failed compressed downloads must not leave partial artifacts behind."""

    @pytest.mark.asyncio
    async def test_download_and_decompress_corrupt_gzip_removes_new_partial_file(
        self, tmp_path: Path
    ) -> None:
        dest = tmp_path / "metrics.json"
        resp = _FakeResponse(body=b"not-a-gzip-stream")

        with pytest.raises(zlib.error):
            await _download_and_decompress(resp, dest, "gzip")

        assert dest.exists() is False

    @pytest.mark.asyncio
    async def test_download_and_decompress_corrupt_gzip_preserves_existing_file(
        self, tmp_path: Path
    ) -> None:
        dest = tmp_path / "profile_export_aiperf.json"
        dest.write_bytes(b'{"previous": true}')
        resp = _FakeResponse(body=b"not-a-gzip-stream")

        with pytest.raises(zlib.error):
            await _download_and_decompress(resp, dest, "gzip")

        assert dest.read_bytes() == b'{"previous": true}'

    @pytest.mark.asyncio
    async def test_download_operator_file_repeated_success_overwrites_atomically(
        self, tmp_path: Path
    ) -> None:
        dest = tmp_path / "metrics.json"
        dest.write_bytes(b'{"old": true}')
        session = _RecordingSession(
            _FakeResponse(
                body=gzip.compress(b'{"new": true}'),
                headers={"Content-Encoding": "gzip"},
            )
        )

        result = await _download_operator_file(
            session,
            api_base="http://localhost:31081",
            namespace="bench-prod",
            job_id="latency-bench-7f2a",
            file_info={"name": "metrics.json"},
            output_dir=tmp_path,
            run="1770000000",
        )

        assert result == ("metrics.json", len(b'{"new": true}'))
        assert dest.read_bytes() == b'{"new": true}'


# ============================================================
# Sweep aggregate artifact nested-path allowlist
# ============================================================


class TestSweepAggregateArtifactPaths:
    """Sweep aggregate downloads allow intended nesting but reject zip-slip names."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "display_name",
        [
            param("../aggregate.json", id="parent-traversal"),
            param("aggregate/../../secret.json", id="nested-parent-traversal"),
            param("/tmp/aggregate.json", id="absolute-path"),
            param(".aiperf_results_ready.json", id="hidden-ready-marker"),
            param("aggregate/.hidden.json", id="hidden-nested-leaf"),
        ],
    )  # fmt: skip
    async def test_download_sweep_operator_file_unsafe_path_refuses_request(
        self, tmp_path: Path, display_name: str
    ) -> None:
        result, session = await _collect_downloaded_sweep_artifact(
            tmp_path, display_name
        )

        assert result is None
        assert session.urls == []
        assert list(tmp_path.rglob("*")) == []

    @pytest.mark.asyncio
    async def test_download_sweep_operator_file_nested_artifact_writes_under_output_dir(
        self, tmp_path: Path
    ) -> None:
        result, session = await _collect_downloaded_sweep_artifact(
            tmp_path, "sweep_aggregate/summary.json"
        )

        assert result == ("sweep_aggregate/summary.json", len(b"aggregate"))
        assert (
            tmp_path / "sweep_aggregate" / "summary.json"
        ).read_bytes() == b"aggregate"
        assert session.urls == [
            "http://localhost:31081/api/v1/sweeps/bench-prod/llama3-sweep/epochs/1770000000/artifacts/sweep_aggregate/summary.json"
        ]


# ============================================================
# Partial body retry contract
# ============================================================


class TestPartialDownloadRetries:
    """Length-mismatched artifact responses retry without writing partial bytes."""

    @pytest.mark.asyncio
    async def test_download_artifact_partial_response_exhausted_leaves_no_file(
        self, tmp_path: Path
    ) -> None:
        session = _RecordingSession(_FakeResponse(body=b"abc", content_length=6))

        result = await _download_artifact(
            session,
            "http://localhost:31090/api/results/files",
            "metrics.json",
            tmp_path,
            max_retries=2,
        )

        assert result is None
        assert session.urls == [
            "http://localhost:31090/api/results/files/metrics.json",
            "http://localhost:31090/api/results/files/metrics.json",
            "http://localhost:31090/api/results/files/metrics.json",
        ]
        assert (tmp_path / "metrics.json").exists() is False
