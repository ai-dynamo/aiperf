# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial tests for Kubernetes results client APIs.

Focuses on:
- operator API discovery and localhost-only client session boundaries;
- redirect and HTTP status handling before writing artifacts;
- list-runs and file-list JSON schema validation at the operator trust boundary;
- reserved filenames from controller artifact listings;
- cleanup/idempotence when multi-file operator downloads encounter bad entries.

Out of scope: chunk-level decompression corruption and sweep nested artifact path
allowlisting, covered by ``tests/unit/kubernetes/test_results_download_adversarial.py``.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Self
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from pytest import param

from aiperf.kubernetes.results_artifacts import (
    API_RESULTS_FILES_PATH,
    _download_all_artifacts,
    _download_artifact,
)
from aiperf.kubernetes.results_operator import (
    _download_all_operator_files,
    _download_operator_file,
    _list_operator_files,
    _resolve_operator_run,
    retrieve_results_from_operator,
)
from aiperf.kubernetes.results_sidecar import READY_MARKER_NAME

# ============================================================
# Helpers
# ============================================================


class _StreamContent:
    """Fake aiohttp streaming body for operator download helpers."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def iter_chunked(self, _chunk_size: int) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _Response:
    """Minimal async response surface shared by results client tests."""

    def __init__(
        self,
        *,
        status: int = 200,
        body: bytes = b"",
        json_data: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        content_length: int | None = None,
    ) -> None:
        self.status = status
        self._body = body
        self._json_data = json_data
        self.headers = headers or {}
        self._content_length = content_length
        self.json_loads: object | None = None
        self.content = _StreamContent([body])

    @property
    def content_length(self) -> int | None:
        return self._content_length

    async def read(self) -> bytes:
        return self._body

    async def json(self, *, loads: object | None = None) -> dict[str, object]:
        self.json_loads = loads
        if self._json_data is not None:
            return self._json_data
        raise aiohttp.ContentTypeError(
            request_info=MagicMock(),
            history=(),
            message="operator returned non-JSON result listing",
        )

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                request_info=MagicMock(),
                history=(),
                status=self.status,
                message="operator rejected result request",
            )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _QueuedSession:
    """Fake aiohttp session that records request URLs and per-call kwargs."""

    def __init__(self, queues: dict[str, list[object]]) -> None:
        self._queues = {url: list(items) for url, items in queues.items()}
        self.get_calls: list[str] = []
        self.get_kwargs: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> object:
        self.get_calls.append(url)
        self.get_kwargs.append(kwargs)
        queue = self._queues.get(url)
        if not queue:
            return _Response(status=404)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _RecordingClientSession:
    """ClientSession replacement that exposes constructor kwargs to assertions."""

    instances: list[_RecordingClientSession] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.session = _QueuedSession(
            {
                "http://localhost:31881/api/v1/results/bench-prod/llama3-latency/runs": [
                    _Response(json_data={"latest_epoch": "1770000000"})
                ],
                "http://localhost:31881/api/v1/results/bench-prod/llama3-latency/runs/1770000000": [
                    _Response(json_data={"files": [{"name": "metrics.json"}]})
                ],
                "http://localhost:31881/api/v1/results/bench-prod/llama3-latency/runs/1770000000/metrics.json": [
                    _Response(body=b'{"throughput": 42}')
                ],
            }
        )
        self.closed = False
        type(self).instances.append(self)

    def get(self, url: str, **kwargs: object) -> object:
        return self.session.get(url, **kwargs)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.closed = True


@asynccontextmanager
async def _operator_port_forward(
    namespace: str,
    pod_name: str,
    local_port: int,
    *,
    remote_port: int,
    verify_api: bool,
    kubeconfig: str | None,
    kube_context: str | None,
) -> AsyncIterator[int]:
    _operator_port_forward.calls.append(
        {
            "namespace": namespace,
            "pod_name": pod_name,
            "local_port": local_port,
            "remote_port": remote_port,
            "verify_api": verify_api,
            "kubeconfig": kubeconfig,
            "kube_context": kube_context,
        }
    )
    yield 31881


_operator_port_forward.calls: list[dict[str, object]] = []


# ============================================================
# Operator API discovery and proxy/auth boundaries
# ============================================================


class TestOperatorApiDiscovery:
    """Operator downloads discover the results server through the operator pod."""

    @pytest.mark.asyncio
    async def test_retrieve_results_from_operator_discovers_operator_pod_and_port(
        self, tmp_path: Path
    ) -> None:
        api = MagicMock()
        download = AsyncMock(return_value=[("metrics.json", 18)])

        with (
            patch(
                "aiperf.kubernetes.results_operator.find_operator_pod",
                AsyncMock(return_value=("aiperf-operator-7f2a", "Running")),
            ) as find_operator_pod,
            patch(
                "aiperf.kubernetes.results_operator.port_forward_with_status",
                _operator_port_forward,
            ),
            patch(
                "aiperf.kubernetes.results_operator._verify_operator_health",
                AsyncMock(return_value=True),
            ) as verify_health,
            patch(
                "aiperf.kubernetes.results_operator._download_all_operator_files",
                download,
            ),
        ):
            ok = await retrieve_results_from_operator(
                "llama3-latency",
                "bench-prod",
                tmp_path,
                api,
                local_port=31081,
                operator_namespace="aiperf-observability",
                results_port=9001,
                kubeconfig="/opt/ci/kubeconfigs/perf-cluster.yaml",
                kube_context="kind-aiperf-smoke",
                run="1770000000",
            )

        assert ok is True
        find_operator_pod.assert_awaited_once_with(
            api, namespace="aiperf-observability"
        )
        assert _operator_port_forward.calls == [
            {
                "namespace": "aiperf-observability",
                "pod_name": "aiperf-operator-7f2a",
                "local_port": 31081,
                "remote_port": 9001,
                "verify_api": False,
                "kubeconfig": "/opt/ci/kubeconfigs/perf-cluster.yaml",
                "kube_context": "kind-aiperf-smoke",
            }
        ]
        verify_health.assert_awaited_once_with("http://localhost:31881")
        download.assert_awaited_once_with(
            api_base="http://localhost:31881",
            namespace="bench-prod",
            job_id="llama3-latency",
            output_dir=tmp_path,
            run="1770000000",
        )

    @pytest.mark.asyncio
    async def test_download_all_operator_files_ignores_proxy_env_for_localhost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HTTP_PROXY", "http://corp-proxy.invalid:3128")
        monkeypatch.setenv("HTTPS_PROXY", "http://corp-proxy.invalid:3128")
        monkeypatch.setenv("AIPERF_KUBE_AUTH_TOKEN", "secret-token-not-for-localhost")
        _RecordingClientSession.instances.clear()

        with (
            patch("aiohttp.ClientSession", _RecordingClientSession),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=MagicMock(name="localhost-connector"),
            ),
        ):
            downloaded = await _download_all_operator_files(
                api_base="http://localhost:31881",
                namespace="bench-prod",
                job_id="llama3-latency",
                output_dir=tmp_path,
            )

        assert downloaded == [("metrics.json", 18)]
        [client] = _RecordingClientSession.instances
        assert client.kwargs["auto_decompress"] is False
        assert client.kwargs.get("trust_env") is not True
        assert "headers" not in client.kwargs
        assert client.closed is True
        assert all(
            url.startswith("http://localhost:31881/")
            for url in client.session.get_calls
        )


# ============================================================
# Redirect and HTTP status handling
# ============================================================


class TestRedirectAndHttpErrorHandling:
    """Redirects are cross-origin trust-boundary events, not artifact bodies."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            param(301, id="moved-permanently"),
            param(302, id="found"),
            param(307, id="temporary-redirect"),
            param(308, id="permanent-redirect"),
        ],
    )  # fmt: skip
    async def test_download_operator_file_redirect_status_returns_none_without_write(
        self, tmp_path: Path, status: int
    ) -> None:
        session = _QueuedSession(
            {
                "http://localhost:31881/api/v1/results/bench-prod/llama3-latency/runs/1770000000/metrics.json": [
                    _Response(
                        status=status,
                        body=b"<a href='https://evil.invalid/metrics.json'>redirect</a>",
                        headers={"Location": "https://evil.invalid/metrics.json"},
                    )
                ]
            }
        )

        result = await _download_operator_file(
            session,  # type: ignore[arg-type]
            api_base="http://localhost:31881",
            namespace="bench-prod",
            job_id="llama3-latency",
            file_info={"name": "metrics.json"},
            output_dir=tmp_path,
            run="1770000000",
        )

        assert result is None
        assert (tmp_path / "metrics.json").exists() is False

    @pytest.mark.asyncio
    async def test_download_artifact_redirect_status_returns_none_without_write(
        self, tmp_path: Path
    ) -> None:
        session = _QueuedSession(
            {
                f"http://localhost:19090{API_RESULTS_FILES_PATH}/metrics.json": [
                    _Response(
                        status=302,
                        body=b"redirect body is not a result artifact",
                        headers={"Location": "https://evil.invalid/metrics.json"},
                    )
                ]
            }
        )

        result = await _download_artifact(
            session,  # type: ignore[arg-type]
            f"http://localhost:19090{API_RESULTS_FILES_PATH}",
            "metrics.json",
            tmp_path,
        )

        assert result is None
        assert (tmp_path / "metrics.json").exists() is False


# ============================================================
# List-runs and file-list schema validation
# ============================================================


class TestOperatorListingSchemaValidation:
    """Operator JSON responses are untrusted and must match the documented shape."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            param({"latest_epoch": {"epoch": "1770000000"}}, id="object-latest-epoch"),
            param({"latest_epoch": ["1770000000"]}, id="list-latest-epoch"),
            param({"latest_epoch": ""}, id="empty-latest-epoch"),
        ],
    )  # fmt: skip
    async def test_resolve_operator_run_malformed_latest_epoch_returns_none(
        self, payload: dict[str, object]
    ) -> None:
        session = _QueuedSession(
            {
                "http://localhost:31881/api/v1/results/bench-prod/llama3-latency/runs": [
                    _Response(json_data=payload)
                ]
            }
        )

        result = await _resolve_operator_run(
            session,  # type: ignore[arg-type]
            api_base="http://localhost:31881",
            namespace="bench-prod",
            job_id="llama3-latency",
            run=None,
        )

        assert result is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            param({"files": "metrics.json"}, id="files-string"),
            param({"files": {"name": "metrics.json"}}, id="files-object"),
            param({"files": [{"path": "metrics.json"}]}, id="entry-missing-name"),
            param({"files": [{"name": 17}]}, id="entry-name-integer"),
        ],
    )  # fmt: skip
    async def test_list_operator_files_malformed_files_payload_returns_none(
        self, payload: dict[str, object]
    ) -> None:
        session = _QueuedSession(
            {
                "http://localhost:31881/api/v1/results/bench-prod/llama3-latency/runs/1770000000": [
                    _Response(json_data=payload)
                ]
            }
        )

        result = await _list_operator_files(
            session,  # type: ignore[arg-type]
            api_base="http://localhost:31881",
            namespace="bench-prod",
            job_id="llama3-latency",
            run="1770000000",
        )

        assert result is None


# ============================================================
# Reserved filenames and idempotence
# ============================================================


class TestReservedFilenamesAndIdempotence:
    """Invalid entries do not poison neighboring downloads or local state."""

    @pytest.mark.asyncio
    async def test_download_all_artifacts_skips_reserved_ready_marker_and_keeps_neighbor(
        self, tmp_path: Path
    ) -> None:
        session = _QueuedSession(
            {
                "http://localhost:19090/api/results/list": [
                    _Response(
                        json_data={
                            "files": [
                                {"name": READY_MARKER_NAME},
                                {"name": "metrics.json"},
                            ]
                        }
                    )
                ],
                "http://localhost:19090/api/results/files/metrics.json": [
                    _Response(body=b'{"ok": true}')
                ],
            }
        )

        with patch("aiohttp.ClientSession", return_value=session):
            downloaded = await _download_all_artifacts(
                "http://localhost:19090", "llama3-latency", tmp_path
            )

        assert downloaded == [("metrics.json", 12)]
        assert (tmp_path / READY_MARKER_NAME).exists() is False
        assert (tmp_path / "metrics.json").read_bytes() == b'{"ok": true}'
        assert (
            "http://localhost:19090/api/results/files/.aiperf_results_ready.json"
            not in session.get_calls
        )

    @pytest.mark.asyncio
    async def test_download_all_operator_files_skips_bad_entry_and_downloads_neighbor(
        self, tmp_path: Path
    ) -> None:
        session = _QueuedSession(
            {
                "http://localhost:31881/api/v1/results/bench-prod/llama3-latency/runs/1770000000": [
                    _Response(
                        json_data={
                            "files": [
                                {"name": READY_MARKER_NAME},
                                {"name": "metrics.json"},
                            ]
                        }
                    )
                ],
                "http://localhost:31881/api/v1/results/bench-prod/llama3-latency/runs/1770000000/metrics.json": [
                    _Response(body=b'{"ok": true}')
                ],
            }
        )

        with (
            patch("aiohttp.ClientSession", return_value=session),
            patch(
                "aiperf.transports.aiohttp_client.create_tcp_connector",
                return_value=MagicMock(name="localhost-connector"),
            ),
        ):
            downloaded = await _download_all_operator_files(
                api_base="http://localhost:31881",
                namespace="bench-prod",
                job_id="llama3-latency",
                output_dir=tmp_path,
                run="1770000000",
            )

        assert downloaded == [("metrics.json", 12)]
        assert (tmp_path / READY_MARKER_NAME).exists() is False
        assert (tmp_path / "metrics.json").read_bytes() == b'{"ok": true}'
        assert all(READY_MARKER_NAME not in url for url in session.get_calls)
