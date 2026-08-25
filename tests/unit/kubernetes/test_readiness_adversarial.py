# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial readiness and endpoint preflight tests.

Focuses on:
- readiness probe HTTP-status and malformed-payload classification
- localhost proxy-env isolation for readiness HTTP clients
- auth/header propagation from endpoint config into the probe
- endpoint URL validator failures before aiohttp sees malformed inputs
- skip semantics when the readiness probe timeout is disabled

Out of scope: generic AioHttpClient transport behavior, covered by
``tests/unit/transports/test_aiohttp_client_edge_cases.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import orjson
import pytest
from kubernetes_asyncio.client.exceptions import ApiException
from pydantic import ValidationError
from pytest import param

from aiperf.cli_runner._preflight import _preflight_endpoint_ready
from aiperf.common import readiness_probe
from aiperf.common.readiness_probe import wait_for_endpoint
from aiperf.config.endpoint import EndpointConfig
from aiperf.kubernetes.preflight import CheckStatus
from aiperf.kubernetes.preflight_checks import check_endpoint_connectivity
from aiperf.transports.aiohttp_client import AioHttpClient
from aiperf.transports.http_defaults import AioHttpDefaults

# =============================================================================
# Helpers
# =============================================================================


@dataclass(slots=True)
class _ProbeError:
    message: str


@dataclass(slots=True)
class _TextBody:
    text: str


class _ProbeRecord:
    def __init__(
        self,
        *,
        status: int | None,
        body: str = "",
        error: _ProbeError | None = None,
    ) -> None:
        self.status = status
        self.error = error
        self.responses = [_TextBody(body)] if body else []


class _SequencedProbeClient:
    def __init__(
        self,
        *,
        get_records: list[_ProbeRecord] | None = None,
        post_records: list[_ProbeRecord] | None = None,
    ) -> None:
        self.get_records = list(get_records or [])
        self.post_records = list(post_records or [])
        self.get_urls: list[str] = []
        self.post_urls: list[str] = []
        self.get_headers: list[dict[str, str]] = []
        self.post_headers: list[dict[str, str]] = []
        self.closed = False

    async def get_request(
        self, url: str, headers: dict[str, str], timeout: object
    ) -> _ProbeRecord:
        del timeout
        self.get_urls.append(url)
        self.get_headers.append(dict(headers))
        return self.get_records.pop(0)

    async def post_request(
        self,
        request_url: str,
        payload: bytes,
        headers: dict[str, str],
        timeout: object,
    ) -> _ProbeRecord:
        decoded = orjson.loads(payload)
        assert isinstance(decoded, dict)
        self.post_urls.append(request_url)
        self.post_headers.append(dict(headers))
        return self.post_records.pop(0)

    async def close(self) -> None:
        self.closed = True


def _model_payload(model_name: str = "meta-llama/Llama-3-8B") -> str:
    return orjson.dumps({"data": [{"id": model_name}]}).decode()


def _plan_with_endpoint(**endpoint_overrides: Any) -> SimpleNamespace:
    endpoint_defaults = {
        "urls": ["http://localhost:8000"],
        "headers": {"X-Trace-ID": "conv-2026-05-18-9c3a"},
        "api_key": "sk-local-readiness",
        "wait_for_model_timeout": 11.0,
        "wait_for_model_interval": 0.25,
        "wait_for_model_mode": "both",
        "type": "chat",
        "path": "/v1/chat/completions",
    }
    endpoint_defaults.update(endpoint_overrides)
    config = SimpleNamespace(
        endpoint=SimpleNamespace(**endpoint_defaults),
        get_model_names=lambda: ["meta-llama/Llama-3-8B"],
    )
    return SimpleNamespace(configs=[config])


@contextmanager
def _patched_aiohttp_session(response: Mock) -> Iterator[Mock]:
    with patch("aiohttp.ClientSession") as session_class:
        session = Mock()
        session_context = AsyncMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=None)
        session_class.return_value = session_context

        request_context = AsyncMock()
        request_context.__aenter__ = AsyncMock(return_value=response)
        request_context.__aexit__ = AsyncMock(return_value=None)
        session.request = Mock(return_value=request_context)
        yield session_class


# =============================================================================
# Readiness probe payload and status classification
# =============================================================================


class TestReadinessProbeStatusClassification:
    """HTTP statuses and malformed model-list bodies drive retry vs ready."""

    @pytest.mark.asyncio
    async def test_wait_models_malformed_json_retries_until_model_appears(self) -> None:
        client = _SequencedProbeClient(
            get_records=[
                _ProbeRecord(status=200, body="{not-json"),
                _ProbeRecord(status=200, body=_model_payload()),
            ]
        )

        await readiness_probe._wait_models(
            client=client,  # type: ignore[arg-type]
            url="http://localhost:8000",
            model_name="meta-llama/Llama-3-8B",
            timeout_s=5.0,
            interval_s=0.1,
            headers={"X-Readiness": "models"},
        )

        assert client.get_urls == [
            "http://localhost:8000/v1/models",
            "http://localhost:8000/v1/models",
        ]
        assert client.get_headers == [
            {"X-Readiness": "models"},
            {"X-Readiness": "models"},
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status",
        [
            param(200, id="success-ready"),
            param(400, id="client-error-ready"),
            param(401, id="auth-error-ready"),
            param(404, id="missing-route-ready"),
            param(429, id="rate-limit-ready"),
        ],
    )  # fmt: skip
    async def test_wait_inference_status_below_500_counts_as_ready(
        self, status: int
    ) -> None:
        client = _SequencedProbeClient(post_records=[_ProbeRecord(status=status)])

        await readiness_probe._wait_inference(
            client=client,  # type: ignore[arg-type]
            url="http://localhost:8000",
            model_name="meta-llama/Llama-3-8B",
            endpoint_type="chat",
            custom_endpoint=None,
            timeout_s=5.0,
            interval_s=0.1,
            headers={"Authorization": "Bearer sk-local-readiness"},
        )

        assert client.post_urls == ["http://localhost:8000/v1/chat/completions"]
        assert client.post_headers == [
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-local-readiness",
            }
        ]

    @pytest.mark.asyncio
    async def test_wait_inference_server_error_retries_until_non_5xx_ready(
        self,
    ) -> None:
        client = _SequencedProbeClient(
            post_records=[_ProbeRecord(status=503), _ProbeRecord(status=204)]
        )

        await readiness_probe._wait_inference(
            client=client,  # type: ignore[arg-type]
            url="http://localhost:8000/v1/chat/completions",
            model_name="meta-llama/Llama-3-8B",
            endpoint_type="chat",
            custom_endpoint=None,
            timeout_s=5.0,
            interval_s=0.1,
            headers={},
        )

        assert client.post_urls == [
            "http://localhost:8000/v1/chat/completions",
            "http://localhost:8000/v1/chat/completions",
        ]

    @pytest.mark.asyncio
    async def test_wait_models_404_base_url_2xx_fallback_returns_ready(self) -> None:
        client = _SequencedProbeClient(
            get_records=[_ProbeRecord(status=404), _ProbeRecord(status=200, body="ok")]
        )

        await readiness_probe._wait_models(
            client=client,  # type: ignore[arg-type]
            url="http://localhost:8000",
            model_name="meta-llama/Llama-3-8B",
            timeout_s=5.0,
            interval_s=0.1,
            headers={},
        )

        assert client.get_urls == [
            "http://localhost:8000/v1/models",
            "http://localhost:8000",
        ]

    def test_models_timeout_error_names_model_url_and_attempt_count(self) -> None:
        with pytest.raises(
            TimeoutError,
            match=r"meta-llama/Llama-3-8B.*http://localhost:8000.*checked 3 time",
        ):
            readiness_probe._models_timeout(
                deadline=0.0,
                request_timeout_base=5.0,
                timeout_s=10.0,
                model_name="meta-llama/Llama-3-8B",
                url="http://localhost:8000",
                checked_attempts=3,
            )

    @pytest.mark.parametrize(
        "payload_text",
        [
            param("null", id="null-json"),
            param("[]", id="list-root"),
            param('{"data": null}', id="null-data"),
            param('{"data": [{"id": "other-model"}]}', id="wrong-model"),
            param('{"data": ["meta-llama/Llama-3-8B"]}', id="string-entry"),
        ],
    )  # fmt: skip
    def test_model_in_payload_malformed_shapes_are_not_ready(
        self, payload_text: str
    ) -> None:
        assert (
            readiness_probe._model_in_payload(payload_text, "meta-llama/Llama-3-8B")
            is False
        )

    def test_response_status_and_error_connection_error_is_explicit(self) -> None:
        status, error = readiness_probe._response_status_and_error(
            _ProbeRecord(
                status=None,
                error=_ProbeError("proxy refused localhost connection"),
            )
        )

        assert status == "connection error"
        assert error == "proxy refused localhost connection"

    @pytest.mark.asyncio
    async def test_sleep_until_next_attempt_uses_asyncio_sleep_not_blocking_sleep(
        self,
    ) -> None:
        with patch(
            "aiperf.common.readiness_probe.asyncio.sleep", new=AsyncMock()
        ) as sleep:
            await readiness_probe._sleep_until_next_attempt(
                deadline=readiness_probe.time.monotonic() + 5.0,
                interval_s=0.25,
            )

        sleep.assert_awaited_once_with(0.25)


# =============================================================================
# Readiness HTTP client proxy and lifecycle behavior
# =============================================================================


class TestReadinessProbeHttpClient:
    """The readiness client must ignore ambient localhost proxy env vars."""

    @pytest.mark.asyncio
    async def test_wait_for_endpoint_closes_aiohttp_client_after_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = _SequencedProbeClient(
            get_records=[_ProbeRecord(status=503, error=_ProbeError("backend loading"))]
        )
        monkeypatch.setattr(
            "aiperf.transports.aiohttp_client.AioHttpClient",
            lambda *args, **kwargs: client,
        )

        with pytest.raises(TimeoutError, match=r"meta-llama/Llama-3-8B"):
            await wait_for_endpoint(
                urls=["http://localhost:8000"],
                model_names=["meta-llama/Llama-3-8B"],
                mode="models",
                endpoint_type="chat",
                custom_endpoint=None,
                timeout_s=0.0,
                interval_s=0.1,
                headers={},
            )

        assert client.closed is True

    @pytest.mark.asyncio
    async def test_wait_models_localhost_ignores_http_proxy_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
        assert AioHttpDefaults.TRUST_ENV is False

        response = Mock(
            spec=aiohttp.ClientResponse,
            status=200,
            reason="OK",
            content_type="application/json",
            text=AsyncMock(return_value=_model_payload()),
        )
        client = AioHttpClient(timeout=1.0)
        try:
            with _patched_aiohttp_session(response) as session_class:
                await readiness_probe._wait_models(
                    client=client,
                    url="http://localhost:8000",
                    model_name="meta-llama/Llama-3-8B",
                    timeout_s=5.0,
                    interval_s=0.1,
                    headers={},
                )
        finally:
            await client.close()

        session_kwargs = session_class.call_args.kwargs
        assert session_kwargs["trust_env"] is False


# =============================================================================
# CLI preflight header propagation and skip semantics
# =============================================================================


class TestCliPreflightEndpointReady:
    """The synchronous CLI preflight wrapper owns auth/header preparation."""

    def test_preflight_endpoint_ready_disabled_timeout_skips_probe(self) -> None:
        plan = _plan_with_endpoint(wait_for_model_timeout=0.0)

        with patch("aiperf.common.readiness_probe.wait_for_endpoint") as wait_probe:
            _preflight_endpoint_ready(plan)

        wait_probe.assert_not_called()

    def test_preflight_endpoint_ready_passes_auth_headers_and_probe_shape(self) -> None:
        plan = _plan_with_endpoint(headers={"X-Trace-ID": "conv-2026-05-18-9c3a"})
        captured: dict[str, Any] = {}

        async def _capture_wait_for_endpoint(**kwargs: Any) -> None:
            captured.update(kwargs)

        with patch(
            "aiperf.common.readiness_probe.wait_for_endpoint",
            side_effect=_capture_wait_for_endpoint,
        ):
            _preflight_endpoint_ready(plan)

        assert captured["urls"] == ["http://localhost:8000"]
        assert captured["model_names"] == ["meta-llama/Llama-3-8B"]
        assert captured["mode"] == "both"
        assert captured["endpoint_type"] == "chat"
        assert captured["custom_endpoint"] == "/v1/chat/completions"
        assert captured["headers"] == {
            "X-Trace-ID": "conv-2026-05-18-9c3a",
            "Authorization": "Bearer sk-local-readiness",
        }


# =============================================================================
# Endpoint URL validation and Kubernetes endpoint classification
# =============================================================================


class TestEndpointUrlValidation:
    """Real EndpointConfig validators catch malformed URLs before the probe runs."""

    @pytest.mark.parametrize(
        "url,match",
        [
            param(
                " http://localhost:8000",
                r"leading or trailing whitespace",
                id="leading-whitespace",
            ),
            param(
                "http://local host:8000",
                r"contains whitespace",
                id="embedded-whitespace",
            ),
            param(
                "ftp://localhost:8000",
                r"unsupported scheme 'ftp'",
                id="unsupported-scheme",
            ),
            param(
                "http://:8000",
                r"missing scheme or host",
                id="missing-host",
            ),
            param(
                "http://localhost:99999",
                r"invalid port",
                id="out-of-range-port",
            ),
        ],
    )  # fmt: skip
    def test_endpoint_config_malformed_url_rejected_with_actionable_message(
        self, url: str, match: str
    ) -> None:
        with pytest.raises(ValidationError, match=match):
            EndpointConfig.model_validate({"urls": [url]})

    def test_endpoint_config_scheme_less_localhost_normalizes_to_http(self) -> None:
        cfg = EndpointConfig.model_validate({"urls": ["localhost:8000"]})

        assert cfg.urls == ["http://localhost:8000"]

    @pytest.mark.asyncio
    async def test_check_endpoint_connectivity_invalid_port_returns_warn(self) -> None:
        result = await check_endpoint_connectivity(
            Mock(), endpoint_url="http://llm-service.default.svc:99999"
        )

        assert result.status == CheckStatus.WARN
        assert "Could not parse endpoint URL" in result.message

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exc",
        [
            param(ApiException(status=403), id="permission-denied"),
            param(TimeoutError(), id="apiserver-timeout"),
            param(OSError("network unreachable"), id="network-unreachable"),
        ],
    )  # fmt: skip
    async def test_check_endpoint_connectivity_service_lookup_error_fails_with_hint(
        self, exc: Exception
    ) -> None:
        core = MagicMock()
        core.read_namespaced_service = AsyncMock(side_effect=exc)

        with patch(
            "aiperf.kubernetes.preflight_checks.client.CoreV1Api", return_value=core
        ):
            result = await check_endpoint_connectivity(
                Mock(), endpoint_url="http://llm-service.inference.svc:8000/v1"
            )

        assert result.status == CheckStatus.FAIL
        assert "llm-service.inference.svc" in result.message
        assert result.hints == [
            "Verify the service exists: kubectl get svc -A | grep llm-service"
        ]
