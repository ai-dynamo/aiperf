# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Control-plane and readiness requests authenticate like the benchmark does.

Against a SigV4-protected endpoint an unsigned probe returns 403, which the
readiness rule ``status < 500 == ready`` misreads as ready, and both control
hooks fail outright. These tests pin the two halves of the fix: the api_key
header is suppressed when ``auth_type`` is set, and every attempt carries a
freshly-computed signature.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import orjson
import pytest
from pytest import param

from aiperf.auth.base_signer import SignedRequest
from aiperf.common.control_hooks import PreparedEndpointControlHooks, run_reset_kv_cache
from aiperf.common.control_plane_http import control_plane_post
from aiperf.common.endpoint_auth import (
    auth_headers_for_endpoint,
    endpoint_signer,
    sign_request,
)
from aiperf.config import EndpointConfig
from aiperf.config.config import BenchmarkConfig


class _CountingSigner:
    """Minimal RequestSignerProtocol stand-in that stamps an attempt counter."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bytes | None]] = []

    async def sign(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> SignedRequest:
        self.calls.append((method, url, body))
        return SignedRequest(
            headers={**headers, "X-Amz-Date": f"stamp-{len(self.calls)}"}
        )


@pytest.mark.parametrize(
    "endpoint_type,suppressed_header",
    [
        param("chat", "Authorization", id="bearer"),
        param("messages", "x-api-key", id="anthropic"),
    ],
)  # fmt: skip
def test_auth_type_suppresses_api_key_header(
    endpoint_type: str, suppressed_header: str
) -> None:
    """A configured auth_type owns the credential on control/readiness paths.

    BaseEndpoint and MessagesEndpoint both drop api_key when auth_type is set
    so the signer's Authorization header survives. Preflight and the control
    hooks must match, or they authenticate differently than the run they gate.
    """
    cfg = EndpointConfig(
        type=endpoint_type,
        urls=["http://server"],
        api_key="leftover-key",
        auth_type="sigv4",
        aws_region="us-east-1",
        aws_service="sagemaker",
    )

    headers = auth_headers_for_endpoint(cfg)

    assert suppressed_header not in headers


def test_api_key_still_applied_without_auth_type() -> None:
    """The suppression is conditional: plain api_key endpoints are unaffected."""
    cfg = EndpointConfig(type="chat", urls=["http://server"], api_key="sk-test")

    assert auth_headers_for_endpoint(cfg)["Authorization"] == "Bearer sk-test"


@pytest.mark.asyncio
async def test_sign_request_is_a_no_op_without_a_signer() -> None:
    """Callers pass ``None`` freely; nothing is rewritten."""
    headers = {"X-Trace": "1"}

    url, out_headers, body = await sign_request(
        None, method="POST", url="http://server/x", headers=headers, body=b"payload"
    )

    assert (url, out_headers, body) == ("http://server/x", headers, b"payload")


@pytest.mark.asyncio
async def test_control_plane_post_sends_signed_headers() -> None:
    """control_plane_post applies the signer before opening the session."""
    signer = _CountingSigner()
    captured: dict[str, Any] = {}

    class _Resp:
        status = 204

        async def read(self) -> bytes:
            return b""

        async def __aenter__(self) -> _Resp:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

    class _Session:
        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *exc: object) -> None:
            return None

        def post(self, url: str, headers: dict[str, str], data: bytes) -> _Resp:
            captured.update(url=url, headers=headers, data=data)
            return _Resp()

    with patch("aiohttp.ClientSession", return_value=_Session()):
        await control_plane_post(
            url="http://server/reset",
            headers={"X-Trace": "1"},
            timeout_s=1.0,
            signer=signer,
        )

    assert signer.calls == [("POST", "http://server/reset", b"")]
    assert captured["headers"]["X-Amz-Date"] == "stamp-1"
    assert captured["headers"]["X-Trace"] == "1"


@pytest.mark.asyncio
async def test_reset_kv_cache_signs_every_retry_attempt() -> None:
    """Each retry re-signs.

    A SigV4 signature is rejected outside a five-minute skew window, and
    reset_kv_cache backoff can run for minutes. Signing once per URL would
    make every retry after the window expire-fail.
    """
    hooks = PreparedEndpointControlHooks(
        timeout_s=1.0,
        reset_urls=["http://server/reset"],
        profiler_start_urls=[],
        profiler_stop_urls=[],
        profiler_timeout_s=1.0,
        reset_max_retry_seconds=60.0,
    )
    signer = _CountingSigner()
    stamps: list[str] = []
    attempts = 0

    async def _post(
        *, url: str, headers: dict[str, str], timeout_s: float, signer: Any
    ):
        nonlocal attempts
        attempts += 1
        _, signed_headers, _ = await sign_request(
            signer, method="POST", url=url, headers=headers, body=b""
        )
        stamps.append(signed_headers["X-Amz-Date"])
        if attempts < 3:
            from aiperf.common.control_plane_http import ControlPlaneHttpError

            raise ControlPlaneHttpError("transient", retryable=True)

    with patch("aiperf.common.control_hooks.control_plane_post", new=_post):
        await run_reset_kv_cache(hooks, {}, signer=signer)

    assert stamps == ["stamp-1", "stamp-2", "stamp-3"]


def _config(**endpoint_overrides: Any) -> BenchmarkConfig:
    """Minimal resolved BenchmarkConfig with an overridable endpoint block."""
    return BenchmarkConfig.model_validate(
        {
            "models": ["test-model"],
            "endpoint": {"urls": ["http://server"], **endpoint_overrides},
            "datasets": [
                {
                    "name": "main",
                    "type": "synthetic",
                    "entries": 1,
                    "prompts": {"isl": 8, "osl": 4},
                }
            ],
            "phases": [
                {
                    "name": "profiling",
                    "type": "concurrency",
                    "requests": 1,
                    "concurrency": 1,
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_endpoint_signer_yields_none_without_auth_type() -> None:
    """No auth_type means no signer object is constructed at all."""
    async with endpoint_signer(_config()) as signer:
        assert signer is None


@pytest.mark.asyncio
async def test_endpoint_signer_starts_and_stops_the_plugin() -> None:
    """The signer is a lifecycle object; it must be started and torn down."""
    cfg = _config(auth_type="sigv4", aws_region="us-east-1", aws_service="sagemaker")
    instance = AsyncMock()

    with patch(
        "aiperf.plugin.plugins.get_class", return_value=lambda **_: instance
    ) as get_class:
        async with endpoint_signer(cfg) as signer:
            assert signer is instance
            instance.initialize_and_start.assert_awaited_once()
            instance.stop.assert_not_awaited()

    instance.stop.assert_awaited_once()
    assert get_class.call_args.args[1] == "sigv4"


@pytest.mark.asyncio
async def test_readiness_inference_probe_signs_each_attempt() -> None:
    """The inference probe re-signs on every retry, over the exact body bytes.

    Signing once would both expire across a long poll and hash a body the
    server never receives.
    """
    from aiperf.common import readiness_probe

    signer = _CountingSigner()
    seen: list[str] = []
    attempts = 0

    class _Record:
        def __init__(self, status: int) -> None:
            self.status = status
            self.error = None

    class _Client:
        async def post_request(
            self,
            request_url: str,
            payload: bytes,
            headers: dict[str, str],
            timeout: object,
        ) -> _Record:
            nonlocal attempts
            del request_url, timeout
            attempts += 1
            seen.append(headers["X-Amz-Date"])
            assert signer.calls[-1][2] == payload
            return _Record(503 if attempts < 2 else 200)

    await readiness_probe._wait_inference(
        client=cast(Any, _Client()),
        url="http://server",
        model_name="model-a",
        endpoint_type="chat",
        custom_endpoint=None,
        timeout_s=5.0,
        interval_s=0.01,
        headers={},
        signer=signer,
    )

    assert seen == ["stamp-1", "stamp-2"]


@pytest.mark.asyncio
async def test_readiness_models_probe_signs_each_attempt() -> None:
    """The /v1/models probe signs the GET it actually issues."""
    from aiperf.common import readiness_probe

    signer = _CountingSigner()
    seen: list[str] = []
    attempts = 0

    class _Record:
        def __init__(self, status: int, body: bytes | None) -> None:
            self.status = status
            self.error = None
            self.responses = [SimpleNamespace(text=body)] if body else []

    class _Client:
        async def get_request(
            self, url: str, headers: dict[str, str], timeout: object
        ) -> _Record:
            nonlocal attempts
            del timeout
            attempts += 1
            seen.append(headers["X-Amz-Date"])
            assert url == "http://server/v1/models"
            if attempts < 2:
                return _Record(503, None)
            return _Record(200, orjson.dumps({"data": [{"id": "model-a"}]}))

    await readiness_probe._wait_models(
        client=cast(Any, _Client()),
        url="http://server",
        model_name="model-a",
        timeout_s=5.0,
        interval_s=0.01,
        headers={},
        signer=signer,
    )

    assert seen == ["stamp-1", "stamp-2"]


def test_from_config_matches_from_run() -> None:
    """from_run must stay a thin delegate so both paths sign identically."""
    from unittest.mock import MagicMock

    from aiperf.common.models.model_endpoint_info import ModelEndpointInfo

    cfg = _config(
        auth_type="sigv4", aws_region="eu-west-1", aws_service="bedrock-runtime"
    )
    run = MagicMock()
    run.cfg = cfg

    from_config = ModelEndpointInfo.from_config(cfg)

    assert ModelEndpointInfo.from_run(run) == from_config
    assert from_config.endpoint.aws_region == "eu-west-1"
    assert from_config.endpoint.aws_service == "bedrock-runtime"
    assert from_config.endpoint.auth_type == "sigv4"
