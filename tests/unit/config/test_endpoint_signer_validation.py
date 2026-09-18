# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-fast validation for endpoints that combine a request signer with a
feature whose requests never reach the signing code path.

Every rejection here stands in for a 403 that would otherwise surface partway
into a benchmark, from a request aiperf sent unsigned.
"""

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.config.control_hooks import ResetKvCacheConfig, ServerProfilerConfig
from aiperf.config.endpoint import EndpointConfig

_URLS = ["http://127.0.0.1:8000"]


def _endpoint(**overrides) -> EndpointConfig:
    """Build a signed chat endpoint, overriding individual fields per test."""
    data: dict = {
        "urls": _URLS,
        "type": "chat",
        "auth_type": "sigv4",
        "aws_region": "us-west-2",
        "aws_signing_service": "sagemaker",
    }
    data.update(overrides)
    return EndpointConfig.model_validate(data)


def test_sigv4_endpoint_with_region_and_signing_service_is_accepted() -> None:
    cfg = _endpoint()
    assert cfg.aws_region == "us-west-2"
    assert cfg.aws_signing_service == "sagemaker"


def test_sigv4_without_region_names_the_missing_flag() -> None:
    with pytest.raises(ValidationError, match="--aws-region"):
        _endpoint(aws_region=None)


def test_sigv4_without_signing_service_names_the_missing_flag() -> None:
    with pytest.raises(ValidationError, match="--aws-signing-service"):
        _endpoint(aws_signing_service=None)


def test_polling_endpoint_with_signer_is_rejected() -> None:
    """video_generation submits and polls an async job outside send_request,
    so its requests would bypass signing entirely."""
    with pytest.raises(ValidationError, match="polling"):
        _endpoint(type="video_generation")


def test_multipart_endpoint_with_signer_is_rejected() -> None:
    """aiohttp streams multipart bodies, so they never materialize as the exact
    bytes SigV4 has to hash."""
    with pytest.raises(ValidationError, match="multipart"):
        _endpoint(type="image_edit")


def test_wait_for_model_with_signer_is_rejected() -> None:
    with pytest.raises(ValidationError, match="--wait-for-model-timeout"):
        _endpoint(wait_for_model_timeout=30.0)


def test_reset_kv_cache_with_signer_is_rejected() -> None:
    with pytest.raises(ValidationError, match="reset_kv_cache"):
        _endpoint(reset_kv_cache=ResetKvCacheConfig())


def test_server_profiler_with_signer_is_rejected() -> None:
    with pytest.raises(ValidationError, match="server_profiler"):
        _endpoint(server_profiler=ServerProfilerConfig())


def test_unsigned_endpoints_keep_every_rejected_feature() -> None:
    """The guards key on the signer, not the feature: without --auth-type all of
    these stay legal, so PR 1 takes nothing away from existing users."""
    cfg = EndpointConfig.model_validate(
        {
            "urls": _URLS,
            "type": "video_generation",
            "wait_for_model_timeout": 30.0,
            "reset_kv_cache": ResetKvCacheConfig(),
        }
    )
    assert cfg.auth_type is None
    assert cfg.wait_for_model_timeout == 30.0
    assert cfg.reset_kv_cache is not None


class TestSigningRequiresTls:
    """Signing puts credential material on the wire -- the SigV4 signature and,
    for temporary credentials, the ``x-amz-security-token`` bearer token. Over
    cleartext HTTP that is a credential disclosure, so it is refused rather than
    sent. Loopback is exempt so the in-repo mock server stays usable."""

    def test_cleartext_http_to_a_remote_host_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="https"):
            _endpoint(urls=["http://sagemaker.example.com"])

    def test_https_is_accepted(self) -> None:
        cfg = _endpoint(urls=["https://sagemaker.example.com"])
        assert cfg.urls == ["https://sagemaker.example.com"]

    @pytest.mark.parametrize(
        "url",
        [
            param("http://localhost:8000", id="localhost"),
            param("http://127.0.0.1:8000", id="ipv4-loopback"),
            param("http://[::1]:8000", id="ipv6-loopback"),
        ],
    )
    def test_loopback_over_http_is_allowed(self, url: str) -> None:
        """The documented mock-server workflow signs against http://localhost."""
        assert _endpoint(urls=[url]).urls == [url]

    def test_one_bad_url_among_several_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="https"):
            _endpoint(urls=["https://good.example.com", "http://bad.example.com"])

    def test_unsigned_endpoints_may_still_use_http(self) -> None:
        """The rule is about protecting credentials, not about mandating TLS."""
        cfg = EndpointConfig.model_validate(
            {"urls": ["http://anywhere.example.com"], "type": "chat"}
        )
        assert cfg.auth_type is None
