# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Config-time validation of SigV4 auth settings.

Each case here previously passed validation and failed at request time in
every worker (internal botocore error, silently unauthenticated requests, or
silently ignored flags) instead of failing fast at startup.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.config.endpoint import EndpointConfig
from aiperf.plugin.enums import EndpointType


@pytest.mark.parametrize(
    "kwargs,expected_message",
    [
        param(
            {"auth_type": "sigv4"},
            "requires --aws-region and --aws-service",
            id="missing-both",
        ),
        param(
            {"auth_type": "sigv4", "aws_service": "sagemaker"},
            "requires --aws-region",
            id="missing-region",
        ),
        param(
            {"auth_type": "sigv4", "aws_region": "us-east-1"},
            "requires --aws-service",
            id="missing-service",
        ),
        param(
            {"auth_type": "sigv4", "aws_region": "  ", "aws_service": "sagemaker"},
            "requires --aws-region",
            id="blank-region",
        ),
        param(
            {"aws_region": "us-east-1"},
            "--aws-region has no effect unless --auth-type is set",
            id="aws-field-without-auth-type",
        ),
        param(
            {"aws_profile": "prod"},
            "--aws-profile has no effect unless --auth-type is set",
            id="aws-profile-without-auth-type",
        ),
        param(
            {
                "type": EndpointType.IMAGE_EDIT,
                "auth_type": "sigv4",
                "aws_region": "us-east-1",
                "aws_service": "sagemaker",
            },
            "does not support multipart/form-data",
            id="sigv4-with-multipart-endpoint",
        ),
    ],
)  # fmt: skip
def test_invalid_sigv4_config_rejected(kwargs: dict, expected_message: str) -> None:
    with pytest.raises(ValueError, match=expected_message):
        EndpointConfig(urls=["http://localhost:8000"], **kwargs)


def test_sigv4_with_plain_http_url_rejected() -> None:
    """Remote cleartext still exposes the signature and any session token."""
    with pytest.raises(ValueError, match="https"):
        EndpointConfig(
            urls=["http://runtime.sagemaker.us-east-1.amazonaws.com"],
            auth_type="sigv4",
            aws_region="us-east-1",
            aws_service="sagemaker",
        )


def test_sigv4_requires_http_transport_gate() -> None:
    """Non-HTTP transports never call _sign_if_needed, so a non-HTTP transport
    plugin would resolve AWS credentials and sign nothing, silently producing
    unauthenticated requests. Mirrors
    test_control_hooks_require_http_transport_gate's model_construct pattern,
    since "grpc" is not a registered transport plugin and would otherwise fail
    normal field validation.
    """
    cfg = EndpointConfig.model_construct(
        urls=["https://localhost:8000"],
        auth_type="sigv4",
        aws_region="us-east-1",
        aws_service="sagemaker",
        transport="grpc",
    )
    with pytest.raises(ValueError, match="transport that signs"):
        cfg._validate_sigv4_auth()


def test_valid_sigv4_config_accepted() -> None:
    config = EndpointConfig(
        urls=["https://localhost:8000"],
        auth_type="sigv4",
        aws_region="us-east-1",
        aws_service="sagemaker",
        aws_profile="prod",
    )
    assert config.aws_region == "us-east-1"
    assert config.aws_service == "sagemaker"


def test_config_without_auth_or_aws_fields_accepted() -> None:
    config = EndpointConfig(urls=["http://localhost:8000"])
    assert config.auth_type is None


class TestTransportGateIsAboutSigningNotIdentity:
    """The gate protects against a transport that resolves AWS credentials and
    then signs nothing. That is a question about capability, not about being
    the ``http`` plugin specifically: signing lives on
    ``AioHttpTransport._sign_if_needed``, which subclasses inherit.

    Comparing against ``TransportType.HTTP`` excluded every future signing
    transport by construction -- notably the SageMaker transport (AIP-1177),
    which subclasses ``AioHttpTransport`` and does sign.
    """

    def test_a_transport_deriving_from_the_http_transport_is_accepted(self) -> None:
        from aiperf.transports.aiohttp_transport import AioHttpTransport
        from tests.harness import mock_plugin

        class _SigningTransport(AioHttpTransport):
            pass

        with mock_plugin("transport", "signing-thing", _SigningTransport):
            cfg = EndpointConfig.model_construct(
                urls=["https://x.example.com"],
                auth_type="sigv4",
                aws_region="us-east-1",
                aws_service="execute-api",
                transport="signing-thing",
            )
            cfg._validate_sigv4_auth()

    def test_a_transport_that_cannot_sign_is_still_rejected(self) -> None:
        """The original protection stands."""
        from tests.harness import mock_plugin

        class _NonSigningTransport:
            pass

        with mock_plugin("transport", "silent-thing", _NonSigningTransport):
            cfg = EndpointConfig.model_construct(
                urls=["https://x.example.com"],
                auth_type="sigv4",
                aws_region="us-east-1",
                aws_service="execute-api",
                transport="silent-thing",
            )
            with pytest.raises(ValueError, match="transport that signs"):
                cfg._validate_sigv4_auth()


class TestLoopbackIsExemptFromTheTlsRule:
    """Signing puts the signature and any ``x-amz-security-token`` into the
    headers, so remote cleartext is a credential disclosure. Headers that never
    leave the machine are not, and the documented mock-server workflow -- which
    is the end-to-end gate for AIP-1177 -- signs against ``http://localhost``.
    """

    @pytest.mark.parametrize(
        "url",
        [
            param("http://localhost:8765", id="localhost"),
            param("http://127.0.0.1:8765", id="ipv4-loopback"),
            param("http://[::1]:8765", id="ipv6-loopback"),
        ],
    )  # fmt: skip
    def test_loopback_over_http_is_allowed(self, url: str) -> None:
        cfg = EndpointConfig(
            urls=[url],
            auth_type="sigv4",
            aws_region="us-east-1",
            aws_service="execute-api",
        )
        assert cfg.urls == [url]

    def test_https_is_always_allowed(self) -> None:
        cfg = EndpointConfig(
            urls=["https://runtime.sagemaker.us-east-1.amazonaws.com"],
            auth_type="sigv4",
            aws_region="us-east-1",
            aws_service="execute-api",
        )
        assert cfg.urls[0].startswith("https://")

    def test_one_remote_cleartext_url_among_several_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="https"):
            EndpointConfig(
                urls=["https://good.example.com", "http://bad.example.com"],
                auth_type="sigv4",
                aws_region="us-east-1",
                aws_service="execute-api",
            )

    def test_unsigned_endpoints_may_still_use_remote_http(self) -> None:
        """The rule protects credentials; it is not a blanket TLS mandate."""
        cfg = EndpointConfig(urls=["http://anywhere.example.com"])
        assert cfg.auth_type is None
