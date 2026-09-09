# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything `--sagemaker-endpoint-name` derives.

The target UX is one flag plus a region:

    aiperf profile -m my-model --sagemaker-endpoint-name my-ep --aws-region us-west-2

Transport, signer, signing scope and base URL all fall out of that. These tests
pin each derivation separately so a failure names which one broke, and they run
against ``EndpointConfig`` rather than a built transport so the derived values
are also what lands in exported config artifacts.
"""

import pytest
from pydantic import ValidationError

from aiperf.config.endpoint import EndpointConfig
from aiperf.plugin.enums import RequestSignerType, TransportType


def _endpoint(**overrides) -> EndpointConfig:
    data: dict = {
        "type": "chat",
        "sagemaker": {"endpoint_name": "my-ep"},
        "aws_region": "us-west-2",
    }
    data.update(overrides)
    return EndpointConfig.model_validate(data)


class TestSelectorDerivation:
    def test_endpoint_name_selects_the_sagemaker_transport(self) -> None:
        assert _endpoint().transport == TransportType.SAGEMAKER

    def test_sagemaker_transport_selects_sigv4_signing(self) -> None:
        assert _endpoint().auth_type == RequestSignerType.SIGV4

    def test_explicit_transport_is_not_overridden(self) -> None:
        """Deriving must never clobber something the user set on purpose.

        Plain http + sigv4 has no transport to derive the signing scope from,
        so this combination legitimately still requires it explicitly.
        """
        cfg = _endpoint(
            transport="http", auth_type="sigv4", aws_signing_service="sagemaker"
        )
        assert cfg.transport == TransportType.HTTP

    def test_missing_region_names_the_flag(self) -> None:
        with pytest.raises(ValidationError, match="--aws-region"):
            _endpoint(aws_region=None)

    def test_signing_scope_is_not_required_from_the_user(self) -> None:
        """The signing name is derivable from the transport's botocore service
        id, so requiring it here would be busywork -- unlike a bare
        `--auth-type sigv4` against an arbitrary host, which still needs it.

        Validating at all is the assertion: a missing scope would raise.
        """
        assert _endpoint().aws_signing_service is None

    def test_explicit_signing_scope_still_wins(self) -> None:
        cfg = _endpoint(aws_signing_service="execute-api")
        assert cfg.aws_signing_service == "execute-api"


class TestBaseUrlDerivation:
    def test_url_is_derived_from_the_region(self) -> None:
        assert _endpoint().urls == ["https://runtime.sagemaker.us-west-2.amazonaws.com"]

    def test_china_region_uses_the_china_suffix(self) -> None:
        cfg = _endpoint(aws_region="cn-north-1")
        assert cfg.urls == ["https://runtime.sagemaker.cn-north-1.amazonaws.com.cn"]

    def test_explicit_url_wins(self) -> None:
        """Required for VPC/PrivateLink endpoints and custom domains."""
        cfg = _endpoint(
            urls=["https://vpce-123.sagemaker.us-west-2.vpce.amazonaws.com"]
        )
        assert cfg.urls == ["https://vpce-123.sagemaker.us-west-2.vpce.amazonaws.com"]


class TestSageMakerFieldsAreOptional:
    def test_multi_model_and_component_fields_default_to_none(self) -> None:
        cfg = _endpoint()
        assert cfg.sagemaker.target_model is None
        assert cfg.sagemaker.inference_component_name is None
        assert cfg.sagemaker.target_variant is None
