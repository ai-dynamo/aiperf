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

    def test_a_non_sagemaker_transport_with_an_endpoint_name_names_the_conflict(
        self,
    ) -> None:
        """``--sagemaker-endpoint-name`` selects the SageMaker transport, so
        naming a different transport is a contradiction, not an override.

        It was already rejected -- but by whichever unrelated guard fired
        first, and both blame the wrong flag. With a region it was "--aws-region
        has no effect unless --auth-type is set to 'sigv4'"; without one,
        "SageMaker endpoints require --aws-region". Neither mentions the
        transport, which is the actual problem.

        Asserting on the message is the whole point: a bare
        ``pytest.raises(ValidationError)`` passes against either of those and
        proves nothing.
        """
        for region in ("us-west-2", None):
            with pytest.raises(ValidationError) as exc_info:
                _endpoint(transport="http", aws_region=region)

            message = str(exc_info.value)
            assert "--transport" in message, message
            assert "--sagemaker-endpoint-name" in message, message
            assert "has no effect" not in message, message

    def test_missing_region_names_the_flag(self) -> None:
        with pytest.raises(ValidationError, match="--aws-region"):
            _endpoint(aws_region=None)

    def test_signing_scope_is_not_required_from_the_user(self) -> None:
        """The signing name is derivable from the transport's botocore service
        id, so requiring it here would be busywork -- unlike a bare
        `--auth-type sigv4` against an arbitrary host, which still needs it.

        Validating at all is the assertion: a missing scope would raise.
        """
        assert _endpoint().aws_service is None

    def test_explicit_signing_scope_still_wins(self) -> None:
        cfg = _endpoint(aws_service="execute-api")
        assert cfg.aws_service == "execute-api"


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
