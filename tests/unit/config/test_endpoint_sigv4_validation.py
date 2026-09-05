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


def test_valid_sigv4_config_accepted() -> None:
    config = EndpointConfig(
        urls=["http://localhost:8000"],
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
