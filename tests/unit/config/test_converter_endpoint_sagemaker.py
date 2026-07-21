# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for AWS/SigV4/SageMaker-related CLI flags flowing through the endpoint converter."""

from __future__ import annotations

from aiperf.config.flags._converter_endpoint import build_endpoint
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.plugin.enums import RequestSignerType


def test_sagemaker_flags_flow_through_converter() -> None:
    cli = CLIConfig(
        urls=["https://my-api.execute-api.us-east-1.amazonaws.com"],
        auth_type=RequestSignerType.SIGV4,
        aws_region="eu-west-1",
        aws_profile="my-profile",
        aws_service="sagemaker",
        sagemaker_inference_component_name="my-component",
        sagemaker_target_model="model.tar.gz",
    )

    endpoint = build_endpoint(cli)

    assert endpoint["auth_type"] == RequestSignerType.SIGV4
    assert endpoint["aws_region"] == "eu-west-1"
    assert endpoint["aws_profile"] == "my-profile"
    assert endpoint["aws_service"] == "sagemaker"
    assert endpoint["sagemaker_inference_component_name"] == "my-component"
    assert endpoint["sagemaker_target_model"] == "model.tar.gz"


def test_sagemaker_fields_absent_when_unset() -> None:
    cli = CLIConfig(urls=["http://localhost:8000"])

    endpoint = build_endpoint(cli)

    assert "auth_type" not in endpoint
    assert "aws_region" not in endpoint
    assert "aws_profile" not in endpoint
    assert "aws_service" not in endpoint
    assert "sagemaker_inference_component_name" not in endpoint
    assert "sagemaker_target_model" not in endpoint


def test_endpoint_config_accepts_sagemaker_fields() -> None:
    from aiperf.config.endpoint import EndpointConfig

    endpoint = EndpointConfig(
        urls=["https://my-api.execute-api.us-east-1.amazonaws.com"],
        auth_type=RequestSignerType.SIGV4,
        aws_region="us-east-1",
        aws_service="sagemaker",
        sagemaker_inference_component_name="my-component",
    )

    assert endpoint.aws_region == "us-east-1"
    assert endpoint.aws_service == "sagemaker"
    assert endpoint.sagemaker_inference_component_name == "my-component"
