# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end threading of the SageMaker flags, CLI through to the transport.

A new endpoint flag has to be declared in several places to reach the worker.
Missing the ``EndpointInfo`` hop fails *silently*: ``AIPerfBaseModel`` allows
extra fields, so tests that construct an ``EndpointInfo`` by hand still pass
while ``from_run`` quietly drops the value in production, and the transport
builds ``/endpoints//invocations``.

These tests therefore assert on declared fields and on ``from_run`` output,
never on hand-built objects.
"""

import pytest
from pytest import param

from aiperf.common.models.model_endpoint_info import EndpointInfo

_SAGEMAKER_FIELDS = (
    "sagemaker_endpoint_name",
    "sagemaker_target_model",
    "sagemaker_inference_component_name",
    "sagemaker_target_variant",
)


@pytest.mark.parametrize("name", [param(f, id=f) for f in _SAGEMAKER_FIELDS])
def test_endpoint_info_declares_the_field(name: str) -> None:
    """Declared, not tolerated as an extra -- an extra would read back fine in
    a hand-built object and be absent everywhere that matters."""
    assert name in EndpointInfo.model_fields


def test_cli_flags_reach_the_endpoint_info() -> None:
    """Drives the whole CLI -> EndpointConfig -> EndpointInfo path, which is
    the only way to catch a missing hop: each one in isolation looks fine."""
    from aiperf.common.models import ModelEndpointInfo
    from aiperf.config.flags.cli_config import CLIConfig
    from tests.unit.conftest import make_run_from_cli

    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            sagemaker_endpoint_name="my-ep",
            sagemaker_target_model="pinned.tar.gz",
            sagemaker_inference_component_name="comp-1",
            sagemaker_target_variant="variant-b",
            aws_region="us-west-2",
        )
    )
    info = ModelEndpointInfo.from_run(run).endpoint

    assert info.sagemaker_endpoint_name == "my-ep"
    assert info.sagemaker_target_model == "pinned.tar.gz"
    assert info.sagemaker_inference_component_name == "comp-1"
    assert info.sagemaker_target_variant == "variant-b"
