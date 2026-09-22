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


def test_the_runtime_url_is_derived_on_the_pure_cli_path() -> None:
    """The documented one-flag quick start must actually reach SageMaker.

    ``--url`` defaults to ``http://localhost:8000`` with ``min_length=1`` and
    ``validate_default=True``, and the converter passed it through
    unconditionally. The SageMaker before-validator only derives the runtime
    host when ``urls`` is empty, so on the pure-CLI path it never fired: the
    benchmark was signed for SageMaker and sent to localhost.

    Asserting the derived host rather than "not localhost" is deliberate --
    the region has to reach the hostname too.
    """
    from aiperf.config.flags.cli_config import CLIConfig
    from tests.unit.conftest import make_run_from_cli

    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            sagemaker_endpoint_name="my-ep",
            aws_region="us-west-2",
        )
    )

    assert run.cfg.endpoint.urls == [
        "https://runtime.sagemaker.us-west-2.amazonaws.com"
    ]


def test_an_explicit_url_still_wins_on_the_cli_path() -> None:
    """Deriving must not override a VPC/PrivateLink endpoint or custom domain
    the user passed deliberately."""
    from aiperf.config.flags.cli_config import CLIConfig
    from tests.unit.conftest import make_run_from_cli

    run = make_run_from_cli(
        CLIConfig(
            model_names=["test-model"],
            sagemaker_endpoint_name="my-ep",
            aws_region="us-west-2",
            urls=["https://vpce-123.execute-api.us-west-2.vpce.amazonaws.com"],
        )
    )

    assert run.cfg.endpoint.urls == [
        "https://vpce-123.execute-api.us-west-2.vpce.amazonaws.com"
    ]
