# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration tests for per-session unique prompt prefixes."""

import pytest
from pydantic import ValidationError

from aiperf.config.endpoint import EndpointConfig
from aiperf.config.flags._converter_endpoint import build_endpoint
from aiperf.config.flags.cli_config import CLIConfig


def test_unique_session_prefix_disabled_by_default() -> None:
    endpoint = EndpointConfig(urls=["http://localhost:8000"])

    assert endpoint.unique_session_prefix_length == 0


def test_cli_converter_preserves_unique_session_prefix_length() -> None:
    endpoint = build_endpoint(CLIConfig(unique_session_prefix_length=64))

    assert endpoint["unique_session_prefix_length"] == 64


def test_camel_case_yaml_contract_resolves() -> None:
    endpoint = EndpointConfig.model_validate(
        {
            "urls": ["http://localhost:8000"],
            "uniqueSessionPrefixLength": 64,
        }
    )

    assert endpoint.unique_session_prefix_length == 64


def test_negative_unique_session_prefix_length_rejected() -> None:
    with pytest.raises(ValidationError):
        EndpointConfig(
            urls=["http://localhost:8000"],
            unique_session_prefix_length=-1,
        )
