# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Server-metrics discovery YAML model tests."""

import pytest
from pydantic import ValidationError

from aiperf.common.enums import ServerMetricsDiscoveryMode
from aiperf.config.server_metrics import (
    ServerMetricsConfig,
    ServerMetricsDiscoveryConfig,
)


def test_discovery_defaults_are_namespaced_auto() -> None:
    config = ServerMetricsDiscoveryConfig()
    assert config.mode == ServerMetricsDiscoveryMode.AUTO
    assert config.namespace is None
    assert config.timeout_seconds == 30.0


def test_discovery_yaml_fields_round_trip() -> None:
    config = ServerMetricsConfig.model_validate(
        {
            "urls": ["http://explicit:9090/metrics"],
            "discovery": {
                "mode": "kubernetes",
                "namespace": "dynamo",
                "label_selector": "app=vllm",
                "timeout_seconds": 5.5,
            },
        }
    )
    assert config.discovery.mode == ServerMetricsDiscoveryMode.KUBERNETES
    assert config.discovery.namespace == "dynamo"
    assert config.discovery.label_selector == "app=vllm"
    assert config.discovery.timeout_seconds == 5.5
    assert ServerMetricsConfig.model_validate(config.model_dump()) == config


def test_disabled_discovery_rejects_kubernetes_options() -> None:
    with pytest.raises(ValidationError, match="label_selector.*namespace"):
        ServerMetricsDiscoveryConfig(
            mode="disabled",
            namespace="dynamo",
            label_selector="app=vllm",
        )


@pytest.mark.parametrize("timeout", [0, -1])
def test_discovery_timeout_must_be_positive(timeout: float) -> None:
    with pytest.raises(ValidationError, match="timeout_seconds"):
        ServerMetricsDiscoveryConfig(timeout_seconds=timeout)
