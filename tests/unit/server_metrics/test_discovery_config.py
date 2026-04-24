# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ServerMetricsDiscoveryConfig and its integration with ServerMetricsConfig."""

import pytest
from pydantic import ValidationError
from pytest import param

from aiperf.common.enums import ServerMetricsDiscoveryMode
from aiperf.config import (
    AIPerfConfig,
    ServerMetricsConfig,
    ServerMetricsDiscoveryConfig,
)

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    datasets={
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
)


class TestServerMetricsDiscoveryConfig:
    """Test ServerMetricsDiscoveryConfig model validation."""

    def test_defaults(self):
        cfg = ServerMetricsDiscoveryConfig()
        assert cfg.mode == ServerMetricsDiscoveryMode.AUTO
        assert cfg.label_selector is None
        assert cfg.namespace is None

    @pytest.mark.parametrize(
        "mode",
        [
            param(ServerMetricsDiscoveryMode.AUTO, id="auto"),
            param(ServerMetricsDiscoveryMode.KUBERNETES, id="kubernetes"),
            param(ServerMetricsDiscoveryMode.DISABLED, id="disabled"),
        ],
    )  # fmt: skip
    def test_mode_values(self, mode: ServerMetricsDiscoveryMode):
        cfg = ServerMetricsDiscoveryConfig(mode=mode)
        assert cfg.mode == mode

    def test_case_insensitive_mode(self):
        cfg = ServerMetricsDiscoveryConfig(mode="AUTO")
        assert cfg.mode == ServerMetricsDiscoveryMode.AUTO

    def test_label_selector_and_namespace(self):
        cfg = ServerMetricsDiscoveryConfig(
            mode="kubernetes",
            label_selector="app=vllm,env=prod",
            namespace="inference",
        )
        assert cfg.label_selector == "app=vllm,env=prod"
        assert cfg.namespace == "inference"

    def test_disabled_rejects_label_selector(self):
        with pytest.raises(ValidationError, match="label_selector"):
            ServerMetricsDiscoveryConfig(mode="disabled", label_selector="app=vllm")

    def test_disabled_rejects_namespace(self):
        with pytest.raises(ValidationError, match="namespace"):
            ServerMetricsDiscoveryConfig(mode="disabled", namespace="inference")

    def test_disabled_rejects_both(self):
        with pytest.raises(ValidationError, match="label_selector.*namespace"):
            ServerMetricsDiscoveryConfig(
                mode="disabled",
                label_selector="app=vllm",
                namespace="inference",
            )

    def test_disabled_without_k8s_options_is_valid(self):
        cfg = ServerMetricsDiscoveryConfig(mode="disabled")
        assert cfg.mode == ServerMetricsDiscoveryMode.DISABLED

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError, match="extra"):
            ServerMetricsDiscoveryConfig(mode="auto", bogus="field")


class TestServerMetricsConfigWithDiscovery:
    """Test discovery field on ServerMetricsConfig."""

    def test_default_discovery(self):
        cfg = ServerMetricsConfig()
        assert cfg.discovery.mode == ServerMetricsDiscoveryMode.AUTO

    def test_explicit_discovery_block(self):
        cfg = ServerMetricsConfig(
            discovery={
                "mode": "kubernetes",
                "label_selector": "app=vllm",
                "namespace": "inference",
            }
        )
        assert cfg.discovery.mode == ServerMetricsDiscoveryMode.KUBERNETES
        assert cfg.discovery.label_selector == "app=vllm"
        assert cfg.discovery.namespace == "inference"

    def test_string_shorthand_preserves_default_discovery(self):
        cfg = ServerMetricsConfig.model_validate("http://localhost:9090/metrics")
        assert cfg.urls == ["http://localhost:9090/metrics"]
        assert cfg.discovery.mode == ServerMetricsDiscoveryMode.AUTO

    def test_urls_plus_discovery(self):
        cfg = ServerMetricsConfig(
            urls=["http://server1:8000/metrics"],
            discovery={"mode": "kubernetes"},
        )
        assert cfg.urls == ["http://server1:8000/metrics"]
        assert cfg.discovery.mode == ServerMetricsDiscoveryMode.KUBERNETES


class TestAIPerfConfigWithDiscovery:
    """Test discovery integration in full AIPerfConfig."""

    def test_default_discovery_in_full_config(self):
        cfg = AIPerfConfig(**_BASE)
        assert cfg.server_metrics.discovery.mode == ServerMetricsDiscoveryMode.AUTO

    def test_yaml_style_discovery_in_full_config(self):
        cfg = AIPerfConfig(
            **_BASE,
            server_metrics={
                "urls": ["http://server:8000/metrics"],
                "discovery": {
                    "mode": "kubernetes",
                    "namespace": "inference",
                },
            },
        )
        assert (
            cfg.server_metrics.discovery.mode == ServerMetricsDiscoveryMode.KUBERNETES
        )
        assert cfg.server_metrics.discovery.namespace == "inference"
        assert cfg.server_metrics.urls == ["http://server:8000/metrics"]

    def test_disabled_discovery_in_full_config(self):
        cfg = AIPerfConfig(
            **_BASE,
            server_metrics={
                "urls": ["http://server:8000/metrics"],
                "discovery": {"mode": "disabled"},
            },
        )
        assert cfg.server_metrics.discovery.mode == ServerMetricsDiscoveryMode.DISABLED

    def test_serialization_roundtrip(self):
        cfg = AIPerfConfig(
            **_BASE,
            server_metrics={
                "discovery": {
                    "mode": "kubernetes",
                    "label_selector": "app=vllm",
                    "namespace": "inference",
                },
            },
        )
        data = cfg.model_dump(exclude_none=True)
        restored = AIPerfConfig.model_validate(data)
        assert (
            restored.server_metrics.discovery.mode
            == ServerMetricsDiscoveryMode.KUBERNETES
        )
        assert restored.server_metrics.discovery.label_selector == "app=vllm"
        assert restored.server_metrics.discovery.namespace == "inference"
