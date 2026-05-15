# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Pydantic Models

Server Metrics - Prometheus scraping configuration and Kubernetes discovery.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.enums import (
    ServerMetricsDiscoveryMode,
    ServerMetricsFormat,
)
from aiperf.config.base import BaseConfig

__all__ = [
    "ServerMetricsConfig",
    "ServerMetricsDiscoveryConfig",
]


class ServerMetricsDiscoveryConfig(BaseConfig):
    """Kubernetes-based auto-discovery of inference-server /metrics endpoints.

    When mode is 'auto' or 'kubernetes', queries the K8s API for pods that
    are recognizable inference servers (vLLM, SGLang, Triton Inference Server,
    TensorRT-LLM, NVIDIA Dynamo). Eligibility (any one is enough):
    1. Dynamo opt-in label: nvidia.com/metrics-enabled=true
    2. AIPerf opt-in annotation: aiperf.nvidia.com/metrics-paths=...
    3. A container image matching a known inference-server signature
    4. User-provided label_selector (server-side filter)

    The broad ``prometheus.io/scrape=true`` annotation is intentionally NOT a
    trigger: Loki, Grafana, kube-state-metrics, and many platform components
    set it without being inference servers. ``prometheus.io/{port,path,scheme}``
    are still honored to construct the scrape URL when an eligible pod sets them.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    mode: Annotated[
        ServerMetricsDiscoveryMode,
        Field(
            default=ServerMetricsDiscoveryMode.AUTO,
            description="Discovery mode: 'auto' detects environment and tries K8s "
            "if in-cluster, 'kubernetes' forces K8s API discovery, "
            "'disabled' uses only explicit URLs.",
        ),
    ]

    label_selector: Annotated[
        str | None,
        Field(
            default=None,
            description="Kubernetes label selector for discovery. "
            "Example: 'app=vllm,env=prod'. Applied in addition to "
            "built-in Dynamo and Prometheus discovery.",
        ),
    ]

    namespace: Annotated[
        str | None,
        Field(
            default=None,
            description="Kubernetes namespace to search. "
            "If not specified, searches all namespaces.",
        ),
    ]

    @model_validator(mode="after")
    def validate_discovery_options(self) -> ServerMetricsDiscoveryConfig:
        """Validate that K8s-specific options aren't set when discovery is disabled."""
        if self.mode == ServerMetricsDiscoveryMode.DISABLED:
            k8s_options = []
            if self.label_selector is not None:
                k8s_options.append("label_selector")
            if self.namespace is not None:
                k8s_options.append("namespace")
            if k8s_options:
                msg = (
                    f"{', '.join(k8s_options)} can only be used when "
                    "discovery mode is 'auto' or 'kubernetes'."
                )
                raise ValueError(msg)
        return self


class ServerMetricsConfig(BaseConfig):
    """
    Server metrics configuration for Prometheus scraping.

    Collects server-side operational metrics (queue depth, KV cache utilization,
    batch sizes, GPU memory) from Prometheus endpoints exposed by inference servers
    like vLLM, TensorRT-LLM, or Triton.

    Accepts shorthand forms:
        - String URL: "http://localhost:9090/metrics"
          → ServerMetricsConfig(enabled=True, urls=["http://localhost:9090/metrics"])
        - Singular url field: {url: "..."}
          → ServerMetricsConfig(urls=["..."])
    """

    # x-kubernetes-preserve-unknown-fields lets apiserver accept the
    # string-URL shorthand (collapsed to the full object form by
    # normalize_before_validation) which a Kubernetes structural schema
    # cannot express as a string|object union.
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
    )

    enabled: Annotated[
        bool,
        Field(
            default=True,
            description="Enable Prometheus metrics scraping. Set to false to disable.",
        ),
    ]

    urls: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Prometheus metrics endpoint URLs to scrape. "
            "Typically the /metrics endpoint on inference servers.",
        ),
    ]

    formats: Annotated[
        list[ServerMetricsFormat],
        Field(
            default_factory=lambda: [
                ServerMetricsFormat.JSON,
                ServerMetricsFormat.CSV,
            ],
            description="Export formats for scraped metrics. "
            "Options: json, csv, parquet, jsonl.",
        ),
    ]

    discovery: Annotated[
        ServerMetricsDiscoveryConfig,
        Field(
            default_factory=ServerMetricsDiscoveryConfig,
            description="Auto-discovery of Prometheus endpoints in Kubernetes. "
            "Discovers pods via Dynamo labels, Prometheus annotations, "
            "or custom label selectors.",
        ),
    ]

    @model_validator(mode="before")
    @classmethod
    def normalize_before_validation(cls, data: Any) -> Any:
        """Normalize shorthand forms before validation.

        Handles:
            - String URL → full config dict with that URL
            - url → urls (singular to plural)
        """
        # String URL → full config with that URL
        if isinstance(data, str):
            return {"enabled": True, "urls": [data]}

        if not isinstance(data, dict):
            return data

        # url → urls (singular to plural)
        if "url" in data and "urls" not in data:
            url = data.pop("url")
            data["urls"] = [url] if isinstance(url, str) else url

        return data
