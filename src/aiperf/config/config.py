# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - Root Configuration Model

This module defines the root AIPerfConfig model that brings together
all configuration sections into a single, validated configuration object.

The AIPerfConfig class is the primary entry point for loading and
working with AIPerf YAML configuration files.

Example Usage:
    >>> from aiperf.config import load_config
    >>> config = load_config("benchmark.yaml")
    >>> print(config.models)
    >>> for name, phase in config.phases.items():
    ...     print(f"{name}: {phase.dataset}")

    Or programmatically:
    >>> from aiperf.config import AIPerfConfig
    >>> config = AIPerfConfig(
    ...     models=["llama-3-8b"],
    ...     endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    ...     datasets={"main": {"type": "synthetic", "count": 1000, "prompts": {"isl": 512}}},
    ...     phases={"profiling": {"type": "concurrency", "dataset": "main", "requests": 100, "concurrency": 8}}
    ... )
"""

from __future__ import annotations

from typing import Annotated, Any, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from aiperf.config._base import BaseConfig
from aiperf.config._benchmark_helpers import BenchmarkHelpersMixin
from aiperf.config._benchmark_normalizers import (
    normalize_benchmark_input,
    parse_datasets_input,
)
from aiperf.config._comm import build_comm_config
from aiperf.config.artifacts import (
    ArtifactsConfig,
    GpuTelemetryConfig,
    ServerMetricsConfig,
)
from aiperf.config.dataset import (
    DatasetConfig,
)
from aiperf.config.endpoint import (
    EndpointConfig,
)
from aiperf.config.metrics import MetricsConfig
from aiperf.config.models import (
    AccuracyConfig,
    LoggingConfig,
    ModelsAdvanced,
    MultiRunConfig,
    RuntimeConfig,
    SLOsConfig,
    TokenizerConfig,
)
from aiperf.config.phases import (
    BasePhaseConfig,
    PhaseConfig,
)
from aiperf.config.sweep import SweepConfig

__all__ = [
    "AIPerfConfig",
    "BenchmarkConfig",
    "build_comm_config",
]


class BenchmarkConfig(BaseConfig, BenchmarkHelpersMixin):
    """Pure runtime configuration - what SystemController and services need.

    Contains all fields required to execute a single benchmark run.
    Does NOT include sweep or multi_run settings (those live on AIPerfConfig).

    Required Sections:
        - models: Model(s) to benchmark
        - endpoint: Server connection settings
        - datasets: Named data sources
        - phases: Benchmark phase configuration (single or named phases)

    Optional Sections:
        - artifacts: Export and console settings
        - slos: SLO-based quality metrics (generic dict)
        - tokenizer: Token counting configuration
        - gpu_telemetry: GPU metrics from DCGM endpoints
        - server_metrics: Server metrics from Prometheus endpoints
        - runtime: Worker and communication settings
        - logging: Logging and debug settings
        - accuracy: Accuracy evaluation configuration

    Global Settings:
        - random_seed: Global seed for reproducibility

    Note:
        When a phase doesn't specify a dataset, the first dataset defined
        in the datasets section is used as the default.

    Validation:
        The configuration is validated in several stages:
        1. Individual field validation (types, ranges, formats)
        2. Dataset reference validation (phase configs reference existing datasets)
        3. Cross-field validation (mutual exclusivity, dependencies)

    Environment Variables:
        Values can reference environment variables using ${VAR} syntax.
        Optional defaults: ${VAR:default_value}

        Example:
            api_key: ${OPENAI_API_KEY}
            api_key: ${OPENAI_API_KEY:sk-default}
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
    )

    # ==========================================================================
    # REQUIRED SECTIONS
    # ==========================================================================

    models: Annotated[
        ModelsAdvanced,
        Field(
            description="Model configuration. Accepts a single model name string, "
            "a list of model names, or an advanced configuration with strategy "
            "and weighted items. All forms are normalized to ModelsAdvanced.",
        ),
    ]

    endpoint: Annotated[
        EndpointConfig,
        Field(
            description="Endpoint configuration for connecting to inference server(s). "
            "Includes URLs, API type, authentication, timeout, and connection settings.",
        ),
    ]

    datasets: Annotated[
        dict[str, DatasetConfig],
        Field(
            min_length=1,
            description="Named dataset configurations. Keys are dataset names that can be "
            "referenced in phases. Values are dataset configs (synthetic, file, public, "
            "or composed with source+augment).",
        ),
    ]

    phases: Annotated[
        dict[str, PhaseConfig],
        Field(
            min_length=1,
            description="Benchmark phase configuration. Can be a single phase config "
            "(with 'type' key) or named phases (dict of phase configs). "
            "Single config is normalized to {'default': config}. "
            "Order is preserved (Python 3.7+) for execution sequence.",
        ),
    ]

    # ==========================================================================
    # OPTIONAL SECTIONS
    # ==========================================================================

    artifacts: Annotated[
        ArtifactsConfig,
        Field(
            default_factory=ArtifactsConfig,
            description="Artifacts configuration for benchmark output. "
            "Controls output directory and export formats.",
        ),
    ]

    slos: Annotated[
        SLOsConfig | None,
        Field(
            default=None,
            description="SLO (Service Level Objectives) configuration as a generic dict. "
            "Maps metric names to threshold values. "
            "A request is counted as good only if it meets ALL specified thresholds.",
        ),
    ]

    tokenizer: Annotated[
        TokenizerConfig | None,
        Field(
            default=None,
            description="Tokenizer configuration for token counting. "
            "Used for ISL/OSL enforcement and accurate metrics. "
            "If not specified, uses the first model name.",
        ),
    ]

    gpu_telemetry: Annotated[
        GpuTelemetryConfig,
        Field(
            default_factory=GpuTelemetryConfig,
            description="GPU telemetry configuration for DCGM metrics collection. "
            "Collects GPU metrics (power, utilization, temperature) from DCGM endpoints. "
            "Enabled by default. Set enabled: false to disable.",
        ),
    ]

    server_metrics: Annotated[
        ServerMetricsConfig,
        Field(
            default_factory=ServerMetricsConfig,
            description="Server metrics configuration for Prometheus scraping. "
            "Collects operational metrics (queue depth, KV cache, batch sizes) "
            "from inference server Prometheus endpoints. "
            "Enabled by default. Set enabled: false to disable.",
        ),
    ]

    runtime: Annotated[
        RuntimeConfig,
        Field(
            default_factory=RuntimeConfig,
            description="Runtime configuration for worker processes and "
            "inter-process communication.",
        ),
    ]

    logging: Annotated[
        LoggingConfig,
        Field(
            default_factory=LoggingConfig,
            description="Logging configuration for verbosity and debug settings.",
        ),
    ]

    metrics: Annotated[
        MetricsConfig,
        Field(
            default_factory=MetricsConfig,
            description="Metrics aggregation configuration for benchmark summaries.",
        ),
    ]

    accuracy: Annotated[
        AccuracyConfig | None,
        Field(
            default=None,
            description="Accuracy benchmarking configuration. "
            "When set, enables accuracy evaluation alongside performance profiling.",
        ),
    ]

    # ==========================================================================
    # GLOBAL SETTINGS
    # ==========================================================================

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Global random seed for reproducibility. "
            "Can be overridden per-dataset. "
            "If not set, uses system entropy.",
        ),
    ]

    # ==========================================================================
    # VALIDATORS
    # ==========================================================================

    @model_validator(mode="before")
    @classmethod
    def normalize_before_validation(cls, data: Any) -> Any:
        """Normalize input data before Pydantic validation.

        Handles singular/plural aliases and warmup/profiling-to-phases
        shorthand. See `_benchmark_normalizers.normalize_benchmark_input`.
        """
        return normalize_benchmark_input(data)

    @field_validator("phases", mode="before")
    @classmethod
    def parse_phases(cls, v: Any) -> dict[str, Any]:
        """Parse phase configurations from dict format.

        Injects the phase name from the dict key into each phase config.
        """
        if not isinstance(v, dict):
            raise ValueError("phases must be a dictionary with phase names as keys")

        result = {}
        for name, config in v.items():
            if isinstance(config, BasePhaseConfig):
                config._name = name
                result[name] = config
            elif isinstance(config, dict):
                result[name] = config
            else:
                raise ValueError(f"Phase config '{name}' must be a dictionary")

        return result

    @field_validator("datasets", mode="before")
    @classmethod
    def parse_datasets(cls, v: Any) -> dict[str, Any]:
        """Parse dataset configurations, handling composed datasets.

        See `_benchmark_normalizers.parse_datasets_input`.
        """
        return parse_datasets_input(v)

    @model_validator(mode="after")
    def inject_phase_names(self) -> Self:
        """Inject phase names from dict keys into PhaseConfig objects."""
        for name, phase in self.phases.items():
            phase._name = name
        return self

    @model_validator(mode="after")
    def validate_dataset_references(self) -> Self:
        """Validate that all dataset references in phase configs exist."""
        dataset_names = set(self.datasets.keys())

        for name, phase in self.phases.items():
            if phase.dataset is not None and phase.dataset not in dataset_names:
                raise ValueError(
                    f"Phase config '{name}' references undefined dataset '{phase.dataset}'. "
                    f"Available datasets: {sorted(dataset_names)}"
                )

        return self

    @model_validator(mode="after")
    def validate_seamless_not_on_first_phase(self) -> Self:
        """Ensure seamless is not enabled on the first phase config."""
        if self.phases:
            first_name = next(iter(self.phases.keys()))
            first_phase = self.phases[first_name]
            if first_phase.seamless:
                raise ValueError(
                    f"Phase config '{first_name}' cannot have seamless=True because it is first. "
                    "Seamless transitions only apply to subsequent phase configs."
                )
        return self

    @model_validator(mode="after")
    def validate_prefill_requires_streaming(self) -> Self:
        """Prefill concurrency requires streaming to measure TTFT boundaries."""
        for name, phase in self.phases.items():
            if phase.prefill_concurrency is not None and not self.endpoint.streaming:
                raise ValueError(
                    f"Phase '{name}': prefill_concurrency requires endpoint.streaming=true"
                )
        return self

    @model_validator(mode="after")
    def validate_phase_dataset_compatibility(self) -> Self:
        """Validate that each phase is compatible with its dataset.

        Checks sampling strategy requirements (e.g., fixed_schedule needs sequential)
        and format requirements (e.g., user_centric needs multi_turn).
        """
        from aiperf.config.resolved import check_phase_dataset_compatibility

        for phase_name, phase in self.phases.items():
            dataset_name = phase.dataset or self.get_default_dataset_name()
            ds = self.datasets.get(dataset_name)
            if ds is None:
                continue
            errors = check_phase_dataset_compatibility(
                phase, ds, phase_name, dataset_name
            )
            if errors:
                raise ValueError(errors[0])
        return self


class AIPerfConfig(BenchmarkConfig):
    """Full YAML schema - adds sweep and multi_run on top of BenchmarkConfig.

    This is the primary entry point for loading YAML configuration files.
    After sweep expansion, each variation becomes a BenchmarkConfig.
    """

    sweep: Annotated[
        SweepConfig | None,
        Field(
            default=None,
            description="Sweep configuration for parameter exploration. "
            "Supports grid (Cartesian product), scenarios (hand-picked), "
            "and sequential (ordered) sweep strategies.",
        ),
    ]

    multi_run: Annotated[
        MultiRunConfig,
        Field(
            default_factory=MultiRunConfig,
            description="Multi-run benchmarking configuration for statistical reporting. "
            "When num_runs > 1, executes multiple runs and computes aggregate statistics.",
        ),
    ]
