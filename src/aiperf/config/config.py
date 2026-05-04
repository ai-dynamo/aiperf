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
    >>> for phase in config.phases:
    ...     print(f"{phase.name}: {phase.dataset}")

    Or programmatically:
    >>> from aiperf.config import AIPerfConfig
    >>> config = AIPerfConfig(
    ...     models=["llama-3-8b"],
    ...     endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    ...     datasets={"main": {"type": "synthetic", "count": 1000, "prompts": {"isl": 512}}},
    ...     phases=[{"name": "profiling", "type": "concurrency", "dataset": "main", "requests": 100, "concurrency": 8}]
    ... )
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

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
    Does NOT include sweep, multi_run, variables, or random_seed settings
    (those live on AIPerfConfig as envelope-level cross-variation fields).

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
            json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
        ),
    ]

    model: Annotated[
        Any | None,
        Field(
            default=None,
            exclude=True,
            json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
            description=(
                "Shorthand sibling for `models`. Accepts a string, list of strings, "
                "or ModelsAdvanced object. Hoisted into `models` by the before-"
                "validator and not present after validation."
            ),
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
        list[DatasetConfig],
        Field(
            min_length=1,
            description="Named dataset configurations. Each entry must have a unique 'name' "
            "(e.g. 'main', 'eval'). Phases reference datasets by name "
            "(`phase.dataset = '<name>'`); when omitted, the FIRST dataset in the list is used. "
            "Singular `dataset:` shorthand at the BenchmarkConfig top level is normalized to "
            "a one-entry list with name='default'.",
        ),
    ]

    dataset: Annotated[
        Any | None,
        Field(
            default=None,
            exclude=True,
            json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
            description=(
                "Shorthand sibling for `datasets`. Accepts a single dataset config "
                "(dict). Hoisted into `datasets` as a one-entry list with "
                "name='default' by the before-validator and not present after "
                "validation."
            ),
        ),
    ]

    phases: Annotated[
        list[PhaseConfig],
        Field(
            min_length=1,
            description="Ordered benchmark phases. Each entry must have a unique 'name' "
            "(e.g. 'warmup', 'profiling'). Order in the list IS the execution order; "
            "the first phase runs first. Single-config shorthand "
            "({'type': 'concurrency', ...}) is normalized to a list of one. "
            "Top-level 'warmup:'/'profiling:' shorthand is normalized to a "
            "[warmup, profiling] list pre-validation.",
        ),
    ]

    warmup: Annotated[
        Any | None,
        Field(
            default=None,
            exclude=True,
            json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
            description=(
                "Shorthand sibling for `phases`. Accepts a phase config dict; "
                "rolled into `phases` as the warmup entry by the before-validator "
                "and not present after validation. Mutually exclusive with "
                "`phases`; requires `profiling` alongside it."
            ),
        ),
    ]

    profiling: Annotated[
        Any | None,
        Field(
            default=None,
            exclude=True,
            json_schema_extra={"x-kubernetes-preserve-unknown-fields": True},
            description=(
                "Shorthand sibling for `phases`. Accepts a phase config dict; "
                "rolled into `phases` as the profiling entry by the before-"
                "validator and not present after validation. Mutually exclusive "
                "with `phases`."
            ),
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
    def parse_phases(cls, v: Any) -> list[Any]:
        """Validate that phases is a list (post-normalizer shape).

        The dict shape is rejected with a migration-pointing message; valid
        shorthand inputs (`warmup:` / `profiling:` top-level, or a single
        flat config under `phases:`) are converted to lists by the
        pre-model normalizers in `_benchmark_normalizers`.
        """
        if isinstance(v, dict):
            raise ValueError(
                "phases must be a list of named phase configs (e.g. "
                "[{name: warmup, ...}, {name: profiling, ...}]); the legacy "
                "dict shape is no longer supported. See "
                "docs/tutorials/yaml-config.md#phases for the new shape."
            )
        if not isinstance(v, list):
            raise ValueError(
                f"phases must be a list of named phase configs, got {type(v).__name__}; "
                "see docs/tutorials/yaml-config.md#phases for the expected shape."
            )
        return v

    @field_validator("datasets", mode="before")
    @classmethod
    def parse_datasets(cls, v: Any) -> list[Any]:
        """Parse dataset configurations into a list shape, validating each item has a name.

        See `_benchmark_normalizers.parse_datasets_input`.
        """
        return parse_datasets_input(v)

    @model_validator(mode="after")
    def validate_phase_names_unique(self) -> Self:
        """Reject duplicate phase names — they must be unique within the list."""
        seen: set[str] = set()
        for phase in self.phases:
            if phase.name in seen:
                raise ValueError(
                    f"duplicate phase name '{phase.name}' — names must be unique. "
                    f"Found names: {[p.name for p in self.phases]}"
                )
            seen.add(phase.name)
        return self

    @model_validator(mode="after")
    def validate_datasets_unique_names(self) -> Self:
        """Reject duplicate dataset names — they must be unique within the list."""
        seen: set[str] = set()
        for ds in self.datasets:
            if ds.name in seen:
                raise ValueError(
                    f"duplicate dataset name '{ds.name}' — names must be unique. "
                    f"Found names: {[d.name for d in self.datasets]}"
                )
            seen.add(ds.name)
        return self

    @model_validator(mode="after")
    def validate_dataset_references(self) -> Self:
        """Validate that all dataset references in phase configs exist."""
        dataset_names = {d.name for d in self.datasets}
        for phase in self.phases:
            if phase.dataset is not None and phase.dataset not in dataset_names:
                raise ValueError(
                    f"Phase config '{phase.name}' references undefined dataset "
                    f"'{phase.dataset}'. Available datasets: {sorted(dataset_names)}"
                )
        return self

    @model_validator(mode="after")
    def validate_seamless_not_on_first_phase(self) -> Self:
        """Ensure seamless is not enabled on the first phase config."""
        if self.phases and self.phases[0].seamless:
            raise ValueError(
                f"Phase config '{self.phases[0].name}' cannot have seamless=True "
                "because it is first. Seamless transitions only apply to "
                "subsequent phase configs."
            )
        return self

    @model_validator(mode="after")
    def validate_prefill_requires_streaming(self) -> Self:
        """Prefill concurrency requires streaming to measure TTFT boundaries."""
        for phase in self.phases:
            if phase.prefill_concurrency is not None and not self.endpoint.streaming:
                raise ValueError(
                    f"Phase '{phase.name}': prefill_concurrency requires "
                    "endpoint.streaming=true"
                )
        return self

    @model_validator(mode="after")
    def validate_phase_dataset_compatibility(self) -> Self:
        """Validate that each phase is compatible with its dataset.

        Checks sampling strategy requirements (e.g., fixed_schedule needs sequential)
        and format requirements (e.g., user_centric needs multi_turn).
        """
        from aiperf.config.resolved import check_phase_dataset_compatibility

        by_name = {d.name: d for d in self.datasets}
        for phase in self.phases:
            dataset_name = phase.dataset or self.get_default_dataset_name()
            ds = by_name.get(dataset_name)
            if ds is None:
                continue
            errors = check_phase_dataset_compatibility(
                phase, ds, phase.name, dataset_name
            )
            if errors:
                raise ValueError(errors[0])
        return self


class AIPerfConfig(BaseConfig):
    """AIPerf YAML envelope.

    Wraps a `BenchmarkConfig` (the swept body) with cross-variation fields
    (`sweep`, `multi_run`, `variables`, `random_seed`). This is the primary
    entry point for loading YAML configuration files. After sweep expansion,
    each variation's body materializes as a separate `BenchmarkConfig`.

    The split (envelope vs body) mirrors how AIPerfSweep CRDs are shaped on
    the K8s side: cross-variation machinery at envelope level, the swept
    workload as a body.
    """

    benchmark: Annotated[
        BenchmarkConfig,
        Field(description="Benchmark workload (the swept body)."),
    ]

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

    variables: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description=(
                "User-defined values exposed to Jinja2 in `{{ ... }}` expressions "
                "during config load. Cross-variation: scenario `runs[i].variables:` "
                "deep-merge over this base. Preserved on the resolved config so "
                "run-time renderers can resolve them again."
            ),
        ),
    ]

    random_seed: Annotated[
        int | None,
        Field(
            default=None,
            description="Global random seed for reproducibility. Base seed for "
            "per-variation derivation in sweep mode (variation N gets base + N).",
        ),
    ]

    @model_validator(mode="after")
    def validate_sweep_no_dashboard_ui(self) -> Self:
        """Reject Dashboard UI when a sweep is active (live UI doesn't multiplex).

        Only fires when the user explicitly set runtime.ui — the default
        ``UIType.DASHBOARD`` is allowed at construction time so test fixtures
        and YAML loads that don't touch runtime.ui can still describe sweeps.
        ``_run_multi_benchmark`` re-checks at execution time and gives the
        same error if the (possibly-defaulted) ui is still dashboard.
        """
        from aiperf.plugin.enums import UIType

        if (
            self.sweep is not None
            and "ui" in self.runtime.model_fields_set
            and self.runtime.ui == UIType.DASHBOARD
        ):
            raise ValueError(
                "Dashboard UI is incompatible with parameter sweeps; sweep "
                "results would overwrite each other in the live console. "
                "Use --ui simple or --ui none with --concurrency <list> / "
                "any sweep configuration."
            )
        return self

    @model_validator(mode="after")
    def validate_sweep_same_seed_requires_seed(self) -> Self:
        """``parameter_sweep_same_seed=true`` only matters when a base seed is set."""
        if self.multi_run.parameter_sweep_same_seed and self.random_seed is None:
            raise ValueError(
                "--parameter-sweep-same-seed requires --random-seed to be set; "
                "without a base seed every variation already gets a fresh draw, "
                "so 'same seed' is meaningless. Either set --random-seed N or "
                "drop --parameter-sweep-same-seed."
            )
        return self

    @model_validator(mode="after")
    def validate_sweep_cooldown_nonneg(self) -> Self:
        """Defensive cooldown check naming the offending CLI flag.

        ``Field(ge=0)`` already enforces this; the explicit validator gives a
        better error message that points at the CLI flag rather than the
        Pydantic field path.
        """
        if self.multi_run.parameter_sweep_cooldown_seconds < 0:
            raise ValueError(
                "--parameter-sweep-cooldown-seconds must be >= 0; "
                f"got {self.multi_run.parameter_sweep_cooldown_seconds}."
            )
        return self

    @model_validator(mode="after")
    def validate_sweep_flags_require_sweep(self) -> Self:
        """Reject parameter-sweep flags when no sweep is configured.

        The v1->v2 converter (``build_multi_run`` in
        ``aiperf.config.v1._converter_optionals``) only emits ``mode``,
        ``parameter_sweep_cooldown_seconds``, and ``parameter_sweep_same_seed``
        into the multi_run dict when the user explicitly set the corresponding
        ``--parameter-sweep-*`` CLI flag. Pydantic's ``model_fields_set`` on
        ``MultiRunConfig`` therefore reflects "user passed this flag" rather
        than "field has its default value".

        We additionally require the value differs from the field default so
        that a YAML round-trip with ``exclude_defaults=False`` (which writes
        every field, then re-loads them as "set") doesn't trip this check —
        only a non-default value actually expresses sweep-specific intent.
        Mirrors origin/main's ``LoadGeneratorConfig.validate_sweep_params``.
        """
        from aiperf.common.enums import SweepMode

        if self.sweep is not None:
            return self

        mr = self.multi_run
        set_fields = mr.model_fields_set

        # mode default flipped to REPEATED; only flag explicit non-default
        # picks (i.e., INDEPENDENT) without a sweep.
        if "mode" in set_fields and mr.mode != SweepMode.REPEATED:
            raise ValueError(
                "--parameter-sweep-mode only applies when sweeping parameters "
                "(e.g., --concurrency 10,20,30); with a single value the flag "
                "has no effect. Either remove --parameter-sweep-mode or "
                "provide a comma-separated list: --concurrency 10,20,30."
            )
        if (
            "parameter_sweep_cooldown_seconds" in set_fields
            and mr.parameter_sweep_cooldown_seconds != 0.0
        ):
            raise ValueError(
                "--parameter-sweep-cooldown-seconds only applies when sweeping "
                "parameters (e.g., --concurrency 10,20,30); with a single "
                "value there is no inter-variation gap to insert. Either "
                "remove --parameter-sweep-cooldown-seconds or provide a "
                "comma-separated list: --concurrency 10,20,30."
            )
        if "parameter_sweep_same_seed" in set_fields and mr.parameter_sweep_same_seed:
            raise ValueError(
                "--parameter-sweep-same-seed only applies when sweeping "
                "parameters (e.g., --concurrency 10,20,30); with a single "
                "value there is only one seed to choose. Either remove "
                "--parameter-sweep-same-seed or provide a comma-separated "
                "list: --concurrency 10,20,30."
            )
        return self
