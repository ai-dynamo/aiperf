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
    >>> print(config.benchmark.models)
    >>> for phase in config.benchmark.phases:
    ...     print(f"{phase.name}: {phase.type}")

    Or programmatically:
    >>> from aiperf.config import AIPerfConfig
    >>> config = AIPerfConfig(
    ...     benchmark={
    ...         "models": ["llama-3-8b"],
    ...         "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    ...         "datasets": [{"name": "main", "type": "synthetic", "entries": 1000, "prompts": {"isl": 512}}],
    ...         "phases": [{"name": "profiling", "type": "concurrency", "requests": 100, "concurrency": 8}],
    ...     }
    ... )
"""

from __future__ import annotations

import difflib
from typing import Annotated, Any, Literal, Self

from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.config.accuracy import (
    AccuracyConfig,
)
from aiperf.config.artifacts import (
    ArtifactsConfig,
)
from aiperf.config.base import BaseConfig
from aiperf.config.comm.build import build_comm_config
from aiperf.config.dataset import (
    DatasetConfig,
)
from aiperf.config.endpoint import (
    EndpointConfig,
)
from aiperf.config.gpu_telemetry import (
    GpuTelemetryConfig,
)
from aiperf.config.loader.helpers import BenchmarkHelpersMixin
from aiperf.config.loader.normalizers import (
    normalize_benchmark_input,
    parse_datasets_input,
)
from aiperf.config.logging import (
    LoggingConfig,
)
from aiperf.config.metrics import MetricsConfig
from aiperf.config.mlflow import (
    MLflowConfig,
)
from aiperf.config.models import (
    ModelsAdvanced,
)
from aiperf.config.network_latency import (
    NetworkLatencyConfig,
)
from aiperf.config.otel import (
    OTelConfig,
)
from aiperf.config.phases import (
    PhaseConfig,
)
from aiperf.config.runtime import (
    RuntimeConfig,
)
from aiperf.config.server_metrics import (
    ServerMetricsConfig,
)
from aiperf.config.slos import (
    SLOsConfig,
)
from aiperf.config.sweep import SweepConfig
from aiperf.config.sweep.multi_run import (
    MultiRunConfig,
)
from aiperf.config.tokenizer import (
    TokenizerConfig,
)
from aiperf.config.wandb import (
    WandbConfig,
)

_logger = AIPerfLogger(__name__)

# `aiperf.config.plot` is imported lazily inside `_resolve_plot_path`. A
# top-level import would cycle: aiperf.config.plot -> aiperf.plot.core ->
# aiperf.common.mixins -> aiperf.common.messages, the last of which can be
# mid-init when this module loads (via common.models.export_models). The
# ``plot`` field is typed ``Any`` for the same reason; the validator owns
# the coercion to ``PlotEnvelopeConfig``.

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
        - datasets: Single-element list of dataset configuration
        - phases: Ordered list of benchmark phases. A single-dict under
          ``phases:`` is rejected; use the envelope-level ``warmup:`` /
          ``profiling:`` shorthand or an explicit list.

    Optional Sections:
        - artifacts: Export and console settings
        - slos: SLO-based quality metrics (generic dict)
        - tokenizer: Token counting configuration
        - gpu_telemetry: GPU metrics from DCGM endpoints
        - server_metrics: Server metrics from Prometheus endpoints
        - runtime: Worker and communication settings
        - logging: Logging and debug settings
        - metrics: Metrics aggregation configuration
        - accuracy: Accuracy evaluation configuration

    Validation:
        The configuration is validated in several stages:
        1. Individual field validation (types, ranges, formats)
        2. Cross-field validation (mutual exclusivity, dependencies)

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

    endpoint_profiles: Annotated[
        dict[str, EndpointConfig],
        Field(
            default_factory=dict,
            description=(
                "Additional named protocol-v2 endpoint profiles. The existing "
                "endpoint section remains the required 'default' profile; evaluation "
                "routes reference either 'default' or one of these names."
            ),
        ),
    ]

    datasets: Annotated[
        list[DatasetConfig],
        Field(
            min_length=1,
            max_length=1,
            description="Dataset configuration as a single-element list. The list "
            "shape exists to share the schema between YAML and the AIPerfSweep CRD; "
            "the runtime currently loads exactly one dataset. Singular `dataset:` "
            "shorthand at the BenchmarkConfig top level is normalized to a one-entry "
            "list with name='default'.",
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

    network_latency: Annotated[
        NetworkLatencyConfig,
        Field(
            default_factory=NetworkLatencyConfig,
            description="Network latency calibration configuration. Probes the endpoint "
            "RTT during profiling and subtracts the mean from latency metrics, emitted as "
            "separate network_adjusted_* metrics. Disabled by default.",
        ),
    ]

    otel: Annotated[
        OTelConfig,
        Field(
            default_factory=OTelConfig,
            description="OpenTelemetry metrics streaming configuration.",
        ),
    ]

    mlflow: Annotated[
        MLflowConfig,
        Field(
            default_factory=MLflowConfig,
            description="MLflow tracking and artifact-upload configuration.",
        ),
    ]

    wandb: Annotated[
        WandbConfig,
        Field(
            default_factory=WandbConfig,
            description="Weights & Biases run-upload configuration.",
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

    failure_policy: Annotated[
        Literal["continue", "abort"] | None,
        Field(
            default=None,
            description="How the run reacts to a failed request. 'continue' "
            "records the failure and keeps running (resilient); 'abort' fails "
            "the whole benchmark on the first non-cancellation failure "
            "(fail-fast). When unset, each execution path applies its historical "
            "default: scheduled runs are resilient, graph (DAG) runs are "
            "fail-fast. Cancellations are never treated as failures.",
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

    scenario: Annotated[
        str | None,
        Field(
            default=None,
            description="Lock all benchmark invariants for a named scenario "
            "(e.g. 'inferencex-agentx-mvp'). Plain data here; the lock is "
            "applied by the ScenarioResolver step in the pre-bootstrap resolver "
            "chain (auto-fills defaults, validates, raises ScenarioLockError on "
            "conflict). Distinct from the sweep ``scenarios`` strategy "
            "(SweepConfig), which expands hand-picked named runs -- this field "
            "is a single invariant LOCK, not a sweep.",
        ),
    ]

    unsafe_override: Annotated[
        bool,
        Field(
            default=False,
            description="Convert scenario lock errors to warnings; stamps "
            "submission_valid=false in the resolved scenario outcome. No-op "
            "without ``scenario``. Plain data; consumed by the ScenarioResolver.",
        ),
    ]

    trajectory_start_min_ratio: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Lower bound of the recorded-graph trajectory-start (t*) "
            "snapshot window, expressed as a fraction of each recorded trace's "
            "total duration. The native runner samples a per-trace start offset in "
            "[min_ratio, max_ratio] of the trace's duration so replay begins mid-"
            "trajectory (an active t* window) rather than at t=0. 0.0 (with a 0.0 "
            "max) disables the window and starts every trace at its beginning.",
        ),
    ]

    trajectory_start_max_ratio: Annotated[
        float,
        Field(
            default=0.0,
            ge=0.0,
            le=1.0,
            description="Upper bound of the recorded-graph trajectory-start (t*) "
            "snapshot window, expressed as a fraction of each recorded trace's "
            "total duration. Paired with trajectory_start_min_ratio; must be >= it. "
            "0.0 disables the window (every trace starts at t=0).",
        ),
    ]

    cache_bust_target: Annotated[
        str,
        Field(
            default="none",
            description="Recorded-graph cache-bust marker target. Auto-filled by "
            "the submission scenario's ``require_cache_bust`` lock "
            '(validator._apply_require_cache_bust). ``"none"`` (the default) '
            'sends recorded content verbatim; ``"first_turn_prefix"`` prepends a '
            "per-conversation ``[rid:<digest>]`` marker to the first user message "
            "so shared-trace lanes send distinct prefixes and the scenario-required "
            "ISL accounting matches. Projected onto the recorded dataset's "
            "``synthesis`` block by ``rust_wire`` and consumed by the native runner "
            "(``CacheBustTarget``); ignored on non-recorded-graph datasets.",
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
                "[{name: warmup, ...}, {name: profiling, ...}]); the dict "
                "shape is not supported. See "
                "docs/tutorials/yaml-config.md#phases for the expected shape."
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
        """Reject duplicate phase names, case-insensitively, within the list."""
        seen: dict[str, str] = {}
        for phase in self.phases:
            key = phase.name.lower()
            if key in seen:
                raise ValueError(
                    f"duplicate phase name '{phase.name}' conflicts with "
                    f"'{seen[key]}' — names must be unique case-insensitively. "
                    f"Found names: {[p.name for p in self.phases]}"
                )
            seen[key] = phase.name
        return self

    @model_validator(mode="after")
    def validate_profiling_phase_required(self) -> Self:
        """Require at least one profiling-kind phase — warmup alone is not a benchmark."""
        if not any(p.kind == "profiling" for p in self.phases):
            raise ValueError(
                "a 'profiling' phase is required; "
                f"got phases: {[(p.name, p.kind) for p in self.phases]}"
            )
        return self

    @model_validator(mode="after")
    def validate_phase_stops(self) -> Self:
        """Require each phase to declare a generic stop condition.

        Every profiling phase needs at least one of ``requests`` / ``duration`` /
        ``sessions`` (or a self-bounding agentic-cache warmup), unless a named
        ``--scenario`` supplies the bound at resolution time.
        """
        # A named ``--scenario`` owns the benchmark invariants and auto-fills the
        # phase stop condition (e.g. profiling ``duration``) at resolution time
        # (``ScenarioResolver`` / ``apply_scenario``), which runs AFTER this
        # construction-time validator. Requiring a bound here would reject a
        # valid scenario run whose bound is supplied a step later — the exact gap
        # the removed ``requests=10`` converter default used to paper over. The
        # scenario (and the runner's own graph-phase validation) enforce the real
        # bound; do not double-require a generic one before the scenario has run.
        if self.scenario is not None:
            return self
        for phase in self.phases:
            if (
                phase._stop_condition_required
                and phase.requests is None
                and phase.duration is None
                and phase.sessions is None
                # The recorded-graph cache-pressure warmup is self-bounding: its
                # sole deadline is agentic_cache_warmup_duration, and a generic
                # stop cap would cancel the cache-pressure recycle. Treat that
                # duration as the phase's stop condition.
                and phase.agentic_cache_warmup_duration is None
            ):
                raise ValueError(
                    f"Phase '{phase.name}': at least one of "
                    "'requests', 'duration', or 'sessions' must be specified"
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
    def default_tokenizer_when_unset(self) -> Self:
        """Materialize a default ``TokenizerConfig`` when the user omitted one.

        Downstream services (DatasetManager, InferenceResultParser,
        RecordProcessorService) dereference ``cfg.tokenizer`` unconditionally;
        defaulting at the config seam keeps ``tokenizer`` non-None for every
        valid ``BenchmarkConfig`` so consumers don't have to None-check at
        every call site. The default ``TokenizerConfig()`` carries
        ``name=None``, which makes ``get_tokenizer_name_for_model(model)``
        fall back to the model name itself -- matching the documented
        "If not specified, uses the first model name." behavior.
        """
        if self.tokenizer is None:
            self.tokenizer = TokenizerConfig()
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
    def validate_endpoint_profile_names(self) -> Self:
        """Keep named endpoint-profile references structural and side-effect free.

        Python checks only identities that are already present in Config v2;
        endpoint dialect capability remains a runner-factory rule.
        """
        for profile_id in self.endpoint_profiles:
            if not profile_id.strip() or profile_id != profile_id.strip():
                raise ValueError(
                    "endpoint_profiles keys must be non-empty and contain no "
                    "surrounding whitespace"
                )
            if profile_id == "default":
                raise ValueError(
                    "endpoint_profiles cannot redefine reserved profile 'default'; "
                    "use benchmark.endpoint for that profile"
                )
        return self

    @model_validator(mode="after")
    def validate_phase_dataset_compatibility(self) -> Self:
        """Validate that each phase is compatible with the dataset.

        Checks sampling strategy requirements (e.g., fixed_schedule needs sequential)
        and format requirements (e.g., user_centric needs multi_turn).
        """
        from aiperf.config.resolution.predicates import (
            check_phase_dataset_compatibility,
        )

        ds = self.get_default_dataset()
        for phase in self.phases:
            errors = check_phase_dataset_compatibility(phase, ds, phase.name, ds.name)
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

    model_config = ConfigDict(
        alias_generator=BaseConfig.model_config["alias_generator"],
        populate_by_name=True,
        extra="forbid",
    )

    schema_version: Annotated[
        Literal["2.0"],
        Field(
            default="2.0",
            description="AIPerf config schema version. Pinned to '2.0' for the "
            "envelope/body shape (envelope keys at top level, swept body under "
            "``benchmark``).",
        ),
    ] = "2.0"

    benchmark: Annotated[
        BenchmarkConfig,
        Field(description="Benchmark workload (the swept body)."),
    ]

    sweep: Annotated[
        SweepConfig | None,
        Field(
            default=None,
            description="Sweep configuration for parameter exploration. "
            "Supports grid (Cartesian product), zip (lock-stepped axes), "
            "scenarios (hand-picked), sobol / latin_hypercube (QMC sampling), "
            "and adaptive_search (Bayesian / monotonic outer-loop) strategies.",
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

    plot: Annotated[
        Any,
        Field(
            default=None,
            description=(
                "Plot configuration for this run/sweep. When set, "
                "``~/.aiperf/plot_config.yaml`` is ignored. May be a bare-string "
                "path to a plot YAML (resolved relative to the AIPerf config "
                "file's directory, or absolute), or an inline mapping mirroring "
                "``default_plot_config.yaml``. Presence implies "
                "``artifacts.auto_plot=True`` unless explicitly set False."
            ),
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
            ge=0,
            description="Global random seed for reproducibility. Base seed for "
            "per-variation derivation in sweep mode (variation N gets base + N). "
            "Must be non-negative; numpy and scipy reject negative seeds.",
        ),
    ]

    no_sweep_table: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Suppress the per-cell streaming sweep table during "
                "multi-variation sweeps. Auto-suppressed when stdout is "
                "non-interactive, when the dashboard UI is active, or for "
                "single-cell sweeps."
            ),
        ),
    ]

    # Pre-Jinja, post-env-var envelope dict captured at load time. Only set by
    # the YAML / dict loaders (load_config, expand_config_dict). Sweep
    # expansion (`build_benchmark_plan`) prefers this over `model_dump()` so
    # `{{ var }}` references in body fields can re-render against each
    # variation's `variables:` block. Direct `AIPerfConfig(...)` callers leave
    # it None and fall back to `model_dump()` (no body templating, but every
    # other path still works).
    _raw_envelope: dict[str, Any] | None = PrivateAttr(default=None)

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_envelope_keys(cls, data: Any) -> Any:
        """Reject typo'd top-level envelope keys with a 'did you mean' hint.

        ``extra="forbid"`` already raises on unknown keys, but its message is
        a generic ``extra_forbidden`` per-field error. This pre-validator runs
        first, intercepts unknown keys, and raises a single ``ValueError``
        listing all unknowns plus suggestions from ``difflib`` so a typo like
        ``sweeps:`` (instead of ``sweep:``) gives an actionable message
        instead of the user's data being silently dropped.
        """
        if not isinstance(data, dict):
            return data
        # Field names plus their alias_generator-derived camelCase aliases.
        known: set[str] = set()
        for name, field in cls.model_fields.items():
            known.add(name)
            if field.alias:
                known.add(field.alias)
        unknown = [k for k in data if isinstance(k, str) and k not in known]
        if not unknown:
            return data
        # Body fields (endpoint, mlflow, otel, gpuTelemetry, …) live under
        # ``benchmark:``, not at the envelope root. A misplaced body key is a
        # common mistake, so point the user at the right nesting explicitly.
        body_keys: set[str] = set()
        for name, field in BenchmarkConfig.model_fields.items():
            body_keys.add(name)
            if field.alias:
                body_keys.add(field.alias)
        suggestions = []
        for key in unknown:
            if key in body_keys:
                suggestions.append(f"{key!r} (nest it under 'benchmark:')")
                continue
            close = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
            if not close:
                close = difflib.get_close_matches(key, body_keys, n=1, cutoff=0.6)
                if close:
                    suggestions.append(
                        f"{key!r} (did you mean 'benchmark.{close[0]}'?)"
                    )
                    continue
            if close:
                suggestions.append(f"{key!r} (did you mean {close[0]!r}?)")
            else:
                suggestions.append(repr(key))
        known_sorted = sorted(known)
        raise ValueError(
            "Unknown top-level envelope key(s): "
            + ", ".join(suggestions)
            + f". Known keys: {known_sorted}"
        )

    @model_validator(mode="before")
    @classmethod
    def _resolve_plot_path(cls, data: Any, info: Any) -> Any:
        """Convert Form A (bare-string path) into a ``PlotEnvelopeConfig``
        instance, or validate Form B (inline dict) into the same type.

        Reads ``info.context["source_dir"]`` to resolve relative paths against
        the AIPerf YAML's directory. ``load_config_from_string`` threads this
        in; programmatic ``AIPerfConfig(...)`` callers leave it None (in which
        case relative paths are rejected by ``load_plot_envelope_from_path``).

        The ``plot`` field is typed ``Any`` (to break an import cycle), so this
        validator owns the typed coercion to ``PlotEnvelopeConfig``.
        """
        if not isinstance(data, dict):
            return data
        plot_value = data.get("plot")
        if plot_value is None:
            return data
        from aiperf.config.plot import (
            PlotEnvelopeConfig,
            load_plot_envelope_from_path,
        )

        if isinstance(plot_value, PlotEnvelopeConfig):
            return data
        if isinstance(plot_value, str):
            source_dir = None
            ctx = getattr(info, "context", None) or {}
            if isinstance(ctx, dict):
                source_dir = ctx.get("source_dir")
            envelope = load_plot_envelope_from_path(plot_value, source_dir=source_dir)
        elif isinstance(plot_value, dict):
            envelope = PlotEnvelopeConfig.model_validate(plot_value)
        else:
            raise ValueError(
                f"plot: must be null, a path string, or an inline mapping; "
                f"got {type(plot_value).__name__}"
            )
        new_data = dict(data)
        new_data["plot"] = envelope
        return new_data

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
            and "ui" in self.benchmark.runtime.model_fields_set
            and self.benchmark.runtime.ui == UIType.DASHBOARD
        ):
            raise ValueError(
                "Dashboard UI is incompatible with parameter sweeps; sweep "
                "results would overwrite each other in the live console. "
                "Use --ui simple or --ui none with --concurrency <list> / "
                "any sweep configuration."
            )
        return self

    @model_validator(mode="after")
    def _apply_consistent_seed_default(self) -> Self:
        # Why: confidence statistics across trials and per-variation comparisons
        # in a sweep require identical workloads. Without a seed, synthetic
        # prompts and session IDs vary every run and the resulting variance is
        # input-noise, not runtime-noise. `multi_run.set_consistent_seed`
        # (default True) advertises an auto-fill of 42 when the user didn't
        # pass `--random-seed`; this validator is what actually performs it.
        if self.random_seed is not None:
            return self
        if not self.multi_run.set_consistent_seed:
            return self
        if self.sweep is None and self.multi_run.num_runs <= 1:
            return self
        self.random_seed = 42
        return self

    @model_validator(mode="after")
    def _plot_implies_auto_plot(self) -> Self:
        """When ``plot:`` is set and the user didn't explicitly set
        ``artifacts.auto_plot``, flip it to True. Explicit ``auto_plot: false``
        wins; we just log an info-level breadcrumb so the silence doesn't
        surprise users.
        """
        if self.plot is None:
            return self
        if "auto_plot" in self.benchmark.artifacts.model_fields_set:
            if self.benchmark.artifacts.auto_plot is False:
                _logger.info(
                    "plot section present but artifacts.auto_plot=false "
                    "explicitly; plots will not auto-render. Use ``aiperf plot "
                    "<artifact_dir>`` after the run to generate them."
                )
            return self
        self.benchmark.artifacts.auto_plot = True
        return self
