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

    @model_validator(mode="after")
    def validate_cache_bust_compatibility(self) -> Self:
        """Refuse cache-bust when endpoint metadata does not support it.

        Cache-bust markers are minted by every timing strategy and injected by
        the worker into structured endpoint turns. Endpoint plugins opt in via
        ``supports_cache_bust`` metadata; timing mode is not a restriction.
        """
        from aiperf.common.enums import CacheBustTarget
        from aiperf.plugin import plugins

        if self.get_cache_bust_target() == CacheBustTarget.NONE:
            return self

        endpoint_metadata = plugins.get_endpoint_metadata(self.endpoint.type)
        if not endpoint_metadata.supports_cache_bust:
            raise ValueError(
                "cache-bust is not supported by endpoint "
                f"{self.endpoint.type}; select an endpoint whose plugin metadata "
                "sets supports_cache_bust=true"
            )

        return self

    @model_validator(mode="after")
    def validate_warmup_isolation_not_agentic_replay(self) -> Self:
        """Reject WARMUP_ISOLATION_* targets when timing mode resolves to AGENTIC_REPLAY.

        ``_inject_marker_at_first_user`` mutates Turn objects in-place inside
        ``session.turn_list``. In agentic replay the same session object spans
        the WARMUP→PROFILING boundary (the session is never evicted at phase
        transition), so a Turn mutated with ``[warmup]`` during WARMUP carries
        that mutation into PROFILING — even though the PROFILING credit has
        ``cache_bust_marker=None`` and ``_apply_cache_bust`` returns early.

        Resolution logic mirrors ``validate_agentic_cache_warmup``: resolve the
        scenario's declared timing_mode when a scenario is present (scenario
        stamps phases post-construction); otherwise inspect the phases directly.
        """
        from aiperf.common.enums import CacheBustTarget
        from aiperf.plugin.enums import TimingMode
        from aiperf.timing.config import _is_agentic_replay

        target = self.get_cache_bust_target()
        _WARMUP_ISOLATION = (
            CacheBustTarget.WARMUP_ISOLATION_SYSTEM,
            CacheBustTarget.WARMUP_ISOLATION_FIRST_TURN,
        )
        if target not in _WARMUP_ISOLATION:
            return self

        profiling_phases = self.get_profiling_phases()
        is_agentic = False
        if self.scenario is not None:
            from aiperf.common.scenario.registry import get_scenario

            is_agentic = (
                get_scenario(self.scenario).timing_mode == TimingMode.AGENTIC_REPLAY
            )
        else:
            is_agentic = _is_agentic_replay(profiling_phases)

        if is_agentic:
            raise ValueError(
                "cache_bust targets warmup_isolation_system and "
                "warmup_isolation_first_turn are not compatible with agentic_replay "
                "timing mode: Turn objects mutated during warmup persist into profiling "
                "sessions. Use cache_bust=none for agentic_replay workloads or one of "
                "the RID-based targets (system_prefix, first_turn_prefix, etc.) for "
                "per-trajectory isolation."
            )
        return self

    @model_validator(mode="after")
    def validate_warmup_isolation_system_requires_shared_system_prompt(self) -> Self:
        """Reject warmup_isolation_system when no system message is statically present.

        ``warmup_isolation_system`` injects ``[warmup]`` into the system message slot
        during warmup. When no system message exists in the dataset config the marker
        silently falls through to the user turn, making it indistinguishable from
        ``warmup_isolation_first_turn``.

        This check is scoped to synthetic datasets where system-message presence is
        statically known. File and trace datasets may carry system messages inside
        their content, which cannot be inspected at config time.
        """
        from aiperf.common.enums import CacheBustTarget, DatasetType

        if self.get_cache_bust_target() != CacheBustTarget.WARMUP_ISOLATION_SYSTEM:
            return self

        dataset = self.get_default_dataset()
        if dataset.type != DatasetType.SYNTHETIC:
            return self

        prefix_prompts = getattr(dataset, "prefix_prompts", None)
        has_shared_system = (
            prefix_prompts is not None
            and getattr(prefix_prompts, "shared_system_length", None) is not None
        )
        if not has_shared_system:
            raise ValueError(
                "cache_bust=warmup_isolation_system requires a shared system prompt, "
                "but no shared_system_length is configured on the synthetic dataset. "
                "The marker would silently fall through to the user turn, making it "
                "indistinguishable from warmup_isolation_first_turn. "
                "Either set prefix_prompts.shared_system_length (--shared-system-prompt-length) "
                "or switch to cache_bust=warmup_isolation_first_turn."
            )
        return self

    @model_validator(mode="after")
    def validate_agentic_cache_warmup(self) -> Self:
        """Restrict accelerated cache warmup to the agentic_replay timing mode.

        ``--agentic-cache-warmup-duration`` is consumed solely by
        ``aiperf.timing.config._build_agentic_warmup_config``, which only runs
        when the profiling phases resolve to AGENTIC_REPLAY. On any other run
        the value is silently dropped, so an unguarded flag is a no-op the user
        cannot see. This guard raises instead.

        A scenario governs the timing_mode: ``apply_scenario`` stamps
        ``spec.timing_mode`` onto the profiling phases POST-construction, so the
        phase ``timing_mode`` is not yet set here. Rather than defer (which would
        leave the flag unchecked -- ``apply_scenario`` never re-runs this gate),
        we resolve the scenario's declared ``timing_mode`` from its
        :class:`ScenarioSpec` and reject when it is not AGENTIC_REPLAY. When no
        scenario is present the phases are final, so we resolve them with the
        same ``_is_agentic_replay``/``_phase_timing_mode`` logic the timing
        builder uses (explicit ``timing_mode`` override, else the phase-type
        mapping) and reject the flag when nothing resolves to AGENTIC_REPLAY.
        """
        from aiperf.plugin.enums import TimingMode
        from aiperf.timing.config import _is_agentic_replay

        profiling_phases = self.get_profiling_phases()
        if not any(
            getattr(phase, "agentic_cache_warmup_duration", None) is not None
            for phase in profiling_phases
        ):
            return self

        if self.scenario is not None:
            from aiperf.common.scenario.registry import get_scenario

            scenario_timing_mode = get_scenario(self.scenario).timing_mode
            if scenario_timing_mode != TimingMode.AGENTIC_REPLAY:
                raise ValueError(
                    "--agentic-cache-warmup-duration requires the agentic_replay "
                    f"timing mode; scenario {self.scenario!r} locks "
                    f"timing_mode={scenario_timing_mode}."
                )
            return self

        if not _is_agentic_replay(profiling_phases):
            raise ValueError(
                "--agentic-cache-warmup-duration requires the agentic_replay "
                "timing mode (set today by --scenario inferencex-agentx-mvp); "
                "the profiling phase(s) are not agentic_replay."
            )
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
        suggestions = []
        for key in unknown:
            close = difflib.get_close_matches(key, known, n=1, cutoff=0.6)
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
    def _reject_scenario_with_sweep(self) -> Self:
        """Reject a parameter sweep when a fixed-spec scenario is locked.

        A ``--scenario`` lock describes ONE fixed configuration; a sweep would
        fan it into N variations with diverging settings, each individually
        satisfying the lock while collectively falsifying the "one scenario =
        one spec" contract. This is the v2 successor to the v1
        ``validate_scenario`` check that rejected a list-shaped ``--concurrency``
        (a parameter sweep): in v2 list-shaped magic-list flags are hoisted to a
        top-level ``sweep:`` block by the CLI converter
        (``_promote_magic_lists_to_sweep_block``) BEFORE this config is built, so
        the rejection lives here at the envelope level where ``scenario`` (under
        ``benchmark``) and ``sweep`` are sibling concerns -- not in
        ``apply_scenario``, which only ever sees one already-expanded variation.

        ``benchmark.unsafe_override`` downgrades the hard error to a warning,
        matching how the scenario resolver treats every other invariant
        violation.
        """
        if self.benchmark.scenario is None or self.sweep is None:
            return self
        message = (
            f"scenario {self.benchmark.scenario!r} does not support parameter "
            "sweeps; "
            "a scenario locks one fixed configuration, so a sweep would "
            "multiply it into diverging runs. Pass a single value for each "
            "swept flag (e.g. one --concurrency value, not a list) or drop "
            "the sweep / --scenario."
        )
        if self.benchmark.unsafe_override:
            _logger.warning("Scenario violation (override active): %s", message)
            return self
        raise ValueError(message)

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
    def validate_adaptive_search_not_nested_with_adaptive_scale(self) -> Self:
        """Reject nested adaptive loops: adaptive_search outside adaptive_scale."""
        if self.sweep is None or self.sweep.type != "adaptive_search":
            return self
        adaptive_phases = [
            phase.name
            for phase in self.benchmark.phases
            if getattr(phase, "adaptive_scale", False)
        ]
        if adaptive_phases:
            raise ValueError(
                "adaptive_search sweeps cannot be combined with adaptive_scale "
                f"phases: {adaptive_phases}. Use either the outer adaptive_search "
                "sweep or per-phase adaptive_scale, not both."
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
