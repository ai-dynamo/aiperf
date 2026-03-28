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

from typing import TYPE_CHECKING, Annotated, Any, Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from aiperf.config._base import BaseConfig

if TYPE_CHECKING:
    from aiperf.common.enums import AIPerfLogLevel, GPUTelemetryMode
    from aiperf.config.zmq import BaseZMQCommunicationConfig
    from aiperf.plugin.enums import UIType

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


def build_comm_config(config: BenchmarkConfig) -> BaseZMQCommunicationConfig:
    """Build a ZMQ communication config from a BenchmarkConfig.

    This is the single source of truth for mapping user-facing communication
    config models to runtime ZMQ config objects. Called by:
    - BenchmarkConfig.comm_config property (backward compat)
    - BenchmarkRun.comm_config property (orchestrator path)

    Handles:
    - Complete field mapping (ports, proxy configs, control ports)
    - K8s env var auto-detection for dual-bind controller_host
    - Fallback to IPC when no communication config is set
    """
    from pathlib import Path

    from aiperf.common.enums import CommunicationType
    from aiperf.config.models import (
        DualBindCommunicationConfig,
        TcpCommunicationConfig,
    )
    from aiperf.config.zmq import (
        ZMQDualBindConfig,
        ZMQDualBindProxyConfig,
        ZMQIPCConfig,
        ZMQTCPConfig,
        ZMQTCPProxyConfig,
    )

    comm = config.runtime.communication
    if comm is None:
        return ZMQIPCConfig()

    if comm.type == CommunicationType.IPC:
        return ZMQIPCConfig(path=comm.path)

    if comm.type == CommunicationType.TCP:
        assert isinstance(comm, TcpCommunicationConfig)
        return ZMQTCPConfig(
            host=comm.host,
            records_push_pull_port=comm.records_port,
            credit_router_port=comm.credit_router_port,
            control_port=comm.control_port,
            event_bus_proxy_config=ZMQTCPProxyConfig(
                frontend_port=comm.event_bus_proxy.frontend_port,
                backend_port=comm.event_bus_proxy.backend_port,
            ),
            dataset_manager_proxy_config=ZMQTCPProxyConfig(
                frontend_port=comm.dataset_manager_proxy.frontend_port,
                backend_port=comm.dataset_manager_proxy.backend_port,
            ),
            raw_inference_proxy_config=ZMQTCPProxyConfig(
                frontend_port=comm.raw_inference_proxy.frontend_port,
                backend_port=comm.raw_inference_proxy.backend_port,
            ),
        )

    if comm.type == CommunicationType.DUAL:
        assert isinstance(comm, DualBindCommunicationConfig)
        controller_host = comm.controller_host
        if controller_host is None:
            from aiperf.kubernetes.environment import K8sEnvironment

            controller_host = K8sEnvironment.ZMQ.CONTROLLER_HOST

        return ZMQDualBindConfig(
            ipc_path=Path(comm.ipc_path),
            tcp_host=comm.tcp_host,
            controller_host=controller_host,
            records_push_pull_tcp_port=comm.records_port,
            credit_router_tcp_port=comm.credit_router_port,
            control_tcp_port=comm.control_port,
            event_bus_proxy_config=ZMQDualBindProxyConfig(
                name="event_bus_proxy",
                tcp_frontend_port=comm.event_bus_proxy.frontend_port,
                tcp_backend_port=comm.event_bus_proxy.backend_port,
            ),
            dataset_manager_proxy_config=ZMQDualBindProxyConfig(
                name="dataset_manager_proxy",
                tcp_frontend_port=comm.dataset_manager_proxy.frontend_port,
                tcp_backend_port=comm.dataset_manager_proxy.backend_port,
            ),
            raw_inference_proxy_config=ZMQDualBindProxyConfig(
                name="raw_inference_proxy",
                tcp_frontend_port=comm.raw_inference_proxy.frontend_port,
                tcp_backend_port=comm.raw_inference_proxy.backend_port,
            ),
        )

    return ZMQIPCConfig()


class BenchmarkConfig(BaseConfig):
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

        Handles:
            - model → models (singular to plural)
            - dataset → datasets (singular to plural, wrapped with key 'default')
            - phases: single config (has 'type') → {'default': config}
            - models: str/list[str] → ModelsAdvanced dict format
        """
        if not isinstance(data, dict):
            return data

        # --- Mutual exclusivity checks (before any normalization) ---

        # warmup/profiling vs phases
        has_warmup = "warmup" in data
        has_profiling = "profiling" in data
        has_phases = "phases" in data

        if has_warmup and has_phases:
            raise ValueError(
                "'warmup' cannot be used with 'phases'. "
                "Use 'warmup'/'profiling' for simple configs "
                "or 'phases' for advanced multi-phase configs."
            )
        if has_profiling and has_phases:
            raise ValueError(
                "'profiling' cannot be used with 'phases'. "
                "Use 'warmup'/'profiling' for simple configs "
                "or 'phases' for advanced multi-phase configs."
            )

        # dataset vs datasets
        if "dataset" in data and "datasets" in data:
            raise ValueError(
                "'dataset' cannot be used with 'datasets'. "
                "Use 'dataset' for a single dataset "
                "or 'datasets' for multiple named datasets."
            )

        # warmup requires profiling
        if has_warmup and not has_profiling:
            raise ValueError(
                "'warmup' requires 'profiling'. "
                "A warmup-only config without a profiling phase would produce no results."
            )

        # --- warmup/profiling -> phases normalization ---

        if has_warmup or has_profiling:
            phases = {}
            if has_warmup:
                warmup = data.pop("warmup")
                if isinstance(warmup, dict):
                    warmup.setdefault("exclude_from_results", True)
                phases["warmup"] = warmup
            if has_profiling:
                phases["profiling"] = data.pop("profiling")
            data["phases"] = phases

        # model → models (singular to plural)
        if "model" in data and "models" not in data:
            data["models"] = data.pop("model")

        # Normalize models to ModelsAdvanced dict format
        if "models" in data:
            models = data["models"]
            if isinstance(models, str):
                # Single string: "llama" → {"items": [{"name": "llama"}]}
                data["models"] = {"items": [{"name": models}]}
            elif isinstance(models, list) and models and isinstance(models[0], str):
                # List of strings: ["llama", "mistral"] → {"items": [...]}
                data["models"] = {"items": [{"name": name} for name in models]}
            # If it's already a dict (ModelsAdvanced format), pass through

        # dataset → datasets (single dataset becomes 'default')
        if "dataset" in data and "datasets" not in data:
            data["datasets"] = {"default": data.pop("dataset")}

        # Normalize phases: single config (has 'type' key) → named phases
        if "phases" in data:
            phases = data["phases"]
            if isinstance(phases, dict) and "type" in phases:
                # Single phase config with 'type' key - wrap with key "default"
                data["phases"] = {"default": phases}
            # Otherwise it's already a dict of named phases

        return data

    @field_validator("phases", mode="before")
    @classmethod
    def parse_phases(cls, v: Any) -> dict[str, Any]:
        """
        Parse phase configurations from dict format.

        Injects the phase name from the dict key into each phase config.
        """
        if not isinstance(v, dict):
            raise ValueError("phases must be a dictionary with phase names as keys")

        result = {}
        for name, config in v.items():
            if isinstance(config, BasePhaseConfig):
                # Already a PhaseConfig, inject name
                config._name = name
                result[name] = config
            elif isinstance(config, dict):
                # Dict will be validated into PhaseConfig, name injected after
                result[name] = config
            else:
                raise ValueError(f"Phase config '{name}' must be a dictionary")

        return result

    @field_validator("datasets", mode="before")
    @classmethod
    def parse_datasets(cls, v: Any) -> dict[str, Any]:
        """
        Parse dataset configurations, handling composed datasets.

        Composed datasets don't have a 'type' field but have 'source' and 'augment'.
        This validator ensures they're properly recognized.

        Also accepts already-constructed Pydantic models for programmatic use.
        """
        from aiperf.config.dataset import (
            ComposedDataset,
            FileDataset,
            PublicDataset,
            SyntheticDataset,
        )

        dataset_types = (SyntheticDataset, FileDataset, PublicDataset, ComposedDataset)

        if not isinstance(v, dict):
            raise ValueError("datasets must be a dictionary")

        result = {}
        for name, config in v.items():
            # Accept already-constructed Pydantic models (for programmatic use)
            if isinstance(config, dataset_types):
                result[name] = config
                continue

            if not isinstance(config, dict):
                raise ValueError(
                    f"Dataset '{name}' configuration must be a dictionary or Pydantic model"
                )

            # --- isl/osl hoisting (only for synthetic or type-absent, not composed) ---
            ds_type = config.get("type")
            is_composed = "source" in config and "augment" in config
            if (
                ds_type in ("synthetic", None)
                and not is_composed
                and ("isl" in config or "osl" in config)
            ):
                prompts = config.setdefault("prompts", {})
                if isinstance(prompts, dict):
                    if "isl" in config:
                        prompts.setdefault("isl", config.pop("isl"))
                    if "osl" in config:
                        prompts.setdefault("osl", config.pop("osl"))

            # Check if this is a composed dataset (has source + augment, no type)
            if "source" in config and "augment" in config and "type" not in config:
                # This is a composed dataset - leave as-is for Pydantic to validate
                result[name] = config
            elif "type" not in config:
                # Default to synthetic if no type specified and not composed
                config["type"] = "synthetic"
                result[name] = config
            else:
                result[name] = config

        return result

    @model_validator(mode="after")
    def inject_phase_names(self) -> Self:
        """Inject phase names from dict keys into PhaseConfig objects."""
        for name, phase in self.phases.items():
            phase._name = name
        return self

    @model_validator(mode="after")
    def validate_dataset_references(self) -> Self:
        """
        Validate that all dataset references in phase configs exist.

        Ensures that every phase config references an existing dataset by name.
        If a phase doesn't specify a dataset, the first dataset is used.
        """
        dataset_names = set(self.datasets.keys())

        for name, phase in self.phases.items():
            if phase.dataset is not None and phase.dataset not in dataset_names:
                raise ValueError(
                    f"Phase config '{name}' references undefined dataset '{phase.dataset}'. "
                    f"Available datasets: {sorted(dataset_names)}"
                )
            # If no dataset specified, first dataset will be used (see get_default_dataset)

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

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def get_model_names(self) -> list[str]:
        """
        Get list of all model names from the configuration.

        Returns:
            List of model name strings.
        """
        return [item.name for item in self.models.items]

    def get_dataset(self, name: str) -> DatasetConfig:
        """
        Get a dataset by name.

        Args:
            name: Dataset name.

        Returns:
            The dataset configuration.

        Raises:
            KeyError: If dataset not found.
        """
        if name not in self.datasets:
            raise KeyError(
                f"Dataset '{name}' not found. Available: {sorted(self.datasets.keys())}"
            )
        return self.datasets[name]

    def get_default_dataset_name(self) -> str:
        """
        Get the default dataset name (first dataset in the datasets dict).

        Returns:
            The name of the first dataset.
        """
        return next(iter(self.datasets.keys()))

    def get_default_dataset(self) -> DatasetConfig:
        """
        Get the default dataset (first dataset in the datasets dict).

        Returns:
            The first dataset configuration.
        """
        return next(iter(self.datasets.values()))

    def get_phase_dataset(self, phase: PhaseConfig) -> DatasetConfig:
        """
        Get the dataset for a specific phase.

        If the phase doesn't specify a dataset, uses the first dataset
        (the default).

        Args:
            phase: The phase configuration.

        Returns:
            The dataset configuration for the phase.
        """
        dataset_name = phase.dataset or self.get_default_dataset_name()
        return self.get_dataset(dataset_name)

    def get_profiling_phases(self) -> dict[str, PhaseConfig]:
        """
        Get phase configs that should be included in results.

        Returns:
            Dict of phase configs with exclude_from_results=False.
        """
        return {
            name: phase
            for name, phase in self.phases.items()
            if not phase.exclude_from_results
        }

    def get_warmup_phases(self) -> dict[str, PhaseConfig]:
        """
        Get warmup phase configs (excluded from results).

        Returns:
            Dict of phase configs with exclude_from_results=True.
        """
        return {
            name: phase
            for name, phase in self.phases.items()
            if phase.exclude_from_results
        }

    # ==========================================================================
    # CONVENIENCE PROPERTIES
    # ==========================================================================

    @property
    def comm_config(self) -> BaseZMQCommunicationConfig:
        """Get the ZMQ communication configuration.

        Cached so all callers get the same IPC paths. Without caching,
        each access creates a new ZMQIPCConfig with a fresh temp directory.
        """
        if not hasattr(self, "_comm_config_cache"):
            object.__setattr__(self, "_comm_config_cache", build_comm_config(self))
        return self._comm_config_cache

    @property
    def ui_type(self) -> UIType:
        """Get the UI type.

        Shortcut for runtime config.ui_type.

        Returns:
            UIType enum value.
        """
        return self.runtime.ui

    @property
    def workers_max(self) -> int | None:
        """Get the maximum number of workers.

        Shortcut for runtime config.workers.max.

        Returns:
            Maximum number of workers or None for auto-detect.
        """
        return self.runtime.workers

    @property
    def record_processor_service_count(self) -> int | None:
        """Get the number of record processor services.

        Shortcut for runtime config.record_processor_service_count.

        Returns:
            Number of record processors or None for auto-detect.
        """
        return self.runtime.record_processors

    @property
    def log_level(self) -> AIPerfLogLevel:
        """Get the logging level.

        Shortcut for runtime config.log_level.

        Returns:
            AIPerfLogLevel enum value.
        """
        return self.logging.level

    @property
    def verbose(self) -> bool:
        """Check if verbose logging is enabled.

        Shortcut for runtime config.verbose.

        Returns:
            True if logging level is DEBUG or more verbose.
        """
        from aiperf.common.enums import AIPerfLogLevel

        return self.logging.level in (AIPerfLogLevel.DEBUG, AIPerfLogLevel.TRACE)

    @property
    def extra_verbose(self) -> bool:
        """Check if extra verbose (trace) logging is enabled.

        Shortcut for runtime config.extra_verbose.

        Returns:
            True if logging level is TRACE.
        """
        from aiperf.common.enums import AIPerfLogLevel

        return self.logging.level == AIPerfLogLevel.TRACE

    @property
    def gpu_telemetry_disabled(self) -> bool:
        """Check if GPU telemetry collection is disabled.

        Shortcut for gpu_telemetry config.gpu_telemetry_disabled.

        Returns:
            True if GPU telemetry is disabled.
        """
        return not self.gpu_telemetry.enabled

    @property
    def gpu_telemetry_mode(self) -> GPUTelemetryMode:
        """Get the GPU telemetry display mode.

        Shortcut for gpu_telemetry config.gpu_telemetry_mode.

        Returns:
            GPUTelemetryMode enum value.
        """
        return self.gpu_telemetry.mode

    @gpu_telemetry_mode.setter
    def gpu_telemetry_mode(self, value: GPUTelemetryMode) -> None:
        """Set the GPU telemetry display mode.

        Shortcut for gpu_telemetry config.gpu_telemetry_mode.

        Args:
            value: GPUTelemetryMode enum value.
        """
        self.gpu_telemetry.mode = value

    @property
    def output(self) -> ArtifactsConfig:
        """Alias for artifacts config.

        Provides convenience access via config.output.* for file paths.
        """
        return self.artifacts

    @property
    def server_metrics_disabled(self) -> bool:
        """Check if server metrics collection is disabled."""
        return not self.server_metrics.enabled

    @property
    def server_metrics_formats(self) -> list:
        """Get the server metrics export formats."""
        return self.server_metrics.formats

    @property
    def benchmark_id(self) -> str:
        """Get the benchmark ID."""
        return self.artifacts.benchmark_id


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
