# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - YAML Configuration System

This package provides a complete, from-scratch implementation of the AIPerf
YAML configuration system. It is designed to support all 150+ configuration
options while providing a clean, composable, and well-documented API.

Key Features:
    - Synthetic, file, and public datasets with optional OSL augmentation
    - Multi-phase benchmark configuration with seamless transitions
    - ISL/OSL distributions for realistic workload modeling
    - Comprehensive multimodal support (images, audio, video)
    - SLO-based goodput tracking
    - Environment variable substitution
    - Full Pydantic validation with detailed error messages

Example Usage:
    >>> from aiperf.config import load_config, AIPerfConfig
    >>> config = load_config("benchmark.yaml")
    >>> print(config.benchmark.models)
    >>> print(config.benchmark.phases[0].name)

    Or programmatically:
    >>> from aiperf.config import AIPerfConfig
    >>> config = AIPerfConfig(
    ...     benchmark={
    ...         "models": ["llama-3-8b"],
    ...         "endpoint": {"urls": ["http://localhost:8000/v1/chat/completions"]},
    ...         "datasets": [{"name": "main", "type": "synthetic", "entries": 1000}],
    ...         "phases": [{"name": "profiling", "type": "concurrency", "requests": 100, "concurrency": 8}],
    ...     }
    ... )

Schema Version: 2.0.0
"""

from typing import Any

from aiperf.config.artifacts import (
    ArtifactsConfig,
    OutputDefaults,
)
from aiperf.config.cli_parameter import (
    CLIParameter,
)
from aiperf.config.comm import (
    BaseZMQCommunicationConfig,
    BaseZMQProxyConfig,
    ZMQDualBindConfig,
    ZMQDualBindProxyConfig,
    ZMQIPCConfig,
    ZMQIPCProxyConfig,
    ZMQTCPConfig,
    ZMQTCPProxyConfig,
)
from aiperf.config.comm.inputs import (
    CommunicationConfig,
    DualBindCommunicationConfig,
    IpcCommunicationConfig,
    TcpCommunicationConfig,
    TcpProxyConfig,
)
from aiperf.config.config import (
    AIPerfConfig,
    BenchmarkConfig,
    build_comm_config,
)
from aiperf.config.dataset import (
    VIDEO_AUDIO_CODEC_MAP,
    AudioConfig,
    DatasetConfig,
    FileDataset,
    ImageConfig,
    PrefixPromptConfig,
    PromptConfig,
    PromptSelectionConfig,
    PublicDataset,
    RankingsConfig,
    SynthesisConfig,
    SyntheticDataset,
    VideoAudioConfig,
    VideoConfig,
)
from aiperf.config.dataset.defaults import (
    InputDefaults,
    InputTokensDefaults,
)
from aiperf.config.endpoint import (
    EndpointConfig,
    EndpointDefaults,
    TemplateConfig,
)
from aiperf.config.gpu_telemetry import (
    GpuTelemetryConfig,
)
from aiperf.config.loader import (
    ENV_VAR_PATTERN,
    ConfigurationError,
    MissingEnvironmentVariableError,
    build_benchmark_plan,
    dump_config,
    load_benchmark_plan,
    load_config,
    load_config_from_mapping,
    load_config_from_string,
    merge_configs,
    save_config,
    substitute_env_vars,
    validate_config_file,
)
from aiperf.config.loader.parsing import (
    coerce_value,
    parse_file,
    parse_service_types,
    parse_str_as_numeric_dict,
    parse_str_or_csv_list,
    parse_str_or_dict_as_tuple_list,
    parse_str_or_list,
    parse_str_or_list_of_positive_values,
    print_str_or_list,
    validate_sequence_distribution,
)
from aiperf.config.logging import (
    LoggingConfig,
)
from aiperf.config.metrics import MetricsConfig
from aiperf.config.mlflow import (
    MLflowConfig,
)
from aiperf.config.models import (
    ModelItem,
    ModelsAdvanced,
    TokenizerOverride,
)
from aiperf.config.otel import (
    OTelConfig,
)
from aiperf.config.phases import (
    BasePhaseConfig,
    CancellationConfig,
    ConcurrencyPhase,
    ConstantPhase,
    FixedSchedulePhase,
    GammaPhase,
    PhaseConfig,
    PhaseType,
    PhaseTypeStr,
    PoissonPhase,
    RampConfig,
    RatePhaseConfig,
    UserCentricPhase,
)
from aiperf.config.resolution import (
    BenchmarkPlan,
    BenchmarkRun,
    ConfigResolver,
    ConfigResolverChain,
    ResolvedConfig,
    build_default_resolver_chain,
)
from aiperf.config.runtime import (
    RuntimeConfig,
    ServiceDefaults,
)
from aiperf.config.server_metrics import (
    ServerMetricsConfig,
    ServerMetricsDiscoveryConfig,
)
from aiperf.config.slos import (
    SLOsConfig,
)
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    GridSweep,
    LatinHypercubeSweep,
    Objective,
    OutcomeConstraint,
    SamplingDimension,
    ScenarioSweep,
    SobolSweep,
    SweepConfig,
    SweepVariation,
)
from aiperf.config.sweep.multi_run import (
    ConvergenceConfig,
)
from aiperf.config.tokenizer import (
    TokenizerConfig,
)
from aiperf.config.types import (
    Distribution,
    EmpiricalDistribution,
    EmpiricalPoint,
    FixedDistribution,
    LogNormalDistribution,
    MultimodalDistribution,
    NormalDistribution,
    PeakEntry,
    SamplingDistribution,
    SequenceDistributionEntry,
    validate_probability_distribution,
)
from aiperf.config.wandb import (
    WandbConfig,
)

# Kubernetes-only config models, resolved on first attribute access. Importing
# them eagerly pulls aiperf.config.deployment and aiperf.config.kube -- and
# through them aiperf.kubernetes.enums -- into all 10 service processes of every
# local multiprocessing run, which never touch them. Direct submodule imports
# (the form every real consumer uses) are unaffected.
_LAZY_EXPORTS = {
    "DeploymentConfig": "aiperf.config.deployment",
    "PodTemplateConfig": "aiperf.config.deployment",
    "SchedulingConfig": "aiperf.config.deployment",
    "KubeManageOptions": "aiperf.config.kube",
    "KubeOptions": "aiperf.config.kube",
    "SecretMountConfig": "aiperf.config.kube",
}


def __getattr__(name: str) -> Any:
    """Resolve a Kubernetes-only config export on first access, then cache it."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = [
    "AIPerfConfig",
    "AdaptiveSearchSweep",
    "ArtifactsConfig",
    "AudioConfig",
    "BasePhaseConfig",
    "BaseZMQCommunicationConfig",
    "BaseZMQProxyConfig",
    "BenchmarkConfig",
    "BenchmarkPlan",
    "BenchmarkRun",
    "CLIParameter",
    "CancellationConfig",
    "CommunicationConfig",
    "ConcurrencyPhase",
    "ConfigResolver",
    "ConfigResolverChain",
    "ConfigurationError",
    "ConstantPhase",
    "ConvergenceConfig",
    "DatasetConfig",
    "DeploymentConfig",
    "Distribution",
    "DualBindCommunicationConfig",
    "ENV_VAR_PATTERN",
    "EmpiricalDistribution",
    "EmpiricalPoint",
    "EndpointConfig",
    "EndpointDefaults",
    "FileDataset",
    "FixedDistribution",
    "FixedSchedulePhase",
    "GammaPhase",
    "GpuTelemetryConfig",
    "GridSweep",
    "ImageConfig",
    "InputDefaults",
    "InputTokensDefaults",
    "IpcCommunicationConfig",
    "KubeManageOptions",
    "KubeOptions",
    "LatinHypercubeSweep",
    "LogNormalDistribution",
    "LoggingConfig",
    "MLflowConfig",
    "MissingEnvironmentVariableError",
    "MetricsConfig",
    "ModelItem",
    "ModelsAdvanced",
    "MultimodalDistribution",
    "NormalDistribution",
    "Objective",
    "OutcomeConstraint",
    "OTelConfig",
    "OutputDefaults",
    "PeakEntry",
    "PhaseConfig",
    "PhaseType",
    "PhaseTypeStr",
    "PoissonPhase",
    "PodTemplateConfig",
    "PrefixPromptConfig",
    "PromptConfig",
    "PromptSelectionConfig",
    "PublicDataset",
    "RampConfig",
    "RankingsConfig",
    "RatePhaseConfig",
    "ResolvedConfig",
    "RuntimeConfig",
    "SLOsConfig",
    "SamplingDimension",
    "SamplingDistribution",
    "ScenarioSweep",
    "SchedulingConfig",
    "SecretMountConfig",
    "SequenceDistributionEntry",
    "ServerMetricsConfig",
    "ServerMetricsDiscoveryConfig",
    "ServiceDefaults",
    "SobolSweep",
    "SweepConfig",
    "SweepVariation",
    "SynthesisConfig",
    "SyntheticDataset",
    "TcpCommunicationConfig",
    "TcpProxyConfig",
    "TemplateConfig",
    "TokenizerConfig",
    "TokenizerOverride",
    "UserCentricPhase",
    "VIDEO_AUDIO_CODEC_MAP",
    "VideoAudioConfig",
    "VideoConfig",
    "WandbConfig",
    "ZMQDualBindConfig",
    "ZMQDualBindProxyConfig",
    "ZMQIPCConfig",
    "ZMQIPCProxyConfig",
    "ZMQTCPConfig",
    "ZMQTCPProxyConfig",
    "build_benchmark_plan",
    "build_comm_config",
    "build_default_resolver_chain",
    "coerce_value",
    "dump_config",
    "load_benchmark_plan",
    "load_config",
    "load_config_from_mapping",
    "load_config_from_string",
    "merge_configs",
    "parse_file",
    "parse_service_types",
    "parse_str_as_numeric_dict",
    "parse_str_or_csv_list",
    "parse_str_or_dict_as_tuple_list",
    "parse_str_or_list",
    "parse_str_or_list_of_positive_values",
    "print_str_or_list",
    "save_config",
    "substitute_env_vars",
    "validate_config_file",
    "validate_probability_distribution",
    "validate_sequence_distribution",
]
