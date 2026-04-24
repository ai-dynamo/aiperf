# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIPerf Configuration v2.0 - YAML Configuration System

This package provides a complete, from-scratch implementation of the AIPerf
YAML configuration system. It is designed to support all 150+ configuration
options while providing a clean, composable, and well-documented API.

Key Features:
    - Named datasets with composition and augmentation
    - Multi-phase benchmark configuration with seamless transitions
    - ISL/OSL distributions for realistic workload modeling
    - Comprehensive multimodal support (images, audio, video)
    - SLO-based goodput tracking
    - Environment variable substitution
    - Full Pydantic validation with detailed error messages

Example Usage:
    >>> from aiperf.config import load_config, AIPerfConfig
    >>> config = load_config("benchmark.yaml")
    >>> print(config.models)
    >>> print(config.phases[0].name)

    Or programmatically:
    >>> from aiperf.config import AIPerfConfig
    >>> config = AIPerfConfig(
    ...     models=["llama-3-8b"],
    ...     endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    ...     datasets={"main": {"type": "synthetic", "count": 1000}},
    ...     phases=[{"name": "profiling", "dataset": "main", "concurrency": 8}]
    ... )

Schema Version: 2.0.0
"""

from aiperf.config.artifacts import (
    ArtifactsConfig,
    GpuTelemetryConfig,
    ServerMetricsConfig,
    ServerMetricsDiscoveryConfig,
)
from aiperf.config.benchmark import (
    BenchmarkPlan,
    BenchmarkRun,
    ResolvedConfig,
)
from aiperf.config.cli_converter import build_aiperf_config
from aiperf.config.cli_model import CLIModel
from aiperf.config.cli_parameter import (
    CLIParameter,
    DisableCLI,
)
from aiperf.config.config import (
    AIPerfConfig,
    BenchmarkConfig,
    build_comm_config,
)
from aiperf.config.dataset import (
    VIDEO_AUDIO_CODEC_MAP,
    AudioConfig,
    AugmentConfig,
    ComposedDataset,
    DatasetConfig,
    FileDataset,
    FileSourceConfig,
    ImageConfig,
    PrefixPromptConfig,
    PromptConfig,
    PublicDataset,
    RankingsConfig,
    SynthesisConfig,
    SyntheticDataset,
    VideoAudioConfig,
    VideoConfig,
)
from aiperf.config.defaults import (
    EndpointDefaults,
    InputDefaults,
    InputTokensDefaults,
    OutputDefaults,
    ServiceDefaults,
)
from aiperf.config.deployment import (
    DeploymentConfig,
    PodTemplateConfig,
    SchedulingConfig,
)
from aiperf.config.endpoint import (
    EndpointConfig,
    TemplateConfig,
)
from aiperf.config.loader import (
    ENV_VAR_PATTERN,
    ConfigurationError,
    MissingEnvironmentVariableError,
    build_benchmark_plan,
    dump_config,
    load_benchmark_plan,
    load_config,
    load_config_from_string,
    merge_configs,
    save_config,
    substitute_env_vars,
    validate_config_file,
)
from aiperf.config.metrics import MetricsConfig
from aiperf.config.models import (
    CommunicationConfig,
    DualBindCommunicationConfig,
    IpcCommunicationConfig,
    LoggingConfig,
    ModelItem,
    ModelsAdvanced,
    RuntimeConfig,
    SLOsConfig,
    TcpCommunicationConfig,
    TcpProxyConfig,
    TokenizerConfig,
    TokenizerOverride,
)
from aiperf.config.parsing import (
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
from aiperf.config.resolvers import (
    ConfigResolver,
    ConfigResolverChain,
    build_default_resolver_chain,
)
from aiperf.config.sweep import (
    GridSweep,
    ScenarioSweep,
    SweepConfig,
    SweepVariation,
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
from aiperf.config.zmq import (
    BaseZMQCommunicationConfig,
    BaseZMQProxyConfig,
    ZMQDualBindConfig,
    ZMQDualBindProxyConfig,
    ZMQIPCConfig,
    ZMQIPCProxyConfig,
    ZMQTCPConfig,
    ZMQTCPProxyConfig,
)

__all__ = [
    "AIPerfConfig",
    "ArtifactsConfig",
    "AudioConfig",
    "AugmentConfig",
    "BasePhaseConfig",
    "BaseZMQCommunicationConfig",
    "BaseZMQProxyConfig",
    "BenchmarkConfig",
    "BenchmarkPlan",
    "BenchmarkRun",
    "CLIModel",
    "CLIParameter",
    "CancellationConfig",
    "CommunicationConfig",
    "ComposedDataset",
    "ConcurrencyPhase",
    "ConfigResolver",
    "ConfigResolverChain",
    "ConfigurationError",
    "ConstantPhase",
    "DatasetConfig",
    "DeploymentConfig",
    "DisableCLI",
    "Distribution",
    "DualBindCommunicationConfig",
    "ENV_VAR_PATTERN",
    "EmpiricalDistribution",
    "EmpiricalPoint",
    "EndpointConfig",
    "EndpointDefaults",
    "FileDataset",
    "FileSourceConfig",
    "FixedDistribution",
    "FixedSchedulePhase",
    "GammaPhase",
    "GpuTelemetryConfig",
    "GridSweep",
    "ImageConfig",
    "InputDefaults",
    "InputTokensDefaults",
    "IpcCommunicationConfig",
    "LogNormalDistribution",
    "LoggingConfig",
    "MissingEnvironmentVariableError",
    "MetricsConfig",
    "ModelItem",
    "ModelsAdvanced",
    "MultimodalDistribution",
    "NormalDistribution",
    "OutputDefaults",
    "PeakEntry",
    "PhaseConfig",
    "PhaseType",
    "PhaseTypeStr",
    "PodTemplateConfig",
    "PoissonPhase",
    "PrefixPromptConfig",
    "PromptConfig",
    "PublicDataset",
    "RampConfig",
    "RankingsConfig",
    "RatePhaseConfig",
    "ResolvedConfig",
    "RuntimeConfig",
    "SLOsConfig",
    "SamplingDistribution",
    "ScenarioSweep",
    "SchedulingConfig",
    "SequenceDistributionEntry",
    "ServerMetricsConfig",
    "ServerMetricsDiscoveryConfig",
    "ServiceDefaults",
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
    "ZMQDualBindConfig",
    "ZMQDualBindProxyConfig",
    "ZMQIPCConfig",
    "ZMQIPCProxyConfig",
    "ZMQTCPConfig",
    "ZMQTCPProxyConfig",
    "build_aiperf_config",
    "build_benchmark_plan",
    "build_comm_config",
    "build_default_resolver_chain",
    "coerce_value",
    "dump_config",
    "load_benchmark_plan",
    "load_config",
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
