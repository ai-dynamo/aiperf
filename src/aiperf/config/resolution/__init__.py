# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolution sub-package: post-load runtime artifacts and resolvers.

Public surface preserved: imports that previously read from
`aiperf.config.resolution.plan`, `aiperf.config.resolution.predicates`, or `aiperf.config.resolution.resolvers`
should now read from `aiperf.config.resolution` (or the submodules).
"""

from aiperf.config.resolution.plan import (
    BenchmarkPlan,
    BenchmarkRun,
    FailurePolicy,
    ResolvedConfig,
)
from aiperf.config.resolution.predicates import (
    check_phase_dataset_compatibility,
    conversations_have_timing_data,
    get_dataset_entries,
    get_dataset_format,
    get_dataset_type,
    get_phase_timing,
    get_random_seed,
    get_sampling_strategy,
    get_stop_condition,
    is_file_dataset,
    is_multi_turn_dataset,
    is_public_dataset,
    is_synthetic_dataset,
    is_trace_dataset,
    requires_multi_turn,
    requires_sequential_sampling,
)
from aiperf.config.resolution.resolvers import (
    ArtifactDirResolver,
    CommConfigResolver,
    ConfigResolver,
    ConfigResolverChain,
    GpuMetricsResolver,
    TimingResolver,
    TokenizerResolver,
    build_default_resolver_chain,
)

__all__ = [
    "ArtifactDirResolver",
    "BenchmarkPlan",
    "BenchmarkRun",
    "CommConfigResolver",
    "ConfigResolver",
    "ConfigResolverChain",
    "FailurePolicy",
    "GpuMetricsResolver",
    "ResolvedConfig",
    "TimingResolver",
    "TokenizerResolver",
    "build_default_resolver_chain",
    "check_phase_dataset_compatibility",
    "conversations_have_timing_data",
    "get_dataset_entries",
    "get_dataset_format",
    "get_dataset_type",
    "get_phase_timing",
    "get_random_seed",
    "get_sampling_strategy",
    "get_stop_condition",
    "is_file_dataset",
    "is_multi_turn_dataset",
    "is_public_dataset",
    "is_synthetic_dataset",
    "is_trace_dataset",
    "requires_multi_turn",
    "requires_sequential_sampling",
]
