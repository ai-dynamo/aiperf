# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataclasses that carry estimation results between layers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from aiperf.kubernetes._memory_estimator.constants import (
    _HEADROOM_WARNING_PCT,
    _PEAK_MARGIN,
    _STEADY_STATE_MARGIN,
)
from aiperf.kubernetes._memory_estimator.params import MemoryEstimationParams


@dataclass(slots=True)
class ComponentEstimate:
    """Memory estimate for one logical component."""

    name: str
    """Display name of the component."""

    base_mib: float
    """Fixed baseline memory in MiB (subprocess + service overhead)."""

    variable_mib: float
    """Workload-dependent memory in MiB (scales with requests, tokens, etc.)."""

    peak_mib: float
    """Worst-case memory including transient spikes."""

    formula: str
    """Human-readable formula explaining how the estimate was computed."""

    dominant_factor: str
    """Primary driver of memory usage for this component."""

    warning: str | None = None
    """Optional warning when estimated usage is unusually high."""

    @property
    def steady_state_mib(self) -> float:
        return self.base_mib + self.variable_mib


@dataclass(slots=True)
class PodEstimate:
    """Aggregated memory estimate for a pod."""

    pod_type: str
    """Pod role: 'controller', 'worker', or 'operator'."""

    components: list[ComponentEstimate]
    """Per-component estimates that sum to the pod total."""

    current_limit_mib: float
    """Configured K8s memory limit for this pod type."""

    replicas: int = 1
    """Number of pod replicas of this type in the cluster."""

    @property
    def total_steady_state_mib(self) -> float:
        return sum(c.steady_state_mib for c in self.components)

    @property
    def total_peak_mib(self) -> float:
        return sum(c.peak_mib for c in self.components)

    @property
    def recommended_request_mib(self) -> int:
        return int(math.ceil(self.total_steady_state_mib * _STEADY_STATE_MARGIN))

    @property
    def recommended_limit_mib(self) -> int:
        return int(math.ceil(self.total_peak_mib * _PEAK_MARGIN))

    @property
    def headroom_pct(self) -> float:
        if self.current_limit_mib <= 0:
            return 0.0
        return (
            (self.current_limit_mib - self.total_peak_mib)
            / self.current_limit_mib
            * 100
        )

    @property
    def at_risk(self) -> bool:
        return self.headroom_pct < _HEADROOM_WARNING_PCT


@dataclass(slots=True)
class ClusterMemoryEstimate:
    """Full cluster memory estimate."""

    params: MemoryEstimationParams
    """Input parameters used to produce this estimate."""

    controller: PodEstimate
    """Memory estimate for the controller pod."""

    worker_pod: PodEstimate
    """Memory estimate for a single worker pod."""

    operator: PodEstimate
    """Memory estimate for the operator pod."""

    warnings: list[str] = field(default_factory=list)
    """Generated warnings about memory risk."""

    recommendations: list[str] = field(default_factory=list)
    """Actionable recommendations for resource tuning."""

    @property
    def total_cluster_mib(self) -> float:
        return (
            self.controller.total_steady_state_mib
            + self.worker_pod.total_steady_state_mib * self.worker_pod.replicas
            + self.operator.total_steady_state_mib
        )
