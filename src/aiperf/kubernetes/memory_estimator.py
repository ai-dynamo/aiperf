# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory estimation framework for AIPerf Kubernetes deployments.

Computes per-pod and cluster-wide memory estimates from an AIPerfConfig
and deployment parameters. Used by ``aiperf kube generate``, ``aiperf kube profile``,
and the operator preflight to detect OOM risk before deployment.

The model is static: aggregate MiB baselines come from real-cluster
working-set sweeps, and the per-object byte constants are measured in-process
against the real model classes (see
``aiperf.kubernetes._memory_estimator.constants``).

This module is a thin facade. Implementation lives in
``aiperf.kubernetes._memory_estimator`` split across several files for
file-size ergonomics. Re-exported symbols — including the underscore-prefixed
calibration helpers consumed by ``tests/unit/kubernetes/test_memory_estimator.py``
— are considered part of the stable surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aiperf.kubernetes._memory_estimator.components import (
    _estimate_dataset_manager as _estimate_dataset_manager,
)
from aiperf.kubernetes._memory_estimator.components import (
    _estimate_fixed_service as _estimate_fixed_service,
)
from aiperf.kubernetes._memory_estimator.components import (
    _estimate_gpu_telemetry as _estimate_gpu_telemetry,
)
from aiperf.kubernetes._memory_estimator.components import (
    _estimate_record_processor as _estimate_record_processor,
)
from aiperf.kubernetes._memory_estimator.components import (
    _estimate_records_manager as _estimate_records_manager,
)
from aiperf.kubernetes._memory_estimator.components import (
    _estimate_server_metrics as _estimate_server_metrics,
)
from aiperf.kubernetes._memory_estimator.components import (
    _estimate_worker as _estimate_worker,
)
from aiperf.kubernetes._memory_estimator.estimates import (
    ClusterMemoryEstimate,
    ComponentEstimate,
    PodEstimate,
)
from aiperf.kubernetes._memory_estimator.estimator import MemoryEstimator
from aiperf.kubernetes._memory_estimator.formatting import format_estimate
from aiperf.kubernetes._memory_estimator.params import MemoryEstimationParams
from aiperf.kubernetes._memory_estimator.utils import (
    _ceil_pow2 as _ceil_pow2,
)
from aiperf.kubernetes._memory_estimator.utils import (
    _mib as _mib,
)
from aiperf.kubernetes.spec_converter import DEFAULT_CONNECTIONS_PER_WORKER

if TYPE_CHECKING:
    from aiperf.config.config import AIPerfConfig

__all__ = [
    "ClusterMemoryEstimate",
    "ComponentEstimate",
    "MemoryEstimationParams",
    "MemoryEstimator",
    "PodEstimate",
    "estimate_memory",
    "format_estimate",
]


def estimate_memory(
    config: AIPerfConfig,
    total_workers: int = 10,
    workers_per_pod: int | None = None,
    connections_per_worker: int = DEFAULT_CONNECTIONS_PER_WORKER,
) -> ClusterMemoryEstimate:
    """Estimate memory usage for an AIPerf Kubernetes deployment.

    This is the primary entry point. Derives all estimation parameters
    from the config and returns a full cluster estimate with warnings.

    Args:
        config: The benchmark configuration.
        total_workers: Total desired workers.
        workers_per_pod: Workers per pod (None = default).
        connections_per_worker: Connections per worker.

    Returns:
        ClusterMemoryEstimate with per-pod and cluster-wide estimates.
    """
    params = MemoryEstimationParams.from_config(
        config, total_workers, workers_per_pod, connections_per_worker
    )
    estimator = MemoryEstimator(params)
    return estimator.estimate()
