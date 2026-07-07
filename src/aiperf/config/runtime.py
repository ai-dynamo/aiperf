# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime configuration models.

Split out of ``models.py`` so each config section lives in its own file.
Re-exported via :mod:`aiperf.config`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Self

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.enums import AIPerfLogLevel
from aiperf.config.base import BaseConfig
from aiperf.config.comm.inputs import CommunicationConfig
from aiperf.plugin.enums import CommunicationBackend, ServiceRunType, UIType


@dataclass(frozen=True)
class ServiceDefaults:
    SERVICE_RUN_TYPE = ServiceRunType.MULTIPROCESSING
    COMM_BACKEND = CommunicationBackend.ZMQ_IPC
    COMM_CONFIG = None
    LOG_LEVEL = AIPerfLogLevel.INFO
    VERBOSE = False
    EXTRA_VERBOSE = False
    LOG_PATH = None
    RECORD_PROCESSOR_SERVICE_COUNT = None
    UI_TYPE = UIType.DASHBOARD


class RuntimeConfig(BaseConfig):
    """Runtime configuration for benchmark execution."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    @property
    def uses_worker_group_manager(self) -> bool:
        """Whether this runtime routes workers through WorkerGroupManager."""
        # Component-integration tests share a single process and one
        # FakeCommunication bus, so pod-lifecycle routing cannot be wired.
        # Treat every service as locally-driven in that mode.
        import os

        if os.environ.get("AIPERF_FAKE_IN_PROCESS_MODE") == "1":
            return False
        return self.service_run_type in {
            ServiceRunType.MULTIPROCESSING,
            ServiceRunType.KUBERNETES,
        }

    @property
    def uses_local_worker_group_manager(self) -> bool:
        """Whether local multiprocessing should launch a group-manager boundary."""
        import os

        if os.environ.get("AIPERF_FAKE_IN_PROCESS_MODE") == "1":
            return False
        return self.service_run_type == ServiceRunType.MULTIPROCESSING

    ui: Annotated[
        UIType,
        Field(
            default=UIType.DASHBOARD,
            description="User interface mode. "
            "dashboard: rich interactive UI, "
            "simple: text progress, "
            "none: silent operation.",
        ),
    ]

    workers: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Total worker processes across the deployment (the maximum, "
            "ramped up to from `workers_min`). null = auto-detect based on CPU cores. "
            "Distinct from `workers_per_pod` (per-Kubernetes-pod fan-out) and "
            "`workers_min` (lower bound for adaptive ramp).",
        ),
    ]

    record_processors: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Total parallel record processors across the deployment. "
            "null = auto-detect based on CPU cores. Distinct from "
            "`record_processors_per_pod` (per-Kubernetes-pod fan-out).",
        ),
    ]

    service_run_type: Annotated[
        ServiceRunType,
        Field(
            default=ServiceRunType.MULTIPROCESSING,
            description="Execution mode. multiprocessing: local multi-process "
            "(default for `aiperf profile`). kubernetes: distributed across pods. "
            "[operator-managed under AIPerfJob — the operator forces 'kubernetes' "
            "on the controller pod; do not set this in a CR spec.]",
        ),
    ]

    communication: Annotated[
        CommunicationConfig | None,
        Field(
            default=None,
            description="Inter-process communication configuration. "
            "Defaults to IPC for single-machine operation.",
        ),
    ]

    api_port: Annotated[
        int | None,
        Field(
            ge=1,
            le=65535,
            default=None,
            description="AIPerf API server port. Enables HTTP and WebSocket endpoints "
            "for real-time metrics and control.",
        ),
    ]

    api_host: Annotated[
        str | None,
        Field(
            default=None,
            description="AIPerf API server host. Requires api_port to be set.",
        ),
    ]

    # Kubernetes-specific runtime fields (set by runner/operator, not user-facing)

    dataset_api_base_url: Annotated[
        str | None,
        Field(
            default=None,
            description="Base URL the operator injects into worker pods so they can "
            "fetch datasets from the controller-pod sidecar in Kubernetes mode. "
            "[operator-managed; do not set in a CR spec.]",
        ),
    ]

    workers_per_pod: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            le=100,
            description="Worker containers packed into each Kubernetes worker pod "
            "(Kubernetes mode only). The total number of worker containers cluster-wide "
            "is `workers` (or auto-detected); this knob controls how that total is "
            "fanned across pods. Ignored in multiprocessing mode.",
        ),
    ]

    record_processors_per_pod: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            le=100,
            description="Record-processor containers packed into each Kubernetes worker pod "
            "(Kubernetes mode only). Sibling of `workers_per_pod`; controls per-pod "
            "fan-out, not the cluster-wide total. Ignored in multiprocessing mode.",
        ),
    ]

    workers_min: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Lower bound on worker processes for the adaptive ramp. "
            "Distinct from `workers` (the upper bound / target total).",
        ),
    ]

    @model_validator(mode="after")
    def _validate_api_host_requires_port(self) -> Self:
        if self.api_host is not None and self.api_port is None:
            raise ValueError("api_host requires api_port to be set")
        return self


__all__ = [
    "RuntimeConfig",
    "ServiceDefaults",
]
