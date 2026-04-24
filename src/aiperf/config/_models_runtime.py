# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime and logging configuration models.

Split out of ``models.py`` so the public module stays under the ergonomics
file-size cap. Re-exported via :mod:`aiperf.config.models`.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from aiperf.common.enums import AIPerfLogLevel
from aiperf.config._base import BaseConfig
from aiperf.config._models_comm import CommunicationConfig
from aiperf.plugin.enums import ServiceRunType, UIType


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
            description="Maximum worker processes. "
            "null = auto-detect based on CPU cores.",
        ),
    ]

    record_processors: Annotated[
        int | None,
        Field(
            ge=1,
            default=None,
            description="Number of parallel record processors. "
            "null = auto-detect based on CPU cores.",
        ),
    ]

    service_run_type: Annotated[
        ServiceRunType,
        Field(
            default=ServiceRunType.MULTIPROCESSING,
            description="Type of service run. "
            "multiprocessing: local multi-process (default), "
            "kubernetes: distributed across pods.",
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
            description="Base URL for dataset API endpoints in Kubernetes mode. "
            "Set by the runner to allow worker pods to download datasets.",
        ),
    ]

    workers_per_pod: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            le=100,
            description="Number of worker service containers per Kubernetes worker pod.",
        ),
    ]

    record_processors_per_pod: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            le=100,
            description="Number of record processor service containers per Kubernetes worker pod.",
        ),
    ]

    workers_min: Annotated[
        int | None,
        Field(
            default=None,
            ge=1,
            description="Minimum number of worker processes.",
        ),
    ]

    @model_validator(mode="after")
    def _validate_api_host_requires_port(self) -> Self:
        if self.api_host is not None and self.api_port is None:
            raise ValueError("api_host requires api_port to be set")
        return self


class LoggingConfig(BaseConfig):
    """Logging configuration for verbosity and debug settings."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    level: Annotated[
        AIPerfLogLevel,
        Field(
            default=AIPerfLogLevel.INFO,
            description="Global logging verbosity level. "
            "trace: most verbose, error: least verbose.",
        ),
    ]
