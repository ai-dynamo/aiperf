# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field

from aiperf.common.config.zmq_config import BaseZMQCommunicationConfig
from aiperf.common.enums import CommunicationBackend, ServiceRunType


class ServiceConfig(BaseModel):
    """Base configuration for all services.

    This class provides the common configuration parameters needed by all services.
    """

    # TODO: this needs to be cleaned up and finalized

    service_run_type: ServiceRunType = Field(
        default=ServiceRunType.MULTIPROCESSING,
        description="Type of service run (MULTIPROCESSING, KUBERNETES)",
    )
    comm_backend: CommunicationBackend = Field(
        default=CommunicationBackend.ZMQ_IPC,
        description="Communication backend to use",
    )
    comm_config: BaseZMQCommunicationConfig | None = Field(
        default=None,
        description="Communication configuration",
    )
    heartbeat_timeout: float = Field(
        default=60.0,
        description="Time in seconds after which a service is considered dead if no "
        "heartbeat received",
    )
    registration_timeout: float = Field(
        default=60.0,
        description="Time in seconds to wait for all required services to register",
    )
    command_timeout: float = Field(
        default=10.0,
        description="Default timeout for command responses",
    )
    heartbeat_interval: float = Field(
        default=10.0,
        description="Interval in seconds between heartbeat messages",
    )
    min_workers: int = Field(
        default=100,
        description="Minimum number of idle workers to maintain",
    )
    max_workers: int = Field(
        default=100,
        description="Maximum number of workers to create",
    )
    target_idle_workers: int = Field(
        default=10,
        description="Target number of idle workers to maintain",
    )
