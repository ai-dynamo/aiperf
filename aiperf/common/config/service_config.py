#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Optional

from pydantic import BaseModel, Field

from aiperf.common.enums import CommBackend, ServiceRunType


class ServiceConfig(BaseModel):
    """Base configuration for all services.

    This class provides the common configuration parameters needed by all services.
    """

    # TODO: this needs to be cleaned up and finalized

    comm_backend: CommBackend = Field(
        default=CommBackend.ZMQ,
        description="Communication backend to use",
    )
    service_run_type: ServiceRunType = Field(
        default=ServiceRunType.ASYNC,
        description="Type of service run (ASYNCIO, MULTIPROCESSING, KUBERNETES)",
    )
    control_topic: str = Field(
        default="control",
        description="Topic for control messages",
    )
    status_topic: str = Field(
        default="status",
        description="Topic for status messages",
    )
    response_topic: str = Field(
        default="response",
        description="Topic for response messages",
    )
    heartbeat_interval: float = Field(
        default=5.0,
        description="Interval in seconds between heartbeat messages",
    )


class ControllerConfig(ServiceConfig):
    """Configuration for controller services.

    This class extends the base service configuration with controller-specific settings.
    """

    # TODO: this needs to be cleaned up and finalized

    heartbeat_timeout: float = Field(
        default=15.0,
        description="Time in seconds after which a service is considered dead if no heartbeat received",
    )
    registration_timeout: float = Field(
        default=60.0,
        description="Time in seconds to wait for all required services to register",
    )
    command_timeout: float = Field(
        default=10.0,
        description="Default timeout for command responses",
    )
    namespace: str = Field(
        default="default",
        description="Kubernetes namespace to use when running with Kubernetes backend",
    )
    workers: int = Field(
        default=1,
        description="Number of worker processes to start",
    )
    port: int = Field(
        default=8000,
        description="Port to run the service on",
    )


class DatasetServiceConfig(ServiceConfig):
    """Configuration for the Dataset Manager service."""

    data_topic: str = "dataset_data"
    # TODO: this needs to be cleaned up and finalized


class TimingServiceConfig(ServiceConfig):
    """Configuration for the Timing Manager service."""

    # TODO: this needs to be cleaned up and finalized

    data_topic: str = "timing_data"
    credit_topic: str = Field(
        default="credit",
        description="Topic for credit messages",
    )


class RecordsServiceConfig(ServiceConfig):
    """Configuration for the Records Manager service."""

    # TODO: this needs to be cleaned up and finalized

    data_topic: str = "records_data"
    results_topic: str = Field(
        default="results",
        description="Topic for results data",
    )


class WorkerServiceConfig(ServiceConfig):
    """Configuration for the Worker Manager service."""

    # TODO: this needs to be cleaned up and finalized

    data_topic: str = "worker_data"
    credit_topic: str = Field(
        default="credit",
        description="Topic for credit messages",
    )
    results_topic: str = Field(
        default="results",
        description="Topic for results data",
    )
    min_workers: int = Field(
        default=5,
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


class WorkerConfig(BaseModel):
    """Configuration for Worker instances."""

    # TODO: this needs to be cleaned up and finalized

    worker_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for this worker",
    )
    manager_id: Optional[str] = Field(
        default=None,
        description="ID of the worker manager that created this worker",
    )
    comm_backend: CommBackend = Field(
        default=CommBackend.ZMQ,
        description="Communication backend to use",
    )
    manager_data_topic: str = Field(
        default="worker_data",
        description="Topic for communication with the worker manager",
    )


class PostProcessorConfig(ServiceConfig):
    """Configuration for Post Processor services."""

    # TODO: this needs to be cleaned up and finalized

    data_topic: str = "post_processor_data"
    records_topic: str = Field(
        default="records_data",
        description="Topic for records data",
    )
    metrics_topic: str = Field(
        default="metrics",
        description="Topic for metrics data",
    )
