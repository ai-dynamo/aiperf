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
from pydantic import BaseModel, Field

from aiperf.common.enums import CommBackend, ServiceRunType


class ServiceConfig(BaseModel):
    """Base configuration for all services.

    This class provides the common configuration parameters needed by all services.
    """

    # TODO: this needs to be cleaned up and finalized

    service_run_type: ServiceRunType = Field(
        default=ServiceRunType.MULTIPROCESSING,
        description="Type of service run (MULTIPROCESSING, KUBERNETES)",
    )
    comm_backend: CommBackend = Field(
        default=CommBackend.ZMQ_TCP,
        description="Communication backend to use",
    )
    heartbeat_timeout: float = Field(
        default=60.0,
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
    heartbeat_interval: float = Field(
        default=10.0,
        description="Interval in seconds between heartbeat messages",
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
