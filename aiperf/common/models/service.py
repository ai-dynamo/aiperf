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
from typing import Union

from pydantic import BaseModel, Field

from aiperf.common.enums import ServiceRunType


class AsyncIORunInfo(BaseModel):
    """Information about a service's run in asyncio."""

    task_id: str = Field(
        ...,
        description="ID of the task",
    )


class MultiProcessingRunInfo(BaseModel):
    """Information about a service's run in multiprocessing."""

    process_id: str = Field(
        ...,
        description="ID of the process",
    )


class KubernetesRunInfo(BaseModel):
    """Information about a service's run in Kubernetes."""

    pod_id: str = Field(
        ...,
        description="ID of the pod",
    )
    node_id: str = Field(
        ...,
        description="ID of the node",
    )
    namespace: str = Field(
        ...,
        description="Namespace of the pod",
    )
    deployment_id: str = Field(
        ...,
        description="ID of the deployment",
    )


class ServiceRunInfo(BaseModel):
    """Information about a running service."""

    service_id: str = Field(
        ...,
        description="ID of the service",
    )
    service_type: str = Field(
        ...,
        description="Type of the service",
    )
    run_type: ServiceRunType = Field(
        ...,
        description="Type of the run",
    )
    run_info: Union[AsyncIORunInfo, MultiProcessingRunInfo, KubernetesRunInfo] = Field(
        ...,
        description="Information about the run",
    )
