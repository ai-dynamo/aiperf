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
from aiperf.common.enums.base import StrEnum


# Service-related enums
class ServiceRunType(StrEnum):
    """Different ways to run a service."""

    MULTIPROCESSING = "process"
    KUBERNETES = "k8s"


class ServiceState(StrEnum):
    """States a service can be in throughout its lifecycle."""

    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class ServiceType(StrEnum):
    """Types of services in the AIPerf system."""

    SYSTEM_CONTROLLER = "system_controller"
    DATASET_MANAGER = "dataset_manager"
    TIMING_MANAGER = "timing_manager"
    WORKER_MANAGER = "worker_manager"
    RECORDS_MANAGER = "records_manager"
    POST_PROCESSOR_MANAGER = "post_processor_manager"
    WORKER = "worker"
    TEST = "test_service"  # Used in tests


class ServiceRegistrationStatus(StrEnum):
    """Status of service registration."""

    UNREGISTERED = "unregistered"
    WAITING = "waiting"
    REGISTERED = "registered"
    TIMEOUT = "timeout"
    ERROR = "error"
