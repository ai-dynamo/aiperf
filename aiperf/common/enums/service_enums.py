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
from aiperf.common.enums.base_enums import StrEnum


class ServiceRunType(StrEnum):
    """The different ways the SystemController should run the component services."""

    MULTIPROCESSING = "process"
    """Run each service as a separate process.
    This is the default way for single-node deployments."""

    KUBERNETES = "k8s"
    """Run each service as a separate Kubernetes pod.
    This is the default way for multi-node deployments."""


class ServiceState(StrEnum):
    """States a service can be in throughout its lifecycle."""

    UNKNOWN = "unknown"
    """The service state is unknown. Placeholder for services that have not yet
    initialized."""

    INITIALIZING = "initializing"
    """The service is currently initializing. This is a temporary state that should be
    followed by READY."""

    READY = "ready"
    """The service has initialized and is ready to be configured or started."""

    STARTING = "starting"
    """The service is starting. This is a temporary state that should be followed
    by RUNNING."""

    RUNNING = "running"
    """The service is running."""

    STOPPING = "stopping"
    """The service is stopping. This is a temporary state that should be followed
    by STOPPED."""

    STOPPED = "stopped"
    """The service is stopped."""

    ERROR = "error"
    """The service is currently in an error state."""


class ServiceType(StrEnum):
    """Types of services in the AIPerf system.

    This is used to identify the service type when registering with the
    SystemController. It can also be used for tracking purposes if multiple
    instances of the same service type are running.
    """

    SYSTEM_CONTROLLER = "system_controller"
    """The SystemController service."""

    DATASET_MANAGER = "dataset_manager"
    """The DatasetManager service."""

    TIMING_MANAGER = "timing_manager"
    """The TimingManager service."""

    RECORDS_MANAGER = "records_manager"
    """The RecordsManager service."""

    POST_PROCESSOR_MANAGER = "post_processor_manager"
    """The PostProcessorManager service."""

    WORKER_MANAGER = "worker_manager"
    """The WorkerManager service."""

    WORKER = "worker"
    """The Worker service."""

    TEST = "test_service"
    """Used in tests."""


class ServiceRegistrationStatus(StrEnum):
    """Defines the various states a service can be in during registration with
    the SystemController."""

    UNREGISTERED = "unregistered"
    """The service is not registered with the SystemController. This is the
    initial state."""

    WAITING = "waiting"
    """The service is waiting for the SystemController to register it.
    This is a temporary state that should be followed by REGISTERED."""

    REGISTERED = "registered"
    """The service is registered with the SystemController."""

    TIMEOUT = "timeout"
    """The service registration timed out."""

    ERROR = "error"
    """The service registration failed."""
