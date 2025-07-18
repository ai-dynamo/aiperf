# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class ServiceRunType(CaseInsensitiveStrEnum):
    """The different ways the SystemController should run the component services."""

    MULTIPROCESSING = "process"
    """Run each service as a separate process.
    This is the default way for single-node deployments."""

    KUBERNETES = "k8s"
    """Run each service as a separate Kubernetes pod.
    This is the default way for multi-node deployments."""


class ServiceState(CaseInsensitiveStrEnum):
    """States a service can be in throughout its lifecycle."""

    UNKNOWN = "unknown"
    """The service is in an unknown state."""

    INITIALIZING = "initializing"
    """The service is currently initializing. This is a temporary state that should be
    followed by PENDING."""

    PENDING = "pending"
    """The service is pending configuration."""

    CONFIGURING = "configuring"
    """The service is currently configuring. This is a temporary state that should be
    followed by READY."""

    READY = "ready"
    """The service has been configured and is ready to be started."""

    STARTING = "starting"
    """The service is starting. This is a temporary state that should be followed
    by RUNNING."""

    RUNNING = "running"
    """The service is running."""

    STOPPING = "stopping"
    """The service is stopping. This is a temporary state that should be followed
    by STOPPED."""

    STOPPED = "stopped"
    """The service has been stopped."""

    SHUTDOWN = "shutdown"
    """The service has been shutdown."""

    ERROR = "error"
    """The service is currently in an error state."""


class ServiceType(CaseInsensitiveStrEnum):
    """Types of services in the AIPerf system.

    This is used to identify the service type when registering with the
    SystemController. It can also be used for tracking purposes if multiple
    instances of the same service type are running.
    """

    SYSTEM_CONTROLLER = "system_controller"
    DATASET_MANAGER = "dataset_manager"
    TIMING_MANAGER = "timing_manager"
    RECORDS_MANAGER = "records_manager"
    INFERENCE_RESULT_PARSER = "inference_result_parser"
    WORKER_MANAGER = "worker_manager"
    WORKER = "worker"

    # For testing purposes only
    TEST = "test_service"


class ServiceRegistrationStatus(CaseInsensitiveStrEnum):
    """Defines the various states a service can be in during registration with
    the SystemController."""

    UNREGISTERED = "unregistered"
    """The service is not registered with the SystemController. This is the
    initial state."""

    WAITING = "waiting"
    """The service is waiting for the SystemController to register it.
    This is a temporary state that should be followed by REGISTERED, TIMEOUT, or ERROR."""

    REGISTERED = "registered"
    """The service is registered with the SystemController."""

    TIMEOUT = "timeout"
    """The service registration timed out."""

    ERROR = "error"
    """The service registration failed."""
