# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class LifecycleState(CaseInsensitiveStrEnum):
    """This is the various states a service can be in during its lifecycle."""

    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


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


class SystemState(CaseInsensitiveStrEnum):
    """State of the system as a whole.

    This is used to track the state of the system as a whole, and is used to
    determine what actions to take when a signal is received.
    """

    INITIALIZING = "initializing"
    """The system is initializing. This is the initial state."""

    CONFIGURING = "configuring"
    """The system is configuring services."""

    READY = "ready"
    """The system is ready to start profiling. This is a temporary state that should be
    followed by PROFILING."""

    PROFILING = "profiling"
    """The system is running a profiling run."""

    PROCESSING = "processing"
    """The system is processing results."""

    STOPPING = "stopping"
    """The system is stopping."""

    SHUTDOWN = "shutdown"
    """The system is shutting down. This is the final state."""


class WorkerStatus(CaseInsensitiveStrEnum):
    """The current status of a worker service.

    NOTE: The order of the statuses is important for the UI.
    """

    HEALTHY = "healthy"
    HIGH_LOAD = "high_load"
    ERROR = "error"
    IDLE = "idle"
    STALE = "stale"


class WorkerStartupState(CaseInsensitiveStrEnum):
    """The current startup lifecycle state of a worker service."""

    STARTING = "starting"
    WAITING_FOR_DATASET = "waiting_for_dataset"
    ROUTER_PROBING = "router_probing"
    READY = "ready"
    SHUTTING_DOWN = "shutting_down"
