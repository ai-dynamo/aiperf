# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class WorkerStartupState(CaseInsensitiveStrEnum):
    """The current startup lifecycle state of a worker service."""

    STARTING = "starting"
    """Worker is initializing and setting up resources."""

    WAITING_FOR_DATASET = "waiting_for_dataset"
    """Worker is waiting for dataset to be available."""

    ROUTER_PROBING = "router_probing"
    """Worker is probing the message router for readiness."""

    READY = "ready"
    """Worker has completed initialization and is ready to accept requests."""

    SHUTTING_DOWN = "shutting_down"
    """Worker is shutting down and cleaning up resources."""
