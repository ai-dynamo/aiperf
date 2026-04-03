# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.types import ServiceTypeT
from aiperf.controller.multiprocess_service_manager import MultiProcessServiceManager


class DirectWorkerServiceManager(MultiProcessServiceManager):
    """Service manager for direct worker mode.

    Extends MultiProcessServiceManager but skips Worker spawning.
    The Worker is co-located inside TimingManager's process and
    receives credits via DirectCreditRouter (direct method calls).
    All other services are spawned normally as separate processes.
    """

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None:
        """Run a service, skipping Worker which is managed in-process."""
        from aiperf.plugin.enums import ServiceType

        if service_type == ServiceType.WORKER:
            self.info("Skipping Worker spawn — co-located in TimingManager process")
            return
        await super().run_service(service_type, num_replicas)
