# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-Side Apply patches for AIPerfSweep status fields owned by the sweep-controller pod.

The operator owns: phase, totalVariations, runEpoch, completedRuns, failedRuns,
runtimeRef, lastChildEvent, conditions[Progressing/Cancelling].
The sweep-controller owns: currentCell, aggregation, aggregateRef,
conditions[Aggregating]. SSA with distinct field managers prevents conflicts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kubernetes_asyncio.client import CustomObjectsApi

__all__ = ["SWEEP_CONTROLLER_FIELD_MANAGER", "SweepStatusWriter"]

SWEEP_CONTROLLER_FIELD_MANAGER = "aiperf-sweep-controller"


class SweepStatusWriter:
    """Patches sweep-controller-owned fields on the AIPerfSweep CR via SSA."""

    def __init__(self, api: Any, *, name: str, namespace: str) -> None:
        self._api = api
        self.name = name
        self.namespace = namespace

    async def current_cell(
        self,
        *,
        variation_index: int,
        label: str,
        trial: int,
        converged: bool = False,
    ) -> None:
        await self._patch(
            {
                "status": {
                    "currentCell": {
                        "variationIndex": variation_index,
                        "label": label,
                        "trial": trial,
                        "converged": converged,
                    }
                }
            }
        )

    async def aggregation_running(self) -> None:
        await self._patch({"status": {"aggregation": {"phase": "Running"}}})

    async def aggregation_complete(
        self, *, aggregate_path: str, controller_host: str, port: int
    ) -> None:
        await self._patch(
            {
                "status": {
                    "aggregation": {
                        "phase": "Complete",
                        "completedAt": _now_iso(),
                        "error": "",
                    },
                    "aggregateRef": {
                        "resultsServerHost": controller_host,
                        "port": port,
                        "apiPath": aggregate_path,
                    },
                }
            }
        )

    async def aggregation_failed(self, *, error: str) -> None:
        await self._patch(
            {
                "status": {
                    "aggregation": {
                        "phase": "Failed",
                        "error": error,
                        "completedAt": _now_iso(),
                    }
                }
            }
        )

    async def _patch(self, body: dict[str, Any]) -> None:
        custom = CustomObjectsApi(self._api)
        await custom.patch_namespaced_custom_object_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=self.namespace,
            plural="aiperfsweeps",
            name=self.name,
            body=body,
        )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
