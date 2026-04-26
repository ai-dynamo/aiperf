# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Status patches for AIPerfSweep fields owned by the sweep-controller pod.

The operator owns: phase, totalVariations, runEpoch, completedRuns, failedRuns,
runtimeRef, lastChildEvent, conditions[Progressing/Cancelling].
The sweep-controller owns: currentCell, aggregation, aggregateRef.

Each writer applies under merge-patch+json with a distinct ``field_manager``
metadata string. Merge-patch is the right primitive for these imperative
event-style writes (each call layers a new fact on top of the existing
state). Server-Side Apply was tried and reverted: SSA's relinquishment
semantics caused a single field manager re-applying to drop its own
previously-set fields between writer methods (e.g. ``aggregation_running``
would erase ``currentCell``). The disjoint top-level field invariant
between operator and controller writers means merge-patch is safe; SSA
would only have buyed us conflict detection that we don't need.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kubernetes_asyncio.client import CustomObjectsApi

__all__ = ["SWEEP_CONTROLLER_FIELD_MANAGER", "SweepStatusWriter"]

SWEEP_CONTROLLER_FIELD_MANAGER = "aiperf-sweep-controller"


class SweepStatusWriter:
    """Patches sweep-controller-owned fields on the AIPerfSweep CR."""

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
        # Force merge-patch content-type — kubernetes_asyncio defaults to
        # application/json-patch+json which expects a list of ops, not the dict
        # body we send here. The api_client kwarg name is `_content_type`.
        # `field_manager` is metadata only here (merge-patch does not enforce
        # SSA semantics); it shows up in `kubectl get ... -o yaml` so operators
        # can tell which writer touched the field last.
        await custom.patch_namespaced_custom_object_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            namespace=self.namespace,
            plural="aiperfsweeps",
            name=self.name,
            body=body,
            field_manager=SWEEP_CONTROLLER_FIELD_MANAGER,
            _content_type="application/merge-patch+json",
        )


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
