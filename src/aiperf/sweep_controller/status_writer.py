# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Status patches for AIPerfSweep fields owned by the sweep-controller pod.

The operator owns: totalVariations, runEpoch, completedRuns, failedRuns,
runtimeRef, lastChildEvent, conditions[Progressing/Cancelling], and the
non-terminal ``phase`` transitions (``Pending`` at create from kopf,
``Aggregating`` from the rollup once every child is terminal).

The sweep-controller owns: currentCell, aggregation, aggregateRef,
aggregate, and the **terminal** ``phase`` transitions (``Succeeded`` /
``Failed`` written from ``aggregation_complete`` after the final
exporters run).

``status.phase`` is therefore co-written by three managers. The rollup
serializes its phase write through a JSON-patch ``test`` op
(``handlers/sweep/child_rollup._conditional_phase_set``) so a concurrent
terminal write from this writer is never clobbered: the apiserver test
fails, the rollup's phase write is dropped, and the terminal phase
stands. The other top-level fields are disjoint between the two writers,
so plain merge-patch is safe for them.

Each writer applies under merge-patch+json with a distinct ``field_manager``
metadata string. Server-Side Apply was tried and reverted: SSA's
relinquishment semantics caused a single field manager re-applying to
drop its own previously-set fields between writer methods (e.g.
``aggregation_running`` would erase ``currentCell``).
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
        self,
        *,
        aggregate_path: str,
        controller_host: str,
        port: int,
        aggregate_doc: dict[str, Any] | None = None,
        terminal_phase: str | None = None,
    ) -> None:
        """Mark aggregation Complete and (optionally) inline the aggregate.

        The aggregate JSON files are small (~50 KB total) and the aggregator
        docstring already commits to the dual-backed model: "the operator
        reads from the CR while live and from the per-epoch directory once
        the sweep has finished and the controller pod is gone." Embedding
        ``aggregate_doc`` here closes the live half of that contract — without
        it, no operator handler observes the disk file and the parent CR
        never advances past ``Aggregating``.

        ``terminal_phase`` should be ``"Succeeded"`` or ``"Failed"`` (members
        of ``PARENT_TERMINAL_PHASES`` in ``child_rollup``) so the rollup
        handler does not clobber the transition on a subsequent child phase
        event. Pass ``None`` to leave ``status.phase`` untouched (e.g. tests).

        Top-level ``status.completionTime`` is also written (CRD-declared name);
        the TTL reaper in ``operator/handlers/sweep/lifecycle.py`` reads it to
        compute ``ttlSecondsAfterFinished``. Without it the reaper falls back
        to ``metadata.creationTimestamp`` and reaps mid-run.
        """
        completed_at = _now_iso()
        body: dict[str, Any] = {
            "status": {
                "aggregation": {
                    "phase": "Complete",
                    "completedAt": completed_at,
                    "error": "",
                },
                "aggregateRef": {
                    "resultsServerHost": controller_host,
                    "port": port,
                    "apiPath": aggregate_path,
                },
                "completionTime": completed_at,
            }
        }
        if aggregate_doc is not None:
            body["status"]["aggregate"] = aggregate_doc
        if terminal_phase is not None:
            body["status"]["phase"] = terminal_phase
        await self._patch(body)

    async def aggregation_failed(self, *, error: str) -> None:
        """Mark aggregation Failed and promote ``status.phase`` to ``Failed``.

        Without the top-level phase write, the parent CR's ``phase`` stays
        ``Aggregating`` forever after an aggregation exception (the rollup
        already advanced phase out of ``Running`` and refuses to clobber its
        own non-terminal write back to ``Failed``). The rollup's
        ``_conditional_phase_set`` skips writes when ``parent_phase`` is
        already in ``PARENT_TERMINAL_PHASES``, so this merge-patch is safe
        against a concurrent rollup tick.

        ``status.completionTime`` is also written so the TTL reaper measures
        retention from the failure timestamp, not creation.
        """
        completed_at = _now_iso()
        await self._patch(
            {
                "status": {
                    "aggregation": {
                        "phase": "Failed",
                        "error": error,
                        "completedAt": completed_at,
                    },
                    "phase": "Failed",
                    "completionTime": completed_at,
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
