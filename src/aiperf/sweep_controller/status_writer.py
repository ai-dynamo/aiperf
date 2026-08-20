# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Status patches for AIPerfSweep fields owned by the sweep-controller pod.

The operator owns: totalVariations, runEpoch, runtimeRef, lastChildEvent,
conditions[Progressing/Cancelling], and the non-terminal ``phase``
transitions (``Pending`` at create from kopf, ``Aggregating`` from the
rollup once every child is terminal). The rollup also maintains live
``completedRuns``, ``failedRuns``, and ``runStates`` during the run;
the sweep-controller overwrites these with authoritative terminal values
in ``aggregation_complete`` (see below).

The sweep-controller owns: currentCell, aggregation, aggregateRef,
aggregate, and the **terminal** ``phase`` transitions (``Succeeded`` /
``Failed`` written from ``aggregation_complete`` after the final
exporters run). At terminal time it also overwrites ``completedRuns``,
``failedRuns``, and ``runStates`` with the authoritative values from the
on-disk parent aggregate, so these fields are correct even when the
kopf rollup handler did not fire for every child (e.g. fast clusters
where children terminate before the field-watch event is processed).

``status.phase`` is therefore co-written by three managers. The rollup
serializes its phase write through a JSON-patch ``test`` op
(``handlers/sweep/child_rollup._conditional_phase_set``) so a concurrent
terminal write from this writer is never clobbered: the apiserver test
fails, the rollup's phase write is dropped, and the terminal phase
stands. Every controller-pod write first tests the immutable parent UID, so a
pod left behind by delete/recreate cannot patch the replacement sweep.

Each writer applies with a distinct ``field_manager`` metadata string. The
controller uses JSON Patch when a parent UID is available and retains the
merge-patch path for compatibility with UID-less callers. Server-Side Apply
was tried and reverted: SSA's
relinquishment semantics caused a single field manager re-applying to
drop its own previously-set fields between writer methods (e.g.
``aggregation_running`` would erase ``currentCell``).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from kubernetes_asyncio.client import CustomObjectsApi

from aiperf.common.endpoint_credentials import (
    redact_sweep_display_label,
    redact_sweep_public_data,
)

__all__ = ["SWEEP_CONTROLLER_FIELD_MANAGER", "SweepStatusWriter"]

SWEEP_CONTROLLER_FIELD_MANAGER = "aiperf-sweep-controller"


class SweepStatusWriter:
    """Patch controller-owned fields, fenced to one AIPerfSweep UID."""

    def __init__(
        self,
        api: Any,
        *,
        name: str,
        namespace: str,
        uid: str | None = None,
    ) -> None:
        self._api = api
        self.name = name
        self.namespace = namespace
        self.uid = uid

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
                        "label": redact_sweep_display_label(label),
                        "trial": trial,
                        "converged": converged,
                    }
                }
            }
        )

    async def aggregation_running(self) -> None:
        await self._patch({"status": {"aggregation": {"phase": "Running"}}})

    async def partial_children(
        self,
        *,
        sweep_run_epoch: str | None,
        children: list[dict[str, Any]],
    ) -> None:
        """Patch ``status.aggregate.children`` with the children seen so far.

        The terminal ``aggregation_complete`` writer embeds the full,
        post-aggregation manifest under ``status.aggregate.children``;
        consumers that read mid-run (the SweepDetail page's
        live-variations rollup, ``aiperf kube list``, third-party watch
        loops) used to see an empty manifest until then. This writer
        emits an incremental snapshot after each terminal child so the
        same shape is available throughout the run — the live rollup
        can render variation pills filling up trial-by-trial.

        Each entry mirrors the ``ChildrenManifestEntry`` schema (snake_case
        keys, matching the disk envelope ``children.json`` writes):
        ``{namespace, name, variation_index, variation_label, trial_index,
        child_run_epoch}``. The terminal writer overwrites this same path
        with the full doc, so a partial snapshot is never load-bearing
        for the post-aggregation read path.
        """
        body = {
            "sweep_run_epoch": str(sweep_run_epoch or ""),
            "children": redact_sweep_public_data(children),
        }
        await self._patch({"status": {"aggregate": {"children": body}}})

    async def parent_running(self) -> None:
        """Promote top-level ``status.phase`` to ``Running`` once child execution begins.

        The CRD enum (``crd-aiperfsweep.yaml``) declares ``Running`` as a phase,
        but no other writer set it — parents jumped straight ``Pending →
        Aggregating``. The rollup's ``_conditional_phase_set`` JSON-patch
        ``test`` op compares against ``Pending``, so this writer uses the same
        atomic primitive: if a peer (the rollup itself) already advanced phase,
        the apiserver returns 422 and we silently skip.

        Called once from sweep-controller ``main.py`` before the orchestrator
        loop begins; idempotent on pod restart (test op fails if already
        ``Running`` / a later phase).
        """
        import aiohttp
        from kubernetes_asyncio.client import ApiException

        custom = CustomObjectsApi(self._api)
        try:
            body: list[dict[str, Any]] = []
            if self.uid is not None:
                body.append({"op": "test", "path": "/metadata/uid", "value": self.uid})
            body.extend(
                [
                    {"op": "test", "path": "/status/phase", "value": "Pending"},
                    {"op": "replace", "path": "/status/phase", "value": "Running"},
                ]
            )
            await custom.patch_namespaced_custom_object_status(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                plural="aiperfsweeps",
                namespace=self.namespace,
                name=self.name,
                body=body,
                field_manager=SWEEP_CONTROLLER_FIELD_MANAGER,
                _content_type="application/json-patch+json",
            )
        except ApiException as e:
            # 422 = test op failed (peer wrote phase already); 404 = CR gone.
            # Both are silent no-ops.
            if e.status in (404, 422):
                return
            raise
        except (aiohttp.ClientError, ConnectionError, TimeoutError):
            # Transient apiserver failure — skip; the next status write
            # (currentCell, aggregation_*) will surface a real error.
            return

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
                "completedAt": completed_at,
                # The inline document makes live inspection possible, but the
                # only durable copy is created by the operator's sidecar
                # harvest. The operator flips this true together with its
                # PVC-backed aggregateRef after every advertised file lands.
                "resultsAvailable": False,
            }
        }
        if aggregate_doc is not None:
            body["status"]["aggregate"] = redact_sweep_public_data(aggregate_doc)
            # Overwrite the rollup-owned run-count fields with authoritative
            # terminal values. The kopf rollup handler fires on each child phase
            # change and is the primary live-update path, but it may miss events
            # on fast clusters where children terminate before field-watch
            # processing catches up. By the time aggregation_complete is called
            # every child is terminal, so these values are final.
            parent = (
                aggregate_doc.get("parent") if isinstance(aggregate_doc, dict) else None
            )
            if isinstance(parent, dict):
                for key in ("completedRuns", "failedRuns", "runStates"):
                    if key in parent:
                        body["status"][key] = parent[key]
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
                        "error": redact_sweep_public_data(error),
                        "completedAt": completed_at,
                    },
                    "phase": "Failed",
                    "completionTime": completed_at,
                    "completedAt": completed_at,
                    "resultsAvailable": False,
                }
            }
        )

    async def _patch(self, body: dict[str, Any]) -> None:
        custom = CustomObjectsApi(self._api)
        if self.uid is not None:
            patch: list[dict[str, Any]] = [
                {"op": "test", "path": "/metadata/uid", "value": self.uid}
            ]
            for key, value in (body.get("status") or {}).items():
                escaped_key = key.replace("~", "~0").replace("/", "~1")
                patch.append(
                    {"op": "add", "path": f"/status/{escaped_key}", "value": value}
                )
            await custom.patch_namespaced_custom_object_status(
                group="aiperf.nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="aiperfsweeps",
                name=self.name,
                body=patch,
                field_manager=SWEEP_CONTROLLER_FIELD_MANAGER,
                _content_type="application/json-patch+json",
            )
            return
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
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
