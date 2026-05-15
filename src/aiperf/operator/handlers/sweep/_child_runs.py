# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-variation run-summary append helpers for the sweep rollup handler.

Carved out of ``child_rollup.py`` to keep that module under the 500-line
ergonomics ceiling. The rollup handler in ``child_rollup`` owns the
counts/currentChildRef/phase logic; this module owns the
``AIPerfSweep.status.runs[]`` summary-entry contract.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

logger = logging.getLogger(__name__)

__all__ = [
    "TERMINAL_CHILD_PHASES",
    "build_run_entry",
    "extract_summary_metrics",
    "append_run_entry",
]

# Lowercased terminal phases from the child's ``status.phase`` that
# trigger a runs[] summary append. Matches AIPerfJob's terminal-phase
# vocabulary; PartiallyFailed intentionally excluded (it is a parent
# rollup state, never a child phase).
TERMINAL_CHILD_PHASES = frozenset({"completed", "succeeded", "failed", "cancelled"})

# Safety threshold: stop appending to runs[] when the entry count would
# push the AIPerfSweep CR past this. Headroom under apiserver's 1 MiB
# limit (per preflight workload validator). 1500 x 600B ~ 900 KB.
_RUNS_SAFETY_THRESHOLD = 1500


def extract_summary_metrics(child_status: dict[str, Any]) -> dict[str, Any]:
    """Extract the slim metric set carried on AIPerfSweep.status.runs[i].metrics.

    Per spec: output_token_throughput, request_throughput, ttft.{p50,p95,p99},
    itl.{p50,p95,p99}, request_count, error_count. Bounded ~400-800 bytes.
    Pulled from child AIPerfJob.status.summary or .liveSummary if present.
    """
    summary = child_status.get("summary") or child_status.get("liveSummary") or {}
    out: dict[str, Any] = {}
    for key in (
        "output_token_throughput",
        "request_throughput",
        "request_count",
        "error_count",
    ):
        if key in summary:
            out[key] = summary[key]
    for stat_key in ("ttft", "itl"):
        if stat_key in summary and isinstance(summary[stat_key], dict):
            out[stat_key] = {
                p: summary[stat_key][p]
                for p in ("p50", "p95", "p99")
                if p in summary[stat_key]
            }
    return out


def build_run_entry(
    *,
    body: dict[str, Any],
    status: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    """Build the slim summary entry to append to ``status.runs[]``.

    ``body`` is the just-transitioned child AIPerfJob; ``status`` is its
    ``status`` subresource; ``name`` is the child's ``metadata.name``.
    Variation index/label/values are read from labels and annotations
    stamped at child-creation time by the sweep-controller's k8s_executor.
    """
    metadata = body.get("metadata") or {}
    labels = metadata.get("labels") or {}
    annotations = metadata.get("annotations") or {}
    try:
        index = int(labels.get("aiperf.nvidia.com/variation-index", "-1"))
    except (TypeError, ValueError):
        index = -1
    return {
        "index": index,
        "label": labels.get("aiperf.nvidia.com/variation-label", ""),
        "values": annotations.get("aiperf.nvidia.com/variation-values", ""),
        "phase": status.get("phase"),
        "childName": name,
        "startedAt": status.get("startTime"),
        "completedAt": status.get("completionTime"),
        "metrics": extract_summary_metrics(status),
    }


async def append_run_entry(
    namespace: str,
    sweep_name: str,
    entry: dict[str, Any],
    *,
    api: ApiClient,
) -> None:
    """Append ``entry`` to ``AIPerfSweep.status.runs`` via JSON-patch.

    Two-step: idempotently initialize ``status.runs = []`` (swallowing the
    409/422 from re-init when it already exists), then ``add`` the entry
    at ``/status/runs/-``.

    Truncation safety net: if the current ``runs[]`` length is at or above
    ``_RUNS_SAFETY_THRESHOLD``, skip the append and stamp
    ``status.runsTruncated`` instead. Keeps the AIPerfSweep CR comfortably
    under the apiserver 1 MiB limit even on huge sweeps; readers fetch the
    full run list from the operator results API.
    """
    from kubernetes_asyncio import client
    from kubernetes_asyncio.client.exceptions import ApiException

    custom_objects = client.CustomObjectsApi(api)
    try:
        await custom_objects.patch_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            namespace=namespace,
            name=sweep_name,
            body=[{"op": "add", "path": "/status/runs", "value": []}],
            _content_type="application/json-patch+json",
        )
    except ApiException as e:
        # 409/422 = path already exists; that's the steady-state case
        # after the very first append. 404 = parent CR was deleted; the
        # subsequent append will see the same 404 and be skipped there.
        if e.status not in (409, 422, 404):
            logger.warning(
                "runs[] init-patch failed for %s/%s: %s",
                namespace,
                sweep_name,
                e.reason,
            )

    current_runs_len, total_variations = await _read_runs_len_and_total(
        custom_objects, namespace, sweep_name
    )
    if current_runs_len >= _RUNS_SAFETY_THRESHOLD:
        await _stamp_runs_truncated(
            custom_objects,
            namespace,
            sweep_name,
            included=current_runs_len,
            total=total_variations or current_runs_len,
        )
        return

    try:
        await custom_objects.patch_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            namespace=namespace,
            name=sweep_name,
            body=[{"op": "add", "path": "/status/runs/-", "value": entry}],
            _content_type="application/json-patch+json",
        )
    except ApiException as e:
        if e.status == 404:
            return
        logger.warning(
            "runs[] append failed for %s/%s: %s",
            namespace,
            sweep_name,
            e.reason,
        )


async def _read_runs_len_and_total(
    custom_objects: Any,
    namespace: str,
    sweep_name: str,
) -> tuple[int, int]:
    """Return ``(len(status.runs), status.totalVariations)`` for the sweep.

    Best-effort: any GET failure returns ``(0, 0)`` so the caller falls
    through to the normal append path. The threshold check is a safety
    net, not a correctness gate.
    """
    from kubernetes_asyncio.client.exceptions import ApiException

    try:
        cr = await custom_objects.get_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            namespace=namespace,
            name=sweep_name,
        )
    except ApiException:
        return 0, 0
    except (ConnectionError, TimeoutError):
        return 0, 0
    status = (cr or {}).get("status") or {}
    runs = status.get("runs") or []
    total = status.get("totalVariations") or 0
    try:
        total_int = int(total)
    except (TypeError, ValueError):
        total_int = 0
    return len(runs), total_int


async def _stamp_runs_truncated(
    custom_objects: Any,
    namespace: str,
    sweep_name: str,
    *,
    included: int,
    total: int,
) -> None:
    """Stamp ``status.runsTruncated`` with ``{total, included, fetchURL}``.

    Best-effort merge-patch; logs and swallows non-404 errors so a stuck
    apiserver does not block the rollup handler. The fetchURL points at
    the operator results API's per-sweep children endpoint.
    """
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.operator.environment import OperatorEnvironment

    base_url = OperatorEnvironment.SERVICE.BASE_URL.rstrip("/")
    fetch_url = f"{base_url}/api/v1/sweeps/{namespace}/{sweep_name}/children"
    body = {
        "status": {
            "runsTruncated": {
                "total": total,
                "included": included,
                "fetchURL": fetch_url,
            }
        }
    }
    try:
        await custom_objects.patch_namespaced_custom_object_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            namespace=namespace,
            name=sweep_name,
            body=body,
            _content_type="application/merge-patch+json",
        )
    except ApiException as e:
        if e.status == 404:
            return
        logger.warning(
            "runsTruncated stamp failed for %s/%s: %s",
            namespace,
            sweep_name,
            e.reason,
        )
