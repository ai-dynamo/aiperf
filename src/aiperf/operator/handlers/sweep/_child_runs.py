# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-variation ``AIPerfSweep.status.runs[]`` append helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, NoReturn

import orjson

from aiperf.common.endpoint_credentials import redact_sweep_public_data

if TYPE_CHECKING:
    from kubernetes_asyncio.client import ApiClient

logger = logging.getLogger(__name__)

__all__ = [
    "TERMINAL_CHILD_PHASES",
    "build_run_entry",
    "extract_summary_metrics",
    "append_run_entry",
]

TERMINAL_CHILD_PHASES = frozenset({"completed", "succeeded", "failed", "cancelled"})
_RUNS_SAFETY_THRESHOLD = 1500
_STATUS_RUNS_MAX_BYTES = 350_000
_STATUS_VARIATION_VALUES_MAX_BYTES = 256


def _status_variation_values_truncated_payload(original_bytes: int) -> dict[str, Any]:
    return {
        "__aiperf_truncated__": True,
        "reason": "variation values exceeded status byte limit",
        "limitBytes": _STATUS_VARIATION_VALUES_MAX_BYTES,
        "originalBytes": original_bytes,
    }


def _bounded_status_variation_values(raw: str) -> str:
    raw = str(redact_sweep_public_data(raw, path="variation_values"))
    encoded = raw.encode()
    if len(encoded) <= _STATUS_VARIATION_VALUES_MAX_BYTES:
        return raw
    return orjson.dumps(
        _status_variation_values_truncated_payload(len(encoded))
    ).decode()


def extract_summary_metrics(child_status: dict[str, Any]) -> dict[str, Any]:
    """Extract the slim metric set carried on AIPerfSweep.status.runs[i].metrics."""
    summary = child_status.get("summary") or child_status.get("liveSummary") or {}
    out: dict[str, Any] = {}
    for key in (
        "output_token_throughput",
        "request_throughput",
        "request_count",
        "error_request_count",
        "error_rate",
        "total_requests",
    ):
        if key in summary:
            out[key] = summary[key]
    for stat_key in ("time_to_first_token", "inter_token_latency"):
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
    """Build the slim summary entry to append to ``status.runs[]``."""
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
        "values": _bounded_status_variation_values(
            annotations.get("aiperf.nvidia.com/variation-values", "")
        ),
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
    expected_uid: str | None = None,
    api: ApiClient,
) -> None:
    """Append ``entry`` to ``AIPerfSweep.status.runs`` via JSON-patch.

    Read first, initialize ``status.runs = []`` only when absent, then
    ``add`` the entry at ``/status/runs/-``. JSON Patch ``add`` to an
    existing object member replaces it, so repeated appends must not send
    the initializer once ``runs`` already exists.

    Truncation safety net: if the current ``runs[]`` length is at or above
    ``_RUNS_SAFETY_THRESHOLD``, skip the append and stamp
    ``status.runsTruncated`` instead. Keeps the AIPerfSweep CR comfortably
    under the apiserver 1 MiB limit even on huge sweeps; readers fetch the
    full run list from the operator results API.

    ``expected_uid`` fences the read/CAS/write cycle to the sweep incarnation
    referenced by the child ownerReference.
    """
    from kubernetes_asyncio import client

    custom_objects = client.CustomObjectsApi(api)
    max_attempts = 20
    for _attempt in range(max_attempts):
        (
            current_runs,
            total_variations,
            runs_present,
            runs_is_list,
            runs_truncated_present,
            resource_version,
        ) = await _read_runs_state(
            custom_objects,
            namespace,
            sweep_name,
            expected_uid=expected_uid,
        )
        if runs_truncated_present:
            return
        if runs_present and not runs_is_list:
            await _truncate_non_list_runs(
                custom_objects,
                namespace,
                sweep_name,
                total_variations,
                expected_uid=expected_uid,
            )
            return
        if not runs_present:
            init_result = await _ensure_runs_present(
                custom_objects,
                namespace,
                sweep_name,
                resource_version=resource_version,
                expected_uid=expected_uid,
            )
            if init_result == "retry":
                continue
            if init_result == "missing":
                return

        if _run_entry_already_present(current_runs, entry):
            return

        if await _truncate_if_over_budget(
            custom_objects,
            namespace,
            sweep_name,
            current_runs,
            entry,
            total_variations=total_variations,
            expected_uid=expected_uid,
        ):
            return

        append_result = await _append_run_entry_patch(
            custom_objects,
            namespace,
            sweep_name,
            entry,
            resource_version=resource_version,
            expected_uid=expected_uid,
        )
        if _append_result_needs_retry(namespace, sweep_name, append_result):
            continue
        return
    _raise_runs_retry(
        namespace,
        sweep_name,
        f"runs[] append raced for {max_attempts} resourceVersion attempts",
    )


async def _truncate_if_over_budget(
    custom_objects: Any,
    namespace: str,
    sweep_name: str,
    current_runs: list[dict[str, Any]],
    entry: dict[str, Any],
    *,
    total_variations: int,
    expected_uid: str | None,
) -> bool:
    current_runs_len = len(current_runs)
    if (
        current_runs_len < _RUNS_SAFETY_THRESHOLD
        and not _runs_payload_would_exceed_budget(current_runs, entry)
    ):
        return False
    await _stamp_runs_truncated(
        custom_objects,
        namespace,
        sweep_name,
        included=current_runs_len,
        total=total_variations or current_runs_len,
        expected_uid=expected_uid,
    )
    return True


async def _ensure_runs_present(
    custom_objects: Any,
    namespace: str,
    sweep_name: str,
    *,
    resource_version: str | None,
    expected_uid: str | None,
) -> Literal["initialized", "missing", "retry"]:
    if resource_version is None:
        _raise_runs_retry(
            namespace,
            sweep_name,
            "status.runs absent but resourceVersion unavailable",
        )
    return await _initialize_runs_if_absent(
        custom_objects,
        namespace,
        sweep_name,
        resource_version=resource_version,
        expected_uid=expected_uid,
    )


def _run_entry_already_present(
    current_runs: list[dict[str, Any]], entry: dict[str, Any]
) -> bool:
    child_name = entry.get("childName")
    if child_name:
        return any(run.get("childName") == child_name for run in current_runs)

    identity = (entry.get("index"), entry.get("label"), entry.get("values"))
    return any(
        (run.get("index"), run.get("label"), run.get("values")) == identity
        for run in current_runs
    )


def _append_result_needs_retry(
    namespace: str,
    sweep_name: str,
    result: Literal["appended", "missing", "retry", "failed"],
) -> bool:
    if result == "retry":
        return True
    if result == "failed":
        _raise_runs_retry(
            namespace,
            sweep_name,
            "runs[] append failed before entry was persisted",
        )
    return False


def _raise_runs_retry(namespace: str, sweep_name: str, reason: str) -> NoReturn:
    import kopf

    raise kopf.TemporaryError(
        f"retry AIPerfSweep {namespace}/{sweep_name} child run rollup: {reason}",
        delay=5,
    )


def _is_resource_version_retry(error: Any, resource_version: str | None) -> bool:
    """Return true when a JSON Patch CAS failure should re-read and retry."""
    if resource_version is None or error.status not in {409, 422}:
        return False
    if error.status == 409:
        return True
    text = " ".join(
        str(part).lower()
        for part in (getattr(error, "reason", ""), getattr(error, "body", ""))
    )
    return "resourceversion" in text or "resource version" in text


async def _truncate_non_list_runs(
    custom_objects: Any,
    namespace: str,
    sweep_name: str,
    total_variations: int,
    *,
    expected_uid: str | None,
) -> None:
    logger.warning(
        "runs[] has non-list value for %s/%s; refusing to replace it",
        namespace,
        sweep_name,
    )
    await _stamp_runs_truncated(
        custom_objects,
        namespace,
        sweep_name,
        included=0,
        total=total_variations,
        expected_uid=expected_uid,
    )


async def _append_run_entry_patch(
    custom_objects: Any,
    namespace: str,
    sweep_name: str,
    entry: dict[str, Any],
    *,
    resource_version: str | None,
    expected_uid: str | None,
) -> Literal["appended", "missing", "retry", "failed"]:
    """Append one run entry with a resourceVersion CAS guard."""
    from kubernetes_asyncio.client.exceptions import ApiException

    body = []
    if expected_uid is not None:
        body.append({"op": "test", "path": "/metadata/uid", "value": expected_uid})
    if resource_version is not None:
        body.append(
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": resource_version,
            }
        )
    body.append({"op": "add", "path": "/status/runs/-", "value": entry})
    try:
        await custom_objects.patch_namespaced_custom_object_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            namespace=namespace,
            name=sweep_name,
            body=body,
            _content_type="application/json-patch+json",
        )
    except ApiException as e:
        if e.status == 404:
            return "missing"
        if _is_resource_version_retry(e, resource_version):
            return "retry"
        logger.warning(
            "runs[] append failed for %s/%s: %s",
            namespace,
            sweep_name,
            e.reason,
        )
        return "failed"
    return "appended"


async def _initialize_runs_if_absent(
    custom_objects: Any,
    namespace: str,
    sweep_name: str,
    *,
    resource_version: str | None,
    expected_uid: str | None,
) -> Literal["initialized", "missing", "retry"]:
    """Initialize absent ``status.runs`` without replacing existing values."""
    from kubernetes_asyncio.client.exceptions import ApiException

    body = []
    if expected_uid is not None:
        body.append({"op": "test", "path": "/metadata/uid", "value": expected_uid})
    if resource_version is not None:
        body.append(
            {
                "op": "test",
                "path": "/metadata/resourceVersion",
                "value": resource_version,
            }
        )
    body.append({"op": "add", "path": "/status/runs", "value": []})
    try:
        await custom_objects.patch_namespaced_custom_object_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            namespace=namespace,
            name=sweep_name,
            body=body,
            _content_type="application/json-patch+json",
        )
    except ApiException as e:
        if e.status == 404:
            return "missing"
        if _is_resource_version_retry(e, resource_version):
            return "retry"
        logger.warning(
            "runs[] init-patch failed for %s/%s: %s",
            namespace,
            sweep_name,
            e.reason,
        )
        return "retry"
    return "initialized"


async def _read_runs_state(
    custom_objects: Any,
    namespace: str,
    sweep_name: str,
    *,
    expected_uid: str | None,
) -> tuple[list[dict[str, Any]], int, bool, bool, bool, str | None]:
    """Return current run-list state plus truncation and resourceVersion flags."""
    from kubernetes_asyncio.client.exceptions import ApiException

    try:
        cr = await custom_objects.get_namespaced_custom_object(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            namespace=namespace,
            name=sweep_name,
        )
    except ApiException as e:
        if e.status == 404:
            return [], 0, True, True, True, None
        _raise_runs_retry(
            namespace,
            sweep_name,
            f"apiserver rejected status read ({e.status}): {e.reason}",
        )
    except (ConnectionError, TimeoutError) as e:
        _raise_runs_retry(
            namespace,
            sweep_name,
            f"apiserver unreachable during status read: {e}",
        )
    resource_version = ((cr or {}).get("metadata") or {}).get("resourceVersion")
    current_uid = ((cr or {}).get("metadata") or {}).get("uid")
    if expected_uid is not None and current_uid != expected_uid:
        return [], 0, True, True, True, resource_version
    status = (cr or {}).get("status") or {}
    runs_present = "runs" in status
    raw_runs = status.get("runs")
    runs_is_list = isinstance(raw_runs, list)
    runs = raw_runs if runs_is_list else []
    runs_truncated_present = "runsTruncated" in status
    total = status.get("totalVariations") or 0
    try:
        total_int = int(total)
    except (TypeError, ValueError):
        total_int = 0
    return (
        runs,
        total_int,
        runs_present,
        runs_is_list,
        runs_truncated_present,
        resource_version,
    )


def _runs_payload_would_exceed_budget(
    current_runs: list[dict[str, Any]], entry: dict[str, Any]
) -> bool:
    return len(orjson.dumps({"runs": [*current_runs, entry]})) > _STATUS_RUNS_MAX_BYTES


async def _stamp_runs_truncated(
    custom_objects: Any,
    namespace: str,
    sweep_name: str,
    *,
    included: int,
    total: int,
    expected_uid: str | None = None,
) -> None:
    """Stamp ``status.runsTruncated`` with ``{total, included, fetchURL}``.

    Missing sweep CRs are ignored because the parent was deleted. Other
    apiserver failures raise ``kopf.TemporaryError`` so the rollup event is
    retried and the durable truncation marker is not silently skipped. The
    fetchURL points at the operator results API's per-sweep children endpoint.
    """
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.operator.environment import OperatorEnvironment

    base_url = OperatorEnvironment.SERVICE.BASE_URL.rstrip("/")
    fetch_url = f"{base_url}/api/v1/sweeps/{namespace}/{sweep_name}/children"
    value = {
        "total": total,
        "included": included,
        "fetchURL": fetch_url,
    }
    body: dict[str, Any] | list[dict[str, Any]]
    content_type: str
    if expected_uid is None:
        body = {"status": {"runsTruncated": value}}
        content_type = "application/merge-patch+json"
    else:
        body = [
            {"op": "test", "path": "/metadata/uid", "value": expected_uid},
            {"op": "add", "path": "/status/runsTruncated", "value": value},
        ]
        content_type = "application/json-patch+json"
    try:
        await custom_objects.patch_namespaced_custom_object_status(
            group="aiperf.nvidia.com",
            version="v1alpha1",
            plural="aiperfsweeps",
            namespace=namespace,
            name=sweep_name,
            body=body,
            _content_type=content_type,
        )
    except ApiException as e:
        if e.status == 404:
            return
        _raise_runs_retry(
            namespace,
            sweep_name,
            f"runsTruncated stamp failed ({e.status}): {e.reason}",
        )
    except (ConnectionError, TimeoutError) as e:
        _raise_runs_retry(
            namespace,
            sweep_name,
            f"apiserver unreachable during runsTruncated stamp: {e}",
        )
