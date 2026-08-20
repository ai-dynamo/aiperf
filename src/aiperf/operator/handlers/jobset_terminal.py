# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Watch-driven JobSet terminal-condition handling.

JobSet terminal conditions cannot prove the controller has fully exported
artifacts, so AIPerfJob success remains controller-driven through its durable
results-ready handshake. Failed AIPerfJob JobSets dispatch directly into the
existing failure classifier. Sweep-controller failure atomically terminalizes
the owning AIPerfSweep.

The kopf decorator binding lives in ``operator/main.py``; this module
is decorator-free so it can be unit-tested without kopf.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp
import kopf

from aiperf.common.endpoint_credentials import redact_sweep_public_data
from aiperf.kubernetes.constants import AIPerfLabels, Annotations
from aiperf.kubernetes.cr_refs import (
    AIPERF_JOB_API_VERSION,
    AIPERF_JOB_KIND,
    AIPERF_SWEEP_API_VERSION,
    AIPERF_SWEEP_GROUP,
    AIPERF_SWEEP_KIND,
    AIPERF_SWEEP_PLURAL,
    AIPERF_SWEEP_VERSION,
)
from aiperf.kubernetes.phase import format_timestamp
from aiperf.operator.handlers.sweep.child_rollup import PARENT_TERMINAL_PHASES

logger = logging.getLogger(__name__)

JOBSET_TERMINAL_FIELD_MANAGER = "aiperf-operator-jobset-terminal"


def _has_true_condition(
    conditions: list[dict[str, Any]] | None, condition_type: str
) -> bool:
    """Return whether a well-formed condition is currently true."""
    return any(
        isinstance(condition, dict)
        and condition.get("type") == condition_type
        and condition.get("status") == "True"
        for condition in conditions or []
    )


def _has_completed_condition(conditions: list[dict[str, Any]] | None) -> bool:
    """Return True if any condition is ``type=Completed status=True``.

    Defensive against non-dict entries (None / strings / numbers) that can
    appear if a malformed JobSet status leaks through the apiserver — kopf
    delivers the conditions list as-is, so we cannot assume well-formedness.
    """
    return _has_true_condition(conditions, "Completed")


def _has_failed_condition(conditions: list[dict[str, Any]] | None) -> bool:
    """Return True if any condition is ``type=Failed status=True``."""
    return _has_true_condition(conditions, "Failed")


def _declared_parent_kind(jobset_body: dict[str, Any] | None) -> str | None:
    """Return the supported controller-owner kind declared by the event."""
    kinds = {
        ref.get("kind")
        for ref in ((jobset_body or {}).get("metadata") or {}).get(
            "ownerReferences", []
        )
        if isinstance(ref, dict)
        and ref.get("controller") is True
        and ref.get("kind") in {AIPERF_JOB_KIND, AIPERF_SWEEP_KIND}
    }
    return kinds.pop() if len(kinds) == 1 else None


async def _lookup_aiperfjob_body(
    namespace: str, jobset_name: str
) -> dict[str, Any] | None:
    """Fetch the parent AIPerfJob CR body. JobSet name pattern: ``aiperf-<aiperfjob-name>``.

    Sweep-owned JobSets resolve to a non-existent AIPerfJob CR (the parent
    there is an AIPerfSweep) and return None -- the handler then silently
    skips. A 404 from the apiserver returns None too.
    """
    from kubernetes_asyncio.client import CustomObjectsApi
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import (
        AIPERF_GROUP,
        AIPERF_PLURAL,
        AIPERF_VERSION,
    )

    if not jobset_name.startswith("aiperf-"):
        return None
    ajob_name = jobset_name.removeprefix("aiperf-")
    try:
        async with k8s_client() as api:
            custom = CustomObjectsApi(api)
            return await custom.get_namespaced_custom_object(
                group=AIPERF_GROUP,
                version=AIPERF_VERSION,
                namespace=namespace,
                plural=AIPERF_PLURAL,
                name=ajob_name,
            )
    except ApiException as e:
        if e.status == 404:
            return None
        raise kopf.TemporaryError(
            f"AIPerfJob owner lookup for {namespace}/{jobset_name} failed "
            f"({e.status}: {e.reason}); retrying",
            delay=5,
        ) from e
    except Exception as e:  # noqa: BLE001 - one-shot watch events must be retried rather than dropped
        raise kopf.TemporaryError(
            f"AIPerfJob owner lookup for {namespace}/{jobset_name} failed: "
            f"{e}; retrying",
            delay=5,
        ) from e


def _is_trusted_aiperf_jobset(
    *,
    jobset_body: dict[str, Any] | None,
    parent_body: dict[str, Any],
    jobset_name: str,
) -> bool:
    """Return True when the JobSet body proves AIPerfJob ownership."""
    metadata = (jobset_body or {}).get("metadata") or {}
    parent_metadata = parent_body.get("metadata") or {}
    parent_name = parent_metadata.get("name")
    parent_uid = parent_metadata.get("uid")
    if not isinstance(parent_name, str) or not isinstance(parent_uid, str):
        return False
    labels = metadata.get("labels") or {}
    if metadata.get("name") != jobset_name:
        return False
    if labels.get(AIPerfLabels.APP_KEY) != AIPerfLabels.APP_VALUE:
        return False
    if labels.get(AIPerfLabels.JOB_ID) != parent_name:
        return False
    owner_refs = metadata.get("ownerReferences") or []
    return any(
        isinstance(ref, dict)
        and ref.get("apiVersion") == AIPERF_JOB_API_VERSION
        and ref.get("kind") == "AIPerfJob"
        and ref.get("name") == parent_name
        and ref.get("uid") == parent_uid
        and ref.get("controller") is True
        for ref in owner_refs
    )


async def _set_benchmark_complete_annotation(
    namespace: str,
    aiperfjob_name: str,
    *,
    aiperfjob_uid: str,
    resource_version: str,
    annotations: dict[str, str],
) -> None:
    """Patch ``metadata.annotations[BENCHMARK_COMPLETE] = "true"`` on the AIPerfJob.

    Setting the annotation makes kopf dispatch ``on_benchmark_complete``,
    which is idempotent (it short-circuits if status.phase is terminal and
    ``try_claim_completion`` returns False if already claimed). Racing the
    controller pod (which also sets this annotation when done) is therefore
    safe -- whichever fires first wins.
    """
    from kubernetes_asyncio.client import CustomObjectsApi
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.client import k8s_client
    from aiperf.kubernetes.cr_refs import (
        AIPERF_GROUP,
        AIPERF_PLURAL,
        AIPERF_VERSION,
    )

    try:
        async with k8s_client() as api:
            custom = CustomObjectsApi(api)
            annotation_path = Annotations.BENCHMARK_COMPLETE.replace("~", "~0").replace(
                "/", "~1"
            )
            await custom.patch_namespaced_custom_object(
                group=AIPERF_GROUP,
                version=AIPERF_VERSION,
                namespace=namespace,
                plural=AIPERF_PLURAL,
                name=aiperfjob_name,
                body=[
                    {
                        "op": "test",
                        "path": "/metadata/uid",
                        "value": aiperfjob_uid,
                    },
                    {
                        "op": "test",
                        "path": "/metadata/resourceVersion",
                        "value": resource_version,
                    },
                    {
                        "op": "add",
                        "path": "/metadata/annotations",
                        "value": annotations,
                    },
                    {
                        "op": "add",
                        "path": f"/metadata/annotations/{annotation_path}",
                        "value": "true",
                    },
                ],
                _content_type="application/json-patch+json",
            )
    except ApiException as e:
        if e.status in (404, 409, 422):
            return
        logger.warning(
            "Failed to set benchmark-complete annotation on AIPerfJob %s/%s: %s",
            namespace,
            aiperfjob_name,
            e,
        )


def _trusted_sweep_owner(
    jobset_body: dict[str, Any] | None, jobset_name: str
) -> tuple[str, str] | None:
    """Return the exact owning AIPerfSweep name and UID for a trusted JobSet."""
    metadata = (jobset_body or {}).get("metadata") or {}
    if metadata.get("name") != jobset_name:
        return None
    for ref in metadata.get("ownerReferences") or []:
        if not isinstance(ref, dict):
            continue
        name = ref.get("name")
        uid = ref.get("uid")
        if (
            ref.get("apiVersion") == AIPERF_SWEEP_API_VERSION
            and ref.get("kind") == "AIPerfSweep"
            and ref.get("controller") is True
            and isinstance(name, str)
            and isinstance(uid, str)
            and jobset_name == f"aiperf-{name}"
        ):
            return name, uid
    return None


async def _lookup_aiperfsweep_body(
    namespace: str, sweep_name: str
) -> dict[str, Any] | None:
    """Fetch the exact parent sweep, returning None only when it is gone."""
    from kubernetes_asyncio.client import CustomObjectsApi
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.client import k8s_client

    try:
        async with k8s_client() as api:
            return await CustomObjectsApi(api).get_namespaced_custom_object(
                group=AIPERF_SWEEP_GROUP,
                version=AIPERF_SWEEP_VERSION,
                namespace=namespace,
                plural=AIPERF_SWEEP_PLURAL,
                name=sweep_name,
            )
    except ApiException as exc:
        if exc.status == 404:
            return None
        raise kopf.TemporaryError(
            f"Failed to read AIPerfSweep {namespace}/{sweep_name}: {exc}", delay=5
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"Failed to read AIPerfSweep {namespace}/{sweep_name}: {exc}", delay=5
        ) from exc


def _jobset_failure_message(
    jobset_name: str, conditions: list[dict[str, Any]] | None
) -> str:
    """Build a bounded, redacted failure message from the JobSet condition."""
    failed = next(
        (
            condition
            for condition in conditions or []
            if isinstance(condition, dict)
            and condition.get("type") == "Failed"
            and condition.get("status") == "True"
        ),
        {},
    )
    detail = ": ".join(
        str(value) for value in (failed.get("reason"), failed.get("message")) if value
    )
    message = f"Sweep controller JobSet {jobset_name} failed"
    if detail:
        message = f"{message}: {detail}"
    return str(redact_sweep_public_data(message))[:32768]


def _sweep_failure_status(error: str) -> dict[str, Any]:
    """Return the terminal status shape owned by ``aggregation_failed``."""
    completed_at = format_timestamp()
    return {
        "aggregation": {
            "phase": "Failed",
            "error": error,
            "completedAt": completed_at,
        },
        "phase": "Failed",
        "completionTime": completed_at,
        "completedAt": completed_at,
        "resultsAvailable": False,
    }


async def _patch_sweep_controller_failure(
    *,
    namespace: str,
    sweep_name: str,
    sweep_uid: str,
    parent_body: dict[str, Any],
    error: str,
) -> None:
    """UID/resourceVersion-fenced terminal failure patch for one parent sweep."""
    from kubernetes_asyncio.client import CustomObjectsApi
    from kubernetes_asyncio.client.exceptions import ApiException

    from aiperf.kubernetes.client import k8s_client

    metadata = parent_body.get("metadata") or {}
    if metadata.get("name") != sweep_name or metadata.get("uid") != sweep_uid:
        return
    resource_version = metadata.get("resourceVersion")
    if not isinstance(resource_version, str):
        raise kopf.TemporaryError(
            f"AIPerfSweep {namespace}/{sweep_name} has no resourceVersion", delay=5
        )
    status = parent_body.get("status")
    current_phase = status.get("phase") if isinstance(status, dict) else None
    if current_phase in PARENT_TERMINAL_PHASES:
        return

    terminal_status = _sweep_failure_status(error)
    patch: list[dict[str, Any]] = [
        {"op": "test", "path": "/metadata/uid", "value": sweep_uid},
        {
            "op": "test",
            "path": "/metadata/resourceVersion",
            "value": resource_version,
        },
    ]
    if isinstance(status, dict):
        if current_phase is not None:
            patch.append(
                {"op": "test", "path": "/status/phase", "value": current_phase}
            )
        patch.extend(
            {"op": "add", "path": f"/status/{key}", "value": value}
            for key, value in terminal_status.items()
        )
    else:
        patch.append({"op": "add", "path": "/status", "value": terminal_status})

    try:
        async with k8s_client() as api:
            await CustomObjectsApi(api).patch_namespaced_custom_object_status(
                group=AIPERF_SWEEP_GROUP,
                version=AIPERF_SWEEP_VERSION,
                namespace=namespace,
                plural=AIPERF_SWEEP_PLURAL,
                name=sweep_name,
                body=patch,
                field_manager=JOBSET_TERMINAL_FIELD_MANAGER,
                _content_type="application/json-patch+json",
            )
    except ApiException as exc:
        if exc.status == 404:
            return
        raise kopf.TemporaryError(
            f"Failed to terminalize AIPerfSweep {namespace}/{sweep_name}: {exc}",
            delay=5,
        ) from exc
    except (aiohttp.ClientError, ConnectionError, TimeoutError) as exc:
        raise kopf.TemporaryError(
            f"Failed to terminalize AIPerfSweep {namespace}/{sweep_name}: {exc}",
            delay=5,
        ) from exc


async def _handle_aiperfjob_completion(
    *, namespace: str, jobset_name: str, jobset_body: dict[str, Any] | None
) -> None:
    """Preserve the existing AIPerfJob terminal-success shortcut."""
    body = await _lookup_aiperfjob_body(namespace, jobset_name)
    if body is None:
        return
    if not _is_trusted_aiperf_jobset(
        jobset_body=jobset_body,
        parent_body=body,
        jobset_name=jobset_name,
    ):
        return
    metadata = body.get("metadata") or {}
    existing = metadata.get("annotations") or {}
    if existing.get(Annotations.BENCHMARK_COMPLETE) == "true":
        return
    parent_uid = metadata.get("uid")
    resource_version = metadata.get("resourceVersion")
    if not isinstance(parent_uid, str) or not isinstance(resource_version, str):
        return
    await _set_benchmark_complete_annotation(
        namespace,
        jobset_name.removeprefix("aiperf-"),
        aiperfjob_uid=parent_uid,
        resource_version=resource_version,
        annotations=dict(existing),
    )


async def _handle_sweep_controller_failure(
    *,
    namespace: str,
    jobset_name: str,
    jobset_body: dict[str, Any] | None,
    conditions: list[dict[str, Any]] | None,
) -> None:
    """Terminalize the exact owning sweep after its controller JobSet fails."""
    owner = _trusted_sweep_owner(jobset_body, jobset_name)
    if owner is None:
        return
    sweep_name, sweep_uid = owner
    parent_body = await _lookup_aiperfsweep_body(namespace, sweep_name)
    if parent_body is None:
        return
    await _patch_sweep_controller_failure(
        namespace=namespace,
        sweep_name=sweep_name,
        sweep_uid=sweep_uid,
        parent_body=parent_body,
        error=_jobset_failure_message(jobset_name, conditions),
    )


async def _handle_aiperfjob_failure(
    *,
    namespace: str,
    jobset_name: str,
    jobset_body: dict[str, Any] | None,
    conditions: list[dict[str, Any]] | None,
) -> bool:
    """Dispatch an exact AIPerfJob-owned JobSet failure to monitor recovery."""
    parent = await _lookup_aiperfjob_body(namespace, jobset_name)
    if parent is None or not _is_trusted_aiperf_jobset(
        jobset_body=jobset_body,
        parent_body=parent,
        jobset_name=jobset_name,
    ):
        return False
    metadata = parent.get("metadata") or {}
    parent_name = metadata.get("name")
    if metadata.get("namespace") != namespace or not isinstance(parent_name, str):
        return False

    event_body = jobset_body or {}
    if not _has_failed_condition((event_body.get("status") or {}).get("conditions")):
        event_body = dict(event_body)
        event_status = dict(event_body.get("status") or {})
        event_status["conditions"] = conditions or []
        event_body["status"] = event_status

    from aiperf.operator.handlers import monitor

    await monitor.handle_jobset_failure_event(
        body=parent,
        jobset_body=event_body,
        namespace=namespace,
        name=parent_name,
    )
    return True


async def handle_jobset_progress(
    *,
    namespace: str,
    jobset_name: str,
    jobset_body: dict[str, Any],
) -> None:
    """Dispatch readiness from an exact AIPerfJob-owned JobSet."""
    if _declared_parent_kind(jobset_body) != AIPERF_JOB_KIND:
        return
    parent = await _lookup_aiperfjob_body(namespace, jobset_name)
    if parent is None or not _is_trusted_aiperf_jobset(
        jobset_body=jobset_body,
        parent_body=parent,
        jobset_name=jobset_name,
    ):
        return
    metadata = parent.get("metadata") or {}
    parent_name = metadata.get("name")
    if metadata.get("namespace") != namespace or not isinstance(parent_name, str):
        return

    from aiperf.operator.handlers import monitor

    await monitor.handle_jobset_progress_event(
        body=parent,
        jobset_body=jobset_body,
        namespace=namespace,
        name=parent_name,
    )


async def handle_jobset_conditions(
    *,
    old: list[dict[str, Any]] | None,
    new: list[dict[str, Any]] | None,
    namespace: str,
    jobset_name: str,
    jobset_body: dict[str, Any] | None = None,
) -> None:
    """React to sweep-controller failure transitions.

    A JobSet ``Completed`` condition proves that its Jobs exited, not that the
    controller has finished writing and exposing its result artifacts. The
    controller alone emits the benchmark-complete annotation after its durable
    results-ready handshake. A new Failed condition dispatches directly to the
    existing AIPerfJob classifier or terminalizes an exact AIPerfSweep parent.
    """
    if not _has_failed_condition(old) and _has_failed_condition(new):
        parent_kind = _declared_parent_kind(jobset_body)
        if parent_kind == AIPERF_SWEEP_KIND:
            await _handle_sweep_controller_failure(
                namespace=namespace,
                jobset_name=jobset_name,
                jobset_body=jobset_body,
                conditions=new,
            )
            return
        if parent_kind == AIPERF_JOB_KIND:
            await _handle_aiperfjob_failure(
                namespace=namespace,
                jobset_name=jobset_name,
                jobset_body=jobset_body,
                conditions=new,
            )
