# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kopf entry point for isolated native-k8s/v1 reconciliation."""

from __future__ import annotations

from typing import Any

import kopf
from kubernetes_asyncio import client
from kubernetes_asyncio.client.exceptions import ApiException

from .contract import ControllerEnvelope, validate_envelope
from .reconciliation import (
    build_jobset,
    submitted_status,
    validate_jobset_identity,
    validate_references,
)

GROUP = "aiperf.nvidia.com"
VERSION = "v1alpha1"
PLURAL = "aiperfjobs"


async def reconcile_job(
    envelope: ControllerEnvelope,
    jobsets: Any,
    metadata_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Create the exact immutable JobSet projection for one accepted envelope."""
    validate_references(envelope, metadata_by_name)
    jobset = build_jobset(envelope)
    try:
        await jobsets.create_namespaced_custom_object(
            group="jobset.x-k8s.io",
            version="v1alpha2",
            namespace=envelope.namespace,
            plural="jobsets",
            body=jobset,
        )
    except ApiException as error:
        if error.status != 409:
            raise
        existing = await jobsets.get_namespaced_custom_object(
            group="jobset.x-k8s.io",
            version="v1alpha2",
            namespace=envelope.namespace,
            plural="jobsets",
            name=envelope.job_id,
        )
        validate_jobset_identity(envelope, existing)
    return submitted_status(envelope)


async def _reference_metadata(
    envelope: ControllerEnvelope, secrets: Any
) -> dict[str, dict[str, Any]]:
    references = [
        *(
            role.bootstrap
            for role in envelope.roles
            if role.name != "cell" and role.bootstrap is not None
        ),
        *envelope.cell_bootstraps,
    ]
    metadata_by_name: dict[str, dict[str, Any]] = {}
    for reference in references:
        secret = await secrets.read_namespaced_secret(
            name=reference.secret_name,
            namespace=envelope.namespace,
        )
        metadata = secret.metadata
        metadata_by_name[reference.secret_name] = {
            "immutable": secret.immutable,
            "metadata": {
                "name": metadata.name,
                "namespace": metadata.namespace,
                "labels": metadata.labels or {},
                "annotations": metadata.annotations or {},
            },
        }
    return metadata_by_name


@kopf.on.create(GROUP, VERSION, PLURAL)
async def create_job(
    spec: dict[str, Any], name: str, namespace: str, **_: Any
) -> dict[str, Any]:
    """Validate a submitted envelope and create its immutable JobSet."""
    envelope = validate_envelope(spec["envelope"])
    if envelope.job_id != name or envelope.namespace != namespace:
        raise ValueError("AIPerfJob metadata does not match envelope identity")
    metadata_by_name = await _reference_metadata(envelope, client.CoreV1Api())
    status = await reconcile_job(envelope, client.CustomObjectsApi(), metadata_by_name)
    return {"status": status}


def main() -> None:
    """Launch kopf without importing the legacy AIPerf Python distribution."""
    kopf.run([__name__])
