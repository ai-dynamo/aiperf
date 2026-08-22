# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kopf entry point for isolated native-k8s/v1 reconciliation."""

from __future__ import annotations

from typing import Any

import kopf
from kubernetes_asyncio import client

from .contract import ControllerEnvelope, validate_envelope
from .reconciliation import build_jobset, submitted_status

GROUP = "aiperf.nvidia.com"
VERSION = "v1alpha1"
PLURAL = "aiperfjobs"


async def reconcile_job(
    envelope: ControllerEnvelope,
    jobsets: Any,
) -> dict[str, Any]:
    """Create the exact immutable JobSet projection for one accepted envelope."""
    jobset = build_jobset(envelope)
    await jobsets.create_namespaced_custom_object(
        group="jobset.x-k8s.io",
        version="v1alpha2",
        namespace=envelope.namespace,
        plural="jobsets",
        body=jobset,
    )
    return submitted_status(envelope)


@kopf.on.create(GROUP, VERSION, PLURAL)
async def create_job(spec: dict[str, Any], **_: Any) -> dict[str, Any]:
    """Validate a submitted envelope and create its immutable JobSet."""
    envelope = validate_envelope(spec["envelope"])
    status = await reconcile_job(envelope, client.CustomObjectsApi())
    return {"status": status}


def main() -> None:
    """Launch kopf without importing the legacy AIPerf Python distribution."""
    kopf.run([__name__])
