# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kopf entry point for isolated native-k8s/v1 reconciliation."""

from __future__ import annotations

from typing import Any

import kopf

from .contract import validate_envelope
from .reconciliation import build_jobset, submitted_status

GROUP = "aiperf.nvidia.com"
VERSION = "v1alpha1"
PLURAL = "aiperfjobs"


@kopf.on.create(GROUP, VERSION, PLURAL)
async def create_job(spec: dict[str, Any], **_: Any) -> dict[str, Any]:
    """Validate the submitted native envelope and return the JobSet projection."""
    envelope = validate_envelope(spec["envelope"])
    return {"status": submitted_status(envelope), "jobSet": build_jobset(envelope)}


def main() -> None:
    """Launch kopf without importing the legacy AIPerf Python distribution."""
    kopf.run([__name__])
