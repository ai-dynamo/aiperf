# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""@kopf.on.update handler for AIPerfSweep spec.cancel.

The sweep-controller pod observes spec.cancel via its own poll; the
operator's job is to mirror the cancel signal into status.conditions
for kubectl observability.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import kopf

__all__ = ["cancel"]


async def cancel(
    *,
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
) -> None:
    """Mirror spec.cancel into status.conditions[Cancelling]."""
    cancelling = bool(spec.get("cancel"))
    if not cancelling:
        return
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    existing = (body.get("status") or {}).get("conditions") or []
    new_conditions = [c for c in existing if c.get("type") != "Cancelling"]
    new_conditions.append(
        {
            "type": "Cancelling",
            "status": "True",
            "reason": "UserRequested",
            "message": "spec.cancel set to true",
            "lastTransitionTime": now,
        }
    )
    patch.status["conditions"] = new_conditions
