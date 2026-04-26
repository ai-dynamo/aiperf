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

TERMINAL_PHASES = frozenset({"Succeeded", "Failed", "Cancelled", "PartiallyFailed"})


async def cancel(
    *,
    body: dict[str, Any],
    spec: dict[str, Any],
    name: str,
    namespace: str,
    patch: kopf.Patch,
    **_: Any,
) -> None:
    """Mirror spec.cancel into status.conditions[Cancelling].

    On cancel=true: append (or replace) Cancelling=True condition.
    On cancel=false: clear any existing Cancelling condition (sticky-flag fix).
    Skips when the sweep has already reached a terminal phase — cancelling a
    finished sweep is a no-op visually.
    """
    cancelling = bool(spec.get("cancel"))
    status_block = body.get("status") or {}
    parent_phase = status_block.get("phase") or ""
    if parent_phase in TERMINAL_PHASES:
        return
    existing = status_block.get("conditions") or []
    new_conditions = [c for c in existing if c.get("type") != "Cancelling"]
    if cancelling:
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        new_conditions.append(
            {
                "type": "Cancelling",
                "status": "True",
                "reason": "UserRequested",
                "message": "spec.cancel set to true",
                "lastTransitionTime": now,
            }
        )
    elif len(new_conditions) == len(existing):
        # spec.cancel=false and no prior Cancelling condition: nothing to do.
        return
    patch.status["conditions"] = new_conditions
