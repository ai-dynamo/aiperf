# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AIPerfJob lifecycle phase vocabulary and Kubernetes timestamp helpers.

These live in ``aiperf.kubernetes`` rather than ``aiperf.operator`` because both
layers need them: the operator stamps them onto CR status, while the CLI-side
watch pollers and diagnosis engine read them back. Keeping them here is what
makes ``aiperf.kubernetes`` importable without pulling in kopf and the operator's
handler tree.
"""

from __future__ import annotations

from datetime import UTC, datetime

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


def format_timestamp() -> str:
    """Generate a Kubernetes-compatible timestamp.

    Returns:
        ISO 8601 timestamp string with Z suffix.
    """
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_timestamp(ts: str) -> datetime:
    """Parse a Kubernetes timestamp string.

    Args:
        ts: ISO 8601 timestamp string.

    Returns:
        datetime object in UTC.
    """
    # Handle both 'Z' suffix and '+00:00'
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


class Phase(CaseInsensitiveStrEnum):
    """AIPerfJob lifecycle phases.

    Phases represent the high-level state of a job:
    - PENDING: Resources created, waiting for pods to start
    - QUEUED: JobSet suspended by Kueue, waiting for quota admission
    - INITIALIZING: Pods starting, services initializing
    - RUNNING: Job actively executing
    - COMPLETED: Job finished successfully
    - FAILED: Job failed due to error
    - CANCELLED: Job was cancelled by user
    """

    PENDING = "Pending"
    QUEUED = "Queued"
    INITIALIZING = "Initializing"
    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"

    @property
    def is_terminal(self) -> bool:
        """Return whether the job has reached a final lifecycle phase."""
        return self in (Phase.COMPLETED, Phase.FAILED, Phase.CANCELLED)


def as_phase(value: Phase | str | None) -> Phase:
    """Coerce a raw ``status.phase`` string into a real ``Phase`` member.

    ``status.get("phase")`` and ``StatusBuilder.get_phase()`` both yield plain
    ``str``. ``Enum.__hash__`` hashes by member *name*, not value, so
    ``"Running" in frozenset({Phase.RUNNING})`` is ``False`` even though
    ``"Running" == Phase.RUNNING`` is ``True``. Tuple membership (``in (...)``)
    uses ``==`` and works; set membership silently does not. Any phase headed
    for a set-membership test must therefore be a genuine member first.

    Unrecognized values fall back to ``Phase.PENDING`` so a future phase string
    written by a newer operator cannot crash an older monitor tick.
    """
    try:
        return Phase(value)
    except ValueError:
        return Phase.PENDING
