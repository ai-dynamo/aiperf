# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for named benchmark phase identity."""

from __future__ import annotations

from aiperf.common.enums import CreditPhase
from aiperf.common.types import PhaseKind

PhaseRuntimeKey = int | CreditPhase


def infer_legacy_phase_kind(
    name: object, kind: PhaseKind | None = None
) -> PhaseKind | None:
    """Infer phase kind from reserved legacy canonical names when omitted."""
    if kind is not None:
        return kind
    if name in {"warmup", "profiling"}:
        return name  # type: ignore[return-value]
    return None


def phase_runtime_key(
    phase: CreditPhase, phase_index: int | None = None
) -> PhaseRuntimeKey:
    """Return the runtime key used for per-phase maps and concurrency slots."""
    return phase_index if phase_index is not None else phase
