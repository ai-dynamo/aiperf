# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed outcomes for credit admission and DAG-child dispatch."""

from __future__ import annotations

from enum import StrEnum


class TurnAdmission(StrEnum):
    """Final admission decision after concurrency slots are acquired."""

    ADMIT = "admit"
    DEFER = "defer"
    REJECT = "reject"

    @classmethod
    def normalize(cls, result: TurnAdmission | bool) -> TurnAdmission:
        """Convert legacy Boolean admission callbacks to an explicit decision."""
        if isinstance(result, cls):
            return result
        return cls.ADMIT if result else cls.REJECT


class ChildDispatchResult(StrEnum):
    """Lifecycle disposition of a DAG-child dispatch attempt."""

    ISSUED = "issued"
    DEFERRED = "deferred"
    REJECTED = "rejected"

    @classmethod
    def normalize(
        cls, result: ChildDispatchResult | bool | None
    ) -> ChildDispatchResult:
        """Convert legacy issuer results without conflating deferral and refusal."""
        if isinstance(result, cls):
            return result
        return cls.ISSUED if result is True else cls.REJECTED

    @property
    def preserves_tracking(self) -> bool:
        """Whether the orchestrator must retain this child's dependency edges."""
        return self is not ChildDispatchResult.REJECTED
