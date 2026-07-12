# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared stock provider adapter contracts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from aiperf.accuracy.evaluation.contracts import (
    EvaluationHostBinding,
    EvaluationPlan,
    EvaluationPlanRequest,
    ResolvedAsset,
    ScopedProxyBinding,
)
from aiperf.accuracy.evaluation.session import EvaluationSession


class EvaluationProviderAdapter(Protocol):
    """Replaceable Python implementation behind evaluator-worker protocol v2."""

    def plan_session(self, request: EvaluationPlanRequest) -> EvaluationPlan:
        """Validate dynamic provider semantics without external effects."""

    async def bind_assets(
        self,
        assets: Sequence[ResolvedAsset],
        proxy: ScopedProxyBinding | None,
        host_binding: EvaluationHostBinding,
        staging_root: Path,
    ) -> EvaluationSession:
        """Freeze provider templates after Rust resolves every effect."""


class ProviderCapabilityError(ValueError):
    """Selected stock task cannot execute through its declared host effects."""


class ProviderRuntimeError(RuntimeError):
    """Provider-owned runtime failed without classifying a case as wrong."""
