# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the canonical Config-v2 benchmark plan for an AIPerfSweep CR."""

from __future__ import annotations

import copy
import hashlib
from typing import Any

from aiperf.common.endpoint_credentials import validate_kubernetes_sweep_credential_axes
from aiperf.config.loader import build_benchmark_plan
from aiperf.config.resolution.plan import BenchmarkPlan
from aiperf.config.sweep import (
    AdaptiveSearchSweep,
    LatinHypercubeSweep,
    SobolSweep,
)
from aiperf.kubernetes.spec_converter import validate_sweep_spec

__all__ = ["build_plan_from_sweep", "validate_sweep_spec"]


def _stable_sweep_seed(sweep_cr: dict[str, Any]) -> int:
    """Derive a restart-stable stochastic-sweep seed from apiserver identity."""
    metadata = sweep_cr.get("metadata") or {}
    uid = metadata.get("uid")
    if not isinstance(uid, str) or not uid:
        raise ValueError(
            "unseeded stochastic AIPerfSweep execution requires metadata.uid "
            "so variations remain stable across sweep-controller pod restarts"
        )
    return int.from_bytes(hashlib.sha256(uid.encode()).digest()[:4], "big")


def build_plan_from_sweep(sweep_cr: dict[str, Any]) -> BenchmarkPlan:
    """Construct a BenchmarkPlan from an AIPerfSweep CR.

    Args:
        sweep_cr: Raw AIPerfSweep dict (typically from kubernetes_asyncio read).

    Returns:
        BenchmarkPlan with one config per variation; trial count comes from
        ``spec.multiRun.numRuns`` (default 1). When convergence is active it
        early-stops within that ``numRuns`` cap.

    Raises:
        ValidationError: If the CR spec fails Pydantic validation.
    """
    spec = validate_sweep_spec(sweep_cr["spec"])
    validate_kubernetes_sweep_credential_axes(spec.sweep)
    if (
        isinstance(spec.sweep, (SobolSweep, LatinHypercubeSweep))
        and spec.sweep.seed is None
    ):
        seed = _stable_sweep_seed(sweep_cr)
        raw_envelope = copy.deepcopy(spec._raw_envelope)
        if raw_envelope is None or not isinstance(raw_envelope.get("sweep"), dict):
            raise RuntimeError("validated QMC sweep is missing its raw sweep envelope")
        raw_envelope["sweep"]["seed"] = seed
        spec = spec.model_copy(
            update={"sweep": spec.sweep.model_copy(update={"seed": seed})}
        )
        spec._raw_envelope = raw_envelope
    plan = build_benchmark_plan(spec)
    plan.failure_policy = spec.failure_policy
    if isinstance(plan.sweep, AdaptiveSearchSweep) and plan.sweep.random_seed is None:
        plan.sweep = plan.sweep.model_copy(
            update={"random_seed": _stable_sweep_seed(sweep_cr)}
        )
    return plan
