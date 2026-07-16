# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the AIPerfSweep CRD.

AIPerfSweep is the parent CR that owns child AIPerfJob CRs and orchestrates
parameter sweeps and multi-run trials. The orchestration loop runs in a
dedicated sweep-controller pod, not in the kopf operator.

`ConvergenceConfig` is re-exported from `aiperf.config.multi_run` so
existing K8s-side callers keep importing from this module while the
canonical class lives with the rest of the v2 config types. The re-export
uses ``__getattr__`` to dodge the circular import: ``aiperf.config.benchmark``
imports ``FailurePolicy`` from this module, so importing
``aiperf.config.multi_run`` at module-load time would deadlock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig
from aiperf.config.resolution.plan import FailurePolicy

if TYPE_CHECKING:
    from aiperf.config.sweep.multi_run import ConvergenceConfig as ConvergenceConfig

__all__ = [
    "ConvergenceConfig",  # noqa: F822 — provided lazily via module __getattr__
    "FailurePolicy",
    "ObjectMetaPartial",
]


def __getattr__(name: str) -> Any:
    if name == "ConvergenceConfig":
        from aiperf.config.sweep.multi_run import ConvergenceConfig

        return ConvergenceConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ObjectMetaPartial(BaseConfig):
    """Subset of Kubernetes ObjectMeta safe to stamp onto child CRs.

    Only labels and annotations are merged into children; name/namespace/uid
    are managed by the controller, so accepting them here would silently lose
    user intent. extra='forbid' surfaces unknown keys at submit time.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Labels merged into every child AIPerfJob.",
    )
    annotations: dict[str, str] = Field(
        default_factory=dict,
        description="Annotations merged into every child AIPerfJob.",
    )


# ``FailurePolicy`` is re-exported from ``aiperf.config.resolution.plan`` so
# operator code and ``BenchmarkPlan.failure_policy`` share the same Pydantic
# class. Importing it at module top-level keeps the symbol available without
# a second class definition that would fail ``isinstance``/field-type checks.
