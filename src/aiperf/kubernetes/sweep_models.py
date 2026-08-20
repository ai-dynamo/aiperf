# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for the AIPerfSweep CRD.

AIPerfSweep is the parent CR that owns child AIPerfJob CRs and orchestrates
parameter sweeps and multi-run trials. The orchestration loop runs in a
dedicated sweep-controller pod, not in the kopf operator.
"""

from __future__ import annotations

from pydantic import ConfigDict, Field

from aiperf.config.base import BaseConfig

__all__ = ["ObjectMetaPartial"]


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
