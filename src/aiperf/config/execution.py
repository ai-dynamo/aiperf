# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner backend, workload, and provider selection models.

The selected ``aiperf-runner`` distribution owns the registries behind these
identifiers.  Config v2 therefore validates only a normalized, non-empty ID
and the shape of its opaque configuration object.  It deliberately does not
copy a runner catalog into a Python enum.
"""

from __future__ import annotations

from typing import Annotated, Any, TypeAlias

from pydantic import AfterValidator, ConfigDict, Field

from aiperf.config.base import BaseConfig

__all__ = [
    "AgenticProviderConfig",
    "RunnerBackendConfig",
    "RunnerComponentId",
    "RunnerWorkloadConfig",
]


def _normalize_runner_component_id(value: str) -> str:
    """Strip presentation whitespace without interpreting registry identity."""
    normalized = value.strip()
    if not normalized:
        raise ValueError("runner component type must be a non-empty string")
    return normalized


RunnerComponentId: TypeAlias = Annotated[
    str,
    Field(min_length=1),
    AfterValidator(_normalize_runner_component_id),
]
"""Open identifier resolved by the exact selected runner distribution."""


class _NamedRunnerComponentConfig(BaseConfig):
    """Shared structural model for a trait-backed runner factory selection."""

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        RunnerComponentId,
        Field(description="Open factory ID resolved by the selected runner registry."),
    ]
    config: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description=(
                "Factory-owned authored configuration. Python preserves this object "
                "without interpreting implementation-specific keys."
            ),
        ),
    ]


class RunnerBackendConfig(_NamedRunnerComponentConfig):
    """Orthogonal execution-backend selection for one native run."""

    type: RunnerComponentId = "online_http"


class RunnerWorkloadConfig(_NamedRunnerComponentConfig):
    """Optional explicit workload-factory selection for one native run."""


class AgenticProviderConfig(_NamedRunnerComponentConfig):
    """Open canonical agentic-provider selection owned by the evaluator worker."""
