# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner backend, workload, and provider selection models.

The selected ``aiperf-runner`` distribution owns the registries behind these
identifiers.  Config v2 therefore validates only a normalized, non-empty ID
and the shape of its opaque configuration object.  It deliberately does not
copy a runner catalog into a Python enum.
"""

from __future__ import annotations

from typing import Annotated, Any, Self, TypeAlias

from pydantic import AfterValidator, ConfigDict, Field, field_validator, model_validator

from aiperf.config.base import BaseConfig
from aiperf.config.dynosim import DynosimBackendConfig

__all__ = [
    "AgenticProviderConfig",
    "DynosimBackendConfig",
    "EvaluationProviderConfig",
    "EvaluationResourceConfig",
    "EvaluationRouteConfig",
    "EvaluationWorkloadConfig",
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
    """Orthogonal execution-backend selection for one native run.

    The open runner registry owns arbitrary backend IDs, whose ``config`` stays an
    opaque dict.  The built-in ``dynosim`` backend (in-process Dynamo mocker
    replay) has a documented structural contract so it can be authored with typed,
    validated, schema-visible fields; it is normalized back to a plain dict for the
    wire, mirroring the ``evaluation`` workload's precedent.
    """

    type: RunnerComponentId = "online_http"
    config: Annotated[
        dict[str, Any] | DynosimBackendConfig,
        Field(
            default_factory=dict,
            description=(
                "Factory-owned authored configuration. The built-in dynosim backend "
                "shape is documented explicitly; other open backend factories retain "
                "opaque objects."
            ),
        ),
    ]

    @model_validator(mode="after")
    def validate_builtin_dynosim_shape(self) -> Self:
        """Normalize the typed dynosim backend envelope when selected.

        Validates against :class:`DynosimBackendConfig`, then dumps back to a plain
        snake_case dict so the wire matches the runner's ``DynosimBackendSpec``.
        Only authored fields are emitted (``exclude_unset``); the runner fills every
        omitted field from its ``serde`` defaults.
        """
        if self.type != "dynosim":
            if isinstance(self.config, DynosimBackendConfig):
                raise ValueError(
                    "DynosimBackendConfig requires backend.type='dynosim'"
                )
            return self
        validated = (
            self.config
            if isinstance(self.config, DynosimBackendConfig)
            else DynosimBackendConfig.model_validate(self.config)
        )
        dumped = validated.model_dump(
            mode="json",
            by_alias=False,
            exclude_unset=True,
            exclude_none=True,
        )
        # Rust decodes required_features into a BTreeSet (order-independent), but the
        # wire bytes must be deterministic across runs, so emit a sorted list rather
        # than a nondeterministic set iteration order.
        if "required_features" in dumped:
            dumped["required_features"] = sorted(dumped["required_features"])
        self.config = dumped
        return self


class AgenticProviderConfig(_NamedRunnerComponentConfig):
    """Open canonical agentic-provider selection owned by the evaluator worker."""


class EvaluationProviderConfig(BaseConfig):
    """Registered evaluator process implementation selected by identity only.

    The selected runner factory owns the executable, module, argv, environment,
    working directory, isolation profile, and attested source/lock closure.
    Those coordinates are deliberately not fields in the authored model.
    """

    model_config = ConfigDict(extra="forbid")

    type: Annotated[
        RunnerComponentId,
        Field(description="Open provider ID resolved by the runner registry."),
    ]
    distribution: Annotated[
        RunnerComponentId,
        Field(
            description=(
                "Immutable provider distribution ID registered by the selected "
                "runner; never an executable or import target."
            )
        ),
    ]


class EvaluationRouteConfig(BaseConfig):
    """One logical evaluator service mapped to a Rust-owned endpoint profile."""

    model_config = ConfigDict(extra="forbid")

    model: Annotated[
        str,
        Field(min_length=1, description="Server-facing model identity."),
    ]
    endpoint_profile: Annotated[
        str,
        Field(min_length=1, description="Run-local endpoint profile reference."),
    ]
    purpose: Annotated[
        RunnerComponentId,
        Field(
            description=(
                "Provider-plan purpose that this logical service is allowed to serve. "
                "Auxiliary routes must set this explicitly."
            )
        ),
    ] = "primary"

    @field_validator("model", "endpoint_profile")
    @classmethod
    def validate_nonempty_identity(cls, value: str) -> str:
        """Reject ambiguous references without resolving either identity."""
        if not value.strip() or value != value.strip():
            raise ValueError("must be non-empty and contain no surrounding whitespace")
        return value


class EvaluationResourceConfig(_NamedRunnerComponentConfig):
    """One explicitly named Rust-hosted resource implementation."""


class EvaluationWorkloadConfig(BaseConfig):
    """Provider-neutral authored shape for the protocol-v2 evaluation workload.

    ``evaluation`` is intentionally opaque to Python.  The exact registered
    provider factory strictly validates it, using its fingerprinted Rust schema,
    before any evaluator process, asset resolver, or sandbox is started.
    """

    model_config = ConfigDict(extra="forbid")

    provider: EvaluationProviderConfig
    evaluation: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Opaque, provider-owned authored evaluation configuration.",
        ),
    ]
    routes: Annotated[
        dict[RunnerComponentId, EvaluationRouteConfig],
        Field(
            min_length=1,
            description="Logical model-service routes; values contain no URL or credential.",
        ),
    ]
    resources: Annotated[
        dict[RunnerComponentId, EvaluationResourceConfig],
        Field(
            default_factory=dict,
            description="Explicit Rust-hosted resource capability bindings.",
        ),
    ]
    unit_concurrency: Annotated[
        int,
        Field(gt=0, description="Maximum concurrently started evaluation units."),
    ] = 1


class RunnerWorkloadConfig(_NamedRunnerComponentConfig):
    """Optional explicit workload-factory selection for one native run.

    The open runner registry still owns arbitrary workload IDs.  The built-in
    ``evaluation`` workload has a product-level structural contract, however,
    because Python must keep launch coordinates out of authored configuration
    and must project logical routes without importing an evaluator package.
    """

    config: Annotated[
        dict[str, Any] | EvaluationWorkloadConfig,
        Field(
            default_factory=dict,
            description=(
                "Factory-owned authored configuration. The built-in evaluation "
                "shape is documented explicitly; other open workload factories "
                "retain opaque objects."
            ),
        ),
    ]

    @model_validator(mode="after")
    def validate_builtin_evaluation_shape(self) -> Self:
        """Normalize the provider-neutral evaluation envelope when selected."""
        if self.type != "evaluation":
            if isinstance(self.config, EvaluationWorkloadConfig):
                raise ValueError(
                    "EvaluationWorkloadConfig requires workload.type='evaluation'"
                )
            return self
        evaluated = EvaluationWorkloadConfig.model_validate(self.config)
        self.config = evaluated.model_dump(
            mode="json",
            by_alias=False,
            exclude_none=True,
        )
        return self
