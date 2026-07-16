# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runner transport, workload, and provider selection models.

The selected ``aiperf`` distribution owns the registries behind these
identifiers.  Config v2 therefore validates only a normalized, non-empty ID
and the shape of its opaque configuration object.  It deliberately does not
copy a runner catalog into a Python enum.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import (
    AfterValidator,
    ConfigDict,
    Discriminator,
    Field,
    Tag,
)

from aiperf.config.base import BaseConfig
from aiperf.config.dynosim import DynosimTransportConfig

__all__ = [
    "AgenticProviderConfig",
    "DynosimOfflineTransport",
    "DynosimOnlineTransport",
    "DynosimTransportConfig",
    "GrpcTransport",
    "HttpTransport",
    "OpenTransport",
    "RunnerComponentId",
    "RunnerTransportConfig",
    "RunnerWorkloadConfig",
]

# Transport IDs Config v2 models with typed, inline, schema-visible fields. Any
# other ID routes to :class:`OpenTransport` and is passed through opaquely, so
# the open runner registry keeps owning arbitrary transport identities.
_BUILTIN_TRANSPORT_TYPES = frozenset({"http", "grpc", "dynosim_offline", "dynosim_online"})
_OPEN_TRANSPORT_TAG = "__open__"


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


# =============================================================================
# TRANSPORT SELECTION — inline discriminated union
# =============================================================================
#
# Transport is authored the same way as every other discriminated Config-v2 union
# (``dataset``, ``phases``, ``endpoint``): ``type`` plus the variant's own fields on
# the same object, never a nested ``{type, config}`` envelope. The
# :func:`aiperf.orchestrator.rust_wire` projector re-nests the selected variant into
# the runner's ``{type, config}`` wire frame, so the strict ``aiperf``
# contract is unchanged.


class HttpTransport(BaseConfig):
    """Native HTTP/SSE transport (wall clock; ``http://`` / ``https://`` URLs).

    ``type`` keeps a default so this is the zero-argument default transport when
    ``benchmark.transport`` is omitted, and so the discriminator falls back here.
    The other variants make ``type`` required — matching the ``phases`` / ``dataset``
    unions — so their discriminator is never dropped by ``exclude_defaults`` dumps.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["http"] = "http"


class GrpcTransport(BaseConfig):
    """Native gRPC transport (wall clock; ``grpc://`` / ``grpcs://`` URLs)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["grpc"]


class DynosimOfflineTransport(DynosimTransportConfig):
    """In-process Dynamo mocker replay on the deterministic virtual clock.

    Inherits the full typed ``dynosim`` field surface from
    :class:`DynosimTransportConfig`; the fields sit directly on the transport
    object, exactly like :class:`~aiperf.config.phases.PoissonPhase` lifts its own
    knobs onto the phase object.  The clock rides on the transport ID, so there is
    no ``replay_mode`` field.
    """

    type: Literal["dynosim_offline"]


class DynosimOnlineTransport(DynosimTransportConfig):
    """In-process Dynamo mocker replay driven under the wall clock."""

    type: Literal["dynosim_online"]


class OpenTransport(BaseConfig):
    """Fallback for a runner-owned transport ID Config v2 does not model.

    ``extra='allow'`` preserves the open-registry property: arbitrary
    factory-owned keys are captured inline and passed through opaquely, without
    Python interpreting them.
    """

    model_config = ConfigDict(extra="allow")

    type: RunnerComponentId


def _transport_discriminator(value: Any) -> str:
    """Route a builtin transport ID to its typed variant, else to the open one."""
    raw = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    tag = raw if isinstance(raw, str) and raw.strip() else "http"
    return tag if tag in _BUILTIN_TRANSPORT_TYPES else _OPEN_TRANSPORT_TAG


RunnerTransportConfig: TypeAlias = Annotated[
    Annotated[HttpTransport, Tag("http")]
    | Annotated[GrpcTransport, Tag("grpc")]
    | Annotated[DynosimOfflineTransport, Tag("dynosim_offline")]
    | Annotated[DynosimOnlineTransport, Tag("dynosim_online")]
    | Annotated[OpenTransport, Tag(_OPEN_TRANSPORT_TAG)],
    Discriminator(_transport_discriminator),
]
"""Orthogonal execution-transport selection for one native run.

Inline discriminated union on ``type``.  Built-in ``http`` / ``grpc`` /
``dynosim_offline`` / ``dynosim_online`` are typed and schema-visible; any other
ID is an :class:`OpenTransport` whose keys stay opaque, so the open runner
registry keeps owning arbitrary transport identities.  The clock rides on the
transport ID (``dynosim_offline`` = deterministic virtual clock, the other three
= wall clock).
"""


class AgenticProviderConfig(_NamedRunnerComponentConfig):
    """Open canonical agentic-provider selection owned by the evaluator worker."""


class RunnerWorkloadConfig(_NamedRunnerComponentConfig):
    """Optional explicit workload-factory selection for one native run.

    The open runner registry owns arbitrary workload IDs; the selected runner
    factory strictly validates the opaque, factory-owned authored configuration.
    """

    config: Annotated[
        dict[str, Any],
        Field(
            default_factory=dict,
            description="Factory-owned authored configuration; opaque to Python.",
        ),
    ]
