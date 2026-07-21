# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-routing transforms: per-session identity on the wire.

One plugin category unifies every mechanism that tells an external router
which session a request belongs to, whether header-based (SGLang Model
Gateway routing keys, Dynamo session headers, generic session-ID headers) or
body-based (Dynamo ``nvext.session_control``). The selected plugin is
instantiated once per worker by ``InferenceClient`` and invoked at the
request-serialization chokepoint.

Contracts:
- Options instances are canonicalized at config resolution; ``self.options``
  is the plugin's own typed model.
- ``transform_body`` must NEVER mutate its input (the structured path
  includes cached ``Turn.raw_payload`` dicts shared with the dataset).
- ``on_session_end`` fires strictly AFTER the session's last worker-side
  activity, on every terminal path (final turn, cancellation, terminal
  context overflow, cancel-before-start). It MUST be idempotent.
- Stateful plugins key instance state on ``ctx.x_correlation_id`` ONLY:
  a session tree deliberately spans workers, so tree-keyed worker state
  fragments. Tree-scoped behavior uses the stateless per-request facts
  (``root_correlation_id`` + ``is_tree_final``) instead.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import ConfigDict, Field

from aiperf.common.models import AIPerfBaseModel


class EmptyRoutingOptions(AIPerfBaseModel):
    """Options model for parameterless plugins; rejects every opt key."""

    model_config = ConfigDict(extra="forbid")


OptionsT = TypeVar("OptionsT", bound=AIPerfBaseModel)


@dataclass(slots=True, frozen=True)
class RoutingContext:
    """Per-request identity facts handed to a routing plugin.

    Field naming mirrors ``RequestInfo`` verbatim. ``is_parent_final`` /
    ``is_tree_final`` are stamped issuer-side from ``SessionTreeRegistry``
    state and are conservative: ``is_tree_final`` is False whenever
    indeterminate, ``is_parent_final`` is None for roots or when unknown.
    """

    x_correlation_id: str
    """This session's stable key (same on every turn)."""
    parent_correlation_id: str | None
    """Immediate parent session's key; None for root sessions."""
    root_correlation_id: str | None
    """Session-tree root key, verbatim from RequestInfo (never None on the
    dispatch path: the worker passes ``credit.effective_root_correlation_id``)."""
    is_final_turn: bool
    """True when this is the current session's last request."""
    is_parent_final: bool | None
    """True when the parent session had already returned its final turn
    at credit-issue time; None for roots or when not determinable."""
    is_tree_final: bool
    """Best-effort: True only when this is provably the last request the
    whole session tree will send."""


class SessionRoutingBase(ABC, Generic[OptionsT]):  # noqa: B024  # ABC marks the plugin protocol; every method has a working default so passthrough subclasses instantiate directly.
    """Base for session-routing plugins (``session_routing`` category)."""

    mutates_body: ClassVar[bool] = False
    """True when ``transform_body`` changes the payload. Gates the plugin off
    the verbatim PAYLOAD_BYTES mmap fast path at dataset build, cache hit,
    and runtime."""

    # ClassVar cannot reference the OptionsT type parameter, so the base type
    # is kept here; subclasses narrow it via their Generic parameterization.
    Options: ClassVar[type[AIPerfBaseModel]] = EmptyRoutingOptions
    """Per-plugin options model, populated from --session-routing-opt
    key=value pairs. Every Options model must set extra='forbid'."""

    def __init__(self, options: OptionsT) -> None:
        self.options: OptionsT = options

    def headers(self, ctx: RoutingContext) -> dict[str, str]:
        """Extra HTTP headers for this request (merged into endpoint headers)."""
        return {}

    def transform_body(
        self, payload: dict[str, Any], ctx: RoutingContext
    ) -> dict[str, Any]:
        """Return a (possibly new) payload dict; never mutate the input."""
        return payload

    def on_session_end(self, x_correlation_id: str) -> None:
        """Post-session cleanup: no further requests will be sent for this
        session by this worker. Idempotent; default no-op."""
        return None


class DynamoHeadersRouting(SessionRoutingBase[EmptyRoutingOptions]):
    """Dynamo session affinity via X-Dynamo-Session-ID / X-Dynamo-Parent-Session-ID.

    Pair with a Dynamo frontend running ``--router-session-affinity-ttl-secs``.
    """

    def headers(self, ctx: RoutingContext) -> dict[str, str]:
        headers = {"X-Dynamo-Session-ID": ctx.x_correlation_id}
        if ctx.parent_correlation_id:
            headers["X-Dynamo-Parent-Session-ID"] = ctx.parent_correlation_id
        return headers


class DynamoNvextOptions(AIPerfBaseModel):
    """Options for the deprecated-upstream nvext.session_control transport."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = Field(
        default=300,
        ge=1,
        description="Dynamo session_control inactivity timeout carried on every bind.",
    )


class DynamoNvextRouting(SessionRoutingBase[DynamoNvextOptions]):
    """Dynamo session affinity via nvext.session_control request-body metadata.

    Modern contract only: 'bind' on every non-final turn (idempotent on the
    router, refreshes the TTL), 'close' on the final turn. Targets Dynamo
    builds that implement session_control; current upstream Dynamo main does
    not (use dynamo_headers there).
    """

    mutates_body: ClassVar[bool] = True
    Options: ClassVar[type[AIPerfBaseModel]] = DynamoNvextOptions

    def transform_body(
        self, payload: dict[str, Any], ctx: RoutingContext
    ) -> dict[str, Any]:
        if ctx.is_final_turn:
            session_control: dict[str, Any] = {
                "session_id": ctx.x_correlation_id,
                "action": "close",
            }
        else:
            session_control = {
                "session_id": ctx.x_correlation_id,
                "action": "bind",
                "timeout": self.options.timeout_seconds,
            }
        merged = dict(payload)
        raw_nvext = merged.get("nvext")
        nvext = dict(raw_nvext) if isinstance(raw_nvext, dict) else {}
        raw_sc = nvext.get("session_control")
        merged_sc = dict(raw_sc) if isinstance(raw_sc, dict) else {}
        merged_sc.update(session_control)
        nvext["session_control"] = merged_sc
        merged["nvext"] = nvext
        return merged


class SmgRoutingKeyRouting(SessionRoutingBase[EmptyRoutingOptions]):
    """SGLang Model Gateway manual-policy stickiness via X-SMG-Routing-Key."""

    def headers(self, ctx: RoutingContext) -> dict[str, str]:
        return {"X-SMG-Routing-Key": ctx.x_correlation_id}


class SessionIdHeaderOptions(AIPerfBaseModel):
    """Options for the generic additive session-ID header."""

    model_config = ConfigDict(extra="forbid")

    header_name: str = Field(
        default="X-Session-ID",
        min_length=1,
        description="Header name carrying the per-session correlation ID.",
    )


class SessionIdHeaderRouting(SessionRoutingBase[SessionIdHeaderOptions]):
    """Generic additive session header for routers expecting a custom name."""

    Options: ClassVar[type[AIPerfBaseModel]] = SessionIdHeaderOptions

    def headers(self, ctx: RoutingContext) -> dict[str, str]:
        return {self.options.header_name: ctx.x_correlation_id}
