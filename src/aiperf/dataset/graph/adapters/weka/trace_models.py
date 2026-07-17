# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter-private Pydantic models for the Weka KV-cache-tester trace format.

``WekaSubagentEntry.requests`` is ``list[WekaRequest]``, allowing inner
subagent markers: subagent-within-subagent nesting, which the trie builder
flattens recursively into ordinary flat ``LlmNode``s.

These models are private to the adapter; nothing outside the weka adapter
imports them, so this file alone defines the adapter's input contract.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import ConfigDict, Field

from aiperf.common.models import AIPerfBaseModel


class WekaTraceAdapterError(ValueError):
    """Base error raised when a Weka trace cannot be converted to a ParsedGraph."""


class EmptyWekaTraceError(WekaTraceAdapterError):
    """The file/directory/HF source yields zero usable traces.

    Raised when a trace's ``requests`` list is empty, or when a directory or
    HuggingFace source produces no parsable traces (no ``.json`` files, zero
    rows, or an empty merge result).
    """


class WekaSchemaError(WekaTraceAdapterError):
    """A Weka trace failed Pydantic schema validation.

    Wraps ``pydantic.ValidationError`` with the offending file path and trace
    ``id`` for clear CLI surfacing.
    """


class WekaHashScopeError(WekaTraceAdapterError):
    """The trace declares an unrecognized ``hash_id_scope``.

    Supported scopes are ``"local"`` (per-trace hash namespace) and
    ``"global"`` (one hash namespace shared across every trace in the corpus,
    reproducing recorded cross-trace KV-cache sharing). Any other value is
    rejected here with the precise cause rather than surfacing as a generic
    Pydantic Literal error.
    """


class WekaNormalRequest(AIPerfBaseModel):
    """One normal (``type: "n"``) API call in a Weka trace."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    t: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Request timestamp in seconds from conversation start.",
    )
    type: Literal["n"] = Field(description="Discriminator: normal API call.")
    model: str = Field(description="Model identifier for this request.")
    input_length: int = Field(ge=0, alias="in", description="Input token count.")
    output_length: int = Field(ge=0, alias="out", description="Output token count.")
    hash_ids: list[Annotated[int, Field(ge=0)]] = Field(
        default_factory=list, description="KV-cache block hash IDs."
    )
    input_types: list[str] = Field(
        default_factory=list, description="Content-type annotations for input."
    )
    output_types: list[str] = Field(
        default_factory=list, description="Content-type annotations for output."
    )
    stop: str = Field(
        default="", description="Stop reason: '', 'tool_use', 'end_turn'."
    )
    api_time: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Server processing time in seconds.",
    )
    think_time: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Client delay in seconds before this request.",
    )


class WekaStreamingRequest(AIPerfBaseModel):
    """One streaming (``type: "s"``) API call in a Weka trace.

    Structurally identical to :class:`WekaNormalRequest` except for the
    discriminator value and an optional ``ttft`` field (recorded
    time-to-first-token in seconds).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    t: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Request timestamp in seconds from conversation start.",
    )
    type: Literal["s"] = Field(description="Discriminator: streaming API call.")
    model: str = Field(description="Model identifier for this request.")
    input_length: int = Field(ge=0, alias="in", description="Input token count.")
    output_length: int = Field(ge=0, alias="out", description="Output token count.")
    hash_ids: list[Annotated[int, Field(ge=0)]] = Field(
        default_factory=list, description="KV-cache block hash IDs."
    )
    input_types: list[str] = Field(
        default_factory=list, description="Content-type annotations for input."
    )
    output_types: list[str] = Field(
        default_factory=list, description="Content-type annotations for output."
    )
    stop: str = Field(
        default="", description="Stop reason: '', 'tool_use', 'end_turn'."
    )
    api_time: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Server processing time in seconds.",
    )
    think_time: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Client delay in seconds before this request.",
    )
    ttft: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="Recorded time-to-first-token in seconds.",
    )


class WekaSubagentEntry(AIPerfBaseModel):
    """A ``type: "subagent"`` marker with its nested inner requests.

    The parent's next ``WekaNormalRequest`` in the outer list is understood
    to occur after this subagent completes (when ``status != "async_launched"``).
    Inner ``requests`` may themselves contain :class:`WekaSubagentEntry` markers;
    the trie builder flattens each recursively into ordinary flat ``LlmNode``s.
    """

    model_config = ConfigDict(extra="forbid")

    t: float = Field(
        ge=0,
        allow_inf_nan=False,
        description="Spawn timestamp in seconds from conversation start.",
    )
    type: Literal["subagent"] = Field(description="Discriminator: subagent marker.")
    agent_id: str = Field(description="Opaque subagent identifier, e.g. 'agent_001'.")
    subagent_type: str = Field(description="Subagent type, e.g. 'Explore'.")
    duration_ms: int | None = Field(
        default=None,
        ge=0,
        description="Wall-clock duration of the subagent. None for subagents "
        "with status='async_launched' (telemetry not captured).",
    )
    total_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Total tokens across all subagent inner requests. None "
        "for status='async_launched'.",
    )
    tool_use_count: int | None = Field(
        default=None,
        ge=0,
        description="Tool calls made by the subagent. None for "
        "status='async_launched'.",
    )
    status: str = Field(description="'completed', 'async_launched', or other status.")
    requests: list[WekaRequest] = Field(
        description="Inner requests of the subagent. May contain nested "
        "WekaSubagentEntry markers (subagent-within-subagent nesting); the "
        "trie builder flattens them recursively.",
    )
    models: list[str] = Field(description="Models used by the subagent.")
    tool_tokens: int = Field(
        default=0, ge=0, description="Subagent's tools prefix token count."
    )
    system_tokens: int = Field(
        default=0, ge=0, description="Subagent's system prefix token count."
    )


WekaRequest: TypeAlias = Annotated[
    WekaNormalRequest | WekaStreamingRequest | WekaSubagentEntry,
    Field(discriminator="type"),
]


class WekaTrace(AIPerfBaseModel):
    """A single Weka trace file."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Trace identifier (session ID).")
    models: list[str] = Field(description="Models used in the trace.")
    block_size: int = Field(gt=0, description="Cache block size in tokens.")
    hash_id_scope: Literal["local", "global"] = Field(
        description=(
            "Hash ID namespace scope. 'local' scopes hashes per-trace: equal "
            "hash ids in different traces are different logical blocks and "
            "synthesize different bytes. 'global' shares ONE hash namespace "
            "across every trace in the corpus: equal hash ids synthesize "
            "byte-identical content, reproducing recorded cross-trace "
            "KV-cache sharing. Any other value is rejected with "
            "WekaHashScopeError."
        )
    )
    tool_tokens: int = Field(default=0, ge=0, description="Tools prefix token count.")
    system_tokens: int = Field(
        default=0, ge=0, description="System prefix token count."
    )
    requests: list[WekaRequest] = Field(
        description="Interleaved normal/streaming requests and subagent markers."
    )
    totals: dict[str, Any] | None = Field(
        default=None, description="Optional trace-level summary; opaque."
    )


# Resolve forward reference inside WekaSubagentEntry.requests.
WekaSubagentEntry.model_rebuild()


__all__ = [
    "EmptyWekaTraceError",
    "WekaHashScopeError",
    "WekaNormalRequest",
    "WekaRequest",
    "WekaSchemaError",
    "WekaStreamingRequest",
    "WekaSubagentEntry",
    "WekaTrace",
    "WekaTraceAdapterError",
]
