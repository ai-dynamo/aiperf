# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed schema for the ``dag_jsonl`` file format.

Each line in a DAG JSONL file validates as a :class:`DagConversation`. Each
turn validates as a :class:`DagTurn`, whose top-level fields map to AIPerf's
native Turn concepts (``messages``, ``model``, ``max_tokens``, ``tools``) plus
three structural scheduling fields (``forks``, ``spawns``, ``delay``). Every
other OpenAI chat-completions or vendor-specific parameter — temperature,
top_p, seed, stop, ignore_eos, min_tokens, etc. — goes in
:attr:`DagTurn.extra_body`, matching the CLI's ``--extra-inputs`` convention.

Messages are stored as ``list[dict[str, Any]]`` with a lightweight validator
(non-empty, each entry must have a ``role`` key), matching ``MooncakeTrace``.
This leaves multimodal content parts, ``tool_calls``, and any future OpenAI
message shape unconstrained so authors can paste their exact wire body.

Unknown top-level keys on either a conversation or a turn are rejected at
load time so typos surface immediately.
"""

from typing import Any

from pydantic import ConfigDict, Field, model_validator

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.loader.models import validate_chat_messages


class DagSpawn(AIPerfBaseModel):
    """Delayed-join SPAWN entry. Object-form alternative to a plain string id.

    Use this when the parent should continue running turns while the spawned
    children execute in parallel. ``join_at`` (default: this turn's index +
    1) authors the turn on which the parent's SPAWN_JOIN prerequisite is
    placed; the parent runs turns [spawn_turn+1 .. join_at-1] concurrently
    with children and suspends only when it's about to dispatch ``join_at``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    children: list[str] = Field(
        min_length=1,
        description="Child session ids to dispatch as SPAWN branches after "
        "this turn completes.",
    )
    join_at: int | None = Field(
        default=None,
        description="Turn index on which the parent's SPAWN_JOIN prerequisite "
        "is placed (delayed-join K>=1). Defaults to (spawn_turn + 1); author "
        "must supply a value strictly greater than the spawn turn index and "
        "less than the conversation's total turn count.",
    )


class DagTurn(AIPerfBaseModel):
    """One turn in a DAG conversation.

    Top-level fields are limited to AIPerf-native Turn concepts plus DAG
    scheduling keys. Any other OpenAI or vendor-specific parameter goes in
    ``extra_body``, where keys are merged into the top level of the wire body
    at dispatch time (matching the OpenAI SDK's ``extra_body=`` keyword and
    AIPerf's CLI ``--extra-inputs`` convention).

    Unknown top-level keys are rejected.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # --- AIPerf-native Turn concepts (top-level) ----------------------------
    messages: list[dict[str, Any]] = Field(
        description="OpenAI-compatible messages authored for this turn. Each "
        "entry must be a dict with a 'role' key; content may be a string or a "
        "multimodal parts list. Concatenated onto the session's accumulator "
        "on each turn (pure append).",
    )
    model: str | None = Field(
        default=None,
        description="Override the model name for this turn (otherwise the "
        "CLI --model wins).",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Maximum completion tokens for this turn.",
    )
    tools: list[dict[str, Any]] | None = Field(
        default=None,
        description="OpenAI-compatible tool definitions. Each entry is a "
        "free-form dict so new tool shapes don't require a loader bump.",
    )

    # --- Everything else (sampling params, vendor tunables) -----------------
    extra_body: dict[str, Any] | None = Field(
        default=None,
        description="Non-native fields sent on the wire: temperature, top_p, "
        "seed, stop, logprobs, response_format, presence/frequency_penalty, "
        "and vendor-specific knobs like ``ignore_eos`` or ``min_tokens``. Keys "
        "are merged into the top level of the request body at dispatch time.",
    )

    # --- Structural (DAG scheduling) fields, not sent on the wire -----------
    forks: list[str] = Field(
        default_factory=list,
        description="Child session ids to dispatch as FORK branches after this "
        "turn completes (children inherit the parent's accumulator and "
        "sticky-route to the parent's worker).",
    )
    spawns: list[str | DagSpawn] = Field(
        default_factory=list,
        description="Child session ids to dispatch as SPAWN branches after "
        "this turn completes (children start fresh, route freely). Each "
        "entry may be a bare string (legacy: auto-join on next turn) or a "
        "``DagSpawn`` object carrying a ``join_at`` index for delayed joins.",
    )
    delay: float = Field(
        default=0.0,
        ge=0.0,
        description="Milliseconds to wait before dispatching this turn. "
        "Matches the unit of ``Turn.delay`` / ``TurnMetadata.delay_ms`` so "
        "the loader can pass the value through without conversion.",
    )

    @model_validator(mode="after")
    def _validate_messages(self) -> "DagTurn":
        validate_chat_messages(self.messages)
        return self


class DagConversation(AIPerfBaseModel):
    """One line of a DAG JSONL file: a session with ordered turns."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    session_id: str = Field(
        min_length=1,
        description="Unique identifier for this conversation within the file.",
    )
    turns: list[DagTurn] = Field(
        min_length=1,
        description="Ordered list of turns (non-empty).",
    )
    pre_session_spawns: list[str] = Field(
        default_factory=list,
        description="Child session ids to dispatch as background SPAWN "
        "branches BEFORE this conversation's turn 0 is issued. Used for "
        "trace-timing fidelity where a captured child first-request "
        "overlaps with parent turn 0's in-flight window. Fire-and-forget "
        "(background SPAWN only); children get a fresh correlation id "
        "with ``parent_correlation_id=None``.",
    )
