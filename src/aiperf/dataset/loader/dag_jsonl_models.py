# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed schema for the ``dag_jsonl`` file format.

Each line in a DAG JSONL file validates as a :class:`DagConversation`. Each
turn validates as a :class:`DagTurn`, whose top-level fields map to AIPerf's
native Turn concepts (``messages``, ``model``, ``max_tokens``, ``tools``) plus
three structural scheduling fields (``forks``, ``spawns``, ``delay``). Every
other OpenAI chat-completions or vendor-specific parameter — temperature,
top_p, seed, stop, ignore_eos, min_tokens, etc. — goes in
:attr:`DagTurn.extra`, matching the CLI's ``--extra-inputs`` convention.

Messages are stored as ``list[dict[str, Any]]`` with a lightweight validator
(non-empty, each entry must have a ``role`` key), matching ``MooncakeTrace``.
This leaves multimodal content parts, ``tool_calls``, and any future OpenAI
message shape unconstrained so authors can paste their exact wire body.

Unknown top-level keys on either a conversation or a turn are rejected at
load time so typos surface immediately.
"""

from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.loader.models import validate_chat_messages


class DagFork(AIPerfBaseModel):
    """Object-form FORK entry. Bare-string ``"<sid>"`` desugars to
    ``DagFork(child="<sid>", background=False)``.

    Use the object form when the parent should keep running its remaining
    turns instead of terminating after the fork dispatches
    (``background=True``).

    Symmetric note: ``DagFork.background`` is to FORK what ``DagSpawn.join_at``
    is to SPAWN — the parent-continuation knob. ``background=True`` is fire-
    and-forget within the parent's session: child inherits the parent's
    accumulator at the spawn point AND sticky-routes to the parent's worker
    (locality preserved), but no SPAWN_JOIN prereq is generated, so the
    parent dispatches its next turn without waiting.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    child: str = Field(
        min_length=1,
        description="Child session id to dispatch as a FORK branch.",
    )
    background: bool = Field(
        default=False,
        description="If True, the parent continues with its remaining turns "
        "after this fork dispatches. Default False (parent terminates after "
        "fork — the bare-string default). Synonym: 'fork-and-continue' (used "
        "in user-facing docs and PR descriptions) — same as ``background=True``.",
    )


class DagSpawn(AIPerfBaseModel):
    """Delayed-join SPAWN entry. Object-form alternative to a plain string id.

    Asymmetric note: ``forks`` entries are ALWAYS bare strings (no equivalent
    ``DagFork`` class). Only ``spawns`` has an object-form because only SPAWN
    branches support delayed-join — FORK terminates the parent so there is no
    later turn to join on.

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
    ``extra``, where keys are merged into the top level of the wire body
    at dispatch time (matching AIPerf's CLI ``--extra-inputs`` convention and
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
    extra: dict[str, Any] | None = Field(
        default=None,
        description="Non-native fields sent on the wire: temperature, top_p, "
        "seed, stop, logprobs, response_format, presence/frequency_penalty, "
        "and vendor-specific knobs like ``ignore_eos`` or ``min_tokens``. Keys "
        "are merged into the top level of the request body at dispatch time.",
    )

    # --- Structural (DAG scheduling) fields, not sent on the wire -----------
    forks: list[str | DagFork] = Field(
        default_factory=list,
        description="Child session ids to dispatch as FORK branches after this "
        "turn completes (children inherit the parent's accumulator and "
        "sticky-route to the parent's worker). Each entry may be a bare "
        "string (parent terminates after fork) or a ``DagFork`` "
        "object carrying ``background=True`` to keep the parent running.",
    )
    spawns: list[str | DagSpawn] = Field(
        default_factory=list,
        description="Child session ids to dispatch as SPAWN branches after "
        "this turn completes (children start fresh, route freely). Each "
        "entry may be a bare string (auto-join on next turn) or a "
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

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _reject_bool_max_tokens(cls, v: Any) -> Any:
        # bool is a subclass of int in Python, so Pydantic accepts True/False
        # for ``int | None`` fields and silently coerces them to 1/0. That
        # makes ``"max_tokens": true`` (a typo or copy-paste from another
        # config shape) round-trip into a 1-token response with no error.
        if isinstance(v, bool):
            raise ValueError(
                "max_tokens must be an integer, not a boolean (got "
                f"{v!r}); check for a typo in your DAG JSONL file"
            )
        return v


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
