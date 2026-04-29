# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed schema for the ``dag_jsonl`` file format.

Each line in a DAG JSONL file validates as a :class:`DagConversation`. Each
turn validates as a :class:`DagTurn`, whose top-level fields map to AIPerf's
native Turn concepts (``messages``, ``model``, ``max_tokens``, ``tools``) plus
two structural scheduling fields (``forks``, ``delay``). Every other OpenAI
chat-completions or vendor-specific parameter — temperature, top_p, seed,
stop, ignore_eos, min_tokens, etc. — goes in :attr:`DagTurn.extra_body`,
matching the CLI's ``--extra-inputs`` convention.

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
    """One line of a DAG JSONL file: a session with ordered turns.

    JSON shape::

        {"session_id": "root",
         "turns": [
             {"messages": [{"role": "user", "content": "Hi"}], "forks": ["child"]}
         ]}

    Unknown top-level keys (e.g. ``"sessoin_id"``) are rejected at load time
    so typos surface immediately rather than being silently dropped.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    session_id: str = Field(
        min_length=1,
        description="Unique identifier for this conversation within the file.",
    )
    turns: list[DagTurn] = Field(
        min_length=1,
        description="Ordered list of turns (non-empty).",
    )
