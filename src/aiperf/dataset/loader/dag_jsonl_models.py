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

from typing import Any, Literal

from pydantic import ConfigDict, Field, ValidationInfo, field_validator, model_validator

from aiperf.common.enums import ConversationContextMode
from aiperf.common.finite import FiniteFloat
from aiperf.common.models import AIPerfBaseModel
from aiperf.dataset.loader.models import validate_chat_messages


class DagMediaSegment(AIPerfBaseModel):
    """Content-addressed geometric descriptor of one synthetic image or video.

    Carries no pixel bytes or file paths. Identical
    ``(kind, media_hash, width, height, fps, num_frames)`` tuples across turns,
    ``FORK`` branches, or ``SPAWN`` sub-agents materialize to byte-identical
    media, enabling server-side vision-encoder and KV prefix cache reuse.
    """

    model_config = ConfigDict(extra="forbid")

    media_hash: str = Field(
        min_length=1,
        max_length=128,
        description="Content-addressed identifier seeding deterministic media synthesis.",
    )
    kind: Literal["image", "video"] = Field(
        description="Modality of this synthetic media segment ('image' or 'video').",
    )
    width: int = Field(
        ge=2,
        le=16384,
        description="Media width in pixels (must be even for video).",
    )
    height: int = Field(
        ge=2,
        le=16384,
        description="Media height in pixels (must be even for video).",
    )
    fps: FiniteFloat = Field(
        default=24.0,
        ge=0.0,
        le=240.0,
        description="Video frame rate, rounded to two decimals (ignored for images).",
    )
    num_frames: int = Field(
        default=1,
        ge=1,
        le=4096,
        description="Number of video frames (must be 1 when kind='image').",
    )
    uuid: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description="Optional multimodal cache uuid sent on the content part. When "
        "omitted and --cache-bust-media-uuids is set, a uuid derived from the "
        "segment identity is sent instead.",
    )

    @field_validator("width", "height", "fps", "num_frames", mode="before")
    @classmethod
    def _reject_bool_numeric_fields(cls, v: Any, info: ValidationInfo) -> Any:
        if isinstance(v, bool):
            raise ValueError(
                f"{info.field_name} must be numeric, not a boolean (got {v!r})"
            )
        return v

    @model_validator(mode="after")
    def _validate_geometry(self) -> "DagMediaSegment":
        if self.kind == "image" and self.num_frames != 1:
            raise ValueError(
                f"image segment '{self.media_hash}' must have num_frames=1 "
                f"(got {self.num_frames})"
            )
        if self.kind == "video":
            if self.width % 2 != 0 or self.height % 2 != 0:
                raise ValueError(
                    f"video segment '{self.media_hash}' requires even dimensions "
                    f"for yuv420p encoding (got {self.width}x{self.height})"
                )
            if round(float(self.fps), 2) <= 0.0:
                raise ValueError(
                    f"video segment '{self.media_hash}' requires fps >= 0.01 "
                    f"(got {self.fps})"
                )
        return self

    @property
    def cache_key(self) -> tuple[str, str, int, int, float, int]:
        """Deduplication key ``(kind, media_hash, width, height, fps, num_frames)``."""
        eff_fps = round(float(self.fps), 2) if self.kind == "video" else 0.0
        return (
            self.kind,
            self.media_hash,
            self.width,
            self.height,
            eff_fps,
            self.num_frames,
        )


class DagPromptSpan(AIPerfBaseModel):
    """Ordered text or media span used to synthesize a turn's messages."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["text", "media"] = Field(
        description="'text' synthesizes tokens; 'media' binds a DagMediaSegment.",
    )
    span_id: str = Field(
        min_length=1,
        max_length=128,
        description="Seed for text synthesis, or the media_hash a media span binds.",
    )
    tokens: int | None = Field(
        default=None,
        ge=1,
        le=1_048_576,
        description="Required for 'text' spans: exact token count to synthesize. "
        "Optional and informational only for 'media' spans (e.g. the recorded "
        "visual token count); media geometry comes from the bound segment.",
    )
    media_index: int | None = Field(
        default=None,
        ge=0,
        le=4096,
        description="Zero-based index into DagTurn.media_segments when kind='media'.",
    )
    role: Literal["system", "user", "assistant"] = Field(
        default="user",
        description="Message role; consecutive spans with the same role merge "
        "into one message. Media spans require 'user'.",
    )

    @field_validator("tokens", "media_index", mode="before")
    @classmethod
    def _reject_bool_span_ints(cls, v: Any, info: ValidationInfo) -> Any:
        if isinstance(v, bool):
            raise ValueError(
                f"{info.field_name} must be an integer, not a boolean (got {v!r})"
            )
        return v

    @model_validator(mode="after")
    def _validate_span_kind(self) -> "DagPromptSpan":
        if self.kind == "text":
            if self.tokens is None:
                raise ValueError(f"text prompt span '{self.span_id}' requires 'tokens'")
            if self.media_index is not None:
                raise ValueError(
                    f"text prompt span '{self.span_id}' must omit media_index"
                )
        elif self.role != "user":
            raise ValueError(
                f"media prompt span '{self.span_id}' requires role='user' "
                f"(got role={self.role!r})"
            )
        return self


def _hash_index(segments: list[DagMediaSegment]) -> dict[str, list[int]]:
    by_hash: dict[str, list[int]] = {}
    for idx, seg in enumerate(segments):
        by_hash.setdefault(seg.media_hash, []).append(idx)
    return by_hash


def _check_unambiguous(
    media_hash: str, matches: list[int], segments: list[DagMediaSegment]
) -> None:
    """Reject a hash-only reference matching non-identical segments."""
    if len({(*segments[i].cache_key, segments[i].uuid) for i in matches}) > 1:
        raise ValueError(
            f"media_hash '{media_hash}' is ambiguous across distinct "
            f"media_segments at indices {matches}; specify 'media_index'"
        )


def _validate_media_ref_part(
    part: dict[str, Any],
    segments: list[DagMediaSegment],
    by_hash: dict[str, list[int]],
) -> None:
    ref_idx, ref_hash = part.get("media_index"), part.get("media_hash")
    if ref_idx is not None:
        if type(ref_idx) is not int or not 0 <= ref_idx < len(segments):
            raise ValueError(
                f"media_ref media_index={ref_idx!r} is not a valid index "
                f"into {len(segments)} media_segments"
            )
        if ref_hash is not None and segments[ref_idx].media_hash != ref_hash:
            raise ValueError(
                f"media_ref media_hash='{ref_hash}' disagrees with "
                f"media_segments[{ref_idx}]"
            )
    elif ref_hash is None:
        raise ValueError("media_ref requires 'media_hash' or 'media_index'")
    elif str(ref_hash) not in by_hash:
        raise ValueError(f"media_ref media_hash='{ref_hash}' not found")
    else:
        _check_unambiguous(str(ref_hash), by_hash[str(ref_hash)], segments)


def _validate_message_media_refs(
    messages: list[dict[str, Any]], segments: list[DagMediaSegment]
) -> None:
    """Validate ``{"type": "media_ref", ...}`` parts inside authored ``messages``."""
    by_hash = _hash_index(segments)
    ref_count = 0
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "media_ref"):
                continue
            ref_count += 1
            if msg.get("role") != "user":
                raise ValueError(
                    "media_ref content part requires message role='user' "
                    f"(got role={msg.get('role')!r})"
                )
            _validate_media_ref_part(part, segments, by_hash)
    if segments and ref_count == 0:
        raise ValueError(
            "media_segments provided alongside authored 'messages', but no "
            "{'type': 'media_ref', ...} content part references them"
        )


def _validate_span_media_refs(
    spans: list[DagPromptSpan], segments: list[DagMediaSegment]
) -> None:
    """Check every media span resolves to exactly one ``media_segments`` entry."""
    by_hash = _hash_index(segments)
    next_idx = 0
    for span in spans:
        if span.kind != "media":
            continue
        if span.media_index is not None:
            idx = span.media_index
        elif span.span_id in by_hash:
            _check_unambiguous(span.span_id, by_hash[span.span_id], segments)
            idx = by_hash[span.span_id][0]
        else:
            idx = next_idx
        if idx >= len(segments):
            raise ValueError(
                f"media prompt span '{span.span_id}' does not resolve to any of "
                f"the {len(segments)} media_segments"
            )
        next_idx = idx + 1


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
    messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="OpenAI-compatible messages authored for this turn. Each "
        "entry must be a dict with a 'role' key; content may be a string or a "
        "multimodal parts list (including ``{'type': 'media_ref', ...}`` "
        "references to ``media_segments``). Concatenated onto the session's "
        "accumulator on each turn (pure append). Required unless "
        "``prompt_spans``, ``media_segments`` or ``in_text_tokens`` is set.",
    )
    media_segments: list[DagMediaSegment] = Field(
        default_factory=list,
        description="Image/video descriptors synthesized deterministically at "
        "load time and sent as inline ``data:`` URIs. Identical "
        "``(kind, media_hash, width, height, fps, num_frames)`` entries across "
        "turns or FORK/SPAWN branches materialize to identical media bytes.",
    )
    prompt_spans: list[DagPromptSpan] = Field(
        default_factory=list,
        description="Ordered text and media spans used to synthesize the "
        "turn's messages when ``messages`` is omitted.",
    )
    in_text_tokens: int | None = Field(
        default=None,
        ge=0,
        le=1_048_576,
        description="Optional text token budget synthesized ahead of "
        "``media_segments`` when both ``messages`` and ``prompt_spans`` are "
        "omitted.",
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
        "this turn completes (children start fresh on the parent worker when "
        "its sticky entry is live, else least-loaded, then stay pinned to "
        "that worker under their own id). "
        "Each entry may be a bare string (auto-join on next turn) or a "
        "``DagSpawn`` object carrying a ``join_at`` index for delayed joins.",
    )
    delay: float = Field(
        default=0.0,
        ge=0.0,
        description="Milliseconds to wait before dispatching this turn. "
        "Matches the unit of ``Turn.delay`` / ``TurnMetadata.delay_ms`` so "
        "the loader can pass the value through without conversion.",
    )

    @model_validator(mode="before")
    @classmethod
    def _validate_messages_presence(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "messages" in data and data["messages"] is None:
            raise ValueError("messages: Input should be a valid list")
        has_structural = bool(
            data.get("prompt_spans")
            or data.get("media_segments")
            or data.get("in_text_tokens") is not None
        )
        if "messages" not in data and not has_structural:
            raise ValueError(
                "messages: Field required (or supply prompt_spans / media_segments)"
            )
        return data

    @model_validator(mode="after")
    def _validate_messages(self) -> "DagTurn":
        if self.messages is not None:
            if self.prompt_spans or self.in_text_tokens is not None:
                raise ValueError(
                    "'messages' is mutually exclusive with 'prompt_spans' and "
                    "'in_text_tokens' on DagTurn"
                )
            validate_chat_messages(self.messages)
            _validate_message_media_refs(self.messages, self.media_segments)
            return self
        if not (self.prompt_spans or self.media_segments or self.in_text_tokens):
            raise ValueError(
                "structural turn requires non-empty prompt_spans, media_segments, "
                "or positive in_text_tokens"
            )
        _validate_span_media_refs(self.prompt_spans, self.media_segments)
        return self

    @field_validator("max_tokens", "in_text_tokens", mode="before")
    @classmethod
    def _reject_bool_max_tokens(cls, v: Any, info: ValidationInfo) -> Any:
        # bool is a subclass of int in Python, so Pydantic accepts True/False
        # for ``int | None`` fields and silently coerces them to 1/0. That
        # makes ``"max_tokens": true`` (a typo or copy-paste from another
        # config shape) round-trip into a 1-token response with no error.
        if isinstance(v, bool):
            raise ValueError(
                f"{info.field_name} must be an integer, not a boolean (got "
                f"{v!r}); check for a typo in your DAG JSONL file"
            )
        return v


class DagRound(AIPerfBaseModel):
    """One independently-authored round of an orchestrator ``rounds`` spine.

    Used in the LIST form of :attr:`DagConversation.rounds`. Each round declares
    its OWN branch session ids (and optional think-time), so the rounds are
    distinct authored stages -- different system prompts, growing pre-baked
    history, different multimodal payloads per round -- instead of the integer
    ``rounds`` form that re-fires one shared branch set every round.
    """

    model_config = ConfigDict(extra="forbid")

    spawns: list[str] = Field(
        min_length=1,
        description="This round's SPAWN branch session ids. Fanned out in "
        "parallel when the round fires; the next round waits (join=all) for ALL "
        "of them before firing. Each id must name another conversation in the "
        "file; authored independently per round. The round is identified in raw "
        "export by these branch session ids (each round uses distinct ones).",
    )
    think_time_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Optional per-round think-time (milliseconds) the coordinator "
        "waits after this round's branches drain, before releasing the next "
        "round. Overrides the conversation-level 'think_time_ms' for this round; "
        "when 'think_time_sigma' is set it is the lognormal median for this round.",
    )

    @field_validator("think_time_ms", mode="before")
    @classmethod
    def _reject_bool_think_time(cls, v: Any) -> Any:
        # bool subclasses int, so Pydantic would coerce `true`/`false` into
        # 1.0/0.0 ms and silently alter pacing. Reject up front.
        if isinstance(v, bool):
            raise ValueError(
                f"think_time_ms must be numeric, not a boolean (got {v!r})"
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
        default_factory=list,
        description="Ordered list of turns. Non-empty for normal conversations; "
        "empty only for an orchestrator conversation (see 'orchestrator').",
    )
    orchestrator: bool = Field(
        default=False,
        description="Mark this as a request-less orchestrator: it sends no request "
        "and re-fires its conversation-level 'spawns' children on every sampled "
        "iteration. The loader synthesizes a single no-op (no_request) turn that "
        "issues a virtual credit and desugars 'spawns' into a post-timing SPAWN "
        "branch. Requires an empty authored turns list, a non-empty 'spawns', and "
        "an empty 'pre_session_spawns'.",
    )
    spawns: list[str] = Field(
        default_factory=list,
        description="Conversation-level SPAWN children for an orchestrator "
        "conversation. Desugared by the loader into a post-timing SPAWN branch on "
        "the synthesized no-op turn, so the orchestrator re-fires them on every "
        "sampled iteration (fire-and-forget). Only valid when 'orchestrator' is "
        "true; per-turn spawning uses 'DagTurn.spawns' instead.",
    )
    think_time_ms: float | None = Field(
        default=None,
        ge=0.0,
        description="Orchestrator-'rounds'-only. Per-round think-time (milliseconds) "
        "the coordinator waits after a round's branches all complete, before "
        "releasing the next round. Applied on turn 0 via the normal turn delay and "
        "on each gated join turn via the branch orchestrator. A fixed value today; "
        "a sampled distribution (per (instance, round)) plugs in at the same seam.",
    )
    think_time_sigma: float | None = Field(
        default=None,
        gt=0.0,
        le=10.0,
        description="When set, per-round think-time is drawn from a lognormal whose "
        "MEDIAN is 'think_time_ms' and log-space stddev is this value (independent "
        "draw per (instance, round), reproducible under --random-seed). Requires "
        "'think_time_ms'. Unset => fixed think-time.",
    )
    think_time_min_ms: float | None = Field(
        default=None, ge=0.0, description="Optional lower clamp on the sampled draw."
    )
    think_time_max_ms: float | None = Field(
        default=None, ge=0.0, description="Optional upper clamp on the sampled draw."
    )
    rounds: int | list[DagRound] | None = Field(
        default=None,
        description="Orchestrator-only. Synthesize a gated 'spine' instead of a "
        "single fire-and-forget turn. Two forms:\n"
        "- INT (N): N rounds that each re-fire the SHARED conversation-level "
        "'spawns' children -- one repeated branch template.\n"
        "- LIST of DagRound: each round authors its OWN 'spawns' (and optional "
        "per-round think-time), so the rounds are distinct stages rather than "
        "repetitions -- different prompts/history/multimodal per round.\n"
        "Either form produces N+1 request-free turns (N spawning turns, each of "
        "which the next round waits on with join=all, plus a terminal gate). "
        "When unset, the orchestrator synthesizes a single fire-and-forget turn.",
    )
    context_mode: ConversationContextMode | None = Field(
        default=None,
        description="Optional per-conversation context mode. When omitted, DAG "
        "conversations default to DELTAS_WITHOUT_RESPONSES (multi-turn chat that "
        "accumulates prior turns and threads live inference responses into the "
        "history). Set 'message_array_with_responses' for payload isolation: each "
        "turn is sent as its OWN complete authored message array with NO "
        "accumulation of prior turns or live responses -- required when every "
        "turn independently authors its own system prompt, pre-baked history, and "
        "multimodal payload (e.g. an orchestrator spine's branch sessions).",
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

    @field_validator(
        "rounds",
        "think_time_ms",
        "think_time_sigma",
        "think_time_min_ms",
        "think_time_max_ms",
        mode="before",
    )
    @classmethod
    def _reject_bool_scheduling_fields(cls, v: Any) -> Any:
        # bool subclasses int, so Pydantic would coerce `true`/`false` for these
        # numeric (and int|list ``rounds``) fields into 1/0 -- silently altering
        # topology (a one-round spine) or pacing (1.0 ms). Reject up front.
        if isinstance(v, bool):
            raise ValueError(
                "rounds / think_time_* must be numeric (or a list for 'rounds'), "
                f"not a boolean (got {v!r}); check for a typo in your DAG JSONL file"
            )
        return v

    @model_validator(mode="after")
    def _validate_orchestrator(self) -> "DagConversation":
        if not self.orchestrator:
            if self.rounds is not None:
                raise ValueError(
                    "'rounds' is only valid on an orchestrator conversation "
                    "(set 'orchestrator': true)"
                )
            if self.spawns:
                raise ValueError(
                    "conversation-level 'spawns' is only valid on an orchestrator "
                    "conversation; a normal conversation uses per-turn "
                    "'DagTurn.spawns' (top-level 'spawns' would be silently ignored)"
                )
            if not self.turns:
                raise ValueError(
                    "'turns' must be non-empty unless 'orchestrator' is true"
                )
            return self

        if self.turns:
            raise ValueError(
                "orchestrator conversation must have an empty authored 'turns' "
                "list (the loader synthesizes its single no-op turn)"
            )
        self._validate_orchestrator_rounds()
        if self.pre_session_spawns:
            raise ValueError(
                "orchestrator conversation must not set 'pre_session_spawns'; "
                "use conversation-level 'spawns' instead"
            )
        return self

    def _validate_orchestrator_rounds(self) -> None:
        """Validate the spawns/rounds shape of an orchestrator conversation.

        LIST-form ``rounds`` authors branches per round (so conversation-level
        ``spawns`` must be omitted); the INT/absent form re-fires the shared
        conversation-level ``spawns``.
        """
        if isinstance(self.rounds, list):
            if not self.rounds:
                raise ValueError("orchestrator 'rounds' list must be non-empty")
            if self.spawns:
                raise ValueError(
                    "with a per-round 'rounds' list, omit conversation-level "
                    "'spawns' (each round declares its own 'spawns')"
                )
            # Each list-form round must own DISTINCT branch sessions: a child id
            # reused across rounds would emit raw records with the same
            # (root, conversation_id, turn_index) key, making per-round latency
            # attribution ambiguous.
            seen: set[str] = set()
            for r in self.rounds:
                for child in r.spawns:
                    if child in seen:
                        raise ValueError(
                            f"child session id '{child}' is spawned by more than "
                            "one round; each list-form round must own distinct "
                            "branch sessions for unambiguous raw attribution"
                        )
                    seen.add(child)
            return
        if isinstance(self.rounds, int) and self.rounds < 1:
            raise ValueError("'rounds' count must be >= 1")
        if not self.spawns:
            raise ValueError(
                "orchestrator conversation requires a non-empty 'spawns' "
                "(or a per-round 'rounds' list)"
            )

    @model_validator(mode="after")
    def _validate_think_time(self) -> "DagConversation":
        """Per-round think-time is exclusive to an orchestrator 'rounds' spine;
        reject orphan/mis-placed/contradictory think-time config up front so it
        can never silently no-op or reach the sampler as a bad value."""
        think_fields = (
            self.think_time_sigma,
            self.think_time_min_ms,
            self.think_time_max_ms,
        )
        if not self.orchestrator:
            if self.think_time_ms is not None or any(
                v is not None for v in think_fields
            ):
                raise ValueError(
                    "'think_time_ms'/'think_time_sigma'/'think_time_min_ms'/"
                    "'think_time_max_ms' are only valid on an orchestrator "
                    "conversation with 'rounds' set"
                )
            return self
        if self.think_time_ms is not None and self.rounds is None:
            raise ValueError(
                "'think_time_ms' requires 'rounds' (it is the per-round wait of the "
                "gated spine); it has no meaning for a single fire-and-forget turn"
            )
        if self.think_time_sigma is not None and self.think_time_ms is None:
            raise ValueError(
                "'think_time_sigma' requires 'think_time_ms' (the lognormal median "
                "to sample around)"
            )
        # Clamps apply to the SAMPLED draw only; without a distribution they would
        # be silently ignored, so reject clamp-only configs.
        clamp_set = (
            self.think_time_min_ms is not None or self.think_time_max_ms is not None
        )
        if clamp_set and self.think_time_sigma is None:
            raise ValueError(
                "'think_time_min_ms'/'think_time_max_ms' require 'think_time_sigma' "
                "(they clamp the sampled draw; a fixed think-time has nothing to clamp)"
            )
        # Ordered clamp: an inverted range silently collapses every draw.
        if (
            self.think_time_min_ms is not None
            and self.think_time_max_ms is not None
            and self.think_time_min_ms > self.think_time_max_ms
        ):
            raise ValueError(
                "'think_time_min_ms' must be <= 'think_time_max_ms' "
                f"(got {self.think_time_min_ms} > {self.think_time_max_ms})"
            )
        return self
