# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed reader for Dynamo request-trace files (`*.jsonl`, `*.jsonl.gz`, segmented).

Schema source (dynamo repo): lib/llm/src/request_trace/types.rs
    and lib/llm/src/protocols/common/extensions.rs (AgentContext)

Dynamo's file sinks (`jsonl` and `jsonl_gz`, selected by
`DYN_REQUEST_TRACE_SINKS`, gated by `DYN_REQUEST_TRACE`) write ONE envelope
per line: `{"timestamp": <ms since writer start>, "event": {<record>}}`
(`telemetry/jsonl_gz.rs::GzipEntry`); only the stderr sink emits the bare
record. `iter_raw_records` unwraps the envelope and also accepts bare records
(hand-authored fixtures). With the jsonl_gz sink, segments roll into files
like `prefix.000000.jsonl.gz`, `prefix.000001.jsonl.gz`; each gzip member is
a complete batch.

Usage:
    for record in iter_trace_records("/tmp/dynamo-trace"):
        # iterates plain JSONL OR all .000000-.NNNNNN.jsonl.gz segments
        ...

    for record in iter_trace_records(
        "/tmp/dynamo-trace",
        event_types={"request_end"},
        session_id="session-42",
    ):
        ...

Note: in the current `dynamo.request.trace.v1` schema, `event_source`,
`agent_context`, and `request.model` are all optional (Option in Rust) and may
be absent on replay-only records, so the models below make them optional too.
"""

from __future__ import annotations

import gzip
import io
import re
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator

from aiperf.common.finite import FiniteFloat

# Wire discriminator for the current Dynamo trace schema. Imported by the sniff
# in ``dynamo/trace.py``; the ``AgentTraceRecord.schema_`` Literal below carries
# the same value for typed decode (a Literal cannot reference a variable).
DYNAMO_TRACE_SCHEMA_V1 = "dynamo.request.trace.v1"


class DynamoTraceAdapterError(ValueError):
    """Raised when a Dynamo trace cannot be converted into a ParsedGraph."""


class EmptyDynamoTraceError(DynamoTraceAdapterError):
    """Raised when the trace file/dir contains no usable records."""


class AgentContext(BaseModel):
    model_config = ConfigDict(extra="ignore")
    session_id: str = Field(description="Stable reasoning/tool session identifier.")
    parent_session_id: str | None = Field(
        default=None,
        description="Legacy parent-session link for subagents. Real captures "
        "leave this unset and carry the linkage in parent_trajectory_id "
        "instead; kept as a fallback for older/hand-authored traces.",
    )
    trajectory_id: str | None = Field(
        default=None,
        description="Per-trajectory identifier; equals session_id in every "
        "observed capture, so a subagent's parent_trajectory_id resolves to the "
        "parent's session_id.",
    )
    parent_trajectory_id: str | None = Field(
        default=None,
        description="Parent trajectory for subagents (the parent's "
        "trajectory_id == its session_id). The AUTHORITATIVE subagent linkage in "
        "real captures; parent_session_id is only a fallback.",
    )
    session_type_id: str | None = Field(
        default=None,
        description="Coarse session kind recorded by the harness (e.g. "
        "'opencode'); carried through for provenance, not used for linkage.",
    )


class WorkerInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    prefill_worker_id: int | None = Field(
        default=None, description="Prefill worker id when recorded."
    )
    prefill_dp_rank: int | None = Field(
        default=None, description="Prefill DP rank when recorded."
    )
    decode_worker_id: int | None = Field(
        default=None, description="Decode worker id when recorded."
    )
    decode_dp_rank: int | None = Field(
        default=None, description="Decode DP rank when recorded."
    )


class AgentReplayMetrics(BaseModel):
    """KV-cache block-hash provenance for replay.

    Emitted unconditionally on every dynamo ``request_end`` at the current
    schema (the record is skipped entirely when the model's KV block size is
    unavailable, ``request_trace/integration.rs``); absent only on older
    captures or hand-authored traces.
    """

    model_config = ConfigDict(extra="ignore")
    trace_block_size: int = Field(
        ge=1,
        description="KV cache block size used to derive replay hashes. Dynamo "
        "never emits a replay block with size 0 (it skips the record instead), "
        "so a non-positive value is a corrupt or hand-mangled trace.",
    )
    input_length: int = Field(
        description="Prompt/input token count represented by the replay hashes."
    )
    input_sequence_hashes: list[int] = Field(
        description="Stable sequence-aware prompt block hashes (u64 in Rust, may be > 2^63)."
    )

    @field_validator("input_sequence_hashes")
    @classmethod
    def _reject_negative_hashes(cls, hashes: list[int]) -> list[int]:
        """Reject negative recorded hashes: they collide with the virtual negative-id namespace.

        Dynamo records these as u64 (values may exceed 2^63, so they stay
        positive Python ints). A negative value would clash with the virtual
        negative ids the trie lowering mints for non-replay turns, silently
        aliasing distinct content.
        """
        # `min()` iterates the list in C (u64 values may exceed 2^63, so an
        # array('q')/numpy int64 scan would overflow -- keep Python ints). The
        # empty guard preserves the prior `any([]) is False` no-raise behavior.
        if hashes and min(hashes) < 0:
            raise ValueError(
                "input_sequence_hashes must be non-negative (u64 in Rust); "
                "negative values collide with the virtual negative-id namespace."
            )
        return hashes


class AgentRequestMetrics(BaseModel):
    model_config = ConfigDict(extra="ignore")
    request_id: str = Field(description="Dynamo request ID for the LLM call.")
    x_request_id: str | None = Field(
        default=None, description="Caller-provided logical request ID when present."
    )
    model: str | None = Field(
        default=None, description="Model name (optional in the current schema)."
    )
    input_tokens: int | None = Field(
        default=None, description="Prompt/input token count when known."
    )
    output_tokens: int | None = Field(
        default=None, description="Final output token count when known."
    )
    cached_tokens: int | None = Field(
        default=None,
        description="Prompt tokens served from prefix/KV cache when known.",
    )
    request_received_ms: int | None = Field(
        default=None, description="Request receive time in Unix epoch milliseconds."
    )
    prefill_wait_time_ms: float | None = Field(
        default=None, description="Time from request receipt to prefill start."
    )
    prefill_time_ms: float | None = Field(
        default=None, description="Time from prefill start to first token."
    )
    ttft_ms: FiniteFloat | None = Field(
        default=None, description="Time from request receipt to first token."
    )
    total_time_ms: float | None = Field(
        default=None, description="Time from request receipt to request completion."
    )
    avg_itl_ms: FiniteFloat | None = Field(
        default=None, description="Average inter-token latency after first token."
    )
    kv_hit_rate: float | None = Field(
        default=None, description="Effective KV-cache hit rate observed by the router."
    )
    kv_transfer_estimated_latency_ms: FiniteFloat | None = Field(
        default=None,
        description="Upper-bound estimated disaggregated KV transfer latency.",
    )
    queue_depth: int | None = Field(
        default=None,
        description="Router queue depth observed when routing the request.",
    )
    worker: WorkerInfo | None = Field(
        default=None,
        description="Prefill/decode worker IDs and DP ranks when recorded.",
    )
    replay: AgentReplayMetrics | None = Field(
        default=None, description="Text-free replay metadata for Mooncake/mocker."
    )


class AgentToolEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tool_call_id: str = Field(description="Harness-provided tool call identifier.")
    tool_class: str = Field(description="Tool class/category name.")
    status: (
        Literal[
            "running",
            "succeeded",
            "ok",
            "success",
            "error",
            "failed",
            "cancelled",
            "canceled",
            "timeout",
        ]
        | None
    ) = Field(
        default=None,
        description="Terminal status when present. Dynamo-written files only "
        "ever contain the canonical serde names (running/succeeded/error/"
        "cancelled) -- the Rust aliases ('ok', 'success', 'failed', "
        "'canceled', 'timeout'; request_trace/types.rs) are deserialize-only "
        "leniency for hand-authored traces and are NOT canonicalized here.",
    )
    duration_ms: float | None = Field(
        default=None, description="Wall-clock duration in milliseconds."
    )


class AgentTraceRecord(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    schema_: Literal["dynamo.request.trace.v1"] = Field(
        alias="schema",
        description="Trace schema discriminator. Aliased to the wire field 'schema' "
        "because it shadows pydantic.BaseModel.schema. Use model_dump(by_alias=True) "
        "to round-trip.",
    )
    event_type: Literal["request_end", "tool_start", "tool_end", "tool_error"] = Field(
        description="Trace event kind."
    )
    event_time_unix_ms: int = Field(
        description="Event wall-clock time in Unix epoch milliseconds."
    )
    event_source: Literal["dynamo", "harness"] | None = Field(
        default=None,
        description="Producer identity. Optional in the current schema and absent "
        "on replay-only records.",
    )
    agent_context: AgentContext | None = Field(
        default=None,
        description="Session identity attached to the event. Optional in the current "
        "schema and absent on replay-only records.",
    )
    request: AgentRequestMetrics | None = Field(
        default=None, description="Populated for request_end events."
    )
    tool: AgentToolEvent | None = Field(
        default=None, description="Populated for tool_* events."
    )


class DynamoTraceReadError(ValueError):
    """Raised when a trace file/directory can't be parsed."""


_SEGMENT_PATTERN = re.compile(r"^(.+?)\.(\d{6,})\.jsonl\.gz$")


def _dir_segment_sort_key(p: Path) -> tuple[str, int]:
    """Numeric segment order within a shared prefix; plain names sort by name.

    Lexicographic name order breaks at the 7-digit rollover
    (``1000000`` < ``999999``); dynamo's writer widens the index naturally, so
    sort segment-shaped names by ``(prefix, int(index))``.
    """
    m = _SEGMENT_PATTERN.match(p.name)
    if m is None:
        return (p.name, -1)
    return (m.group(1), int(m.group(2)))


def discover_segments(path: Path) -> list[Path]:
    """Return ordered list of `.NNNNNN.jsonl.gz` segments sharing the prefix.

    If `path` is a plain `.jsonl` or `.jsonl.gz` file, returns `[path]`.
    If `path` is a directory, returns all sorted `*.jsonl` and `*.jsonl.gz` files
    inside (segments share a prefix and sort numerically by segment number).
    If `path` matches a segmented prefix (e.g. `/tmp/dynamo-trace`), returns all
    segments whose names match `<prefix>.NNNNNN.jsonl.gz`. Mirroring dynamo's
    own segment naming (`telemetry/jsonl_gz.rs::segment_path`), a trailing
    `.jsonl.gz` / `.jsonl` on a non-existent prefix path is stripped first, so
    the configured `DYN_REQUEST_TRACE_OUTPUT_PATH` value works verbatim.
    """
    p = path
    if p.is_file():
        return [p]
    if p.is_dir():
        out = sorted(
            list(p.glob("*.jsonl")) + list(p.glob("*.jsonl.gz")),
            key=_dir_segment_sort_key,
        )
        if not out:
            raise DynamoTraceReadError(
                f"{p}: no .jsonl or .jsonl.gz files in directory"
            )
        return out
    parent = p.parent
    if not parent.is_dir():
        raise DynamoTraceReadError(f"{p}: not a file/dir, and parent isn't a directory")
    prefix = p.name
    for suffix in (".jsonl.gz", ".jsonl"):
        if prefix.endswith(suffix):
            prefix = prefix[: -len(suffix)]
            break
    candidates = [
        c for c in parent.glob(f"{prefix}.*.jsonl.gz") if _SEGMENT_PATTERN.match(c.name)
    ]
    out = sorted(
        candidates,
        key=lambda x: int(_SEGMENT_PATTERN.match(x.name).group(2)),  # type: ignore[union-attr]
    )
    if not out:
        raise DynamoTraceReadError(
            f"{p}: no matching segments found ({prefix}.*.jsonl.gz)"
        )
    return out


def _open_segment(path: Path) -> io.TextIOBase:
    # .lower(): detection sniffing is case-insensitive, so a .JSONL.GZ capture
    # that passed can_load must not be opened as utf-8 text here.
    if path.suffix.lower() == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def unwrap_sink_envelope(raw: dict[str, Any]) -> dict[str, Any]:
    """Unwrap dynamo's file-sink line envelope, passing bare records through.

    Both dynamo file sinks (``jsonl`` and ``jsonl_gz``) wrap every line as
    ``{"timestamp": <ms since writer start>, "event": {<record>}}``
    (``telemetry/jsonl_gz.rs::GzipEntry``, ``recorder.rs``); only the stderr
    sink emits the bare record. Bare records (hand-authored fixtures, older
    captures) are detected by their required top-level ``schema`` key.
    """
    if "schema" not in raw and isinstance(raw.get("event"), dict):
        return raw["event"]
    return raw


def iter_raw_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream raw JSON dicts (sink envelope unwrapped) from a file/prefix/dir."""
    p = Path(path)
    for segment in discover_segments(p):
        with _open_segment(segment) as f:
            try:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        rec = unwrap_sink_envelope(orjson.loads(line))
                    except orjson.JSONDecodeError as e:
                        raise DynamoTraceReadError(
                            f"{segment}: invalid JSON line: {e}"
                        ) from e
                    # Skip non-record marker lines the S3 uploader sidecar appends
                    # (e.g. a trailing {"verification": "trace-s3-uploader"}): a real
                    # dynamo record always carries the schema discriminator, so a
                    # schema-less line is a control marker, not a dropped record.
                    if "schema" not in rec:
                        continue
                    yield rec
            except (EOFError, gzip.BadGzipFile, zlib.error) as e:
                # A SIGKILL mid-append leaves a truncated final gzip member
                # (EOFError); bit rot or a partial copy corrupts the deflate
                # stream (zlib.error); a non-gzip file behind a .gz name fails
                # the header check (BadGzipFile).
                raise DynamoTraceReadError(
                    f"{segment}: truncated or corrupt gzip stream (capture "
                    f"interrupted mid-flush, partial copy, or not gzip?): {e}"
                ) from e
            except UnicodeDecodeError as e:
                raise DynamoTraceReadError(
                    f"{segment}: not valid UTF-8 JSONL (binary or gzip bytes "
                    f"behind a .jsonl name?): {e}"
                ) from e


def iter_trace_records(
    path: Path | str,
    *,
    event_types: Iterable[str] | None = None,
    session_id: str | None = None,
    time_range_ms: tuple[int, int] | None = None,
) -> Iterator[AgentTraceRecord]:
    """Stream typed `AgentTraceRecord` from a trace file/dir/prefix.

    Filters apply at parse time so unwanted records skip Pydantic validation.
    `time_range_ms` is inclusive on both bounds.
    """
    et_set = set(event_types) if event_types is not None else None
    lo, hi = time_range_ms if time_range_ms is not None else (None, None)
    for raw in iter_raw_records(path):
        if et_set is not None and raw.get("event_type") not in et_set:
            continue
        ts = raw.get("event_time_unix_ms")
        if lo is not None and (not isinstance(ts, int) or ts < lo):
            continue
        if hi is not None and (not isinstance(ts, int) or ts > hi):
            continue
        ac = raw.get("agent_context") or {}
        if session_id is not None and ac.get("session_id") != session_id:
            continue
        try:
            yield AgentTraceRecord.model_validate(raw)
        except Exception as e:
            raise DynamoTraceReadError(
                f"failed to parse trace record: {e!s} (raw keys: {sorted(raw.keys())})"
            ) from e


__all__ = [
    "AgentContext",
    "AgentReplayMetrics",
    "AgentRequestMetrics",
    "AgentToolEvent",
    "AgentTraceRecord",
    "DynamoTraceAdapterError",
    "DynamoTraceReadError",
    "EmptyDynamoTraceError",
    "WorkerInfo",
    "discover_segments",
    "iter_raw_records",
    "iter_trace_records",
]
