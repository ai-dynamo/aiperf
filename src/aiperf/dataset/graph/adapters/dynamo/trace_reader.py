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
import hashlib
import io
import re
import sys
import zlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
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
        default=None, ge=0, description="Prefill DP rank when recorded."
    )
    decode_worker_id: int | None = Field(
        default=None, description="Decode worker id when recorded."
    )
    decode_dp_rank: int | None = Field(
        default=None, ge=0, description="Decode DP rank when recorded."
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
        ge=0,
        description="Prompt/input token count represented by the replay hashes.",
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
        default=None, ge=0, description="Prompt/input token count when known."
    )
    output_tokens: int | None = Field(
        default=None, ge=0, description="Final output token count when known."
    )
    cached_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Prompt tokens served from prefix/KV cache when known.",
    )
    request_received_ms: int | None = Field(
        default=None,
        ge=0,
        description="Request receive time in Unix epoch milliseconds.",
    )
    prefill_wait_time_ms: FiniteFloat | None = Field(
        default=None, ge=0, description="Time from request receipt to prefill start."
    )
    prefill_time_ms: FiniteFloat | None = Field(
        default=None, ge=0, description="Time from prefill start to first token."
    )
    ttft_ms: FiniteFloat | None = Field(
        default=None, description="Time from request receipt to first token."
    )
    total_time_ms: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description="Time from request receipt to request completion.",
    )
    avg_itl_ms: FiniteFloat | None = Field(
        default=None, description="Average inter-token latency after first token."
    )
    kv_hit_rate: FiniteFloat | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Effective KV-cache hit rate observed by the router.",
    )
    kv_transfer_estimated_latency_ms: FiniteFloat | None = Field(
        default=None,
        description="Upper-bound estimated disaggregated KV transfer latency.",
    )
    queue_depth: int | None = Field(
        default=None,
        ge=0,
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
    duration_ms: FiniteFloat | None = Field(
        default=None, ge=0, description="Wall-clock duration in milliseconds."
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
        ge=0, description="Event wall-clock time in Unix epoch milliseconds."
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


def parse_trace_line(
    line: bytes | str, *, source: str = "trace"
) -> dict[str, Any] | None:
    """Parse one wire line with JSON and envelope semantics.

    Returns ``None`` for schema-less uploader/control markers. Callers that need
    typed validation should pass the returned mapping to ``AgentTraceRecord``.
    """
    try:
        raw = orjson.loads(line)
    except orjson.JSONDecodeError as exc:
        raise DynamoTraceReadError(f"{source}: invalid JSON line: {exc}") from exc
    if not isinstance(raw, dict):
        raise DynamoTraceReadError(f"{source}: trace line must decode to an object")
    record = unwrap_sink_envelope(raw)
    if "schema" not in record:
        return None
    return record


def record_identity(record: AgentTraceRecord) -> tuple[object, ...] | None:
    """Return the serial collector's stable de-duplication key for a record."""
    ctx = record.agent_context
    session_id = ctx.session_id if ctx is not None else None
    if record.event_type == "request_end":
        if record.request is None:
            return None
        return ("request_end", session_id, record.request.request_id)
    if record.tool is None:
        return None
    return (
        record.event_type,
        session_id,
        record.tool.tool_call_id,
        record.event_time_unix_ms,
        record.tool.status,
    )


def resolve_parent(ctx: AgentContext) -> str | None:
    """Return the session's parent id, or None when it is a forest root.

    Real captures carry the subagent link in ``parent_trajectory_id``
    (``trajectory_id == session_id`` in every observed record, so the parent's
    trajectory id IS its session id); ``parent_session_id`` is a fallback for
    older / hand-authored traces and is consulted only when no
    ``parent_trajectory_id`` is present. A self-parent (parent == this session)
    is passed through verbatim by dynamo's generic header mapping and means "no
    parent", never a cycle, so it resolves to None here.

    This is the SINGLE parent authority for every consumer of the trace -- the
    chain parser's forest guard and the trace report both call it, so the
    subagent hierarchy they each report can never diverge.
    """
    parent = ctx.parent_trajectory_id or ctx.parent_session_id
    if not parent or parent == ctx.session_id:
        return None
    return parent


def synthetic_agent_context(record: AgentTraceRecord) -> AgentContext | None:
    """Create a deterministic root context for an identity-bearing request.

    Replay-only captures omit ``agent_context`` but retain Dynamo's request id.
    Such a request is an independent root session; context-free tool records
    remain ungroupable and return ``None``.
    """
    if record.event_type != "request_end" or record.request is None:
        return None
    session_id = f"request-{record.request.request_id}"
    return AgentContext(session_id=session_id, trajectory_id=session_id)


@dataclass(slots=True, frozen=True)
class DynamoNormalizedRecord:
    """Canonical session identity and parent metadata for one trace record."""

    record: AgentTraceRecord
    session_id: str
    parent_session_id: str | None
    synthetic_session: bool
    identity: tuple[object, ...] | None


@dataclass(slots=True)
class DynamoSessionSummary:
    """Metadata-only summary for one canonical session."""

    session_id: str
    parent_session_id: str | None = None
    request_end_count: int = 0
    byte_weight: int = 0
    peak_context: int = 0
    first_request_end_ms: int = sys.maxsize
    """Earliest ``request_end`` event time for this session; ``sys.maxsize`` when
    it has none. Feeds the time-ordered corpus cap, and counts REQUEST_END
    records only so it matches the serial path's per-tree arrival instant
    (whose chains are built from request_end turns)."""


@dataclass(slots=True)
class DynamoIngestScan:
    """Metadata collected by the canonical trace preflight scan."""

    physical_record_count: int = 0
    canonical_record_count: int = 0
    request_end_count: int = 0
    duplicate_count: int = 0
    skipped_record_count: int = 0
    synthetic_session_count: int = 0
    block_size: int = 16
    source_fingerprint: str | None = None
    segments: list[dict[str, int | str]] = field(default_factory=list)
    block_sizes: set[int] = field(default_factory=set)
    sessions: dict[str, DynamoSessionSummary] = field(default_factory=dict)


def normalize_dynamo_record(
    record: AgentTraceRecord,
) -> DynamoNormalizedRecord | None:
    """Apply canonical session, parent, and identity semantics to a record."""
    context = record.agent_context or synthetic_agent_context(record)
    if context is None:
        return None
    identity = record_identity(record)
    if record.agent_context is None and identity is not None:
        identity = (identity[0], context.session_id, *identity[2:])
    return DynamoNormalizedRecord(
        record=record,
        session_id=context.session_id,
        parent_session_id=resolve_parent(context),
        synthetic_session=record.agent_context is None,
        identity=identity,
    )


def scan_dynamo_trace(
    path: Path | str,
    *,
    session_id: str | None = None,
    on_record: Callable[[AgentTraceRecord], None] | None = None,
    capture_peak: bool = False,
) -> DynamoIngestScan:
    """Preflight a trace using the canonical normalization and dedup fold."""
    source_descriptor = _source_descriptors(path)
    fingerprint = hashlib.sha256(
        orjson.dumps(source_descriptor, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()
    scan = DynamoIngestScan(
        source_fingerprint=fingerprint,
        segments=[
            {"path": item["path"], "size": item["size"]} for item in source_descriptor
        ],
    )
    seen: set[tuple[object, ...]] = set()
    for record, line_size in iter_trace_records_with_size(path, session_id=session_id):
        scan.physical_record_count += 1
        if on_record is not None:
            on_record(record)
        normalized = normalize_dynamo_record(record)
        if normalized is None:
            scan.skipped_record_count += 1
            continue
        identity = normalized.identity
        if identity is not None and identity in seen:
            scan.duplicate_count += 1
            continue
        if identity is not None:
            seen.add(identity)
        scan.canonical_record_count += 1
        scan.synthetic_session_count += int(normalized.synthetic_session)
        summary = scan.sessions.setdefault(
            normalized.session_id,
            DynamoSessionSummary(
                session_id=normalized.session_id,
                parent_session_id=normalized.parent_session_id,
            ),
        )
        summary.byte_weight += line_size
        if (
            summary.parent_session_id is None
            and normalized.parent_session_id is not None
        ):
            summary.parent_session_id = normalized.parent_session_id
        if record.event_type == "request_end":
            scan.request_end_count += 1
            summary.request_end_count += 1
            summary.first_request_end_ms = min(
                summary.first_request_end_ms, record.event_time_unix_ms
            )
            if record.request is not None and record.request.replay is not None:
                scan.block_sizes.add(record.request.replay.trace_block_size)
            if capture_peak:
                from aiperf.dataset.graph.adapters.shared.peak_context import (
                    dynamo_tree_peak_context,
                )

                summary.peak_context = max(
                    summary.peak_context,
                    dynamo_tree_peak_context((record,)),
                )
    scan.block_size = next(iter(scan.block_sizes), 16)
    return scan


def ingest_sidecar_path(path: Path | str) -> Path:
    """Return the optional metadata sidecar path for a trace source."""
    return Path(f"{Path(path)}.aiperf-ingest.json")


def write_ingest_sidecar(path: Path | str, scan: DynamoIngestScan) -> Path:
    """Persist metadata-only ingest results beside a trace source."""
    sidecar = ingest_sidecar_path(path)
    payload = {
        "schema": "aiperf.dynamo.ingest.v1",
        "source_fingerprint": scan.source_fingerprint,
        "segments": scan.segments,
        "physical_record_count": scan.physical_record_count,
        "canonical_record_count": scan.canonical_record_count,
        "request_end_count": scan.request_end_count,
        "duplicate_count": scan.duplicate_count,
        "skipped_record_count": scan.skipped_record_count,
        "synthetic_session_count": scan.synthetic_session_count,
        "block_sizes": sorted(scan.block_sizes),
        "sessions": {
            session_id: {
                "session_id": summary.session_id,
                "parent_session_id": summary.parent_session_id,
                "request_end_count": summary.request_end_count,
                "byte_weight": summary.byte_weight,
                "peak_context": summary.peak_context,
                "first_request_end_ms": summary.first_request_end_ms,
            }
            for session_id, summary in scan.sessions.items()
        },
    }
    sidecar.write_bytes(
        orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE)
    )
    return sidecar


def load_ingest_sidecar(path: Path | str) -> DynamoIngestScan | None:
    """Load a matching metadata sidecar, returning ``None`` when stale/invalid."""
    from aiperf.common.path_safety import safe_read_template_path

    text = safe_read_template_path(str(ingest_sidecar_path(path)))
    if text is None:
        return None
    try:
        payload = orjson.loads(text)
        current = scan_dynamo_trace_metadata(path)
    except (TypeError, ValueError, OSError, orjson.JSONDecodeError):
        return None
    if payload.get("schema") != "aiperf.dynamo.ingest.v1":
        return None
    if payload.get("source_fingerprint") != current["source_fingerprint"]:
        return None
    try:
        sessions = {
            session_id: DynamoSessionSummary(
                session_id=str(value["session_id"]),
                parent_session_id=value.get("parent_session_id"),
                request_end_count=int(value["request_end_count"]),
                byte_weight=int(value.get("byte_weight", 0)),
                peak_context=int(value.get("peak_context", 0)),
                first_request_end_ms=int(
                    value.get("first_request_end_ms", sys.maxsize)
                ),
            )
            for session_id, value in payload["sessions"].items()
        }
        return DynamoIngestScan(
            physical_record_count=int(payload["physical_record_count"]),
            canonical_record_count=int(payload["canonical_record_count"]),
            request_end_count=int(payload["request_end_count"]),
            duplicate_count=int(payload["duplicate_count"]),
            skipped_record_count=int(payload["skipped_record_count"]),
            synthetic_session_count=int(payload["synthetic_session_count"]),
            block_size=(
                int(payload["block_sizes"][0]) if payload["block_sizes"] else 16
            ),
            block_sizes={int(size) for size in payload["block_sizes"]},
            source_fingerprint=payload["source_fingerprint"],
            segments=list(payload["segments"]),
            sessions=sessions,
        )
    except (KeyError, TypeError, ValueError):
        return None


def scan_dynamo_trace_metadata(
    path: Path | str,
) -> dict[str, str | list[dict[str, int | str]]]:
    """Return source identity metadata without parsing trace records."""
    descriptors = _source_descriptors(path)
    return {
        "source_fingerprint": hashlib.sha256(
            orjson.dumps(descriptors, option=orjson.OPT_SORT_KEYS)
        ).hexdigest(),
        "segments": [
            {"path": item["path"], "size": item["size"]} for item in descriptors
        ],
    }


def _source_descriptors(path: Path | str) -> list[dict[str, int | str]]:
    """Describe source segments for sidecar identity and display metadata."""
    return [
        {
            "path": str(segment),
            "size": segment.stat().st_size,
            "mtime_ns": segment.stat().st_mtime_ns,
        }
        for segment in discover_segments(Path(path))
    ]


def iter_session_records(
    path: Path | str,
    *,
    session_id: str | None = None,
    on_duplicate: Callable[[AgentTraceRecord], None] | None = None,
    on_no_context: Callable[[AgentTraceRecord], None] | None = None,
    synthesize_contextless_requests: bool = True,
    on_record: Callable[[AgentTraceRecord], None] | None = None,
) -> Iterator[tuple[AgentContext, AgentTraceRecord]]:
    """Stream session records, optionally synthesizing context-free roots.

    The SHARED pre-lowering fold: every consumer that groups dynamo records by
    session goes through here, so the dedup identity and the two skip classes
    are defined once rather than reimplemented per caller.

    When enabled, context-free request-end records receive a deterministic root
    context from their request id. Otherwise, or for other context-free records,
    the record is skipped. Dynamo's dual file sinks can write the SAME
    record into two files of one capture dir, so records are folded once by
    :func:`record_identity`. Both skip classes are reported through the
    optional callbacks -- callers that only need a count pass a counter
    increment; callers that need the record itself get it.
    """
    seen: set[tuple] = set()
    for record in iter_trace_records(path, session_id=session_id):
        if on_record is not None:
            on_record(record)
        normalized = normalize_dynamo_record(record)
        if normalized is None or (
            normalized.synthetic_session and not synthesize_contextless_requests
        ):
            if on_no_context is not None:
                on_no_context(record)
            continue
        ctx = AgentContext(
            session_id=normalized.session_id,
            parent_session_id=normalized.parent_session_id,
            trajectory_id=normalized.session_id,
        )
        identity = normalized.identity
        if identity is not None:
            if identity in seen:
                if on_duplicate is not None:
                    on_duplicate(record)
                continue
            seen.add(identity)
        yield ctx, record


def iter_raw_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream raw JSON dicts (sink envelope unwrapped) from a file/prefix/dir."""
    for record, _line_size in _iter_raw_records_with_size(path):
        yield record


def _iter_raw_records_with_size(
    path: Path | str,
) -> Iterator[tuple[dict[str, Any], int]]:
    """Stream raw records and their physical source line sizes."""
    p = Path(path)
    for segment in discover_segments(p):
        with _open_segment(segment) as f:
            try:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    rec = parse_trace_line(line, source=str(segment))
                    if rec is None:
                        continue
                    # Skip non-record marker lines the S3 uploader sidecar appends
                    # (e.g. a trailing {"verification": "trace-s3-uploader"}): a real
                    # dynamo record always carries the schema discriminator, so a
                    # schema-less line is a control marker, not a dropped record.
                    if "schema" not in rec:
                        continue
                    yield rec, len(raw_line)
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
    for record, _line_size in iter_trace_records_with_size(
        path,
        event_types=event_types,
        session_id=session_id,
        time_range_ms=time_range_ms,
    ):
        yield record


def iter_trace_records_with_size(
    path: Path | str,
    *,
    event_types: Iterable[str] | None = None,
    session_id: str | None = None,
    time_range_ms: tuple[int, int] | None = None,
) -> Iterator[tuple[AgentTraceRecord, int]]:
    """Stream typed records with their physical source line sizes."""
    et_set = set(event_types) if event_types is not None else None
    lo, hi = time_range_ms if time_range_ms is not None else (None, None)
    for raw, line_size in _iter_raw_records_with_size(path):
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
            yield AgentTraceRecord.model_validate(raw), line_size
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
    "DynamoIngestScan",
    "DynamoNormalizedRecord",
    "DynamoSessionSummary",
    "DynamoTraceAdapterError",
    "DynamoTraceReadError",
    "EmptyDynamoTraceError",
    "WorkerInfo",
    "discover_segments",
    "iter_raw_records",
    "iter_session_records",
    "iter_trace_records",
    "iter_trace_records_with_size",
    "ingest_sidecar_path",
    "load_ingest_sidecar",
    "parse_trace_line",
    "record_identity",
    "resolve_parent",
    "normalize_dynamo_record",
    "scan_dynamo_trace",
    "scan_dynamo_trace_metadata",
    "synthetic_agent_context",
    "write_ingest_sidecar",
]
