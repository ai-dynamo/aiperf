# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Msgspec worker->record-processor wire model.

This module defines the trimmed MessagePack/msgspec payload used on the
worker->record-processor channel. Projection + codec helpers live in
:mod:`aiperf.common.inference_wire_codec` and are re-exported here for
backwards compatibility.
"""

from __future__ import annotations

from typing import Any, TypeAlias

from msgspec import Struct

from aiperf.common.enums import MessageType
from aiperf.common.metric_records_wire import WireErrorDetails
from aiperf.common.models.trace_models import AioHttpTraceData, BaseTraceData

TraceDataWireT = BaseTraceData | AioHttpTraceData


class WireText(Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Wire-format text content from a conversation turn."""

    contents: tuple[str, ...] = ()
    """Text content strings for this text block."""


class WireTurn(Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Wire-format conversation turn."""

    model: str | None = None
    """Model name override for this turn."""

    role: str | None = None
    """Conversation role (e.g. user, assistant, system)."""

    max_tokens: int | None = None
    """Maximum tokens requested for this turn's completion."""

    texts: tuple[WireText, ...] = ()
    """Text content blocks in this turn."""

    image_count: int = 0
    """Number of images referenced in this turn (flattened from Image objects)."""


class WirePromptProjection(Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Wire-format prompt projection containing turns and system/context messages."""

    turns: tuple[WireTurn, ...] = ()
    """Conversation turns included in this prompt."""

    system_message: str | None = None
    """System message prepended to the conversation."""

    user_context_message: str | None = None
    """User context message appended to the conversation."""


class WireRequestMetadata(Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Wire-format request metadata for credit and conversation tracking."""

    credit_num: int
    """Sequential credit number (0-based request index within the phase)."""

    session_num: int | None = None
    """Sequential session/conversation number (0-based)."""

    credit_phase: str
    """Credit phase name (e.g. warmup, profile)."""

    x_request_id: str
    """Unique request identifier (X-Request-ID header)."""

    x_correlation_id: str
    """Conversation instance identifier for sticky routing (X-Correlation-ID header)."""

    conversation_id: str
    """Template conversation ID from the dataset."""

    turn_index: int
    """Index of the turn in the conversation (0-based)."""

    credit_issued_ns: int | None = None
    """Wall clock timestamp when the credit was issued."""

    credit_received_ns: int | None = None
    """Wall clock timestamp when the credit was received by the worker."""

    requested_max_tokens: int | None = None
    """Maximum tokens requested for the final turn's completion."""


class WireSSEField(Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Wire-format SSE field name/value pair."""

    name: str
    """SSE field name."""

    value: str | None = None
    """SSE field value, when present."""


class WireSSEMessage(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="k",
    tag="sse",
):
    """Wire-format SSE streaming response."""

    perf_ns: int
    """Performance counter timestamp when this SSE message was received."""

    packets: tuple[WireSSEField, ...] = ()
    """SSE field packets contained in this message."""


class WireTextResponse(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="k",
    tag="txt",
):
    """Wire-format text response."""

    perf_ns: int
    """Performance counter timestamp when this text response was received."""

    text: str
    """Response text body."""

    content_type: str | None = None
    """HTTP Content-Type of the response, when available."""


class WireBinaryResponse(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="k",
    tag="bin",
):
    """Wire-format binary response."""

    perf_ns: int
    """Performance counter timestamp when this binary response was received."""

    raw_bytes: bytes
    """Raw binary response body."""

    content_type: str | None = None
    """HTTP Content-Type of the response, when available."""


WireResponse: TypeAlias = WireSSEMessage | WireTextResponse | WireBinaryResponse


class InferenceWireRecord(Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Wire-format inference record carrying a single request's full lifecycle."""

    metadata: WireRequestMetadata
    """Credit and conversation tracking metadata."""

    prompt: WirePromptProjection | None = None
    """Prompt projection containing turns and system messages."""

    model_name: str | None = None
    """Model name used for this inference request."""

    timestamp_ns: int
    """Wall clock timestamp in nanoseconds when the request was created."""

    start_perf_ns: int
    """Performance counter timestamp at request start."""

    end_perf_ns: int | None = None
    """Performance counter timestamp at request end."""

    recv_start_perf_ns: int | None = None
    """Performance counter timestamp when the first response byte was received."""

    responses: tuple[WireResponse, ...] = ()
    """Ordered response chunks received from the inference server."""

    error: WireErrorDetails | None = None
    """Error details if the request failed."""

    credit_drop_latency: int | None = None
    """Latency in nanoseconds between credit issuance and request dispatch."""

    cancellation_perf_ns: int | None = None
    """Performance counter timestamp when cancellation was triggered."""

    clock_offset_ns: int | None = None
    """Estimated clock offset in nanoseconds for cross-process time alignment."""

    trace_data: TraceDataWireT | None = None
    """Native trace data (msgspec Struct) for plugin-specific trace fields."""

    request_headers: dict[str, str] | None = None
    """HTTP request headers sent to the inference server."""

    status: int | None = None
    """HTTP status code from the inference server response."""

    raw_payload: dict[str, Any] | None = None
    """Raw JSON payload from the inference server response."""


class InferenceResultsWireMessage(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="t",
    tag="iwr",
):
    """Wire envelope for a single inference result on the worker->RP channel."""

    message_type: MessageType = MessageType.INFERENCE_RESULTS
    """Message type discriminator."""

    service_id: str
    """Worker service identifier that produced this result."""

    record: InferenceWireRecord
    """Full inference record for a single request."""


# Re-export projection + codec helpers so existing
# ``from aiperf.common.inference_wire import ...`` callsites keep working.
from aiperf.common.inference_wire_codec import (  # noqa: E402
    build_inference_results_wire_message,
    decode_inference_results_wire_message,
    encode_inference_results_wire_message,
    wire_message_to_request_record,
    wire_record_to_request_record,
)

__all__ = [
    "InferenceResultsWireMessage",
    "InferenceWireRecord",
    "TraceDataWireT",
    "WireBinaryResponse",
    "WirePromptProjection",
    "WireRequestMetadata",
    "WireResponse",
    "WireSSEField",
    "WireSSEMessage",
    "WireText",
    "WireTextResponse",
    "WireTurn",
    "build_inference_results_wire_message",
    "decode_inference_results_wire_message",
    "encode_inference_results_wire_message",
    "wire_message_to_request_record",
    "wire_record_to_request_record",
]
