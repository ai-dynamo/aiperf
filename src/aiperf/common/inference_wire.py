# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Msgspec worker->record-processor wire model.

This module defines the trimmed MessagePack/msgspec payload used on the
worker->record-processor channel.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import msgspec
import orjson
from msgspec import Struct

from aiperf.common.enums import MessageType
from aiperf.common.metric_records_wire import WireErrorDetails
from aiperf.common.models import ErrorDetails, RequestInfo, RequestRecord, Turn
from aiperf.common.models.dataset_models import Image, Text
from aiperf.common.models.record_models import (
    BinaryResponse,
    SSEField,
    SSEMessage,
    TextResponse,
)
from aiperf.common.models.trace_models import AioHttpTraceData, BaseTraceData

TraceDataWireT = BaseTraceData | AioHttpTraceData


def _json_safe(value: Any) -> Any:
    """Convert dynamic values to a JSON-safe representation.

    The current live path serializes via Pydantic ``mode="json"`` + ``orjson``.
    For alternate wire measurements we mirror that constraint by round-tripping
    through orjson for dynamic payloads.
    """

    if value is None:
        return None
    return orjson.loads(orjson.dumps(value))


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


_wire_encoder = msgspec.msgpack.Encoder()
_wire_decoder = msgspec.msgpack.Decoder(type=InferenceResultsWireMessage)


def _error_to_wire(error: ErrorDetails | None) -> WireErrorDetails | None:
    if error is None:
        return None
    cause_chain = tuple(error.cause_chain) if error.cause_chain else None
    return WireErrorDetails(
        code=error.code,
        type=error.type,
        message=error.message,
        cause=error.cause,
        cause_chain=cause_chain,
        details=_json_safe(error.details),
    )


def _wire_to_error(error: WireErrorDetails | None) -> ErrorDetails | None:
    if error is None:
        return None
    return ErrorDetails(
        code=error.code,
        type=error.type,
        message=error.message,
        cause=error.cause,
        cause_chain=list(error.cause_chain) if error.cause_chain else None,
        details=error.details,
    )


def _turn_to_wire(turn: Turn) -> WireTurn:
    return WireTurn(
        model=turn.model,
        role=turn.role,
        max_tokens=turn.max_tokens,
        texts=tuple(WireText(contents=tuple(text.contents)) for text in turn.texts),
        image_count=sum(len(image.contents) for image in turn.images),
    )


def _wire_to_turn(turn: WireTurn) -> Turn:
    images = []
    if turn.image_count > 0:
        images.append(Image(contents=[f"image_{i}" for i in range(turn.image_count)]))
    return Turn(
        model=turn.model,
        role=turn.role,
        max_tokens=turn.max_tokens,
        texts=[Text(contents=list(text.contents)) for text in turn.texts],
        images=images,
    )


def _response_to_wire(
    response: SSEMessage | TextResponse | BinaryResponse,
) -> WireResponse:
    if isinstance(response, SSEMessage):
        return WireSSEMessage(
            perf_ns=response.perf_ns,
            packets=tuple(
                WireSSEField(name=str(packet.name), value=packet.value)
                for packet in response.packets
            ),
        )
    if isinstance(response, TextResponse):
        return WireTextResponse(
            perf_ns=response.perf_ns,
            text=response.text,
            content_type=response.content_type,
        )
    if isinstance(response, BinaryResponse):
        return WireBinaryResponse(
            perf_ns=response.perf_ns,
            raw_bytes=response.raw_bytes,
            content_type=response.content_type,
        )
    raise TypeError(f"Unsupported response type for wire projection: {type(response)}")


def _wire_to_response(
    response: WireResponse,
) -> SSEMessage | TextResponse | BinaryResponse:
    if isinstance(response, WireSSEMessage):
        return SSEMessage(
            perf_ns=response.perf_ns,
            packets=[
                SSEField(name=packet.name, value=packet.value)
                for packet in response.packets
            ],
        )
    if isinstance(response, WireTextResponse):
        return TextResponse(
            perf_ns=response.perf_ns,
            text=response.text,
            content_type=response.content_type,
        )
    if isinstance(response, WireBinaryResponse):
        return BinaryResponse(
            perf_ns=response.perf_ns,
            raw_bytes=response.raw_bytes,
            content_type=response.content_type,
        )
    raise TypeError(f"Unsupported wire response type: {type(response)}")


def build_inference_results_wire_message(
    *,
    service_id: str,
    record: RequestRecord,
    raw_payload: dict[str, Any] | None = None,
    include_request_headers: bool = False,
    include_status: bool = False,
    include_trace_data: bool = False,
) -> InferenceResultsWireMessage:
    """Project a full RequestRecord into the worker->RP wire model."""
    request_info = record.request_info
    if request_info is None:
        raise ValueError("RequestRecord.request_info is required for wire projection")

    source_turns = record.turns or request_info.turns or ()
    prompt_turns = tuple(_turn_to_wire(turn) for turn in source_turns)
    requested_max_tokens = source_turns[-1].max_tokens if source_turns else None

    prompt = None
    if (
        prompt_turns
        or request_info.system_message is not None
        or request_info.user_context_message is not None
    ):
        prompt = WirePromptProjection(
            turns=prompt_turns,
            system_message=request_info.system_message,
            user_context_message=request_info.user_context_message,
        )

    trace_data = record.trace_data if include_trace_data else None

    wire_record = InferenceWireRecord(
        metadata=WireRequestMetadata(
            credit_num=request_info.credit_num,
            session_num=request_info.session_num,
            credit_phase=request_info.credit_phase,
            x_request_id=request_info.x_request_id,
            x_correlation_id=request_info.x_correlation_id,
            conversation_id=request_info.conversation_id,
            turn_index=request_info.turn_index,
            credit_issued_ns=request_info.credit_issued_ns,
            credit_received_ns=request_info.credit_received_ns,
            requested_max_tokens=requested_max_tokens,
        ),
        prompt=prompt,
        model_name=record.model_name,
        timestamp_ns=record.timestamp_ns,
        start_perf_ns=record.start_perf_ns,
        end_perf_ns=record.end_perf_ns,
        recv_start_perf_ns=record.recv_start_perf_ns,
        responses=tuple(_response_to_wire(response) for response in record.responses),
        error=_error_to_wire(record.error),
        credit_drop_latency=record.credit_drop_latency,
        cancellation_perf_ns=record.cancellation_perf_ns,
        clock_offset_ns=record.clock_offset_ns,
        trace_data=trace_data,
        request_headers=dict(record.request_headers)
        if include_request_headers and record.request_headers is not None
        else None,
        status=record.status if include_status else None,
        raw_payload=_json_safe(raw_payload),
    )
    return InferenceResultsWireMessage(service_id=service_id, record=wire_record)


def encode_inference_results_wire_message(
    message: InferenceResultsWireMessage,
) -> bytes:
    """Encode the worker->RP wire message as MessagePack bytes."""
    return _wire_encoder.encode(message)


def decode_inference_results_wire_message(
    data: bytes,
) -> InferenceResultsWireMessage:
    """Decode MessagePack bytes into the worker->RP wire message."""
    return _wire_decoder.decode(data)


def wire_record_to_request_record(
    *,
    wire_record: InferenceWireRecord,
) -> RequestRecord:
    """Rehydrate the wire projection back into the current RequestRecord shape."""
    prompt = wire_record.prompt
    turns = [_wire_to_turn(turn) for turn in (prompt.turns if prompt else ())]

    if turns and wire_record.metadata.requested_max_tokens is not None:
        turns[-1].max_tokens = wire_record.metadata.requested_max_tokens

    request_info = RequestInfo(
        turns=turns,
        turn_index=wire_record.metadata.turn_index,
        credit_num=wire_record.metadata.credit_num,
        session_num=wire_record.metadata.session_num,
        credit_phase=wire_record.metadata.credit_phase,
        x_request_id=wire_record.metadata.x_request_id,
        x_correlation_id=wire_record.metadata.x_correlation_id,
        conversation_id=wire_record.metadata.conversation_id,
        system_message=prompt.system_message if prompt else None,
        user_context_message=prompt.user_context_message if prompt else None,
        credit_issued_ns=wire_record.metadata.credit_issued_ns,
        credit_received_ns=wire_record.metadata.credit_received_ns,
    )

    request_record = RequestRecord(
        request_info=request_info,
        request_headers=dict(wire_record.request_headers)
        if wire_record.request_headers is not None
        else None,
        model_name=wire_record.model_name,
        timestamp_ns=wire_record.timestamp_ns,
        start_perf_ns=wire_record.start_perf_ns,
        end_perf_ns=wire_record.end_perf_ns,
        recv_start_perf_ns=wire_record.recv_start_perf_ns,
        status=wire_record.status,
        responses=[_wire_to_response(response) for response in wire_record.responses],
        error=_wire_to_error(wire_record.error),
        credit_drop_latency=wire_record.credit_drop_latency,
        cancellation_perf_ns=wire_record.cancellation_perf_ns,
        clock_offset_ns=wire_record.clock_offset_ns,
        trace_data=wire_record.trace_data,
        turns=turns,
    )
    if wire_record.raw_payload is not None:
        request_record.raw_payload = wire_record.raw_payload
    return request_record


def wire_message_to_request_record(
    *,
    message: InferenceResultsWireMessage,
) -> tuple[str, RequestRecord]:
    """Decode the wire envelope into the current runtime types."""
    return (
        message.service_id,
        wire_record_to_request_record(wire_record=message.record),
    )
