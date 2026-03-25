# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parallel msgspec worker->record-processor wire model.

This module defines a trimmed, msgspec-based projection of the current
``InferenceResultsMessage`` payload. It is intentionally implemented in
parallel with the existing Pydantic/JSON path so we can measure size and
compatibility before switching the live transport.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import msgspec
import orjson
from msgspec import Struct

from aiperf.common.models import ErrorDetails, RequestInfo, RequestRecord, Turn
from aiperf.common.models.dataset_models import Image, Text
from aiperf.common.models.record_models import (
    BinaryResponse,
    SSEField,
    SSEMessage,
    TextResponse,
)
from aiperf.common.models.trace_models import BaseTraceData

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig


def _json_safe(value: Any) -> Any:
    """Convert dynamic values to a JSON-safe representation.

    The current live path serializes via Pydantic ``mode="json"`` + ``orjson``.
    For alternate wire measurements we mirror that constraint by round-tripping
    through orjson for dynamic payloads.
    """

    if value is None:
        return None
    return orjson.loads(orjson.dumps(value))


class WireErrorDetails(Struct, frozen=True, kw_only=True, omit_defaults=True):
    code: int | None = None
    type: str | None = None
    message: str
    cause: str | None = None
    cause_chain: tuple[str, ...] | None = None
    details: Any | None = None


class WireText(Struct, frozen=True, kw_only=True, omit_defaults=True):
    contents: tuple[str, ...] = ()


class WireTurn(Struct, frozen=True, kw_only=True, omit_defaults=True):
    model: str | None = None
    role: str | None = None
    max_tokens: int | None = None
    texts: tuple[WireText, ...] = ()
    image_count: int = 0


class WirePromptProjection(Struct, frozen=True, kw_only=True, omit_defaults=True):
    turns: tuple[WireTurn, ...] = ()
    system_message: str | None = None
    user_context_message: str | None = None


class WireRequestMetadata(Struct, frozen=True, kw_only=True, omit_defaults=True):
    credit_num: int
    session_num: int | None = None
    credit_phase: str
    x_request_id: str
    x_correlation_id: str
    conversation_id: str
    turn_index: int
    credit_issued_ns: int | None = None
    credit_received_ns: int | None = None
    requested_max_tokens: int | None = None


class WireSSEField(Struct, frozen=True, kw_only=True, omit_defaults=True):
    name: str
    value: str | None = None


class WireSSEMessage(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="k",
    tag="sse",
):
    perf_ns: int
    packets: tuple[WireSSEField, ...] = ()


class WireTextResponse(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="k",
    tag="txt",
):
    perf_ns: int
    text: str
    content_type: str | None = None


class WireBinaryResponse(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="k",
    tag="bin",
):
    perf_ns: int
    raw_bytes: bytes
    content_type: str | None = None


WireResponse: TypeAlias = WireSSEMessage | WireTextResponse | WireBinaryResponse


class InferenceWireRecord(Struct, frozen=True, kw_only=True, omit_defaults=True):
    metadata: WireRequestMetadata
    prompt: WirePromptProjection | None = None
    model_name: str | None = None
    timestamp_ns: int
    start_perf_ns: int
    end_perf_ns: int | None = None
    recv_start_perf_ns: int | None = None
    responses: tuple[WireResponse, ...] = ()
    error: WireErrorDetails | None = None
    credit_drop_latency: int | None = None
    cancellation_perf_ns: int | None = None
    clock_offset_ns: int | None = None
    trace_data: dict[str, Any] | None = None
    request_headers: dict[str, str] | None = None
    status: int | None = None
    raw_payload: dict[str, Any] | None = None


class InferenceResultsWireMessage(
    Struct,
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    tag_field="t",
    tag="iwr",
):
    service_id: str
    record: InferenceWireRecord


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
    """Project a full RequestRecord into the alternate msgspec wire model."""
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

    trace_data = None
    if include_trace_data and record.trace_data is not None:
        trace_data = record.trace_data.model_dump(exclude_none=True, mode="json")

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
    """Encode the alternate msgspec wire message as MessagePack bytes."""
    return _wire_encoder.encode(message)


def decode_inference_results_wire_message(
    data: bytes,
) -> InferenceResultsWireMessage:
    """Decode MessagePack bytes into the alternate wire message."""
    return _wire_decoder.decode(data)


def wire_record_to_request_record(
    *,
    config: BenchmarkConfig,
    wire_record: InferenceWireRecord,
) -> RequestRecord:
    """Rehydrate the alternate wire projection back into the current RequestRecord shape."""
    prompt = wire_record.prompt
    turns = [_wire_to_turn(turn) for turn in (prompt.turns if prompt else ())]

    if turns and wire_record.metadata.requested_max_tokens is not None:
        turns[-1].max_tokens = wire_record.metadata.requested_max_tokens

    request_info = RequestInfo(
        config=config,
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
        trace_data=BaseTraceData.from_json(wire_record.trace_data)
        if wire_record.trace_data is not None
        else None,
        turns=turns,
    )
    if wire_record.raw_payload is not None:
        request_record.raw_payload = wire_record.raw_payload
    return request_record


def wire_message_to_request_record(
    *,
    config: BenchmarkConfig,
    message: InferenceResultsWireMessage,
) -> tuple[str, RequestRecord]:
    """Decode the alternate envelope into the current runtime types."""
    return (
        message.service_id,
        wire_record_to_request_record(config=config, wire_record=message.record),
    )
