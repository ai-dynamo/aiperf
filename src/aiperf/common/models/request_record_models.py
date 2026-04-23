# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import time
from typing import Any

import msgspec

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.enums import CreditPhase
from aiperf.common.exceptions import InvalidInferenceResultError
from aiperf.common.models.dataset_models import Turn
from aiperf.common.models.error_models import ErrorDetails
from aiperf.common.models.inference_response_models import (
    BinaryResponse,
    SSEMessage,
    TextResponse,
)
from aiperf.common.models.trace_models import BaseTraceData

_logger = AIPerfLogger(__name__)


class RequestInfo(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
):
    """Info about a request.

    Mutable msgspec struct: the worker populates many fields
    incrementally after construction. Endpoints/transports access the
    benchmark config via their own ``self.run.cfg`` — it is never
    embedded here, which keeps the struct msgspec-native.
    """

    turn_index: int
    credit_num: int
    credit_phase: CreditPhase
    x_request_id: str
    x_correlation_id: str
    conversation_id: str
    turns: list[Turn] = msgspec.field(default_factory=list)
    endpoint_headers: dict[str, str] = msgspec.field(default_factory=dict)
    endpoint_params: dict[str, str] = msgspec.field(default_factory=dict)
    session_num: int | None = None
    cancel_after_ns: int | None = None
    system_message: str | None = None
    user_context_message: str | None = None
    drop_perf_ns: int | None = None
    credit_issued_ns: int | None = None
    credit_received_ns: int | None = None
    is_final_turn: bool = True
    url_index: int | None = None


class RequestRecord(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
):
    """Record of a request with its associated responses.

    Mutable msgspec struct: transport code sets fields (``end_perf_ns``,
    ``status``, ``error``, ``cancellation_perf_ns`` etc.) on the record
    as the request progresses. The records-manager later drops
    ``responses`` / ``turns`` to None to free memory.
    """

    request_info: RequestInfo | None = None
    request_headers: dict[str, str] | None = None
    model_name: str | None = None
    timestamp_ns: int = msgspec.field(default_factory=time.time_ns)
    start_perf_ns: int = msgspec.field(default_factory=time.perf_counter_ns)
    end_perf_ns: int | None = None
    recv_start_perf_ns: int | None = None
    status: int | None = None
    # Msgspec-tagged union (tag_field="response_type") — each leaf is a
    # msgspec.Struct. The records-manager nulls
    # the field after parsing, so it must accept None.
    responses: list[SSEMessage | TextResponse | BinaryResponse] | None = msgspec.field(
        default_factory=list
    )
    error: ErrorDetails | None = None
    credit_drop_latency: int | None = None
    cancellation_perf_ns: int | None = None
    clock_offset_ns: int | None = None
    trace_data: BaseTraceData | None = None
    # Records-manager nulls this to free memory after parsing.
    turns: list[Turn] | None = msgspec.field(default_factory=list)
    # Populated by the inference_wire rehydration path when raw-export is
    # enabled; stored here temporarily before the raw-record writer moves it
    # to RawRecordInfo. Omitted from the wire by default.
    raw_payload: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Parity with the former Pydantic field_validator on trace_data:
        # route dict payloads to the correct BaseTraceData subclass via
        # msgspec tagged-union decoding (see BaseTraceData.from_json).
        if isinstance(self.trace_data, dict):
            self.trace_data = BaseTraceData.from_json(self.trace_data)

    @property
    def was_cancelled(self) -> bool:
        """Check if the request was cancelled."""
        return self.cancellation_perf_ns is not None

    # TODO: Most of these properties will be removed once we have proper record handling and metrics.

    @property
    def has_error(self) -> bool:
        """Check if the request record has an error."""
        return self.error is not None

    @property
    def valid(self) -> bool:
        """Check if the request record is valid by ensuring that the start time
        and response timestamps are within valid ranges.

        Returns:
            bool: True if the record is valid, False otherwise.
        """
        return not self.has_error and (
            0 <= self.start_perf_ns < sys.maxsize
            and self.responses is not None
            and len(self.responses) > 0
            and all(0 < response.perf_ns < sys.maxsize for response in self.responses)
        )

    def create_error_from_invalid(self) -> None:
        """Convert any invalid request records to error records for combined processing."""
        if not self.valid and not self.has_error:
            _logger.debug(
                lambda: f"Converting invalid request record to error record: {self}"
            )
            err = InvalidInferenceResultError("Invalid inference result")
            if not self.responses:
                err.add_note("No responses were received")
            if self.start_perf_ns <= 0 or self.start_perf_ns >= sys.maxsize:
                err.add_note(
                    f"Start perf ns timestamp is invalid: {self.start_perf_ns}"
                )
            for i, response in enumerate(self.responses or ()):
                if response.perf_ns <= 0 or response.perf_ns >= sys.maxsize:
                    err.add_note(
                        f"Response {i} perf ns timestamp is invalid: {response.perf_ns}"
                    )
            self.error = ErrorDetails.from_exception(err)
