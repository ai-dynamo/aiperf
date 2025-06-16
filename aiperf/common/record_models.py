# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from typing import Any

from pydantic import BaseModel, Field, SerializeAsAny
from pydantic.dataclasses import dataclass

from aiperf.common.enums import CaseInsensitiveStrEnum


# Temporary Record class to be used by the ConsoleExporter.
# TODO: Remove once the actual Records classes are fully implemented.
@dataclass
class Record:
    name: str
    unit: str
    avg: float | None = None
    min: float | None = None
    max: float | None = None
    p1: float | None = None
    p5: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    std: float | None = None
    count: int | None = None
    streaming_only: bool = False


################################################################################
# Inference Client Models
################################################################################


class BaseClientConfig(BaseModel):
    """Base configuration options for all clients."""

    ...


class GenericHTTPClientConfig(BaseClientConfig):
    """Configuration options for a generic HTTP inference client."""

    url: str = Field(
        default=f"http://localhost:{os.getenv('AIPERF_PORT', 8080)}",
        description="The URL of the inference client.",
    )
    protocol: str = Field(
        default="http", description="The protocol to use for the inference client."
    )
    ssl_options: dict[str, Any] | None = Field(
        default=None,
        description="The SSL options to use for the inference client.",
    )
    timeout_ms: int = Field(
        default=300000,
        description="The timeout in milliseconds for the inference client.",
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="The headers to use for the inference client.",
    )
    api_key: str | None = Field(
        default=None,
        description="The API key to use for the inference client.",
    )


################################################################################
# Inference Client Response Models
################################################################################


class InferenceServerResponse(BaseModel):
    """Response from a inference client."""

    perf_ns: int = Field(
        ...,
        description="The timestamp of the response in nanoseconds (perf_counter_ns).",
    )


class TextResponse(InferenceServerResponse):
    """Response from a inference client."""

    text: str = Field(
        ...,
        description="The text of the response.",
    )


class ErrorDetails(BaseModel):
    """Encapsulates details about an error."""

    code: int | None = Field(
        default=None,
        description="The error code.",
    )
    type: str | None = Field(
        default=None,
        description="The error type.",
    )
    message: str = Field(
        ...,
        description="The error message.",
    )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ErrorDetails):
            return False
        return (
            self.code == other.code
            and self.type == other.type
            and self.message == other.message
        )

    def __hash__(self) -> int:
        return hash((self.code, self.type, self.message))


class ErrorDetailsCount(BaseModel):
    """Count of error details."""

    error_details: ErrorDetails
    count: int = Field(
        ...,
        description="The count of the error details.",
    )


class SSEFieldType(CaseInsensitiveStrEnum):
    """Field types in an SSE message."""

    DATA = "data"
    EVENT = "event"
    ID = "id"
    RETRY = "retry"
    COMMENT = "comment"


class SSEEventType(CaseInsensitiveStrEnum):
    """Event types in an SSE message. Many of these are custom and not defined by the SSE spec."""

    ERROR = "error"
    LLM_METRICS = "llm_metrics"


class SSEField(BaseModel):
    """Base model for a single field in an SSE message."""

    name: SSEFieldType | str = Field(
        ...,
        description="The name of the field. e.g. 'data', 'event', 'id', 'retry', 'comment'.",
    )
    value: str | None = Field(
        default=None,
        description="The value of the field.",
    )


class SSEMessage(InferenceServerResponse):
    """Individual SSE message from an SSE stream. Delimited by \n\n."""

    # Note: "fields" is a restricted keyword in pydantic
    packets: list[SSEField] = Field(
        default_factory=list,
        description="The fields contained in the message.",
    )


################################################################################
# Inference Data Models
################################################################################


class InferResult(BaseModel):
    """Result of an inference request."""

    id: str
    model_name: str
    model_version: str | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    client_id: int | None = None
    request_id: int | None = None
    raw_response: Any | None = None


class InferInput(BaseModel):
    """Input for an inference request."""

    name: str
    shape: list[int] | None = None
    datatype: str | None = None
    data: Any | None = None


class InferRequestOptions(BaseModel):
    """Options for an inference request."""

    sequence_id: int | None = None
    sequence_start: bool = False
    sequence_end: bool = False
    priority: int | None = None
    timeout_ms: int | None = None
    headers: dict[str, str] = Field(default_factory=dict)


################################################################################
# Worker Internal Models
################################################################################


class RequestRecord(BaseModel):
    """Record of a request."""

    start_perf_ns: int = Field(
        default_factory=time.perf_counter_ns,
        description="The start time of the request in perf_counter_ns.",
    )
    end_perf_ns: int | None = Field(
        default=None,
        description="The end time of the request in perf_counter_ns.",
    )
    recv_start_perf_ns: int | None = Field(
        default=None,
        description="The start time of the response in perf_counter_ns.",
    )

    status: int | None = Field(
        default=None,
        description="The HTTPstatus code of the request.",
    )
    # Note: we need to use SerializeAsAny to allow for generic subclass support
    responses: SerializeAsAny[
        list[InferenceServerResponse | SSEMessage | TextResponse]
    ] = Field(
        default_factory=list,
        description="The raw responses received from the request.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )
    sequence_end: bool = Field(
        default=True, description="Whether the sequence has ended."
    )
    delayed: bool = Field(default=False, description="Whether the request was delayed.")

    # TODO: Most of these properties will be removed once we have proper record handling.

    @property
    def has_null_last_response(self) -> bool:
        """Whether the last response received was null."""
        return len(self.responses) > 0 and self.responses[-1] is None

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
            and len(self.responses) > 0
            and all(0 < response.perf_ns < sys.maxsize for response in self.responses)
        )

    @property
    def time_to_first_response_ns(self) -> int | None:
        """Get the time to the first response in nanoseconds."""
        if not self.valid:
            return None
        return self.responses[0].perf_ns - self.recv_start_perf_ns

    @property
    def time_to_second_response_ns(self) -> int | None:
        """Get the time to the second response in nanoseconds."""
        if not self.valid or len(self.responses) < 2:
            return None
        # first_data = 0
        # while first_data < len(self.responses) - 1:
        #     if self.responses[first_data].packets[-1].value == "[DONE]":
        #         break
        #     first_data += 1
        return self.responses[1].perf_ns - self.responses[0].perf_ns

    @property
    def time_to_last_response_ns(self) -> int | None:
        """Get the time to the last response in nanoseconds."""
        if not self.valid:
            return None
        if self.end_perf_ns is None or self.start_perf_ns is None:
            return None
        return self.end_perf_ns - self.recv_start_perf_ns

    @property
    def inter_token_latency_ns(self) -> float | None:
        """Get the interval between responses in nanoseconds."""
        if not self.valid or len(self.responses) < 2:
            return None

        if (
            hasattr(self.responses[-1], "packets")
            and self.responses[-1].packets[-1].value == "[DONE]"
        ):
            return (self.responses[-2].perf_ns - self.responses[0].perf_ns) / (
                len(self.responses) - 2
            )

        return (self.responses[-1].perf_ns - self.responses[0].perf_ns) / (
            len(self.responses) - 1
        )

    def token_latency_ns(self, index: int) -> float | None:
        """Get the latency of a token in nanoseconds."""
        if not self.valid or len(self.responses) < 1:
            return None
        if index == 0:
            return self.responses[0].perf_ns - self.recv_start_perf_ns
        return self.responses[index].perf_ns - self.responses[index - 1].perf_ns

    def time_string(self) -> str:
        """Return a string representation of the request record."""
        lt = [
            # f"start_perf_ns={self.start_perf_ns / 1000000:.3f}ms",
            f"recv_start_perf_ns={(self.recv_start_perf_ns - self.start_perf_ns) / 1000000:.3f}ms",
            f"end_perf_ns={(self.end_perf_ns - self.start_perf_ns) / 1000000:.3f}ms",
            f"recv_duration={(self.end_perf_ns - self.recv_start_perf_ns) / 1000000:.3f}ms",
            f"duration={(self.end_perf_ns - self.start_perf_ns) / 1000000:.3f}ms",
        ]
        tt = [
            f"{self.token_latency_ns(i) / 1000000:.3f}ms"
            for i in range(len(self.responses))
        ]
        return f"RequestRecord({', '.join(lt)}, [{', '.join(tt)}])"


class Transaction(BaseModel):
    """
    Represents a request/response with a timestamp and associated payload.

    Attributes:
        timestamp: The time at which the transaction was recorded.
        payload: The data or content of the transaction.
    """

    timestamp: int = Field(description="The timestamp of the transaction")
    payload: Any = Field(description="The payload of the transaction")


# class Record(BaseModel):
#     """
#     Represents a record containing a request transaction and its associated response transactions.
#     Attributes:
#         request: The input transaction for the record.
#         responses A list of response transactions corresponding to the request.
#     """

#     request: Transaction = Field(description="The request transaction for the record")
#     responses: list[Transaction] = Field(
#         description="A list of response transactions for the record",
#     )
