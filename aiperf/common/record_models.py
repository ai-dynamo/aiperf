# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import time
from typing import Any

from pydantic import BaseModel, Field, SerializeAsAny
from pydantic.dataclasses import dataclass

from aiperf.common.enums import SSEFieldType


# Temporary Record class to be used by the ConsoleExporter.
# TODO: Remove once the actual Records classes are fully implemented.
@dataclass
class ResultsRecord:
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
    """Raw text response from a inference client including an optional content type."""

    content_type: str | None = Field(
        default=None,
        description="The content type of the response. e.g. 'text/plain', 'application/json'.",
    )
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
        """Check if the error details are equal by comparing the code, type, and message."""
        if not isinstance(other, ErrorDetails):
            return False
        return (
            self.code == other.code
            and self.type == other.type
            and self.message == other.message
        )

    def __hash__(self) -> int:
        """Hash the error details by hashing the code, type, and message."""
        return hash((self.code, self.type, self.message))

    @classmethod
    def from_exception(cls, e: Exception) -> "ErrorDetails":
        """Create an error details object from an exception."""
        return cls(
            type=e.__class__.__name__,
            message=str(e),
        )


class ErrorDetailsCount(BaseModel):
    """Count of error details."""

    error_details: ErrorDetails
    count: int = Field(
        ...,
        description="The count of the error details.",
    )


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

    def extract_data_content(self) -> list[str]:
        """Extract the data contents from the SSE message as a list of strings. Note that the SSE spec specifies
        that each data content should be combined and delimited by a single \n. We have left
        it as a list to allow the caller to decide how to handle the data.

        Returns:
            list[str]: A list of strings containing the data contents of the SSE message.
        """
        return [
            packet.value
            for packet in self.packets
            if packet.name == SSEFieldType.DATA and packet.value is not None
        ]


################################################################################
# Worker Internal Models
################################################################################


class RequestRecord(BaseModel):
    """Record of a request with its associated responses.

    Attributes:
        request: The request payload.
        start_perf_ns: The start time of the request in nanoseconds (perf_counter_ns).
        end_perf_ns: The end time of the request in nanoseconds (perf_counter_ns).
        recv_start_perf_ns: The start time of the response in nanoseconds (perf_counter_ns).
        status: The HTTP status code of the request.
        responses: The raw responses received from the request.
        error: The error details if the request failed.
        delayed: Whether the request was delayed from when it was expected to be sent.
    """

    request: Any = Field(
        default=None,
        description="The request payload.",
    )
    start_perf_ns: int = Field(
        default_factory=time.perf_counter_ns,
        description="The start time of the request in nanoseconds (perf_counter_ns).",
    )
    end_perf_ns: int | None = Field(
        default=None,
        description="The end time of the request in nanoseconds (perf_counter_ns).",
    )
    recv_start_perf_ns: int | None = Field(
        default=None,
        description="The start time of the streaming response in nanoseconds (perf_counter_ns).",
    )
    status: int | None = Field(
        default=None,
        description="The HTTP status code of the response.",
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
    delayed: bool = Field(
        default=False,
        description="Whether the request was delayed from when it was expected to be sent.",
    )

    # TODO: Most of these properties will be removed once we have proper record handling.

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
        return (
            self.responses[0].perf_ns - self.start_perf_ns
            if self.start_perf_ns
            else None
        )

    @property
    def time_to_second_response_ns(self) -> int | None:
        """Get the time to the second response in nanoseconds."""
        if not self.valid or len(self.responses) < 2:
            return None
        return (
            self.responses[1].perf_ns - self.responses[0].perf_ns
            if self.responses[1].perf_ns and self.responses[0].perf_ns
            else None
        )

    @property
    def time_to_last_response_ns(self) -> int | None:
        """Get the time to the last response in nanoseconds."""
        if not self.valid:
            return None
        if self.end_perf_ns is None or self.start_perf_ns is None:
            return None
        return self.end_perf_ns - self.start_perf_ns if self.start_perf_ns else None

    @property
    def inter_token_latency_ns(self) -> float | None:
        """Get the interval between responses in nanoseconds."""
        if not self.valid or len(self.responses) < 2:
            return None

        if (
            isinstance(self.responses[-1], SSEMessage)
            and self.responses[-1].packets[-1].value == "[DONE]"
        ):
            return (
                (self.responses[-2].perf_ns - self.responses[0].perf_ns)
                / (len(self.responses) - 2)
                if self.responses[-2].perf_ns and self.responses[0].perf_ns
                else None
            )

        return (
            (self.responses[-1].perf_ns - self.responses[0].perf_ns)
            / (len(self.responses) - 1)
            if self.responses[-1].perf_ns and self.responses[0].perf_ns
            else None
        )

    def token_latency_ns(self, index: int) -> float | None:
        """Get the latency of a token in nanoseconds."""
        if not self.valid or len(self.responses) < 1:
            return None
        if index == 0:
            return (
                self.responses[0].perf_ns - self.recv_start_perf_ns
                if self.recv_start_perf_ns
                else None
            )
        return (
            self.responses[index].perf_ns - self.responses[index - 1].perf_ns
            if self.responses[index].perf_ns and self.responses[index - 1].perf_ns
            else None
        )


class ResponseData(BaseModel):
    """Base class for all response data."""

    perf_ns: int = Field(description="The performance timestamp of the response.")
    raw_text: list[str] = Field(description="The raw text of the response.")
    parsed_text: list[str | None] = Field(
        description="The parsed text of the response."
    )
    token_count: int | None = Field(
        default=None,
        description="The total number of tokens in the response from the parsed text.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="The metadata of the response."
    )


class Transaction(BaseModel):
    """
    Represents a request/response with a timestamp and associated payload.

    Attributes:
        timestamp: The time at which the transaction was recorded.
        payload: The data or content of the transaction.
    """

    timestamp: int = Field(description="The timestamp of the transaction")
    payload: Any = Field(description="The payload of the transaction")


class Record(BaseModel):
    """
    Represents a record containing a request transaction and its associated response transactions.
    Attributes:
        request: The input transaction for the record.
        responses A list of response transactions corresponding to the request.
    """

    request: Transaction = Field(description="The request transaction for the record")
    responses: list[Transaction] = Field(
        description="A list of response transactions for the record",
    )
