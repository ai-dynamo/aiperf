# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from functools import cached_property
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    SerializeAsAny,
)

from aiperf.common.constants import NANOS_PER_SECOND
from aiperf.common.enums import CreditPhase, SSEFieldType
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.dataset_models import Turn
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount
from aiperf.common.types import MetricTagT


class MetricResult(AIPerfBaseModel):
    """The result values of a single metric."""

    tag: MetricTagT = Field(description="The unique identifier of the metric")
    # NOTE: We do not use a MetricUnitT here, as that is harder to de-serialize from JSON strings with pydantic.
    #       If we need an instance of a MetricUnitT, lookup the unit based on the tag in the MetricRegistry.
    unit: str = Field(description="The unit of the metric, e.g. 'ms'")
    header: str = Field(
        description="The user friendly name of the metric (e.g. 'Inter Token Latency')"
    )
    avg: float | None = None
    min: int | float | None = None
    max: int | float | None = None
    p1: float | None = None
    p5: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    std: float | None = None
    count: int | None = Field(
        default=None,
        description="The total number of records used to calculate the metric",
    )

    def to_display_unit(self) -> "MetricResult":
        """Convert the metric result to its display unit."""
        from aiperf.exporters.display_units_utils import to_display_unit
        from aiperf.metrics.metric_registry import MetricRegistry

        return to_display_unit(self, MetricRegistry)


class ProfileResults(AIPerfBaseModel):
    records: list[MetricResult] | None = Field(
        ..., description="The records of the profile results"
    )
    total_expected: int | None = Field(
        default=None,
        description="The total number of inference requests expected to be made (if known)",
    )
    completed: int = Field(
        ..., description="The number of inference requests completed"
    )
    start_ns: int = Field(
        ..., description="The start time of the profile run in nanoseconds"
    )
    end_ns: int = Field(
        ..., description="The end time of the profile run in nanoseconds"
    )
    was_cancelled: bool = Field(
        default=False,
        description="Whether the profile run was cancelled early",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )

    def get(self, tag: MetricTagT) -> MetricResult | None:
        """Get a metric result by tag, if it exists."""
        for record in self.records or []:
            if record.tag == tag:
                return record
        return None


class ProcessRecordsResult(AIPerfBaseModel):
    """Result of the process records command."""

    results: ProfileResults = Field(..., description="The profile results")
    errors: list[ErrorDetails] = Field(
        default_factory=list,
        description="Any error that occurred while processing the profile results",
    )

    def get(self, tag: MetricTagT) -> MetricResult | None:
        """Get a metric result by tag, if it exists."""
        return self.results.get(tag)


################################################################################
# Inference Client Response Models
################################################################################


class InferenceServerResponse(AIPerfBaseModel):
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


class SSEField(AIPerfBaseModel):
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

    def extract_data_content(self) -> str:
        """Extract the data contents from the SSE message as a list of strings. Note that the SSE spec specifies
        that each data content should be combined and delimited by a single \n.

        Returns:
            list[str]: A list of strings containing the data contents of the SSE message.
        """
        return "\n".join(
            [
                packet.value
                for packet in self.packets
                if packet.name == SSEFieldType.DATA and packet.value
            ]
        )


class RequestRecord(AIPerfBaseModel):
    """Record of a request with its associated responses."""

    turn: Turn | None = Field(
        default=None,
        description="The turn of the request, if applicable.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="The ID of the conversation (if applicable).",
    )
    turn_index: int | None = Field(
        default=None,
        ge=0,
        description="The index of the turn in the conversation (if applicable).",
    )
    model_name: str | None = Field(
        default=None,
        description="The name of the model targeted by the request.",
    )
    timestamp_ns: int = Field(
        default_factory=time.time_ns,
        description="The wall clock timestamp of the request in nanoseconds. DO NOT USE FOR LATENCY CALCULATIONS. (time.time_ns).",
    )
    start_perf_ns: int = Field(
        default_factory=time.perf_counter_ns,
        description="The start reference time of the request in nanoseconds used for latency calculations (perf_counter_ns).",
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
    # TODO: Maybe we could improve this with subclassing the responses to allow for more specific types.
    #       This would allow us to remove the SerializeAsAny and use a more specific type. Look at how we handle
    #       the CommandMessage and CommandResponse classes for an example.
    # NOTE: We need to use SerializeAsAny to allow for generic subclass support
    # NOTE: The order of the types is important, as that is the order they are type checked.
    #       Start with the most specific types and work towards the most general types.
    responses: SerializeAsAny[
        list[SSEMessage | TextResponse | InferenceServerResponse | Any]
    ] = Field(
        default_factory=list,
        description="The raw responses received from the request.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )
    delayed_ns: int | None = Field(
        default=None,
        ge=0,
        description="The number of nanoseconds the request was delayed from when it was expected to be sent, "
        "or None if the request was sent on time, or did not have a credit_drop_ns timestamp.",
    )
    credit_phase: CreditPhase = Field(
        default=CreditPhase.PROFILING,
        description="The type of credit phase (either warmup or profiling)",
    )
    credit_drop_latency: int | None = Field(
        default=None,
        description="The latency of the credit drop in nanoseconds from when it was first received by a Worker to when the inference request was actually sent. "
        "This can be used to trace internal latency in order to identify bottlenecks or other issues.",
        ge=0,
    )
    was_cancelled: bool = Field(
        default=False,
        description="Whether the request was cancelled during execution.",
    )
    cancel_after_ns: int = Field(
        default=0,
        ge=0,
        description="The delay in nanoseconds after which the request should be cancelled, as specified in the credit drop message.",
    )
    cancellation_time_ns: int | None = Field(
        default=None,
        ge=0,
        description="The time in nanoseconds (perf_counter_ns) when the request was actually cancelled, if applicable.",
    )

    @property
    def delayed(self) -> bool:
        """Check if the request was delayed."""
        return self.delayed_ns is not None and self.delayed_ns > 0

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


class BaseResponseData(BaseModel):
    """Base class for all response data."""

    def get_text(self) -> str:
        """Get the text of the response."""
        return ""


class TextResponseData(BaseResponseData):
    """Parsed text response data."""

    text: str = Field(..., description="The parsed text of the response.")

    def get_text(self) -> str:
        """Get the text of the response."""
        return self.text


class ReasoningResponseData(BaseResponseData):
    """Parsed reasoning response data."""

    content: str | None = Field(
        default=None, description="The parsed content of the response."
    )
    reasoning: str | None = Field(
        default=None, description="The parsed reasoning of the response."
    )

    def get_text(self) -> str:
        """Get the text of the response."""
        return "".join([self.reasoning or "", self.content or ""])


class EmbeddingResponseData(BaseResponseData):
    """Parsed embedding response data."""

    embeddings: list[list[float]] = Field(
        ..., description="The embedding vectors from the response."
    )

    def get_text(self) -> str:
        """Get the text of the response (empty for embeddings)."""
        return ""


class RankingsResponseData(BaseResponseData):
    """Parsed rankings response data."""

    rankings: list[dict[str, Any]] = Field(
        ..., description="The rankings results from the response."
    )

    def get_text(self) -> str:
        """Get the text of the response (empty for rankings)."""
        return ""


class ParsedResponse(AIPerfBaseModel):
    """Parsed response from a inference client."""

    perf_ns: int = Field(description="The performance timestamp of the response.")
    data: SerializeAsAny[
        ReasoningResponseData
        | TextResponseData
        | EmbeddingResponseData
        | RankingsResponseData
        | BaseResponseData
    ] = Field(..., description="The parsed response data.")


class ParsedResponseRecord(AIPerfBaseModel):
    """Record of a request and its associated responses, already parsed and ready for metrics."""

    request: RequestRecord = Field(description="The original request record")
    responses: list[ParsedResponse] = Field(description="The parsed responses.")
    input_token_count: int | None = Field(
        default=None,
        description="The number of tokens in the input. If None, the number of tokens could not be calculated.",
    )
    output_token_count: int | None = Field(
        default=None,
        description="The number of output tokens across all responses. If None, the number of tokens could not be calculated.",
    )
    reasoning_token_count: int | None = Field(
        default=None,
        description="The number of reasoning tokens across all responses. If None, the number of tokens could not be calculated, or the model does not support reasoning.",
    )

    @cached_property
    def start_perf_ns(self) -> int:
        """Get the start time of the request in nanoseconds (perf_counter_ns)."""
        return self.request.start_perf_ns

    @cached_property
    def timestamp_ns(self) -> int:
        """Get the wall clock timestamp of the request in nanoseconds. DO NOT USE FOR LATENCY CALCULATIONS. (time.time_ns)."""
        return self.request.timestamp_ns

    # TODO: How do we differentiate the end of the request vs the time of the last response?
    #       Which one should we use for the latency metrics?
    @cached_property
    def end_perf_ns(self) -> int:
        """Get the end time of the request in nanoseconds (perf_counter_ns).
        If request.end_perf_ns is not set, use the time of the last response.
        If there are no responses, use sys.maxsize.
        """
        return (
            self.request.end_perf_ns
            if self.request.end_perf_ns
            else self.responses[-1].perf_ns
            if self.responses
            else sys.maxsize
        )

    @cached_property
    def request_duration_ns(self) -> int:
        """Get the duration of the request in nanoseconds."""
        return self.end_perf_ns - self.start_perf_ns

    @cached_property
    def tokens_per_second(self) -> float | None:
        """Get the number of tokens per second of the request."""
        if self.output_token_count is None or self.request_duration_ns == 0:
            return None
        return self.output_token_count / (self.request_duration_ns / NANOS_PER_SECOND)

    @cached_property
    def has_error(self) -> bool:
        """Check if the response record has an error."""
        return self.request.has_error

    @cached_property
    def valid(self) -> bool:
        """Check if the response record is valid.

        Checks:
        - Request has no errors
        - Has at least one response
        - Start time is before the end time
        - Response timestamps are within valid ranges

        Returns:
            bool: True if the record is valid, False otherwise.
        """
        return (
            not self.has_error
            and len(self.responses) > 0
            and 0 <= self.start_perf_ns < self.end_perf_ns < sys.maxsize
            and all(0 < response.perf_ns < sys.maxsize for response in self.responses)
        )
