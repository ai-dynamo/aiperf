# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import time
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, AnyStr, Protocol, runtime_checkable

import orjson
from pydantic import (
    Field,
    RootModel,
    SerializeAsAny,
    field_validator,
)
from typing_extensions import Self

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import STAT_KEYS
from aiperf.common.enums import CreditPhase, MetricValueTypeT, SSEFieldType
from aiperf.common.exceptions import InvalidInferenceResultError
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.dataset_models import Turn
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.trace_models import BaseTraceData, TraceDataExport
from aiperf.common.models.usage_models import Usage
from aiperf.common.types import JsonObject, MetricTagT, TimeSliceT
from aiperf.common.utils import load_json_str

_logger = AIPerfLogger(__name__)


class MetricResult(JsonMetricResult):
    """The result values of a single metric."""

    tag: MetricTagT = Field(description="The unique identifier of the metric")
    # NOTE: We do not use a MetricUnitT here, as that is harder to de-serialize from JSON strings with pydantic.
    #       If we need an instance of a MetricUnitT, lookup the unit based on the tag in the MetricRegistry.
    header: str = Field(
        description="The user friendly name of the metric (e.g. 'Inter Token Latency')"
    )
    count: int | None = Field(
        default=None,
        description="The total number of records used to calculate the metric",
    )
    current: float | None = Field(
        default=None,
        description="The most recent value of the metric (used for realtime dashboard display only)",
    )

    def to_display_unit(self) -> "MetricResult":
        """Convert the metric result to its display unit."""
        from aiperf.exporters.display_units_utils import to_display_unit
        from aiperf.metrics.metric_registry import MetricRegistry

        return to_display_unit(self, MetricRegistry)

    def to_json_result(self) -> JsonMetricResult:
        """Convert the metric result to a JsonMetricResult."""
        result = JsonMetricResult(unit=self.unit)
        for stat in STAT_KEYS:
            setattr(result, stat, getattr(self, stat, None))
        return result


class MetricValue(AIPerfBaseModel):
    """The value of a metric converted to display units for export."""

    value: MetricValueTypeT
    unit: str


class MetricRecordMetadata(AIPerfBaseModel):
    """The metadata of a metric record for export."""

    session_num: int = Field(
        ...,
        description="The sequential number of the session in the benchmark. For single-turn datasets, this will be the"
        " request index. For multi-turn datasets, this will be the session index.",
    )
    x_request_id: str | None = Field(
        default=None,
        description="The X-Request-ID header of the request. This is a unique ID for the request.",
    )
    x_correlation_id: str | None = Field(
        default=None,
        description="The X-Correlation-ID header of the request. This is a shared ID for each user session/conversation in multi-turn.",
    )
    conversation_id: str | None = Field(
        default=None,
        description="The ID of the conversation (if applicable). This can be used to lookup the original request data from the inputs.json file.",
    )
    turn_index: int | None = Field(
        default=None,
        description="The index of the turn in the conversation (if applicable). This can be used to lookup the original request data from the inputs.json file.",
    )
    credit_issued_ns: int | None = Field(
        default=None,
        description="Wall clock timestamp (time.time_ns) when the credit was issued by the rate limiter. "
        "This is the control point for accurate rate measurement, before ZeroMQ transit to workers.",
    )
    request_start_ns: int = Field(
        ...,
        description="The wall clock timestamp of the request start time measured as time.time_ns().",
    )
    request_ack_ns: int | None = Field(
        default=None,
        description="The wall clock timestamp of the request acknowledgement from the server, measured as time.time_ns(), if applicable. "
        "This is only applicable to streaming requests, and servers that send 200 OK back immediately after the request is received.",
    )
    request_end_ns: int = Field(
        ...,
        description="The wall clock timestamp of the request end time measured as time.time_ns(). If the request failed, "
        "this will be the time of the error.",
    )
    worker_id: str = Field(
        ..., description="The ID of the AIPerf worker that processed the request."
    )
    record_processor_id: str = Field(
        ...,
        description="The ID of the AIPerf record processor that processed the record.",
    )
    benchmark_phase: CreditPhase = Field(
        ...,
        description="The benchmark phase of the record, either warmup or profiling.",
    )
    was_cancelled: bool = Field(
        default=False,
        description="Whether the request was cancelled during execution.",
    )
    cancellation_time_ns: int | None = Field(
        default=None,
        description="The wall clock timestamp of the request cancellation time measured as time.time_ns(), if applicable. "
        "This is only applicable to requests that were cancelled.",
    )


class ProfileResults(AIPerfBaseModel):
    """The results of a profile run."""

    records: list[MetricResult] | None = Field(
        ..., description="The records of the profile results"
    )
    timeslice_metric_results: dict[TimeSliceT, list[MetricResult]] | None = Field(
        default=None,
        description="The timeslice metric results of the profile (if using timeslice mode)",
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


@runtime_checkable
class InferenceServerResponse(Protocol):
    """Protocol for inference server response objects.

    Defines the interface for response objects that can parse themselves
    into different formats. Any object implementing these methods can be
    used as a response in the inference pipeline.

    This protocol-based approach allows for:
    - Duck typing (structural subtyping)
    - Easier testing with mocks
    - Flexibility in implementation
    - No concrete inheritance required
    """

    perf_ns: int
    """Timestamp of the response in nanoseconds (perf_counter_ns)."""

    def get_raw(self) -> Any | None:
        """Get the raw representation of the response.

        Returns:
            Raw response data or None
        """
        ...

    def get_text(self) -> str | None:
        """Get the text representation of the response.

        Returns:
            Text content or None
        """
        ...

    def get_json(self) -> JsonObject | None:
        """Get the JSON representation of the response.

        Automatically parses text content as JSON if applicable.

        Returns:
            Parsed JSON dict or None if parsing fails
        """
        ...


class BaseInferenceServerResponse(AIPerfBaseModel, ABC):
    """Base class for inference server response implementations.

    Provides shared functionality for all response types. The interface
    is defined by the InferenceServerResponse Protocol in protocols.py,
    allowing for flexible implementations and easy mocking.

    This base class ensures concrete implementations provide:
    - perf_ns: Timestamp in nanoseconds
    - get_raw(): Raw response data
    - get_text(): Text representation
    - get_json(): Parsed JSON representation
    """

    perf_ns: int = Field(
        ...,
        description="The timestamp of the response in nanoseconds (perf_counter_ns).",
    )

    @abstractmethod
    def get_raw(self) -> Any | None:
        """Get the raw representation of the response."""
        ...

    @abstractmethod
    def get_text(self) -> str | None:
        """Get the text representation of the response."""
        ...

    @abstractmethod
    def get_json(self) -> JsonObject | None:
        """Get the JSON representation of the response."""
        ...


class TextResponse(BaseInferenceServerResponse):
    """Raw text response from a inference client including an optional content type."""

    content_type: str | None = Field(
        default=None,
        description="The content type of the response. e.g. 'text/plain', 'application/json'.",
    )
    text: str = Field(
        ...,
        description="The text of the response.",
    )

    def get_raw(self) -> Any | None:
        """Get the raw representation of the response."""
        return self.text

    def get_text(self) -> str | None:
        """Get the text representation of the response."""
        return self.text

    def get_json(self) -> JsonObject | None:
        """Get the JSON representation of the response."""
        try:
            if not self.text:
                return None
            return load_json_str(self.text)
        except orjson.JSONDecodeError:
            return None


class BinaryResponse(BaseInferenceServerResponse):
    """Raw binary response from an inference client for non-text content types."""

    content_type: str | None = Field(
        default=None,
        description="The content type of the response. e.g. 'video/mp4', 'application/octet-stream'.",
    )
    raw_bytes: bytes = Field(
        ...,
        description="The raw binary content of the response.",
    )

    def get_raw(self) -> Any | None:
        """Get the raw representation of the response."""
        return self.raw_bytes

    def get_text(self) -> str | None:
        """Get the text representation of the response."""
        return None

    def get_json(self) -> JsonObject | None:
        """Get the JSON representation of the response."""
        return None


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


class SSEMessage(BaseInferenceServerResponse):
    """Individual SSE message from an SSE stream. Delimited by \n\n."""

    # Note: "fields" is a restricted keyword in pydantic
    packets: list[SSEField] = Field(
        default_factory=list,
        description="The fields contained in the message.",
    )

    @classmethod
    def parse(cls, raw_message: AnyStr, perf_ns: int) -> Self:
        """Parse a raw SSE message into an SSEMessage object.

        Parsing logic based on the official HTML SSE Living Standard:
        https://html.spec.whatwg.org/multipage/server-sent-events.html#parsing-an-event-stream

        Args:
            raw_message: The raw SSE message to parse. Can be a string or a bytes object.
            perf_ns: The performance timestamp of the response.

        Returns:
            The parsed SSEMessage.
        """
        if isinstance(raw_message, bytes):
            raw_message = raw_message.decode("utf-8")

        message = cls(perf_ns=perf_ns)
        for line in raw_message.splitlines():
            if not (line := line.strip()):
                continue

            parts = line.split(":", 1)
            if len(parts) < 2:
                # Fields without a colon have no value, so the whole line is the field name
                message.packets.append(SSEField(name=parts[0].strip(), value=None))
                continue

            field_name, value = parts

            if field_name == "":
                # Field name is empty, so this is a comment
                field_name = SSEFieldType.COMMENT

            message.packets.append(
                SSEField(name=field_name.strip(), value=value.strip())
            )

        return message

    def extract_data_content(self) -> str:
        """Extract and combine the data contents from the SSE message.

        Per the SSE spec, multiple data fields are combined and delimited by a single newline.

        Returns:
            str: The combined data contents of the SSE message, joined by newlines.
        """
        return "\n".join(
            packet.value
            for packet in self.packets
            if packet.name == SSEFieldType.DATA and packet.value
        )

    def get_raw(self) -> Any | None:
        """Get the raw representation of the SSE message."""
        return self.packets

    def get_text(self) -> str | None:
        """Get the text representation of the SSE message."""
        if data_content := self.extract_data_content():
            return data_content
        return None

    def get_json(self) -> JsonObject | None:
        """Get the JSON representation of the response."""
        data_content = None
        try:
            data_content = self.get_text()
            if data_content in ("", None, "[DONE]"):
                return None
            return load_json_str(data_content)
        except orjson.JSONDecodeError:
            return None


class RequestInfo(AIPerfBaseModel):
    """Info about a request."""

    model_endpoint: ModelEndpointInfo = Field(
        ...,
        description="The model endpoint that the request was sent to.",
    )
    turns: list[Turn] = Field(
        default_factory=list,
        description="The actual turns of the request. This will include assistant turns as well as user turns in multi-turn conversations.",
    )
    turn_index: int = Field(
        ...,
        description="The index of the turn in the conversation (if applicable).",
    )
    endpoint_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Endpoint-specific headers (auth, API keys, custom headers).",
    )
    endpoint_params: dict[str, str] = Field(
        default_factory=dict,
        description="Endpoint-specific URL query parameters.",
    )
    credit_num: int = Field(
        ...,
        ge=0,
        description="The sequential number of the credit in the credit phase. This is used to track the progress of the credit phase,"
        " as well as the order that requests are sent in.",
    )
    credit_phase: CreditPhase = Field(
        ...,
        description="The type of credit phase (either warmup or profiling)",
    )
    cancel_after_ns: int | None = Field(
        default=None,
        ge=0,
        description="The delay in nanoseconds after which the request should be cancelled, or None if the request should not be cancelled.",
    )
    x_request_id: str = Field(
        ...,
        description="The X-Request-ID header of the request. This is a unique ID for the request.",
    )
    x_correlation_id: str = Field(
        ...,
        description="The X-Correlation-ID header of the request. This is the ID of the credit drop.",
    )
    conversation_id: str = Field(
        ...,
        description="The ID of the conversation (if applicable).",
    )
    system_message: str | None = Field(
        default=None,
        description="Optional shared system message to prepend to the first turn. "
        "Extracted from conversation.system_message at request time.",
    )
    user_context_message: str | None = Field(
        default=None,
        description="Optional per-conversation user context message to prepend to the first turn. "
        "Extracted from conversation.user_context_message at request time.",
    )
    drop_perf_ns: int | None = Field(
        default=None,
        ge=0,
        description="The time in nanoseconds (perf_counter_ns) when the credit was dropped by the timing manager. "
        "This is used to calculate the credit drop latency.",
    )
    credit_issued_ns: int | None = Field(
        default=None,
        ge=0,
        description="Wall clock timestamp (time.time_ns) when the credit was issued by the rate limiter. "
        "This is the control point for accurate rate measurement, before ZeroMQ transit to workers.",
    )
    is_final_turn: bool = Field(
        default=True,
        description="Whether this is the final turn in the conversation. "
        "Used by per-conversation connection strategy to release the connection lease.",
    )
    url_index: int | None = Field(
        default=None,
        ge=0,
        description="Index of the URL to use when multiple --url values are configured. "
        "None means use the default (first) URL. Used for round-robin load balancing.",
    )


class RequestRecord(AIPerfBaseModel):
    """Record of a request with its associated responses."""

    request_info: RequestInfo | None = Field(
        default=None,
        description="The original request info.",
    )
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the request.",
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
    responses: SerializeAsAny[list[SSEMessage | TextResponse | BinaryResponse]] = Field(
        default_factory=list,
        description="The raw responses received from the request.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )
    credit_drop_latency: int | None = Field(
        default=None,
        description="The latency of the credit drop in nanoseconds from when it was first received by a Worker to when the inference request was actually sent. "
        "This can be used to trace internal latency in order to identify bottlenecks or other issues.",
        ge=0,
    )
    cancellation_perf_ns: int | None = Field(
        default=None,
        ge=0,
        description="The time in nanoseconds (perf_counter_ns) when the request was actually cancelled, if applicable.",
    )
    trace_data: SerializeAsAny[BaseTraceData] | None = Field(
        default=None,
        description="Comprehensive trace data captured via a trace config. "
        "Includes detailed timing for connection establishment, DNS resolution, request/response events, etc. "
        "The type of the trace data is determined by the transport and library used.",
    )
    turns: list[Turn] = Field(
        default_factory=list,
        description="Deep copy of the request turns. This is a copy of the turns from request_info, "
        "made to avoid mutating the original session data when stripping multimodal content.",
    )

    @field_validator("trace_data", mode="before")
    @classmethod
    def route_trace_data(cls, v: Any) -> BaseTraceData | None:
        """Route nested trace_data to correct subclass based on trace_type discriminator."""
        if isinstance(v, dict):
            return BaseTraceData.from_json(v)
        return v

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
            if len(self.responses) == 0:
                err.add_note("No responses were received")
            if self.start_perf_ns <= 0 or self.start_perf_ns >= sys.maxsize:
                err.add_note(
                    f"Start perf ns timestamp is invalid: {self.start_perf_ns}"
                )
            for i, response in enumerate(self.responses):
                if response.perf_ns <= 0 or response.perf_ns >= sys.maxsize:
                    err.add_note(
                        f"Response {i} perf ns timestamp is invalid: {response.perf_ns}"
                    )
            self.error = ErrorDetails.from_exception(err)


class BaseResponseData(AIPerfBaseModel):
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


class RAGSources(RootModel[dict[str, Any] | list[Any]]):
    """RAG sources can be either a dictionary or list format."""


class EmbeddingResponseData(BaseResponseData):
    """Parsed embedding response data."""

    embeddings: list[list[float]] = Field(
        ..., description="The embedding vectors from the response."
    )


class RankingsResponseData(BaseResponseData):
    """Parsed rankings response data."""

    rankings: list[dict[str, Any]] = Field(
        ..., description="The rankings results from the response."
    )


class ImageDataItem(AIPerfBaseModel):
    """Parsed image item response data."""

    url: str | None = Field(
        default=None,
        description="The URL of the generated image.",
    )
    b64_json: str | None = Field(
        default=None,
        description="The base64 encoded image.",
    )
    revised_prompt: str | None = Field(
        default=None,
        description="The revised prompt that was used for image generation.",
    )
    partial_image_index: int | None = Field(
        default=None,
        description="The index of the partial image in the response.",
    )


class ImageResponseData(BaseResponseData):
    """Parsed image response data."""

    images: list[ImageDataItem] = Field(
        default_factory=list, description="The generated images from the response."
    )
    size: str | None = Field(
        default=None,
        description="The size of the generated images.",
    )
    quality: str | None = Field(
        default=None,
        description="The quality of the generated images.",
    )
    output_format: str | None = Field(
        default=None,
        description="The output format of the generated images.",
    )
    background: str | None = Field(
        default=None,
        description="The background of the generated images.",
    )


class VideoResponseData(BaseResponseData):
    """Parsed video generation response data.

    Matches SGLang/OpenAI VideoResponse schema for async job-based video generation.
    """

    video_id: str | None = Field(
        default=None,
        description="Unique identifier for the video job.",
    )
    object: str | None = Field(
        default=None,
        description="Object type, always 'video'.",
    )
    status: str | None = Field(
        default=None,
        description="Job status: queued, in_progress, completed, failed.",
    )
    progress: int | None = Field(
        default=None,
        description="Completion percentage (0-100).",
    )
    url: str | None = Field(
        default=None,
        description="URL to download completed video (only when status=completed).",
    )
    size: str | None = Field(
        default=None,
        description="Video resolution (e.g., '1280x720').",
    )
    seconds: str | None = Field(
        default=None,
        description="Video duration in seconds.",
    )
    quality: str | None = Field(
        default=None,
        description="Quality setting for the generated video.",
    )
    model: str | None = Field(
        default=None,
        description="Model used for generation.",
    )
    created_at: int | None = Field(
        default=None,
        description="Unix timestamp of job creation.",
    )
    completed_at: int | None = Field(
        default=None,
        description="Unix timestamp of job completion.",
    )
    expires_at: int | None = Field(
        default=None,
        description="Unix timestamp when video assets expire.",
    )
    inference_time_s: float | None = Field(
        default=None,
        description="Generation time in seconds (SGLang metric).",
    )
    peak_memory_mb: float | None = Field(
        default=None,
        description="Peak memory usage in MB (SGLang metric).",
    )
    error: dict[str, Any] | None = Field(
        default=None,
        description="Error details if job failed.",
    )


class ParsedResponse(AIPerfBaseModel):
    """Parsed response from a inference client."""

    perf_ns: int = Field(description="The performance timestamp of the response.")
    # NOTE: SerializeAsAny is used to allow for generic subclass support at runtime,
    #       allowing for user-defined response data classes.
    data: SerializeAsAny[
        ReasoningResponseData
        | TextResponseData
        | EmbeddingResponseData
        | RankingsResponseData
        | ImageResponseData
        | VideoResponseData
        | BaseResponseData
        | None
    ] = Field(
        default=None,
        description="The parsed response data. This can be any of the response data classes, "
        "or a user-defined response data class that inherits from BaseResponseData. "
        "May be None for usage-only responses in streaming mode.",
    )
    usage: Usage | None = Field(
        default=None,
        description="API-reported usage information. Structure varies by provider. "
        "Access token counts via properties like: usage.prompt_tokens, usage.completion_tokens, or "
        "by accessing the usage dictionary directly.",
    )
    sources: RAGSources | None = Field(
        default=None,
        description="The sources used in the RAG query of the response. This can be a dictionary of source documents, "
        "a list of sources, or None. Only applicable to responses with RAG response data.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata from the response that is useful for analysis (rate limits, content filters, etc.)",
    )


class TokenCounts(AIPerfBaseModel):
    """Token counts for a record."""

    input: int | None = Field(
        default=None,
        description="The number of tokens in the input. If None, the number of tokens could not be calculated.",
    )
    output: int | None = Field(
        default=None,
        description="The number of output tokens across all responses. If None, the number of tokens could not be calculated.",
    )
    reasoning: int | None = Field(
        default=None,
        description="The number of reasoning tokens across all responses. If None, the number of tokens could not be calculated, or the model does not support reasoning.",
    )


class ParsedResponseRecord(AIPerfBaseModel):
    """Record of a request and its associated responses, already parsed and ready for metrics."""

    request: RequestRecord = Field(..., description="The original request record")
    responses: list[ParsedResponse] = Field(..., description="The parsed responses.")
    token_counts: TokenCounts | None = Field(
        default=None,
        description="The token counts for the response. If None, the token counts could not be calculated.",
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
    def content_responses(self) -> list[ParsedResponse]:
        """Get only responses with actual content (data is not None or empty).

        This excludes usage-only or [DONE] responses that may appear at the end of streaming responses.
        Useful for timing metrics that should measure content delivery.
        """
        return [response for response in self.responses if response.data]

    @property
    def has_error(self) -> bool:
        """Check if the response record has an error."""
        return self.request.has_error

    @cached_property
    def valid(self) -> bool:
        """Check if the response record is valid.

        Checks:
        - Request has no errors
        - Has at least one content response
        - Start time is before the end time
        - Response timestamps are within valid ranges

        Returns:
            bool: True if the record is valid, False otherwise.
        """
        return (
            not self.has_error
            and len(self.content_responses) > 0
            and 0 <= self.start_perf_ns < self.end_perf_ns < sys.maxsize
            and all(0 < response.perf_ns < sys.maxsize for response in self.responses)
        )

    def create_error_from_invalid(self) -> None:
        """Convert any invalid request records to error records for combined processing."""
        if not self.valid and not self.has_error:
            _logger.debug(
                lambda: f"Converting invalid request record to error record: {self}"
            )
            err = InvalidInferenceResultError("Invalid inference result")
            if len(self.responses) == 0 or len(self.content_responses) == 0:
                err.add_note(
                    "No responses with actual content were received from the server (only usage/metadata, null/empty data, or [DONE] markers)"
                )
            if self.start_perf_ns <= 0 or self.start_perf_ns >= sys.maxsize:
                err.add_note(
                    f"Start perf ns timestamp is invalid: {self.start_perf_ns}"
                )
            for i, response in enumerate(self.responses):
                if response.perf_ns <= 0 or response.perf_ns >= sys.maxsize:
                    err.add_note(
                        f"Response {i} perf ns timestamp is invalid: {response.perf_ns}"
                    )
            self.request.error = ErrorDetails.from_exception(err)


class MetricRecordInfo(AIPerfBaseModel):
    """The full info of a metric record including the metadata, metrics, and error for export."""

    metadata: MetricRecordMetadata = Field(
        ...,
        description="The metadata of the record. Should match the metadata in the MetricRecordsMessage.",
    )
    metrics: dict[str, MetricValue] = Field(
        ...,
        description="A dictionary containing all metric values along with their units.",
    )
    trace_data: SerializeAsAny[TraceDataExport] | None = Field(
        default=None,
        description="Comprehensive trace data captured via a trace config with wall-clock timestamps. "
        "Includes detailed timing for connection establishment, DNS resolution, request/response events, etc. "
        "The type of the trace data is determined by the transport and library used.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )


class RawRecordInfo(AIPerfBaseModel):
    """The full info of a raw record including the request record for export."""

    metadata: MetricRecordMetadata = Field(
        ...,
        description="The metadata of the record. Should match the metadata in the MetricRecordsMessage.",
    )
    start_perf_ns: int = Field(
        default_factory=time.perf_counter_ns,
        description="The start reference time of the request in nanoseconds used for latency calculations (perf_counter_ns).",
    )
    payload: dict[str, Any] = Field(
        ...,
        description="The raw request payload sent to the server.",
    )
    request_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the request.",
    )
    status: int | None = Field(
        default=None,
        description="The status code of the response.",
    )
    response_headers: dict[str, str] | None = Field(
        default=None,
        description="The headers of the response.",
    )
    responses: SerializeAsAny[list[SSEMessage | TextResponse | BinaryResponse]] = Field(
        ...,
        description="The raw responses received from the request.",
    )
    error: ErrorDetails | None = Field(
        default=None,
        description="The error details if the request failed.",
    )
