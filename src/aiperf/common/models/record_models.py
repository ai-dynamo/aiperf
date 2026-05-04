# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Annotated, Any, AnyStr, Protocol, runtime_checkable

import orjson
from pydantic import (
    ConfigDict,
    Field,
    PlainSerializer,
    RootModel,
    SerializeAsAny,
    field_validator,
)
from pydantic.functional_validators import AfterValidator

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
    sum: int | float | None = Field(
        default=None,
        description="The sum of all the metric values across all records",
    )

    def to_display_unit(self) -> MetricResult:
        """Convert the metric result to its display unit."""
        from aiperf.metrics.display_units import to_display_unit
        from aiperf.metrics.metric_registry import MetricRegistry

        return to_display_unit(self, MetricRegistry)

    def to_json_result(self) -> JsonMetricResult:
        """Convert the metric result to a JsonMetricResult."""
        result = JsonMetricResult(unit=self.unit)
        for stat in [
            s for s in STAT_KEYS if s != "sum"
        ]:  # sum is not included in the JsonMetricResult
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


@dataclass(slots=True)
class SSEField:
    """Lightweight field in an SSE message.

    Using dataclass(slots=True) instead of Pydantic for memory efficiency during
    high-throughput streaming. Each SSE message can have multiple fields, and with
    thousands of concurrent requests each generating hundreds of chunks, Pydantic overhead
    was the #1 memory allocator.
    """

    name: SSEFieldType | str
    """The name of the field. e.g. 'data', 'event', 'id', 'retry', 'comment'."""

    value: str | None = None
    """The value of the field."""


@dataclass(slots=True)
class TextResponse:
    """Raw text response from an inference client including an optional content type."""

    # Reject extra fields so Pydantic's union discrimination (e.g. in
    # RequestRecord.responses) doesn't match the wrong dataclass type.
    __pydantic_config__ = ConfigDict(extra="forbid")

    perf_ns: int
    """The performance timestamp of the response in nanoseconds (perf_counter_ns)."""

    text: str
    """The raw text body of the response."""

    content_type: str | None = None
    """The content type of the response. e.g. 'text/plain', 'application/json'."""

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


@dataclass(slots=True)
class BinaryResponse:
    """Raw binary response from an inference client for non-text content types."""

    # Reject extra fields so Pydantic's union discrimination (e.g. in
    # RequestRecord.responses) doesn't match the wrong dataclass type.
    __pydantic_config__ = ConfigDict(extra="forbid")

    perf_ns: int
    """The performance timestamp of the response in nanoseconds (perf_counter_ns)."""

    raw_bytes: bytes
    """The raw binary body of the response."""

    content_type: str | None = None
    """The content type of the response. e.g. 'video/mp4', 'application/octet-stream'."""

    def get_raw(self) -> Any | None:
        """Get the raw representation of the response."""
        return self.raw_bytes

    def get_text(self) -> str | None:
        """Get the text representation of the response."""
        return None

    def get_json(self) -> JsonObject | None:
        """Get the JSON representation of the response."""
        return None


@dataclass(slots=True)
class SSEMessage:
    """Individual SSE message from an SSE stream. Delimited by \\n\\n.

    Uses dataclass(slots=True) instead of Pydantic for ~6x faster construction
    and ~10x smaller memory footprint per instance. Pydantic handles serialization
    and deserialization automatically when this appears inside Pydantic model fields.
    """

    # Reject extra fields so Pydantic's union discrimination (e.g. in
    # RequestRecord.responses) doesn't match the wrong dataclass type.
    __pydantic_config__ = ConfigDict(extra="forbid")

    perf_ns: int
    """The performance timestamp of the message in nanoseconds (perf_counter_ns)."""

    packets: list[SSEField] = field(default_factory=list)
    """The parsed SSE fields (data, event, id, retry, comment) in this message."""

    @classmethod
    def parse(cls, raw_message: AnyStr, perf_ns: int) -> SSEMessage:
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

            prev_value = message.packets[-1].value if message.packets else None
            # Detect continuation: if the previous packet's value is an incomplete
            # JSON object (starts with '{' but doesn't end with '}') and this line
            # isn't a new data field, the server embedded a literal newline in the
            # JSON value. Append this line as a continuation. This can happen when
            # ignore_eos=True and the model emits weird tokens.
            if (
                prev_value
                and prev_value.startswith("{")
                and not prev_value.endswith("}")
                and not line.startswith("data:")
            ):
                # Use \\n (JSON escape) not \n (raw newline) — the original raw 0x0A
                # byte is illegal in JSON strings; \n is the valid encoding.
                message.packets[-1].value = f"{prev_value}\\n{line}"
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

            # Spec says strip only one leading space; we strip() all whitespace
            # to normalize inconsistent servers for downstream exact comparisons
            # (e.g. "[DONE]", SSEEventType.ERROR).
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


@dataclass(slots=True)
class BaseResponseData:
    """Base class for all response data."""

    # Reject extra fields so Pydantic's union discrimination (e.g. in
    # ParsedResponse.data) doesn't match the wrong dataclass type.
    __pydantic_config__ = ConfigDict(extra="forbid")

    def get_text(self) -> str:
        """Get the text of the response."""
        return ""


@dataclass(slots=True)
class TextResponseData(BaseResponseData):
    """Parsed text response data."""

    text: str
    """The parsed text of the response."""

    def get_text(self) -> str:
        """Get the text of the response."""
        return self.text


@dataclass(slots=True)
class ReasoningResponseData(BaseResponseData):
    """Parsed reasoning response data."""

    content: str | None = None
    """The parsed content of the response."""

    reasoning: str | None = None
    """The parsed reasoning of the response."""

    def get_text(self) -> str:
        """Get the text of the response."""
        return "".join([self.reasoning or "", self.content or ""])


@dataclass(slots=True)
class ToolCallResponseData(BaseResponseData):
    """Parsed tool call response data (streaming delta or complete message)."""

    text: str
    """Combined model-generated text from tool calls (function names and arguments)."""

    def get_text(self) -> str:
        """Get the text of the response."""
        return self.text


class RAGSources(RootModel[dict[str, Any] | list[Any]]):
    """RAG sources can be either a dictionary or list format."""


@dataclass(slots=True)
class EmbeddingResponseData(BaseResponseData):
    """Parsed embedding response data."""

    embeddings: list[list[float]]
    """The embedding vectors from the response."""


@dataclass(slots=True)
class RankingsResponseData(BaseResponseData):
    """Parsed rankings response data."""

    rankings: list[dict[str, Any]]
    """The rankings results from the response."""


@dataclass(slots=True)
class ImageRetrievalResponseData(BaseResponseData):
    """Parsed image retrieval response data."""

    data: list[dict[str, Any]]
    """The image retrieval data from the response."""

    def get_text(self) -> str:
        """Get the text of the response (empty for image retrieval)."""
        return ""


@dataclass(slots=True)
class ImageDataItem:
    """Parsed image item response data."""

    url: str | None = None
    """The URL of the generated image."""

    b64_json: str | None = None
    """The base64 encoded image."""

    revised_prompt: str | None = None
    """The revised prompt that was used for image generation."""

    partial_image_index: int | None = None
    """The index of the partial image in the response."""


@dataclass(slots=True)
class ImageResponseData(BaseResponseData):
    """Parsed image response data."""

    images: list[ImageDataItem] = field(default_factory=list)
    """The generated images from the response."""

    size: str | None = None
    """The size of the generated images."""

    quality: str | None = None
    """The quality of the generated images."""

    output_format: str | None = None
    """The output format of the generated images."""

    background: str | None = None
    """The background of the generated images."""


@dataclass(slots=True)
class VideoResponseData(BaseResponseData):
    """Parsed video generation response data.

    Matches SGLang/OpenAI VideoResponse schema for async job-based video generation.
    """

    video_id: str | None = None
    """Unique identifier for the video job."""

    object: str | None = None
    """Object type, always 'video'."""

    status: str | None = None
    """Job status: queued, in_progress, completed, failed."""

    progress: int | None = None
    """Completion percentage (0-100)."""

    url: str | None = None
    """URL to download completed video (only when status=completed)."""

    size: str | None = None
    """Video resolution (e.g., '1280x720')."""

    seconds: str | None = None
    """Video duration in seconds."""

    quality: str | None = None
    """Quality setting for the generated video."""

    model: str | None = None
    """Model used for generation."""

    created_at: int | None = None
    """Unix timestamp of job creation."""

    completed_at: int | None = None
    """Unix timestamp of job completion."""

    expires_at: int | None = None
    """Unix timestamp when video assets expire."""

    inference_time_s: float | None = None
    """Generation time in seconds (SGLang metric)."""

    peak_memory_mb: float | None = None
    """Peak memory usage in MB (SGLang metric)."""

    error: dict[str, Any] | None = None
    """Error details if job failed."""


@dataclass(slots=True)
class ParsedResponse:
    """Parsed response from a inference client."""

    perf_ns: int
    """The performance timestamp of the response in nanoseconds (perf_counter_ns)."""

    # NOTE: SerializeAsAny is used to allow for generic subclass support at runtime,
    #       allowing for user-defined response data classes.
    data: SerializeAsAny[
        ReasoningResponseData
        | TextResponseData
        | ToolCallResponseData
        | EmbeddingResponseData
        | RankingsResponseData
        | ImageRetrievalResponseData
        | ImageResponseData
        | VideoResponseData
        | BaseResponseData
        | None
    ] = None
    """The parsed response data. Can be any of the response data classes,
    or a user-defined class inheriting from BaseResponseData.
    May be None for usage-only responses in streaming mode."""

    usage: (
        Annotated[dict[str, Any], AfterValidator(Usage), PlainSerializer(dict)] | None
    ) = None
    """API-reported usage information. Structure varies by provider.
    Access token counts via properties like usage.prompt_tokens, usage.completion_tokens,
    or by accessing the usage dictionary directly."""

    sources: RAGSources | None = None
    """The sources used in the RAG query of the response. Can be a dictionary of source
    documents, a list of sources, or None. Only applicable to RAG responses."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata from the response useful for analysis (rate limits, content filters, etc.)."""

    def __post_init__(self) -> None:
        # Coerce raw dicts to Usage, since dataclass __init__ doesn't run
        # Pydantic validation like BaseModel did.
        if self.usage is not None and not isinstance(self.usage, Usage):
            self.usage = Usage(self.usage)


@dataclass(slots=True)
class TokenCounts:
    """Token counts for a record."""

    input: int | None = None
    """The number of input tokens. None if token count could not be calculated."""

    output: int | None = None
    """The number of output tokens across all responses. None if token count could not be calculated."""

    reasoning: int | None = None
    """The number of reasoning tokens. None if token count could not be calculated or the model does not support reasoning."""


@dataclass
class ParsedResponseRecord:
    """Record of a request and its associated responses, already parsed and ready for metrics.

    Uses @dataclass without slots to allow @cached_property (requires __dict__).
    """

    request: RequestRecord
    """The original request record."""

    responses: list[ParsedResponse]
    """The parsed responses."""

    token_counts: TokenCounts | None = None
    """The token counts for the response. None if the token counts could not be calculated."""

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
