# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    AnyStr,
    ClassVar,
    Protocol,
    runtime_checkable,
)

import msgspec
import orjson
from pydantic import (
    ConfigDict,
    PlainSerializer,
    RootModel,
    SerializeAsAny,
)
from pydantic.functional_validators import AfterValidator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import STAT_KEYS
from aiperf.common.enums import CreditPhase, MetricValueTypeT, SSEFieldType
from aiperf.common.exceptions import InvalidInferenceResultError

# NOTE: MetricRecordMetadata and metric_record_metadata_from_model live in
# aiperf.common.metric_records_wire, which imports aiperf.common.models at
# module load time. A top-level import here forms a circular that fails when
# metric_records_wire is the first-entry module in a load chain. Type
# annotations below reference MetricRecordMetadata as a string thanks to
# `from __future__ import annotations`; the only runtime use is inside
# decode_metric_record_info_json, which does a local import.
from aiperf.common.models.dataset_models import Turn
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.common.models.trace_models import BaseTraceData, TraceDataExport
from aiperf.common.models.usage_models import Usage
from aiperf.common.types import JsonObject, MetricTagT, TimeSliceT
from aiperf.common.utils import load_json_str

if TYPE_CHECKING:
    from aiperf.common.metric_records_wire import MetricRecordMetadata

_logger = AIPerfLogger(__name__)


@dataclass(slots=True, kw_only=True)
class MetricResult:
    """The result values of a single metric.

    Slotted dataclass — shared type for msgspec envelopes
    (``RealtimeMetricsMessage.metrics``, ``ProfileResults.records``) and
    Pydantic (``ProfileResults`` under ``BenchmarkResultsResponse``) via
    ``__pydantic_config__``.

    Carries every JsonMetricResult percentile/stat directly — historically
    inherited, but a msgspec.Struct cannot subclass Pydantic BaseModel, so the
    fields are duplicated here (see ``to_json_result`` for the conversion
    back to the Pydantic JSON-export shape).
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    tag: MetricTagT
    header: str
    unit: str
    count: int | None = None
    # The most recent value of the metric (realtime dashboard display only).
    current: float | int | None = None
    sum: int | float | None = None
    avg: float | None = None
    p1: float | None = None
    p5: float | None = None
    p10: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    p90: float | None = None
    p95: float | None = None
    p99: float | None = None
    min: int | float | None = None
    max: int | float | None = None
    std: float | None = None

    def __post_init__(self) -> None:
        # Callers sometimes pass a BaseMetricUnit enum (str-backed but with a
        # custom __hash__) where a plain str is expected. Pydantic used to
        # coerce this on validation; msgspec does not. Collapse to str so
        # downstream set/dict comparisons keyed on the unit continue to work.
        if type(self.unit) is not str and isinstance(self.unit, str):
            self.unit = str.__str__(self.unit)

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


@dataclass(frozen=True, slots=True)
class MetricValue:
    """The value of a metric converted to display units for export."""

    value: MetricValueTypeT
    """The numeric metric value in display units."""

    unit: str
    """The display unit label (e.g. 'ms', 'tokens/s')."""


@dataclass(slots=True, kw_only=True)
class ProfileResults:
    """The results of a profile run.

    Slotted dataclass — shared type for msgspec
    (``ProfileResultsMessage.profile_results``, the /api/results HTTP
    payload encoded via msgspec) and Pydantic
    (``BenchmarkResultsResponse.results`` via ``ProcessRecordsResult``).

    Every field including ``was_cancelled=False`` and ``error_summary=[]``
    is serialized on the wire (historical ``omit_defaults=False`` semantics)
    because downstream consumers expect them — dataclasses always emit
    every field.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    completed: int
    start_ns: int
    end_ns: int
    records: list[MetricResult] | None = None
    timeslice_metric_results: dict[TimeSliceT, list[MetricResult]] | None = None
    total_expected: int | None = None
    was_cancelled: bool = False
    error_summary: list[ErrorDetailsCount] = field(default_factory=list)

    def get(self, tag: MetricTagT) -> MetricResult | None:
        """Get a metric result by tag, if it exists."""
        for record in self.records or []:
            if record.tag == tag:
                return record
        return None


@dataclass(slots=True, kw_only=True)
class ProcessRecordsResult:
    """Result of the process records command.

    Slotted dataclass — the last user of ``PydanticStructMixin`` and the
    leaf that held it in place. Now shared natively between msgspec
    (``ProcessRecordsResultMessage.results``) and Pydantic
    (``BenchmarkResultsResponse.results``).
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    results: ProfileResults
    errors: list[ErrorDetails] = field(default_factory=list)

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


class SSEField(
    msgspec.Struct,
    kw_only=True,
    omit_defaults=True,
):
    """Lightweight field in an SSE message.

    A msgspec.Struct for memory efficiency under streaming load: each SSE
    message can have multiple fields, and with thousands of concurrent
    requests each generating hundreds of chunks, any per-field allocation
    overhead shows up in peak RSS.
    """

    name: str
    """The name of the field. e.g. 'data', 'event', 'id', 'retry', 'comment'."""

    value: str | None = None
    """The value of the field."""


class TextResponse(
    msgspec.Struct,
    tag_field="response_type",
    tag="text",
    kw_only=True,
    omit_defaults=True,
):
    """Raw text response from an inference client including an optional content type.

    Carries a ``response_type`` tag so RequestRecord.responses can route
    dicts to the correct tagged-union variant on decode.
    """

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


class BinaryResponse(
    msgspec.Struct,
    tag_field="response_type",
    tag="binary",
    kw_only=True,
    omit_defaults=True,
):
    """Raw binary response from an inference client for non-text content types."""

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


class SSEMessage(
    msgspec.Struct,
    tag_field="response_type",
    tag="sse",
    kw_only=True,
    omit_defaults=True,
):
    """Individual SSE message from an SSE stream. Delimited by \\n\\n.

    Uses msgspec.Struct for memory efficiency under streaming load.
    """

    perf_ns: int
    """The performance timestamp of the message in nanoseconds (perf_counter_ns)."""

    packets: list[SSEField] = msgspec.field(default_factory=list)
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
                field_name = str(SSEFieldType.COMMENT)

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


class MetricRecordInfo(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """The full info of a metric record including the metadata, metrics, and error for export."""

    metadata: MetricRecordMetadata
    """Record metadata (timestamps, credit info, phase)."""

    metrics: dict[str, MetricValue]
    """Computed metric values keyed by metric tag."""

    trace_data: TraceDataExport | None = None
    """Optional trace data captured via a trace config."""

    error: ErrorDetails | None = None
    """Error details if the underlying request failed."""

    def to_json_bytes(self) -> bytes:
        return _METRIC_RECORD_INFO_ENCODER.encode(self)


class RawRecordInfo(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """The full info of a raw record including the request record for export."""

    metadata: MetricRecordMetadata
    """Record metadata (timestamps, credit info, phase)."""

    start_perf_ns: int
    """Request start timestamp in nanoseconds (perf_counter_ns)."""

    payload: dict[str, Any]
    """The serialized request payload sent to the inference server."""

    request_headers: dict[str, str] | None = None
    """HTTP request headers, if captured."""

    status: int | None = None
    """HTTP response status code."""

    response_headers: dict[str, str] | None = None
    """HTTP response headers, if captured."""

    responses: list[Any]
    """Raw response objects from the inference server."""

    error: ErrorDetails | None = None
    """Error details if the request failed."""

    def to_json_bytes(self) -> bytes:
        return _RAW_RECORD_INFO_ENCODER.encode(self)


def _record_info_enc_hook(obj: Any) -> Any:
    # MetricValue is a dataclass, which msgspec encodes natively — no hook
    # needed. Only the Pydantic fallback below is load-bearing: TraceDataExport
    # (and its AioHttpTraceDataExport subtype) plus ErrorDetails are Pydantic
    # final-export models, and msgspec can't serialize them directly.
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True, mode="json")
    raise TypeError(f"Unsupported record artifact type: {type(obj)}")


_METRIC_RECORD_INFO_ENCODER = msgspec.json.Encoder(enc_hook=_record_info_enc_hook)
_RAW_RECORD_INFO_ENCODER = msgspec.json.Encoder(enc_hook=_record_info_enc_hook)


def decode_metric_record_info_json(data: str | bytes) -> MetricRecordInfo:
    from aiperf.common.metric_records_wire import metric_record_metadata_from_model

    payload = orjson.loads(data)
    trace_data = payload.get("trace_data")
    return MetricRecordInfo(
        metadata=metric_record_metadata_from_model(payload["metadata"]),
        metrics={
            key: MetricValue(**value) for key, value in payload["metrics"].items()
        },
        trace_data=TraceDataExport.model_validate(trace_data) if trace_data else None,
        error=msgspec.convert(payload["error"], ErrorDetails)
        if payload.get("error")
        else None,
    )


def decode_raw_record_info_json(data: str | bytes) -> RawRecordInfo:
    from aiperf.common.metric_records_wire import metric_record_metadata_from_model

    payload = orjson.loads(data)
    return RawRecordInfo(
        metadata=metric_record_metadata_from_model(payload["metadata"]),
        start_perf_ns=payload["start_perf_ns"],
        payload=payload["payload"],
        request_headers=payload.get("request_headers"),
        status=payload.get("status"),
        response_headers=payload.get("response_headers"),
        responses=payload["responses"],
        error=msgspec.convert(payload["error"], ErrorDetails)
        if payload.get("error")
        else None,
    )
