# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import Annotated, Any, AnyStr, Protocol, runtime_checkable

import msgspec
import orjson
from pydantic import (
    ConfigDict,
    Field,
    PlainSerializer,
    RootModel,
    SerializeAsAny,
    field_serializer,
    field_validator,
)
from pydantic.functional_validators import AfterValidator

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.common.constants import STAT_KEYS
from aiperf.common.enums import CreditPhase, MetricValueTypeT, SSEFieldType
from aiperf.common.exceptions import InvalidInferenceResultError
from aiperf.common.models.base_models import AIPerfBaseModel
from aiperf.common.models.branch_stats import BranchStats
from aiperf.common.models.dataset_models import Turn, TurnMetadata
from aiperf.common.models.error_models import ErrorDetails, ErrorDetailsCount
from aiperf.common.models.export_models import JsonMetricResult
from aiperf.common.models.extracted_payload import ExtractedPayload
from aiperf.common.models.metric_inputs import MetricInputs
from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
from aiperf.common.models.trace_models import (
    BaseTraceDataUnion,
    TraceDataExport,
)
from aiperf.common.models.usage_models import Usage
from aiperf.common.types import JsonObject, MetricTagT, TimeSliceT
from aiperf.common.utils import load_json_str

_logger = AIPerfLogger(__name__)


class MetricResult(msgspec.Struct, kw_only=True, omit_defaults=True):
    """The result values of a single metric.

    Wire- and storage-compatible ``msgspec.Struct`` (no Pydantic inheritance).
    Shares the on-wire field shape with ``JsonMetricResult`` so downstream
    JSON consumers continue to parse the same dict; ``JsonMetricResult`` stays
    Pydantic because it is exporter-side only (file-writer output) and bumping
    its schema would require a coordinated GenAI-Perf compatibility note.

    ``omit_defaults=True`` strips None-valued stat fields from the JSON wire
    so the encoded payload matches the prior Pydantic ``exclude_none=True``
    shape used by ``RealtimeMetricsMessage`` and the metric exporters.
    """

    tag: MetricTagT
    """The unique identifier of the metric."""

    header: str
    """The user friendly name of the metric (e.g. 'Inter Token Latency')."""

    # NOTE: We do not use a MetricUnitT here, as that is harder to de-serialize
    #       from JSON strings. If we need an instance of a MetricUnitT, lookup
    #       the unit based on the tag in the MetricRegistry.
    unit: str
    """The unit of the metric, e.g. 'ms' or 'requests/sec'."""

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
    count: int | None = None
    """The total number of records used to calculate the metric."""

    current: float | None = None
    """The most recent value of the metric (used for realtime dashboard display only)."""

    sum: int | float | None = None
    """The sum of all the metric values across all records."""

    def __post_init__(self) -> None:
        # Tests and downstream consumers depend on ``unit`` being a plain
        # ``str`` (the prior Pydantic shape coerced enum subclass values via
        # ``__str__`` during validation). ``MetricUnitT`` enums subclass
        # ``str`` but compare against bare strings only inside their own
        # equality; set membership of bare strings against enum members
        # fails. Force ``str()`` so callers see the wire-stable form.
        if type(self.unit) is not str:
            self.unit = str(self.unit)

    def to_display_unit(self) -> MetricResult:
        """Convert the metric result to its display unit."""
        from aiperf.metrics.display_units import to_display_unit
        from aiperf.metrics.metric_registry import MetricRegistry

        return to_display_unit(self, MetricRegistry)

    def to_json_result(self) -> JsonMetricResult:
        """Convert the metric result to a JsonMetricResult.

        `count` is omitted for non-RECORD metrics (derived/aggregate scalars),
        where it would trivially be 1 and risks being misread as the request
        count. Tags from other registries (e.g. GPU telemetry) are not in
        MetricRegistry; those keep `count` as-is. Future MetricType members
        also keep `count` by default — opt them in here explicitly.
        """
        from aiperf.common.enums import MetricType
        from aiperf.metrics.metric_registry import MetricRegistry

        metric_class = MetricRegistry.get_class_or_none(self.tag)
        is_scalar = metric_class is not None and metric_class.type in {
            MetricType.AGGREGATE,
            MetricType.DERIVED,
        }

        result = JsonMetricResult(
            unit=self.unit,
            count=None if is_scalar else self.count,
        )
        for stat in STAT_KEYS:
            setattr(result, stat, getattr(self, stat, None))
        return result


# Bridge helpers for the residual Pydantic models that still embed
# ``MetricResult`` (``ProfileResults``, ``RealtimeTelemetryMetricsMessage``,
# ``TimesliceCollectionExportData``). After every embedding model has migrated
# to msgspec these can disappear; until then a shared encoder/decoder pair
# keeps the on-wire shape consistent.
_METRIC_RESULT_ENCODER = msgspec.json.Encoder()
_METRIC_RESULT_DECODER = msgspec.json.Decoder(MetricResult)


def _metric_result_to_jsonable(value: MetricResult) -> dict[str, Any]:
    """Encode a ``MetricResult`` Struct to a plain JSON-safe dict."""
    return orjson.loads(_METRIC_RESULT_ENCODER.encode(value))


def _metric_result_from_value(value: Any) -> MetricResult:
    """Decode a Pydantic-supplied value into a ``MetricResult`` Struct."""
    if isinstance(value, MetricResult):
        return value
    if isinstance(value, dict):
        return _METRIC_RESULT_DECODER.decode(orjson.dumps(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _METRIC_RESULT_DECODER.decode(bytes(value))
    raise TypeError(
        f"MetricResult must be a MetricResult Struct, dict, or bytes; "
        f"got {type(value).__name__}"
    )


def _metric_result_list_from_value(value: Any) -> list[MetricResult]:
    """Decode a list of ``MetricResult`` from a Pydantic-supplied value."""
    if value is None:
        return []
    return [_metric_result_from_value(item) for item in value]


def _metric_result_list_to_jsonable(
    value: list[MetricResult],
) -> list[dict[str, Any]]:
    """Encode a list of ``MetricResult`` to plain JSON-safe dicts."""
    return [_metric_result_to_jsonable(item) for item in value]


class MetricValue(AIPerfBaseModel):
    """The value of a metric converted to display units for export."""

    value: MetricValueTypeT
    unit: str


class MetricRecordMetadata(msgspec.Struct, kw_only=True, omit_defaults=True):
    """The metadata of a metric record for export.

    Wire-compatible ``msgspec.Struct`` embedded by ``MetricRecordsData`` /
    ``MetricRecordsMessage`` on the records-pipeline channel and by the
    exporter-side ``MetricRecordInfo`` / ``RawRecordInfo`` JSONL records.
    """

    session_num: int
    """The sequential number of the session in the benchmark. For single-turn
    datasets, this will be the request index. For multi-turn datasets, this
    will be the session index."""

    request_start_ns: int
    """The wall clock timestamp of the request start time measured as
    time.time_ns()."""

    request_end_ns: int
    """The wall clock timestamp of the request end time measured as
    time.time_ns(). If the request failed, this will be the time of the error."""

    worker_id: str
    """The ID of the AIPerf worker that processed the request."""

    record_processor_id: str
    """The ID of the AIPerf record processor that processed the record."""

    benchmark_phase: CreditPhase
    """The benchmark phase of the record, either warmup or profiling."""

    x_request_id: str | None = None
    """The X-Request-ID header of the request. This is a unique ID for the request."""

    x_correlation_id: str | None = None
    """The X-Correlation-ID header of the request. This is a shared ID for each
    user session/conversation in multi-turn."""

    conversation_id: str | None = None
    """The ID of the conversation (if applicable). This can be used to lookup
    the original request data from the inputs.json file."""

    turn_index: int | None = None
    """The index of the turn in the conversation (if applicable). This can be
    used to lookup the original request data from the inputs.json file."""

    credit_issued_ns: int | None = None
    """Wall clock timestamp (time.time_ns) when the credit was issued by the
    rate limiter. This is the control point for accurate rate measurement,
    before ZeroMQ transit to workers."""

    request_ack_ns: int | None = None
    """The wall clock timestamp of the request acknowledgement from the server,
    measured as time.time_ns(), if applicable. This is only applicable to
    streaming requests, and servers that send 200 OK back immediately after
    the request is received."""

    was_cancelled: bool = False
    """Whether the request was cancelled during execution."""

    cancellation_time_ns: int | None = None
    """The wall clock timestamp of the request cancellation time measured as
    time.time_ns(), if applicable. This is only applicable to requests that
    were cancelled."""

    agent_depth: int = 0
    """The DAG agent depth of the session that produced this record. 0 for root
    sessions, incremented by 1 for each nested subagent fork. Use to filter
    records by DAG layer."""

    parent_correlation_id: str | None = None
    """The x_correlation_id of the parent session that spawned this record's
    session via a DAG subagent fork. None for root sessions. Use to group
    sibling branches of the same DAG."""


# Bridge helpers for the residual Pydantic models that still embed
# ``MetricRecordMetadata`` (``MetricRecordInfo``). After every embedding model
# has migrated to msgspec these can disappear.
_METRIC_RECORD_METADATA_ENCODER = msgspec.json.Encoder()
_METRIC_RECORD_METADATA_DECODER = msgspec.json.Decoder(MetricRecordMetadata)


def _metric_record_metadata_to_jsonable(
    value: MetricRecordMetadata,
) -> dict[str, Any]:
    """Encode a ``MetricRecordMetadata`` Struct to a plain JSON-safe dict."""
    return orjson.loads(_METRIC_RECORD_METADATA_ENCODER.encode(value))


def _metric_record_metadata_from_value(value: Any) -> MetricRecordMetadata:
    """Decode a Pydantic-supplied value into a ``MetricRecordMetadata`` Struct."""
    if isinstance(value, MetricRecordMetadata):
        return value
    if isinstance(value, dict):
        return _METRIC_RECORD_METADATA_DECODER.decode(orjson.dumps(value))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return _METRIC_RECORD_METADATA_DECODER.decode(bytes(value))
    raise TypeError(
        f"MetricRecordMetadata must be a MetricRecordMetadata Struct, dict, "
        f"or bytes; got {type(value).__name__}"
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
    successful_request_count: int = Field(
        default=0,
        ge=0,
        description="The number of inference requests that returned successful responses",
    )
    error_request_count: int = Field(
        default=0,
        ge=0,
        description="The number of inference requests that returned errors",
    )
    error_summary: list[ErrorDetailsCount] = Field(
        default_factory=list,
        description="A list of the unique error details and their counts",
    )
    branch_stats: BranchStats | None = Field(
        default=None,
        description="DAG branch orchestration counters for the run. "
        "None for non-DAG runs; a populated snapshot for DAG-shaped "
        "runs. Forwarded to profile_export_aiperf.json under the "
        "``branch_stats`` key when present.",
    )

    @field_validator("records", mode="before")
    @classmethod
    def _route_records(cls, v: Any) -> list[MetricResult] | None:
        """Decode the ``records`` list back into ``MetricResult`` Structs."""
        if v is None:
            return None
        return _metric_result_list_from_value(v)

    @field_serializer("records", when_used="json")
    def _encode_records(
        self, value: list[MetricResult] | None
    ) -> list[dict[str, Any]] | None:
        """Encode each ``MetricResult`` via msgspec.json for the JSON wire."""
        if value is None:
            return None
        return _metric_result_list_to_jsonable(value)

    @field_validator("timeslice_metric_results", mode="before")
    @classmethod
    def _route_timeslice_metric_results(
        cls, v: Any
    ) -> dict[TimeSliceT, list[MetricResult]] | None:
        """Decode each timeslice's ``MetricResult`` list back into Structs."""
        if v is None:
            return None
        return {
            slice_key: _metric_result_list_from_value(items)
            for slice_key, items in v.items()
        }

    @field_serializer("timeslice_metric_results", when_used="json")
    def _encode_timeslice_metric_results(
        self, value: dict[TimeSliceT, list[MetricResult]] | None
    ) -> dict[TimeSliceT, list[dict[str, Any]]] | None:
        """Encode each timeslice's ``MetricResult`` list via msgspec.json."""
        if value is None:
            return None
        return {
            slice_key: _metric_result_list_to_jsonable(items)
            for slice_key, items in value.items()
        }

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

    name: str
    """The name of the field. e.g. 'data', 'event', 'id', 'retry', 'comment'.

    Stored as plain ``str`` so msgspec can decode SSE packets back from the
    JSON wire without a union dispatch (msgspec rejects ``SSEFieldType | str``
    because both are str-like). Equality against ``SSEFieldType`` values still
    works via ``CaseInsensitiveStrEnum.__eq__``.
    """

    value: str | None = None
    """The value of the field."""


class TextResponse(msgspec.Struct, kw_only=True, tag="text"):
    """Raw text response from an inference client including an optional content type.

    msgspec.Struct with a string ``tag="text"`` so it discriminates against the
    other ``InferenceServerResponse`` variants on the records-pipeline wire.
    Constructed kwargs-only — every callsite already does this.
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


class BinaryResponse(msgspec.Struct, kw_only=True, tag="binary"):
    """Raw binary response from an inference client for non-text content types.

    msgspec.Struct: bytes are base64-encoded by msgspec.json by default, so the
    ``raw_bytes`` field survives the JSON wire trip without a custom serializer.
    """

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


class SSEMessage(msgspec.Struct, kw_only=True, tag="sse"):
    """Individual SSE message from an SSE stream. Delimited by \\n\\n.

    msgspec.Struct discriminates against the other ``InferenceServerResponse``
    variants on the records-pipeline wire via ``tag="sse"``. The embedded
    ``packets`` list of ``SSEField`` dataclasses round-trips through msgspec
    natively (msgspec handles plain dataclasses inside Structs).
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


# Tagged union for ``RequestRecord.responses`` and ``RawRecordInfo.responses``.
# msgspec dispatches by the top-level ``"type"`` field which each Struct's
# ``tag="..."`` populates. Distinct from the ``InferenceServerResponse``
# Protocol above: the Protocol describes the get_raw/get_text/get_json
# duck-typed surface that endpoint parsers consume; this union is the
# exhaustive set of concrete Structs that ride the records-pipeline wire and
# need tag-discriminated decode.
InferenceServerResponseUnion = SSEMessage | TextResponse | BinaryResponse


class RequestInfo(AIPerfBaseModel):
    """Full request info used Worker-side for transport dispatch.

    Carries everything ``inference_client`` and the endpoint plugins need to
    format and send a request: routing identity, the worker-built turn list,
    transport headers and URL params, timing fields, and the
    pre-encoded wire payload bytes. ``RequestInfo`` is purely worker-side —
    it never crosses ZMQ. The records pipeline reads its inputs from the
    flat ``MetricInputs`` struct attached to ``RequestRecord.metric_inputs``.

    Disambiguation note: aiperf has several "Context" types that are easy to
    confuse but live in distinct subsystems:

    - ``CreditContext`` (``aiperf.credit.structs``): timing-side struct the
      credit issuer attaches to a credit before the worker picks it up.
    - ``PhaseCallbackContext`` (``aiperf.credit.callback_handler``): inputs
      passed to credit-phase begin/end callbacks (phase + stats snapshot).
    - ``MetricContext`` (``aiperf.metrics.prometheus_formatter``):
      NamedTuple of label values used to format a single Prometheus sample.

    They do not interconvert; pick the one named for the subsystem you are in.
    """

    # --- Identity / routing ---------------------------------------------------

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
    conversation_id: str = Field(
        ...,
        description="The ID of the conversation (if applicable).",
    )
    turn_index: int = Field(
        ...,
        description="The index of the turn in the conversation (if applicable).",
    )
    x_request_id: str = Field(
        ...,
        description="The X-Request-ID header of the request. This is a unique ID for the request.",
    )
    x_correlation_id: str = Field(
        ...,
        description="The X-Correlation-ID header of the request. This is the ID of the credit drop.",
    )
    credit_issued_ns: int | None = Field(
        default=None,
        ge=0,
        description="Wall clock timestamp (time.time_ns) when the credit was issued by the rate limiter. "
        "This is the control point for accurate rate measurement, before ZeroMQ transit to workers.",
    )

    # --- DAG ------------------------------------------------------------------

    agent_depth: int = Field(
        default=0,
        description="The DAG agent depth of the session that produced this request. 0 for root sessions, "
        "incremented by 1 for each nested subagent fork. Sourced from the originating Credit.",
    )
    parent_correlation_id: str | None = Field(
        default=None,
        description="The x_correlation_id of the parent session that spawned this session via a DAG "
        "subagent fork. None for root sessions. Sourced from the originating Credit.",
    )

    # --- Wire payload addressing ----------------------------------------------

    payload_bytes: bytes | None = Field(
        default=None,
        description="Canonical pre-encoded JSON bytes of the request body sent to the server. "
        "Populated by ``inference_client`` before transport dispatch (or by the worker when "
        "the bytes were fetched from mmap).",
    )

    # --- Worker-side request shape (used by endpoint.format_payload) ----------

    turns: list[Turn] = Field(
        default_factory=list,
        description="The actual turns of the request. Includes assistant turns from prior "
        "rounds in multi-turn conversations. Consumed by endpoint plugins to format the wire "
        "payload. Empty for PAYLOAD_BYTES sessions (the worker fetches verbatim bytes from mmap).",
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

    # --- Transport extras ------------------------------------------------------

    model_endpoint: ModelEndpointInfo = Field(
        ...,
        description="The model endpoint that the request was sent to.",
    )
    endpoint_headers: dict[str, str] = Field(
        default_factory=dict,
        description="Endpoint-specific headers (auth, API keys, custom headers).",
    )
    endpoint_params: dict[str, str] = Field(
        default_factory=dict,
        description="Endpoint-specific URL query parameters.",
    )
    cancel_after_ns: int | None = Field(
        default=None,
        ge=0,
        description="The delay in nanoseconds after which the request should be cancelled, or None if the request should not be cancelled.",
    )
    drop_perf_ns: int | None = Field(
        default=None,
        ge=0,
        description="The time in nanoseconds (perf_counter_ns) when the credit was dropped by the timing manager. "
        "This is used to calculate the credit drop latency.",
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
    from_mmap: bool = Field(
        default=False,
        description="True when payload_bytes was fetched from mmap by the worker. "
        "Records-process can read the same bytes via its own mmap client, so the "
        "wire-side MetricInputs.payload_bytes is None on this path.",
    )


class RequestRecord(msgspec.Struct, kw_only=True, omit_defaults=True):
    """Record of a request with its associated responses.

    Wire-only msgspec.Struct. Carries the slim ``MetricInputs`` over the
    records-pipeline ZMQ channel along with response data, headers, error
    state, and trace data. All embedded msgspec.Struct types
    (``MetricInputs``, ``InferenceServerResponseUnion`` tagged union,
    ``ErrorDetails``, ``BaseTraceDataUnion`` tagged union) round-trip
    natively through ``msgspec.json`` -- no Pydantic bridges needed.

    Field semantics (descriptions moved here from per-field ``Field(...)`` notes):

    - ``metric_inputs``: Wire-only record-pipeline input populated by
      ``InferenceClient._finalize_request_record`` before the ZMQ hop. Carries
      routing identity (credit/conversation/turn IDs), DAG fields, and
      optionally inline payload bytes (``None`` on the PAYLOAD_BYTES path).
    - ``timestamp_ns``: Wall-clock timestamp (``time.time_ns``). DO NOT USE
      FOR LATENCY CALCULATIONS.
    - ``start_perf_ns`` / ``end_perf_ns``: ``perf_counter_ns`` references for
      latency math.
    - ``recv_start_perf_ns``: Start time of the streaming response.
    - ``credit_drop_latency``: Nanoseconds from worker credit receipt to
      transport dispatch. Used for internal-latency tracing.
    - ``cancellation_perf_ns``: ``perf_counter_ns`` at actual cancellation.
    - ``trace_data``: Comprehensive transport-level trace (connection
      establishment, DNS, request/response events). Subtype depends on
      transport/library.

    ``omit_defaults=True`` drops None-valued / default-empty fields from the
    JSON wire (mirrors the prior Pydantic ``exclude_none=True`` shape).
    """

    metric_inputs: MetricInputs | None = None
    request_headers: dict[str, str] | None = None
    model_name: str | None = None
    timestamp_ns: int = msgspec.field(default_factory=time.time_ns)
    start_perf_ns: int = msgspec.field(default_factory=time.perf_counter_ns)
    end_perf_ns: int | None = None
    recv_start_perf_ns: int | None = None
    status: int | None = None
    responses: list[InferenceServerResponseUnion] = msgspec.field(default_factory=list)
    error: ErrorDetails | None = None
    credit_drop_latency: int | None = None
    cancellation_perf_ns: int | None = None
    trace_data: BaseTraceDataUnion | None = None

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
    """Parsed tool-call response data (streaming delta or complete message).

    Mirrors the ``ReasoningResponseData`` shape - two fields, one for the
    type's primary content and one for any prose that arrived alongside
    it. Both contribute to client-side OSL (Output Sequence Length) via
    :meth:`get_text`; the distinct fields let downstream metrics that
    want to categorise output (e.g. "what fraction of OSL was tool-call
    dispatch?") read each portion separately.
    """

    tool_call_text: str
    """Combined model-generated text from tool calls - every call's
    ``function.name`` and ``function.arguments`` concatenated in
    ``output[]`` order."""

    content: str | None = None
    """Optional prose ``content`` emitted alongside the tool calls in the
    same chunk/message. Carries the prose portion when the model talks
    while dispatching a tool (~18% of turns in agentic traffic) so
    client-side OSL counts both portions and matches the server's
    ``usage.completion_tokens``. ``None`` when the response is pure
    tool-call (no prose accompanying the dispatch)."""

    def get_text(self) -> str:
        """Return ``content`` followed by ``tool_call_text`` - the
        combined string the tokeniser sees for this response."""
        return (self.content or "") + self.tool_call_text


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


def find_last_non_empty_usage(responses: list[ParsedResponse]) -> Usage | None:
    """Return the last response chunk's usage that has any data, walking
    the list backwards.

    Streaming chunks fall into two real-world patterns: (a) `usage = None`
    until a single final chunk carries the full usage, or (b) cumulative
    running totals where the last chunk holds the final values. Both
    collapse to "find the last non-empty Usage." A vendor never changes
    shape mid-stream and never explicitly nulls a field it had previously
    set, so a per-field walkback into earlier chunks would only matter
    for synthetic adversarial cases that don't occur in practice.

    Returns None if no chunk had any usage data. An empty Usage (`{}`) is
    falsy and treated the same as no usage.

    Used by:
    - `ParsedResponseRecord.final_usage` (cached at the record level so
      every metric reading the merged usage walks at most once per record)
    - `InferenceResultParser._compute_server_token_counts` (called before
      the record is constructed; reads input/reasoning/completion token
      counts off the same Usage to keep them mutually consistent)
    """
    for response in reversed(responses):
        if response.usage:
            return response.usage
    return None


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
    """A ``RequestRecord`` after the parser has resolved + extracted its inputs.

    Produced by ``InferenceResultParser.parse_request_record``, which is the
    single chokepoint for payload IO and JSON decode in the records pipeline:
    bytes are resolved (inline or mmap), parsed once, walked once via
    ``endpoint.extract_payload_inputs``, and the ``TurnMetadata`` is looked
    up once. The stashed ``payload_inputs`` / ``turn_metadata`` /
    ``payload_dict`` fields are how every downstream consumer (tokenizer,
    raw-record exporter, image/audio/OSL metrics) reads its data — they
    never re-resolve or re-parse.

    Uses @dataclass without slots to allow @cached_property (requires __dict__).
    """

    request: RequestRecord
    """The original request record."""

    responses: list[ParsedResponse]
    """The parsed responses."""

    token_counts: TokenCounts | None = None
    """The token counts for the response. None if the token counts could not be calculated."""

    payload_inputs: ExtractedPayload | None = None
    """Single-pass payload extraction (texts + image_count + audio_count +
    video_count + pretokenised_token_count). Populated once by the parser; read
    by tokenizer + image/audio/video metrics. None when payload couldn't be
    resolved (no mmap client, no inline bytes, invalid JSON)."""

    turn_metadata: TurnMetadata | None = None
    """TurnMetadata looked up by (conversation_id, turn_index) at parse time.
    Read by osl_mismatch (max_tokens) and audio_duration_metric. None for
    records that arrived without metric_inputs."""

    payload_dict: dict[str, Any] | None = None
    """Parsed wire payload (same content extract_payload_inputs walked).
    Stashed so raw_record_writer doesn't re-parse. None when extraction was skipped."""

    @cached_property
    def final_usage(self) -> Usage | None:
        """API-reported usage from the last streaming response chunk that had any.

        Thin wrapper around `find_last_non_empty_usage`. Cached, so the walk
        happens at most once per record regardless of how many metrics consult
        it. See the helper's docstring for the rationale behind "last
        non-empty chunk wins" instead of a per-key merge.
        """
        return find_last_non_empty_usage(self.responses)

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
    """The full info of a metric record including the metadata, metrics, and error for export.

    Remains Pydantic because it embeds Pydantic-side types
    (``dict[str, MetricValue]``, ``TraceDataExport``) that the exporter-side
    JSON-export schema owns; ``MetricRecordMetadata`` still bridges through
    msgspec while ``ErrorDetails`` is a Pydantic-compatible dataclass.
    """

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

    @field_validator("metadata", mode="before")
    @classmethod
    def _route_metadata(cls, v: Any) -> MetricRecordMetadata:
        """Decode a dict / bytes value into a ``MetricRecordMetadata`` Struct."""
        return _metric_record_metadata_from_value(v)

    @field_serializer("metadata", when_used="json")
    def _encode_metadata(self, value: MetricRecordMetadata) -> dict[str, Any]:
        """Encode the embedded ``MetricRecordMetadata`` via msgspec.json."""
        return _metric_record_metadata_to_jsonable(value)


class RawRecordInfo(msgspec.Struct, kw_only=True, omit_defaults=True):
    """The full info of a raw record including the request record for export.

    Wire-compatible ``msgspec.Struct``. All embedded types (``MetricRecordMetadata``,
    ``InferenceServerResponseUnion`` tagged union, ``ErrorDetails``) round-trip
    natively through msgspec.json -- no Pydantic bridges needed.
    """

    metadata: MetricRecordMetadata
    """The metadata of the record. Should match the metadata in the MetricRecordsMessage."""

    payload: dict[str, Any]
    """The raw request payload sent to the server."""

    responses: list[InferenceServerResponseUnion]
    """The raw responses received from the request."""

    start_perf_ns: int = msgspec.field(default_factory=time.perf_counter_ns)
    """The start reference time of the request in nanoseconds used for latency
    calculations (perf_counter_ns)."""

    request_headers: dict[str, str] | None = None
    """The headers of the request."""

    status: int | None = None
    """The status code of the response."""

    response_headers: dict[str, str] | None = None
    """The headers of the response."""

    error: ErrorDetails | None = None
    """The error details if the request failed."""
