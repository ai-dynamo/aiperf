# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from typing_extensions import Self

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class AIPerfLogLevel(CaseInsensitiveStrEnum):
    """Logging levels for AIPerf output verbosity."""

    TRACE = "TRACE"
    """Most verbose. Logs all operations including ZMQ messages and internal state changes."""

    DEBUG = "DEBUG"
    """Detailed debugging information. Logs function calls and important state transitions."""

    INFO = "INFO"
    """General informational messages. Default level showing benchmark progress and results."""

    NOTICE = "NOTICE"
    """Important informational messages that are more significant than INFO but not warnings."""

    WARNING = "WARNING"
    """Warning messages for potentially problematic situations that don't prevent execution."""

    SUCCESS = "SUCCESS"
    """Success messages for completed operations and milestones."""

    ERROR = "ERROR"
    """Error messages for failures that prevent specific operations but allow continued execution."""

    CRITICAL = "CRITICAL"
    """Critical errors that may cause the benchmark to fail or produce invalid results."""


class AudioFormat(CaseInsensitiveStrEnum):
    """Audio file formats for synthetic audio generation."""

    WAV = "wav"
    """WAV format. Uncompressed audio, larger file sizes, best quality."""

    MP3 = "mp3"
    """MP3 format. Compressed audio, smaller file sizes, good quality."""


class CommAddress(CaseInsensitiveStrEnum):
    """Enum for specifying the address type for communication clients.
    This is used to lookup the address in the communication config."""

    EVENT_BUS_PROXY_FRONTEND = "event_bus_proxy_frontend"
    """Frontend address for services to publish messages to."""

    EVENT_BUS_PROXY_BACKEND = "event_bus_proxy_backend"
    """Backend address for services to subscribe to messages."""

    CREDIT_ROUTER = "credit_router"
    """Address for bidirectional ROUTER-DEALER credit routing (all timing modes)."""

    RECORDS = "records"
    """Address to send parsed records from InferenceParser to RecordManager."""

    DATASET_MANAGER_PROXY_FRONTEND = "dataset_manager_proxy_frontend"
    """Frontend address for sending requests to the DatasetManager."""

    DATASET_MANAGER_PROXY_BACKEND = "dataset_manager_proxy_backend"
    """Backend address for the DatasetManager to receive requests from clients."""

    RAW_INFERENCE_PROXY_FRONTEND = "raw_inference_proxy_frontend"
    """Frontend address for sending raw inference messages to the InferenceParser from Workers."""

    RAW_INFERENCE_PROXY_BACKEND = "raw_inference_proxy_backend"
    """Backend address for the InferenceParser to receive raw inference messages from Workers."""


class CommandType(CaseInsensitiveStrEnum):
    REALTIME_METRICS = "realtime_metrics"
    PROCESS_RECORDS = "process_records"
    PROFILE_CANCEL = "profile_cancel"
    PROFILE_COMPLETE = "profile_complete"
    PROFILE_CONFIGURE = "profile_configure"
    PROFILE_START = "profile_start"
    REGISTER_SERVICE = "register_service"
    SHUTDOWN = "shutdown"
    SHUTDOWN_WORKERS = "shutdown_workers"
    SPAWN_WORKERS = "spawn_workers"
    START_REALTIME_TELEMETRY = "start_realtime_telemetry"


class CommandResponseStatus(CaseInsensitiveStrEnum):
    ACKNOWLEDGED = "acknowledged"
    FAILURE = "failure"
    SUCCESS = "success"
    UNHANDLED = "unhandled"  # The command was received but not handled by any hook


class ConnectionReuseStrategy(CaseInsensitiveStrEnum):
    """Transport connection reuse strategy. Controls how and when connections are reused across requests."""

    POOLED = "pooled"
    """Connections are pooled and reused across all requests"""

    NEVER = "never"
    """New connection for each request, closed after response"""

    STICKY_USER_SESSIONS = "sticky-user-sessions"
    """Connection persists across turns of a multi-turn conversation, closed on final turn (enables sticky load balancing)"""


class CreditPhase(CaseInsensitiveStrEnum):
    """The type of credit phase. This is used to identify which phase of the
    benchmark the credit is being used in, for tracking and reporting purposes."""

    WARMUP = "warmup"
    """The credit phase while the warmup is active. This is used to warm up the model and
    ensure that the model is ready to be profiled."""

    PROFILING = "profiling"
    """The credit phase while profiling is active. This is the primary phase of the
    benchmark, and what is used to calculate the final results."""


class ExportLevel(CaseInsensitiveStrEnum):
    """Export level for benchmark data."""

    SUMMARY = "summary"
    """Export only aggregated/summarized metrics (default, most compact)"""

    RECORDS = "records"
    """Export per-record metrics after aggregation with display unit conversion"""

    RAW = "raw"
    """Export raw parsed records with full request/response data (most detailed)"""


class GPUTelemetryMode(CaseInsensitiveStrEnum):
    """GPU telemetry display mode."""

    SUMMARY = "summary"
    REALTIME_DASHBOARD = "realtime_dashboard"


class ImageFormat(CaseInsensitiveStrEnum):
    """Image file formats for synthetic image generation."""

    PNG = "png"
    """PNG format. Lossless compression, larger file sizes, best quality."""

    JPEG = "jpeg"
    """JPEG format. Lossy compression, smaller file sizes, good for photos."""

    RANDOM = "random"
    """Randomly select PNG or JPEG for each image."""


class IPVersion(CaseInsensitiveStrEnum):
    """IP version for HTTP socket connections."""

    V4 = "4"
    """Use IPv4 only (AF_INET). Default for most environments."""

    V6 = "6"
    """Use IPv6 only (AF_INET6). Use when connecting to IPv6-only servers."""

    AUTO = "auto"
    """Let the system choose (AF_UNSPEC). Supports both IPv4 and IPv6."""


class LifecycleState(CaseInsensitiveStrEnum):
    """This is the various states a service can be in during its lifecycle."""

    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class MediaType(CaseInsensitiveStrEnum):
    """The various types of media (e.g. text, image, audio, video)."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class MessageType(CaseInsensitiveStrEnum):
    """The various types of messages that can be sent between services.

    The message type is used to determine what Pydantic model the message maps to,
    based on the message_type field in the message model. For detailed explanations
    of each message type, go to its definition in :mod:`aiperf.common.messages`.
    """

    ALL_RECORDS_RECEIVED = "all_records_received"
    CANCEL_CREDITS = "cancel_credits"
    COMMAND = "command"
    COMMAND_RESPONSE = "command_response"
    CONNECTION_PROBE = "connection_probe"
    CONVERSATION_REQUEST = "conversation_request"
    CONVERSATION_RESPONSE = "conversation_response"
    CONVERSATION_TURN_REQUEST = "conversation_turn_request"
    CONVERSATION_TURN_RESPONSE = "conversation_turn_response"
    CREDIT_PHASE_COMPLETE = "credit_phase_complete"
    CREDIT_PHASE_PROGRESS = "credit_phase_progress"
    CREDIT_PHASE_SENDING_COMPLETE = "credit_phase_sending_complete"
    CREDIT_PHASE_START = "credit_phase_start"
    CREDIT_PHASES_CONFIGURED = "credit_phases_configured"
    CREDITS_COMPLETE = "credits_complete"
    DATASET_CONFIGURED_NOTIFICATION = "dataset_configured_notification"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    INFERENCE_RESULTS = "inference_results"
    METRIC_RECORDS = "metric_records"
    PARSED_INFERENCE_RESULTS = "parsed_inference_results"
    PROCESSING_STATS = "processing_stats"
    PROCESS_RECORDS_RESULT = "process_records_result"
    PROCESS_TELEMETRY_RESULT = "process_telemetry_result"
    PROCESS_SERVER_METRICS_RESULT = "process_server_metrics_result"
    PROFILE_PROGRESS = "profile_progress"
    PROFILE_RESULTS = "profile_results"
    REALTIME_METRICS = "realtime_metrics"
    REALTIME_TELEMETRY_METRICS = "realtime_telemetry_metrics"
    REGISTRATION = "registration"
    SERVICE_ERROR = "service_error"
    STATUS = "status"
    TELEMETRY_RECORDS = "telemetry_records"
    TELEMETRY_STATUS = "telemetry_status"
    SERVER_METRICS_RECORD = "server_metrics_record"
    SERVER_METRICS_STATUS = "server_metrics_status"
    WORKER_HEALTH = "worker_health"
    WORKER_STATUS_SUMMARY = "worker_status_summary"


class ModelSelectionStrategy(CaseInsensitiveStrEnum):
    """Strategy for selecting the model to use for the request."""

    ROUND_ROBIN = "round_robin"
    """Cycle through models in order. The nth prompt is assigned to model at index (n mod number_of_models)."""

    RANDOM = "random"
    """Randomly select a model for each prompt using uniform distribution."""


class PrometheusMetricType(CaseInsensitiveStrEnum):
    """Prometheus metric types as defined in the Prometheus exposition format.

    See: https://prometheus.io/docs/concepts/metric_types/
    """

    COUNTER = "counter"
    """Counter: A cumulative metric that represents a single monotonically increasing counter."""

    GAUGE = "gauge"
    """Gauge: A metric that represents a single numerical value that can arbitrarily go up and down."""

    HISTOGRAM = "histogram"
    """Histogram: Samples observations and counts them in configurable buckets."""

    SUMMARY = "summary"
    """Summary: Not supported for collection (quantiles are cumulative over server lifetime).

    Note: Summary metrics are intentionally skipped during collection because their
    quantiles are computed cumulatively over the entire server lifetime, making them
    unsuitable for benchmark-specific analysis. No major LLM inference servers use
    Summary metrics - they all use Histograms instead.
    """

    UNKNOWN = "unknown"
    """Unknown: Untyped metric (prometheus_client uses 'unknown' instead of 'untyped')."""

    @classmethod
    def _missing_(cls, value: Any) -> Self:
        """Handle unrecognized metric type values by returning UNKNOWN.

        Called automatically when constructing a PrometheusMetricType with a value
        that doesn't match any defined member. Attempts case-insensitive matching
        via parent class, then falls back to UNKNOWN for unrecognized types.

        This ensures robust parsing of Prometheus metrics where servers may expose
        non-standard or future metric types.

        Args:
            value: The value to match against enum members

        Returns:
            Matching enum member (case-insensitive) or PrometheusMetricType.UNKNOWN
        """
        try:
            return super()._missing_(value)
        except ValueError:
            return cls.UNKNOWN


class PromptSource(CaseInsensitiveStrEnum):
    SYNTHETIC = "synthetic"
    FILE = "file"
    PAYLOAD = "payload"


class PublicDatasetType(CaseInsensitiveStrEnum):
    """Public datasets available for benchmarking."""

    SHAREGPT = "sharegpt"
    """ShareGPT dataset from HuggingFace. Multi-turn conversational dataset with user/assistant exchanges."""


class ServerMetricsFormat(CaseInsensitiveStrEnum):
    """Format options for server metrics export.

    Controls which output files are generated for server metrics data.
    Default selection is JSON + CSV (JSONL excluded to avoid large files).
    """

    JSON = "json"
    """Export aggregated statistics in JSON hybrid format with metrics keyed by name.
    Best for: Programmatic access, CI/CD pipelines, automated analysis."""

    CSV = "csv"
    """Export aggregated statistics in CSV tabular format organized by metric type.
    Best for: Spreadsheet analysis, Excel/Google Sheets, pandas DataFrames."""

    JSONL = "jsonl"
    """Export raw time-series records in line-delimited JSON format.
    Best for: Time-series analysis, debugging, visualizing metric evolution.
    Warning: Can generate very large files for long-running benchmarks."""

    PARQUET = "parquet"
    """Export raw time-series data with delta calculations in Parquet columnar format.
    Best for: Analytics with DuckDB/pandas/Polars, efficient storage, SQL queries.
    Includes cumulative deltas from reference point for counters and histograms."""


class ServiceRegistrationStatus(CaseInsensitiveStrEnum):
    """Defines the various states a service can be in during registration with
    the SystemController."""

    UNREGISTERED = "unregistered"
    """The service is not registered with the SystemController. This is the
    initial state."""

    WAITING = "waiting"
    """The service is waiting for the SystemController to register it.
    This is a temporary state that should be followed by REGISTERED, TIMEOUT, or ERROR."""

    REGISTERED = "registered"
    """The service is registered with the SystemController."""

    TIMEOUT = "timeout"
    """The service registration timed out."""

    ERROR = "error"
    """The service registration failed."""


class SSEEventType(CaseInsensitiveStrEnum):
    """Event types in an SSE message."""

    ERROR = "error"


class SSEFieldType(CaseInsensitiveStrEnum):
    """Field types in an SSE message."""

    DATA = "data"
    EVENT = "event"
    ID = "id"
    RETRY = "retry"
    COMMENT = "comment"


class SystemState(CaseInsensitiveStrEnum):
    """State of the system as a whole.

    This is used to track the state of the system as a whole, and is used to
    determine what actions to take when a signal is received.
    """

    INITIALIZING = "initializing"
    """The system is initializing. This is the initial state."""

    CONFIGURING = "configuring"
    """The system is configuring services."""

    READY = "ready"
    """The system is ready to start profiling. This is a temporary state that should be
    followed by PROFILING."""

    PROFILING = "profiling"
    """The system is running a profiling run."""

    PROCESSING = "processing"
    """The system is processing results."""

    STOPPING = "stopping"
    """The system is stopping."""

    SHUTDOWN = "shutdown"
    """The system is shutting down. This is the final state."""


class VideoFormat(CaseInsensitiveStrEnum):
    """Video container formats for synthetic video generation."""

    MP4 = "mp4"
    """MP4 container. Widely compatible, good for H.264/H.265 codecs."""

    WEBM = "webm"
    """WebM container. Open format, optimized for web, good for VP9 codec."""


class VideoJobStatus(CaseInsensitiveStrEnum):
    """Status values for async video generation jobs."""

    QUEUED = "queued"
    """Job is queued and waiting to start."""

    IN_PROGRESS = "in_progress"
    """Job is currently being processed."""

    COMPLETED = "completed"
    """Job completed successfully."""

    FAILED = "failed"
    """Job failed with an error."""


class VideoSynthType(CaseInsensitiveStrEnum):
    MOVING_SHAPES = "moving_shapes"
    """Generate videos with animated geometric shapes moving across the frame"""

    GRID_CLOCK = "grid_clock"
    """Generate videos with a grid pattern and timestamp overlay for frame-accurate verification"""


class WorkerStatus(CaseInsensitiveStrEnum):
    """The current status of a worker service.

    NOTE: The order of the statuses is important for the UI.
    """

    HEALTHY = "healthy"
    HIGH_LOAD = "high_load"
    ERROR = "error"
    IDLE = "idle"
    STALE = "stale"
