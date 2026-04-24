# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum
from aiperf.common.enums.communication_enums import (
    CommAddress,
    CommandType,
    CommunicationType,
    MessageType,
)
from aiperf.common.enums.dataset_enums import (
    ConnectionReuseStrategy,
    ConversationContextMode,
    CreditPhase,
    DatasetFormat,
    DatasetType,
    ModelSelectionStrategy,
    OslMode,
    PromptSource,
    SweepType,
)
from aiperf.common.enums.export_enums import (
    ExportFormat,
    ExportLevel,
    ListMetricAggregationMode,
    RecordExportFormat,
    ServerMetricsFormat,
    SummaryFormat,
)
from aiperf.common.enums.lifecycle_enums import (
    LifecycleState,
    ServiceRegistrationStatus,
    SystemState,
    WorkerStartupState,
    WorkerStatus,
)
from aiperf.common.enums.media_enums import (
    AudioFormat,
    ContentType,
    ImageFormat,
    MediaType,
    VideoAudioCodec,
    VideoFormat,
    VideoJobStatus,
    VideoSynthType,
)
from aiperf.common.enums.server_metrics_enums import (
    ConvergenceMode,
    ConvergenceStat,
    GPUTelemetryMode,
    GpuTelemetryType,
    PrometheusMetricType,
    ServerMetricsDiscoveryMode,
)


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


class IPVersion(CaseInsensitiveStrEnum):
    """IP version for HTTP socket connections."""

    V4 = "4"
    """Use IPv4 only (AF_INET). Default for most environments."""

    V6 = "6"
    """Use IPv6 only (AF_INET6). Use when connecting to IPv6-only servers."""

    AUTO = "auto"
    """Let the system choose (AF_UNSPEC). Supports both IPv4 and IPv6."""


class RequestContentType(CaseInsensitiveStrEnum):
    """Content type for HTTP request body serialization."""

    APPLICATION_JSON = "application/json"
    """Standard JSON encoding. Default for all endpoints."""

    MULTIPART_FORM_DATA = "multipart/form-data"
    """Multipart form encoding. Required by some video generation servers (e.g., vLLM)."""


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


__all__ = [
    "AIPerfLogLevel",
    "AudioFormat",
    "CommAddress",
    "CommandType",
    "CommunicationType",
    "ConnectionReuseStrategy",
    "ContentType",
    "ConvergenceMode",
    "ConvergenceStat",
    "ConversationContextMode",
    "CreditPhase",
    "DatasetFormat",
    "DatasetType",
    "ExportFormat",
    "ExportLevel",
    "GPUTelemetryMode",
    "GpuTelemetryType",
    "IPVersion",
    "ImageFormat",
    "LifecycleState",
    "ListMetricAggregationMode",
    "MediaType",
    "MessageType",
    "ModelSelectionStrategy",
    "OslMode",
    "PrometheusMetricType",
    "PromptSource",
    "RecordExportFormat",
    "RequestContentType",
    "SSEEventType",
    "SSEFieldType",
    "ServerMetricsDiscoveryMode",
    "ServerMetricsFormat",
    "ServiceRegistrationStatus",
    "SummaryFormat",
    "SweepType",
    "SystemState",
    "VideoAudioCodec",
    "VideoFormat",
    "VideoJobStatus",
    "VideoSynthType",
    "WorkerStartupState",
    "WorkerStatus",
]
