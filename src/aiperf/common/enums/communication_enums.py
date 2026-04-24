# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class CommAddress(CaseInsensitiveStrEnum):
    """Enum for specifying the address type for communication clients.
    This is used to lookup the address in the communication config."""

    EVENT_BUS_PROXY_FRONTEND = "event_bus_proxy_frontend"
    """Frontend address for services to publish messages to."""

    EVENT_BUS_PROXY_BACKEND = "event_bus_proxy_backend"
    """Backend address for services to subscribe to messages."""

    CREDIT_ROUTER = "credit_router"
    """Address for bidirectional ROUTER-DEALER credit routing (all timing modes)."""

    CREDIT_RETURN_ROUTER = "credit_return_router"
    """Address for dedicated ROUTER-DEALER credit return channel (Worker -> Router acks)."""

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

    CONTROL = "control"
    """Address for direct DEALER/ROUTER control channel communication with the controller."""

    GROUP_LIFECYCLE = "group_lifecycle"
    """Address for group-local DEALER/ROUTER lifecycle coordination owned by WorkerGroupManager."""


class CommandType(CaseInsensitiveStrEnum):
    PROCESS_RECORDS = "process_records"
    PROFILE_CANCEL = "profile_cancel"
    PROFILE_COMPLETE = "profile_complete"
    PROFILE_CONFIGURE = "profile_configure"
    PROFILE_START = "profile_start"
    REALTIME_METRICS = "realtime_metrics"
    REPORT_WORKER_STATUS_SUMMARY = "report_worker_status_summary"
    SHUTDOWN = "shutdown"
    SHUTDOWN_WORKERS = "shutdown_workers"
    SPAWN_WORKERS = "spawn_workers"
    START_REALTIME_TELEMETRY = "start_realtime_telemetry"
    ABORT = "abort"
    """Signal sibling pod peers (workers/record-processors) to exit the process
    with a non-zero status so kubelet restarts them. Used by WorkerGroupManager
    when its own lifecycle failed — a clean SHUTDOWN would let siblings exit 0
    and leave the pod permanently half-dead at 1/13 Ready."""


class CommunicationType(CaseInsensitiveStrEnum):
    """Type of inter-process communication transport."""

    IPC = "ipc"
    """Unix domain sockets (single machine)."""

    TCP = "tcp"
    """TCP sockets (multi-machine)."""

    DUAL = "dual"
    """Dual-bind: IPC for co-located services, TCP for remote workers (Kubernetes)."""


class MessageType(CaseInsensitiveStrEnum):
    """The various types of messages that can be sent between services.

    The message type is used to determine what Pydantic model the message maps to,
    based on the message_type field in the message model. For detailed explanations
    of each message type, go to its definition in :mod:`aiperf.common.messages`.
    """

    ALL_RECORDS_RECEIVED = "all_records_received"
    BENCHMARK_COMPLETE = "benchmark_complete"
    CANCEL_CREDITS = "cancel_credits"
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
    DATASET_DOWNLOADED_NOTIFICATION = "dataset_downloaded_notification"
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
    MEMORY_REPORT = "memory_report"
    REALTIME_METRICS = "realtime_metrics"
    REALTIME_TELEMETRY_METRICS = "realtime_telemetry_metrics"
    REALTIME_SERVER_METRICS = "realtime_server_metrics"
    REGISTRATION = "registration"
    SERVICE_ERROR = "service_error"
    STATUS = "status"
    TELEMETRY_STATUS = "telemetry_status"
    SERVER_METRICS_STATUS = "server_metrics_status"
    WORKER_HEALTH = "worker_health"
    WORKER_POD_STATE = "worker_pod_state"
    WORKER_STARTUP_STATE = "worker_startup_state"
    WORKER_STATUS_SUMMARY = "worker_status_summary"
