# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


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
