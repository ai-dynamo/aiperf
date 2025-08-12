# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class MessageType(CaseInsensitiveStrEnum):
    """The various types of messages that can be sent between services.

    The message type is used to determine what Pydantic model the message maps to,
    based on the message_type field in the message model. For detailed explanations
    of each message type, go to its definition in :mod:`aiperf.common.messages`.
    """

    ALL_RECORDS_RECEIVED = "all_records_received"
    COMMAND = "command"
    COMMAND_RESPONSE = "command_response"
    CONNECTION_PROBE = "connection_probe"
    CONVERSATION_REQUEST = "conversation_request"
    CONVERSATION_RESPONSE = "conversation_response"
    CONVERSATION_TURN_REQUEST = "conversation_turn_request"
    CONVERSATION_TURN_RESPONSE = "conversation_turn_response"
    CREDITS_COMPLETE = "credits_complete"
    CREDIT_DROP = "credit_drop"
    CREDIT_PHASE_COMPLETE = "credit_phase_complete"
    CREDIT_PHASE_PROGRESS = "credit_phase_progress"
    CREDIT_PHASE_SENDING_COMPLETE = "credit_phase_sending_complete"
    CREDIT_PHASE_START = "credit_phase_start"
    CREDIT_RETURN = "credit_return"
    DATASET_CONFIGURED_NOTIFICATION = "dataset_configured_notification"
    DATASET_TIMING_REQUEST = "dataset_timing_request"
    DATASET_TIMING_RESPONSE = "dataset_timing_response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    INFERENCE_RESULTS = "inference_results"
    METRIC_RECORDS = "metric_records"
    PARSED_INFERENCE_RESULTS = "parsed_inference_results"
    PROCESSING_STATS = "processing_stats"
    PROCESS_RECORDS_RESULT = "process_records_result"
    PROFILE_PROGRESS = "profile_progress"
    PROFILE_RESULTS = "profile_results"
    REGISTRATION = "registration"
    SERVICE_ERROR = "service_error"
    STATUS = "status"
    WORKER_HEALTH = "worker_health"
    WORKER_STATUS_SUMMARY = "worker_status_summary"
