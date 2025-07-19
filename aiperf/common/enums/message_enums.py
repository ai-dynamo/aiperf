# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from aiperf.common.enums.base_enums import CaseInsensitiveStrEnum


class MessageType(CaseInsensitiveStrEnum):
    """The various types of messages that can be sent between services.

    The message type is used to determine what Pydantic model the message maps to,
    based on the message_type field in the message model.
    """

    UNKNOWN = "unknown"
    """A placeholder value for when the message type is not known."""

    ERROR = "error"
    """A generic error message."""

    #########################################################
    # Service messages
    #########################################################

    REGISTRATION = "registration"
    """A message sent by a component service to register itself with the
    system controller."""

    HEARTBEAT = "heartbeat"
    """A message sent by a component service to the system controller to indicate it
    is still running."""

    COMMAND = "command"
    """A message sent by the system controller to a component service to command it
    to do something."""

    COMMAND_RESPONSE = "command_response"
    """A message sent by a component service to the system controller to respond
    to a command."""

    STATUS = "status"
    """A notification sent by a component service to the system controller to
    report its status."""

    SERVICE_ERROR = "service_error"
    """A message sent by a component service to the system controller to
    report an error."""

    #########################################################
    # Sweep run messages
    #########################################################

    SWEEP_CONFIGURE = "sweep_configure"
    """A message sent to configure a sweep run."""

    SWEEP_BEGIN = "sweep_begin"
    """A message sent to indicate that a sweep has begun."""

    SWEEP_PROGRESS = "sweep_progress"
    """A message containing sweep run progress."""

    SWEEP_END = "sweep_end"
    """A message sent to indicate that a sweep has ended."""

    SWEEP_RESULTS = "sweep_results"
    """A message containing sweep run results."""

    SWEEP_ERROR = "sweep_error"
    """A message containing an error from a sweep run."""

    #########################################################
    # Profile run messages
    #########################################################

    PROFILE_PROGRESS = "profile_progress"
    """A message containing profile run progress."""

    PROCESSING_STATS = "processing_stats"
    """A message containing processing stats from the records manager."""

    PROFILE_RESULTS = "profile_results"
    """A message containing profile run results."""

    PROFILE_ERROR = "profile_error"
    """A message containing an error from a profile run."""

    NOTIFICATION = "notification"
    """A message containing a notification from a service. This is used to notify other services of events."""

    WORKER_HEALTH = "worker_health"
    """A message sent by a worker to the worker manager to report its health."""

    #########################################################
    # Credit messages
    #########################################################

    CREDIT_DROP = "credit_drop"
    """A message sent by the Timing Manager service to allocate credits
    for a worker."""

    CREDIT_RETURN = "credit_return"
    """A message sent by the Worker services to return credits to the credit pool."""

    CREDIT_PHASE_START = "credit_phase_start"
    """A message sent by the TimingManager to report that a phase has started."""

    CREDIT_PHASE_PROGRESS = "credit_phase_progress"
    """A message sent by the TimingManager to report the progress of a credit phase."""

    CREDIT_PHASE_SENDING_COMPLETE = "credit_phase_sending_complete"
    """A message sent by the TimingManager to report that a phase has completed sending (but not necessarily all credits have been returned)."""

    CREDIT_PHASE_COMPLETE = "credit_phase_complete"
    """A message sent by the TimingManager to report that a phase has completed."""

    CREDITS_COMPLETE = "credits_complete"
    """A message sent by the Timing Manager services to signify all requests have completed."""

    #########################################################
    # Dataset messages
    #########################################################

    DATASET_CONFIGURED_NOTIFICATION = "dataset_configured_notification"
    """A notification sent to notify other services that the dataset has been configured."""

    DATASET_TIMING_REQUEST = "dataset_timing_request"
    """A message sent by a service to request timing information from a dataset."""

    DATASET_TIMING_RESPONSE = "dataset_timing_response"
    """A message sent by a service to respond to a dataset timing request."""

    CONVERSATION_REQUEST = "conversation_request"
    """A message sent by one service to the DatasetManager to request a conversation."""

    CONVERSATION_RESPONSE = "conversation_response"
    """A message sent by the DatasetManager to a service, containing the requested conversation data."""

    CONVERSATION_TURN_REQUEST = "conversation_turn_request"
    """A message sent by one service to the DatasetManager to request a single turn from a conversation."""

    CONVERSATION_TURN_RESPONSE = "conversation_turn_response"
    """A message sent by the DatasetManager to a service, containing the requested turn data."""

    #########################################################
    # Post processor / Records manager messages
    #########################################################

    PROCESS_RECORDS_REQUEST = "process_records_request"
    """A message sent by the system controller to a component service to request
    that it process records."""

    PROCESS_RECORDS_RESPONSE = "process_records_response"
    """A message sent by a component service to the system controller to respond
    to a process records request."""

    INFERENCE_RESULTS = "inference_results"
    """A message containing inference results from a worker."""

    PARSED_INFERENCE_RESULTS = "parsed_inference_results"
    """A message containing parsed inference results from a post processor."""


class NotificationType(CaseInsensitiveStrEnum):
    """Types of notifications that can be sent to other services."""

    DATASET_CONFIGURED = "dataset_configured"
    """A notification sent to notify other services that the dataset has been configured."""
