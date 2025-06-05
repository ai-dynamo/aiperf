# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


################################################################################
# Base Enums
################################################################################
class CaseInsensitiveStrEnum(str, Enum):
    """
    CaseInsensitiveStrEnum is a custom enumeration class that extends `str` and `Enum` to provide case-insensitive
    lookup functionality for its members.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.{self.name}"

    @classmethod
    def _missing_(cls, value):
        """
        Handles cases where a value is not directly found in the enumeration.

        This method is called when an attempt is made to access an enumeration
        member using a value that does not directly match any of the defined
        members. It provides custom logic to handle such cases.

        Returns:
            The matching enumeration member if a case-insensitive match is found
            for string values; otherwise, returns None.
        """
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        return None


################################################################################
# Communication Enums
################################################################################


class CommunicationBackend(CaseInsensitiveStrEnum):
    """Supported communication backends."""

    ZMQ_TCP = "zmq_tcp"
    """ZeroMQ backend using TCP sockets."""


class Topic(CaseInsensitiveStrEnum):
    """Communication topics for the main messaging bus.
    Right now, there is some overlap between Topic and MessageType."""

    CREDIT_DROP = "credit_drop"
    CREDIT_RETURN = "credit_return"
    REGISTRATION = "registration"
    COMMAND = "command"
    RESPONSE = "response"
    STATUS = "status"
    HEARTBEAT = "heartbeat"


# TODO: Is this separation needed? Or should we just use the Topic enum?
class DataTopic(CaseInsensitiveStrEnum):
    """TBD. Specific data topics for use in the future."""

    DATASET = "dataset_data"
    RECORDS = "records_data"
    WORKER = "worker_data"
    POST_PROCESSOR = "post_processor_data"
    RESULTS = "results"
    METRICS = "metrics"
    CONVERSATION = "conversation_data"


TopicType = Topic | DataTopic
"""Union of all the various different topic types supported by the system, for use in
type hinting."""


################################################################################
# Data Format Enums
################################################################################


class ImageFormat(CaseInsensitiveStrEnum):
    PNG = "png"
    JPEG = "jpeg"


class AudioFormat(CaseInsensitiveStrEnum):
    WAV = "wav"
    MP3 = "mp3"


################################################################################
# Message-related enums
################################################################################


class MessageType(CaseInsensitiveStrEnum):
    """The various types of messages that can be sent between services.

    The message type is used to determine what Pydantic model the message maps to,
    based on the message_type field in the message model.
    """

    UNKNOWN = "unknown"
    """A placeholder value for when the message type is not known."""

    REGISTRATION = "registration"
    """A message sent by a component service to register itself with the
    system controller."""

    HEARTBEAT = "heartbeat"
    """A message sent by a component service to the system controller to indicate it
    is still running."""

    COMMAND = "command"
    """A message sent by the system controller to a component service to command it
    to do something."""

    RESPONSE = "response"
    """A message sent by a component service to the system controller to respond
    to a command."""

    STATUS = "status"
    """A notification sent by a component service to the system controller to
    report its status."""

    ERROR = "error"
    """A message sent by a component service to the system controller to
    report an error."""

    CREDIT_DROP = "credit_drop"
    """A message sent by the Timing Manager service to allocate credits
    for a worker."""

    CREDIT_RETURN = "credit_return"
    """A message sent by the Worker services to return credits to the credit pool."""


################################################################################
# Command Enums
################################################################################


class CommandType(CaseInsensitiveStrEnum):
    """List of commands that the SystemController can send to component services."""

    START = "start"
    STOP = "stop"
    CONFIGURE = "configure"


################################################################################
# Service Enums
################################################################################


class ServiceRunType(CaseInsensitiveStrEnum):
    """The different ways the SystemController should run the component services."""

    MULTIPROCESSING = "process"
    """Run each service as a separate process.
    This is the default way for single-node deployments."""

    KUBERNETES = "k8s"
    """Run each service as a separate Kubernetes pod.
    This is the default way for multi-node deployments."""


class ServiceState(CaseInsensitiveStrEnum):
    """States a service can be in throughout its lifecycle."""

    UNKNOWN = "unknown"

    INITIALIZING = "initializing"
    """The service is currently initializing. This is a temporary state that should be
    followed by READY."""

    READY = "ready"
    """The service has initialized and is ready to be configured or started."""

    STARTING = "starting"
    """The service is starting. This is a temporary state that should be followed
    by RUNNING."""

    RUNNING = "running"

    STOPPING = "stopping"
    """The service is stopping. This is a temporary state that should be followed
    by STOPPED."""

    STOPPED = "stopped"

    ERROR = "error"
    """The service is currently in an error state."""


class ServiceType(CaseInsensitiveStrEnum):
    """Types of services in the AIPerf system.

    This is used to identify the service type when registering with the
    SystemController. It can also be used for tracking purposes if multiple
    instances of the same service type are running.
    """

    SYSTEM_CONTROLLER = "system_controller"
    DATASET_MANAGER = "dataset_manager"
    TIMING_MANAGER = "timing_manager"
    RECORDS_MANAGER = "records_manager"
    POST_PROCESSOR_MANAGER = "post_processor_manager"
    WORKER_MANAGER = "worker_manager"
    WORKER = "worker"
    TEST = "test_service"


class ServiceRegistrationStatus(CaseInsensitiveStrEnum):
    """Defines the various states a service can be in during registration with
    the SystemController."""

    UNREGISTERED = "unregistered"
    """The service is not registered with the SystemController. This is the
    initial state."""

    WAITING = "waiting"
    """The service is waiting for the SystemController to register it.
    This is a temporary state that should be followed by REGISTERED."""

    REGISTERED = "registered"
    """The service is registered with the SystemController."""

    TIMEOUT = "timeout"
    """The service registration timed out."""

    ERROR = "error"
    """The service registration failed."""


################################################################################
# Output Format Enums
################################################################################


class OutputFormat(CaseInsensitiveStrEnum):
    TENSORRTLLM = "tensorrtllm"
    VLLM = "vllm"


#################################################################################
# Model Selection Strategy Enums
################################################################################


class ModelSelectionStrategy(CaseInsensitiveStrEnum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


####################################################################################
# Data Exporter Enums
####################################################################################


class DataExporterType(CaseInsensitiveStrEnum):
    CONSOLE = "console"


#################################################################################
# Post Processor Enums
################################################################################


class PostProcessorType(CaseInsensitiveStrEnum):
    METRIC_SUMMARY = "metric_summary"
