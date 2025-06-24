# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from enum import Enum, auto
from typing import Any


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

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, str):
            return self.value.lower() == other.lower()
        return super().__eq__(other)

    def __hash__(self) -> int:
        return hash(self.value.lower())

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

    ZMQ_IPC = "zmq_ipc"
    """ZeroMQ backend using IPC sockets."""

    ZMQ_INPROC = "zmq_inproc"
    """ZeroMQ backend using in-process communication."""


class Topic(CaseInsensitiveStrEnum):
    """Communication topics for the main messaging bus.
    Right now, there is some overlap between Topic and MessageType.

    NOTE: If you add a new topic, you must also add handlers for it in the
    ClientType enums so the system knows what type of client to use for that topic.
    """

    CREDIT_DROP = "credit_drop"
    CREDIT_RETURN = "credit_return"
    CREDITS_COMPLETE = "credits_complete"
    PROFILE_PROGRESS = "profile_progress"
    PROFILE_STATS = "profile_stats"
    PROFILE_RESULTS = "profile_results"
    REGISTRATION = "registration"
    COMMAND = "command"
    RESPONSE = "response"
    STATUS = "status"
    HEARTBEAT = "heartbeat"
    INFERENCE_RESULTS = "inference_results"
    CONVERSATION_DATA = "conversation_data"


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
    """A generic error message."""

    SERVICE_ERROR = "service_error"
    """A message sent by a component service to the system controller to
    report an error."""

    CREDIT_DROP = "credit_drop"
    """A message sent by the Timing Manager service to allocate credits
    for a worker."""

    CREDIT_RETURN = "credit_return"
    """A message sent by the Worker services to return credits to the credit pool."""

    CREDITS_COMPLETE = "credits_complete"
    """A message sent by the Timing Manager services to signify all requests have completed."""

    CONVERSATION_REQUEST = "conversation_request"
    """A message sent by one service to another to request a conversation."""

    CONVERSATION_RESPONSE = "conversation_response"
    """A message sent by one service to another to respond to a conversation request."""

    INFERENCE_RESULTS = "inference_results"
    """A message containing inference results from a worker."""

    # Sweep run messages

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

    # Profile run messages

    PROFILE_PROGRESS = "profile_progress"
    """A message containing profile run progress."""

    PROFILE_STATS = "profile_stats"
    """A message containing profile run stats such as error rates, etc."""

    PROFILE_RESULTS = "profile_results"
    """A message containing profile run results."""

    PROFILE_ERROR = "profile_error"
    """A message containing an error from a profile run."""


################################################################################
# Command Enums
################################################################################


class CommandType(CaseInsensitiveStrEnum):
    """List of commands that the SystemController can send to component services."""

    PROFILE_CONFIGURE = "profile_configure"
    """A command sent to configure a service in preparation for a profile run. This will
    override the current configuration."""

    PROFILE_START = "profile_start"
    """A command sent to indicate that a service should begin profiling using the
    current configuration."""

    PROFILE_STOP = "profile_stop"
    """A command sent to indicate that a service should stop doing profile related
    work, as the profile run is complete."""

    PROFILE_CANCEL = "profile_cancel"
    """A command sent to cancel a profile run. This will stop the current profile run and
    process the partial results."""

    SHUTDOWN = "shutdown"
    """A command sent to shutdown a service. This will stop the service gracefully
    no matter what state it is in."""

    PROCESS_RECORDS = "process_records"
    """A command sent to process records. This will process the records and return
    the services to their pre-record processing state."""


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
    MULTI_WORKER_PROCESS = "multi_worker_process"
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
# System State Enums
################################################################################


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


################################################################################
# Inference Client Enums
################################################################################


class InferenceClientType(CaseInsensitiveStrEnum):
    """Inference client types."""

    GRPC = "grpc"
    HTTP = "http"
    OPENAI = "openai"
    DYNAMO = "dynamo"


################################################################################
# Converter Enums
################################################################################


class PromptSource(CaseInsensitiveStrEnum):
    """Source of prompts for the model."""

    SYNTHETIC = "synthetic"
    FILE = "file"
    PAYLOAD = "payload"


class Modality(CaseInsensitiveStrEnum):
    """Modality of the model. Can be used to determine the type of data to send to the model in
    conjunction with the ModelSelectionStrategy.MODALITY_AWARE."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"


class ModelSelectionStrategy(CaseInsensitiveStrEnum):
    """Strategy for selecting the model to use for the request."""

    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    MODALITY_AWARE = "modality_aware"


class MeasurementMode(CaseInsensitiveStrEnum):
    REQUEST_COUNT = "request_count"
    INTERVAL = "interval"


class RequestPayloadType(CaseInsensitiveStrEnum):
    """Request payload types.

    These determine the format of the request payload to send to the model.
    """

    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    OPENAI_COMPLETIONS = "openai_completions"
    OPENAI_EMBEDDINGS = "openai_embeddings"
    OPENAI_MULTIMODAL = "openai_multimodal"
    OPENAI_RESPONSES = "openai_responses"

    HUGGINGFACE_GENERATE = "huggingface_generate"
    HUGGINGFACE_RANKINGS = "huggingface_rankings"

    IMAGE_RETRIEVAL = "image_retrieval"
    DYNAMIC_GRPC = "dynamic_grpc"
    NVCLIP = "nvclip"

    RANKINGS = "rankings"
    TEMPLATE = "template"

    TENSORRTLLM = "tensorrtllm"
    VLLM = "vllm"


class ResponsePayloadType(CaseInsensitiveStrEnum):
    """Response payload types.

    These determine the format of the response payload that the model will return.
    """

    HUGGINGFACE_GENERATE = "huggingface_generate"
    HUGGINGFACE_RANKINGS = "huggingface_rankings"

    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    OPENAI_COMPLETIONS = "openai_completions"
    OPENAI_EMBEDDINGS = "openai_embeddings"
    OPENAI_MULTIMODAL = "openai_multimodal"
    OPENAI_RESPONSES = "openai_responses"

    RANKINGS = "rankings"
    IMAGE_RETRIEVAL = "image_retrieval"

    TRITON = "triton"
    TRITON_GENERATE = "triton_generate"


####################################################################################
# Data Exporter Enums
####################################################################################


class DataExporterType(CaseInsensitiveStrEnum):
    CONSOLE = "console"
    CONSOLE_ERROR = "console_error"


#################################################################################
# Post Processor Enums
################################################################################


class PostProcessorType(CaseInsensitiveStrEnum):
    METRIC_SUMMARY = "metric_summary"


#################################################################################
# Metric Enums
################################################################################


class MetricTimeType(Enum):
    """Defines the time types for metrics."""

    NANOSECONDS = 9
    MILLISECONDS = 3
    SECONDS = 0


class MetricType(Enum):
    METRIC_OF_RECORDS = auto()
    METRIC_OF_METRICS = auto()
    METRIC_OF_BOTH = auto()
