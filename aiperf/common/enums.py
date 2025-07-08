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


class CommandResponseStatus(CaseInsensitiveStrEnum):
    """Status of a command response."""

    SUCCESS = "success"
    FAILURE = "failure"


class CommunicationClientType(CaseInsensitiveStrEnum):
    """Enum for specifying the communication client type for communication clients."""

    PUB = "pub"
    SUB = "sub"
    PUSH = "push"
    PULL = "pull"
    REQUEST = "request"
    REPLY = "reply"


class CommunicationClientAddressType(CaseInsensitiveStrEnum):
    """Enum for specifying the address type for communication clients.
    This is used to lookup the address in the communication config."""

    EVENT_BUS_PROXY_FRONTEND = "event_bus_proxy_frontend"
    """Frontend address for services to publish messages to."""

    EVENT_BUS_PROXY_BACKEND = "event_bus_proxy_backend"
    """Backend address for services to subscribe to messages."""

    CREDIT_DROP = "credit_drop"
    """Address to send CreditDrop messages from the TimingManager to the Worker."""

    CREDIT_RETURN = "credit_return"
    """Address to send CreditReturn messages from the Worker to the TimingManager."""

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


class ZMQProxyType(CaseInsensitiveStrEnum):
    """Types of ZMQ proxys."""

    DEALER_ROUTER = "dealer_router"
    XPUB_XSUB = "xpub_xsub"
    PUSH_PULL = "push_pull"


################################################################################
# Dataset Enums
################################################################################


class ComposerType(CaseInsensitiveStrEnum):
    """
    The type of composer to use for the dataset.
    """

    SYNTHETIC = "synthetic"
    CUSTOM = "custom"
    PUBLIC_DATASET = "public_dataset"


class CustomDatasetType(CaseInsensitiveStrEnum):
    """Defines the type of JSONL custom dataset from the user."""

    SINGLE_TURN = "single_turn"
    MULTI_TURN = "multi_turn"
    RANDOM_POOL = "random_pool"
    TRACE = "trace"


class ImageFormat(CaseInsensitiveStrEnum):
    """Types of image formats supported by AIPerf."""

    PNG = "png"
    JPEG = "jpeg"


class AudioFormat(CaseInsensitiveStrEnum):
    """Types of audio formats supported by AIPerf."""

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

    COMMAND_RESPONSE = "command_response"
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

    PARSED_INFERENCE_RESULTS = "parsed_inference_results"
    """A message containing parsed inference results from a post processor."""

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

    PROCESSING_STATS = "processing_stats"
    """A message containing processing stats from the records manager."""

    PROFILE_RESULTS = "profile_results"
    """A message containing profile run results."""

    PROFILE_ERROR = "profile_error"
    """A message containing an error from a profile run."""

    NOTIFICATION = "notification"
    """A message containing a notification from a service. This is used to notify other services of events."""

    DATASET_TIMING_REQUEST = "dataset_timing_request"
    """A message sent by a service to request timing information from a dataset."""

    DATASET_TIMING_RESPONSE = "dataset_timing_response"
    """A message sent by a service to respond to a dataset timing request."""

    WORKER_HEALTH = "worker_health"
    """A message sent by a worker to the worker manager to report its health."""


################################################################################
# Command Enums
################################################################################


class CommandType(CaseInsensitiveStrEnum):
    """List of commands that the SystemController can send to component services."""

    SHUTDOWN = "shutdown"
    """A command sent to shutdown a service. This will stop the service gracefully
    no matter what state it is in."""

    PROCESS_RECORDS = "process_records"
    """A command sent to process records. This will process the records and return
    the services to their pre-record processing state."""

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


################################################################################
# Notification Enums
################################################################################


class NotificationType(CaseInsensitiveStrEnum):
    """Types of notifications that can be sent to other services."""

    DATASET_CONFIGURED = "dataset_configured"
    """A notification sent to notify other services that the dataset has been configured."""


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
    """The service is in an unknown state."""

    INITIALIZING = "initializing"
    """The service is currently initializing. This is a temporary state that should be
    followed by PENDING."""

    PENDING = "pending"
    """The service is pending configuration."""

    CONFIGURING = "configuring"
    """The service is currently configuring. This is a temporary state that should be
    followed by READY."""

    READY = "ready"
    """The service has been configured and is ready to be started."""

    STARTING = "starting"
    """The service is starting. This is a temporary state that should be followed
    by RUNNING."""

    RUNNING = "running"
    """The service is running."""

    STOPPING = "stopping"
    """The service is stopping. This is a temporary state that should be followed
    by STOPPED."""

    STOPPED = "stopped"
    """The service has been stopped."""

    SHUTDOWN = "shutdown"
    """The service has been shutdown."""

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
    INFERENCE_RESULT_PARSER = "inference_result_parser"
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


####################################################################################
# Data Exporter Enums
####################################################################################


class DataExporterType(CaseInsensitiveStrEnum):
    CONSOLE = "console"
    CONSOLE_ERROR = "console_error"
    JSON = "json"


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


################################################################################
# SSE Enums
################################################################################


class SSEFieldType(CaseInsensitiveStrEnum):
    """Field types in an SSE message."""

    DATA = "data"
    EVENT = "event"
    ID = "id"
    RETRY = "retry"
    COMMENT = "comment"


class SSEEventType(CaseInsensitiveStrEnum):
    """Event types in an SSE message. Many of these are custom and not defined by the SSE spec."""

    ERROR = "error"
    LLM_METRICS = "llm_metrics"


class ResponsePayloadType(CaseInsensitiveStrEnum):
    """Response payload types.

    These determine the format of the response payload that the model will return.

    Equivalent to `output_format` from `genai-perf`.
    """

    OPENAI_CHAT_COMPLETIONS = "openai_chat_completions"
    OPENAI_COMPLETIONS = "openai_completions"
    OPENAI_EMBEDDINGS = "openai_embeddings"
    OPENAI_MULTIMODAL = "openai_multimodal"
    OPENAI_RESPONSES = "openai_responses"

    HUGGINGFACE_GENERATE = "huggingface_generate"

    RANKINGS = "rankings"

    IMAGE_RETRIEVAL = "image_retrieval"

    @classmethod
    def from_endpoint_type(cls, endpoint_type: "EndpointType") -> "ResponsePayloadType":
        """Get the response payload type for the endpoint type."""
        endpoint_to_payload_map = {
            EndpointType.OPENAI_CHAT_COMPLETIONS: ResponsePayloadType.OPENAI_CHAT_COMPLETIONS,
            EndpointType.OPENAI_MULTIMODAL: ResponsePayloadType.OPENAI_CHAT_COMPLETIONS,
            EndpointType.OPENAI_COMPLETIONS: ResponsePayloadType.OPENAI_COMPLETIONS,
            EndpointType.OPENAI_EMBEDDINGS: ResponsePayloadType.OPENAI_EMBEDDINGS,
            EndpointType.OPENAI_RESPONSES: ResponsePayloadType.OPENAI_RESPONSES,
            EndpointType.HUGGINGFACE_GENERATE: ResponsePayloadType.HUGGINGFACE_GENERATE,
            EndpointType.RANKINGS: ResponsePayloadType.RANKINGS,
            EndpointType.IMAGE_RETRIEVAL: ResponsePayloadType.IMAGE_RETRIEVAL,
        }

        if endpoint_type not in endpoint_to_payload_map:
            raise NotImplementedError(
                f"Payload type not implemented for {endpoint_type}"
            )

        return endpoint_to_payload_map[endpoint_type]


class EndpointType(CaseInsensitiveStrEnum):
    """Endpoint types.

    These determine the format of request payload to send to the model.

    Similar to `endpoint_type_map` and `OutputFormat` from `genai-perf`.
    """

    OPENAI_CHAT_COMPLETIONS = "chat"
    OPENAI_COMPLETIONS = "completions"
    OPENAI_EMBEDDINGS = "embeddings"
    OPENAI_MULTIMODAL = "multimodal"
    OPENAI_RESPONSES = "responses"

    HUGGINGFACE_GENERATE = "generate"

    DYNAMIC_GRPC = "dynamic_grpc"
    NVCLIP = "nvclip"
    TEMPLATE = "template"

    RANKINGS = "rankings"
    IMAGE_RETRIEVAL = "image_retrieval"

    TENSORRTLLM = "tensorrtllm"
    TENSORRTLLM_ENGINE = "tensorrtllm_engine"

    TRITON_GENERATE = "triton_generate"

    DYNAMO_ENGINE = "dynamo_engine"

    def endpoint_path(self) -> str | None:
        """Get the endpoint path for the endpoint type."""
        endpoint_path_map = {
            # OpenAI endpoints
            EndpointType.OPENAI_CHAT_COMPLETIONS: "/v1/chat/completions",
            EndpointType.OPENAI_MULTIMODAL: "/v1/chat/completions",
            EndpointType.OPENAI_COMPLETIONS: "/v1/completions",
            EndpointType.OPENAI_EMBEDDINGS: "/v1/embeddings",
            EndpointType.OPENAI_RESPONSES: "/v1/responses",
            # Other
            EndpointType.NVCLIP: "/v1/embeddings",
            EndpointType.HUGGINGFACE_GENERATE: "/",  # HuggingFace TGI only exposes root endpoint
            EndpointType.RANKINGS: "/v1/ranking",  # TODO: Not implemented yet
            EndpointType.IMAGE_RETRIEVAL: "/v1/infer",  # TODO: Not implemented yet
            EndpointType.TRITON_GENERATE: "/v2/models/{MODEL_NAME}/generate",  # TODO: Not implemented yet
            # These endpoints do not have a specific path
            EndpointType.DYNAMIC_GRPC: None,  # TODO: Not implemented yet
            EndpointType.TEMPLATE: None,  # TODO: Not implemented yet
            EndpointType.TENSORRTLLM: None,  # TODO: Not implemented yet
            EndpointType.TENSORRTLLM_ENGINE: None,  # TODO: Not implemented yet
            EndpointType.DYNAMO_ENGINE: None,  # TODO: Not implemented yet
        }

        if self not in endpoint_path_map:
            raise NotImplementedError(f"Endpoint not implemented for {self}")

        return endpoint_path_map[self]

    def response_payload_type(self) -> ResponsePayloadType:
        """Get the response payload type for the request payload type."""
        return ResponsePayloadType.from_endpoint_type(self)


class RequestRateMode(CaseInsensitiveStrEnum):
    """The different ways the request rate scheduler should generate requests."""

    FIXED = "fixed"
    """Generate requests at a fixed rate. This is the default mode."""

    # DYNAMIC = "dynamic"
    # """Generate requests at a dynamic rate based on the average response times of the previous requests. TBD."""
