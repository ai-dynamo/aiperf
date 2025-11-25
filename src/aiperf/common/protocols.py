# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from aiperf.common.enums import CommClientType, LifecycleState
from aiperf.common.environment import Environment
from aiperf.common.hooks import Hook, HookType
from aiperf.common.models import (
    Conversation,
    MetricRecordMetadata,
    ParsedResponse,
    ParsedResponseRecord,
    RequestInfo,
    RequestRecord,
    ServiceRunInfo,
    TelemetryRecord,
)
from aiperf.common.types import (
    CommAddressType,
    JsonObject,
    MessageCallbackMapT,
    MessageOutputT,
    MessageT,
    MessageTypeT,
    RequestInputT,
    RequestOutputT,
    ServiceTypeT,
)

if TYPE_CHECKING:
    import multiprocessing
    from pathlib import Path

    from rich.console import Console

    from aiperf.common.config import ServiceConfig, UserConfig
    from aiperf.common.enums import DatasetSamplingStrategy
    from aiperf.common.messages.inference_messages import MetricRecordsData
    from aiperf.common.models.metadata import EndpointMetadata, TransportMetadata
    from aiperf.common.models.model_endpoint_info import ModelEndpointInfo
    from aiperf.common.models.record_models import MetricResult
    from aiperf.dataset.loader.models import CustomDatasetT
    from aiperf.exporters.exporter_config import ExporterConfig, FileExportInfo
    from aiperf.metrics.metric_dicts import MetricRecordDict
    from aiperf.timing.config import TimingManagerConfig


################################################################################
# Core Base Protocols (Cannot be sorted)
################################################################################


@runtime_checkable
class AIPerfLoggerProtocol(Protocol):
    @property
    def is_trace_enabled(self) -> bool: ...
    @property
    def is_debug_enabled(self) -> bool: ...

    def __init__(self, logger_name: str | None = None, **kwargs) -> None: ...
    def log(
        self, level: int, message: str | Callable[..., str], *args, **kwargs
    ) -> None: ...
    def trace(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def debug(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def info(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def notice(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def warning(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def success(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def error(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def exception(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def critical(self, message: str | Callable[..., str], *args, **kwargs) -> None: ...
    def is_enabled_for(self, level: int) -> bool: ...


@runtime_checkable
class TaskManagerProtocol(AIPerfLoggerProtocol, Protocol):
    def execute_async(self, coro: Coroutine) -> asyncio.Task: ...

    async def cancel_all_tasks(self, timeout: float) -> None: ...

    async def wait_for_tasks(self) -> list[BaseException | None]: ...

    def start_background_task(
        self,
        method: Callable,
        interval: float | Callable[["TaskManagerProtocol"], float] | None = None,
        immediate: bool = False,
        stop_on_error: bool = False,
    ) -> None: ...


@runtime_checkable
class AIPerfLifecycleProtocol(TaskManagerProtocol, Protocol):
    """Protocol for AIPerf lifecycle methods.
    see :class:`aiperf.common.mixins.aiperf_lifecycle_mixin.AIPerfLifecycleMixin` for more details.
    """

    @property
    def was_initialized(self) -> bool: ...
    @property
    def was_started(self) -> bool: ...
    @property
    def was_stopped(self) -> bool: ...
    @property
    def is_running(self) -> bool: ...

    initialized_event: asyncio.Event
    started_event: asyncio.Event
    stopped_event: asyncio.Event

    @property
    def state(self) -> LifecycleState: ...

    async def initialize(self) -> None: ...
    async def start(self) -> None: ...
    async def initialize_and_start(self) -> None: ...
    async def stop(self) -> None: ...


################################################################################
# Communication Client Protocols (sorted alphabetically)
################################################################################


@runtime_checkable
class CommunicationClientProtocol(AIPerfLifecycleProtocol, Protocol):
    def __init__(
        self,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        **kwargs,
    ) -> None: ...


@runtime_checkable
class PubClientProtocol(CommunicationClientProtocol, Protocol):
    async def publish(self, message: MessageT) -> None: ...


@runtime_checkable
class PullClientProtocol(CommunicationClientProtocol, Protocol):
    def register_pull_callback(
        self,
        message_type: MessageTypeT,
        callback: Callable[[MessageT], Coroutine[Any, Any, None]],
    ) -> None: ...


@runtime_checkable
class PushClientProtocol(CommunicationClientProtocol, Protocol):
    async def push(self, message: MessageT) -> None: ...


@runtime_checkable
class ReplyClientProtocol(CommunicationClientProtocol, Protocol):
    def register_request_handler(
        self,
        service_id: str,
        message_type: MessageTypeT,
        handler: Callable[[MessageT], Coroutine[Any, Any, MessageOutputT | None]],
    ) -> None: ...


@runtime_checkable
class RequestClientProtocol(CommunicationClientProtocol, Protocol):
    async def request(
        self,
        message: MessageT,
        timeout: float = Environment.SERVICE.COMMS_REQUEST_TIMEOUT,
    ) -> MessageOutputT: ...

    async def request_async(
        self,
        message: MessageT,
        callback: Callable[[MessageOutputT], Coroutine[Any, Any, None]],
    ) -> None: ...


@runtime_checkable
class StreamingRouterClientProtocol(CommunicationClientProtocol, Protocol):
    """Protocol for ROUTER socket client with bidirectional streaming."""

    def register_receiver(
        self,
        handler: Callable[[str, MessageT], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register handler for incoming messages from DEALER clients.

        Args:
            handler: Async function that takes (identity: str, message: Message)
        """
        ...

    async def send_to(self, identity: str, message: MessageT) -> None:
        """
        Send message to specific DEALER client by identity.

        Args:
            identity: The DEALER client's identity (routing key)
            message: The message to send
        """
        ...


@runtime_checkable
class StreamingDealerClientProtocol(CommunicationClientProtocol, Protocol):
    """Protocol for DEALER socket client with bidirectional streaming."""

    def register_receiver(
        self,
        handler: Callable[[MessageT], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Register handler for incoming messages from ROUTER.

        Args:
            handler: Async function that takes (message: Message)
        """
        ...

    async def send(self, message: MessageT) -> None:
        """
        Send message to ROUTER.

        Args:
            message: The message to send
        """
        ...


@runtime_checkable
class SubClientProtocol(CommunicationClientProtocol, Protocol):
    async def subscribe(
        self,
        message_type: MessageTypeT,
        callback: Callable[[MessageT], Coroutine[Any, Any, None]],
    ) -> None: ...

    async def subscribe_all(
        self,
        message_callback_map: MessageCallbackMapT,
    ) -> None: ...


################################################################################
# Communication Protocol (must come after the clients)
################################################################################


@runtime_checkable
class CommunicationProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for the base communication layer.
    see :class:`aiperf.common.comms.base_comms.BaseCommunication` for more details.
    """

    def get_address(self, address_type: CommAddressType) -> str: ...

    """Get the address for the given address type can be an enum value for lookup, or a string for direct use."""

    def create_client(
        self,
        client_type: CommClientType,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
        max_pull_concurrency: int | None = None,
        **kwargs: Any,
    ) -> CommunicationClientProtocol:
        """Create a client for the given client type and address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_pub_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> PubClientProtocol:
        """Create a PUB client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_sub_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> SubClientProtocol:
        """Create a SUB client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_push_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> PushClientProtocol:
        """Create a PUSH client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_pull_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
        max_pull_concurrency: int | None = None,
    ) -> PullClientProtocol:
        """Create a PULL client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_request_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> RequestClientProtocol:
        """Create a REQUEST client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_reply_client(
        self,
        address: CommAddressType,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> ReplyClientProtocol:
        """Create a REPLY client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_streaming_router_client(
        self,
        address: CommAddressType,
        bind: bool = True,
        socket_ops: dict | None = None,
    ) -> StreamingRouterClientProtocol:
        """Create a STREAMING_ROUTER client for the given address, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...

    def create_streaming_dealer_client(
        self,
        address: CommAddressType,
        identity: str,
        bind: bool = False,
        socket_ops: dict | None = None,
    ) -> StreamingDealerClientProtocol:
        """Create a STREAMING_DEALER client for the given address and identity, which will be automatically
        started and stopped with the CommunicationProtocol instance."""
        ...


@runtime_checkable
class MessageBusClientProtocol(PubClientProtocol, SubClientProtocol, Protocol):
    """A message bus client is a client that can publish and subscribe to messages
    on the event bus/message bus."""

    comms: CommunicationProtocol
    sub_client: SubClientProtocol
    pub_client: PubClientProtocol


################################################################################
# General Protocols (sorted alphabetically)
################################################################################


@runtime_checkable
class AIPerfUIProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol interface definition for AIPerf UI implementations.

    Basically a UI can be any class that implements the AIPerfLifecycleProtocol. However, in order to provide
    progress tracking and worker tracking, the simplest way would be to inherit from the :class:`aiperf.ui.base_ui.BaseAIPerfUI`.
    """


@runtime_checkable
class ConsoleExporterProtocol(Protocol):
    """Protocol for console exporters.
    Any class implementing this protocol will be provided an ExporterConfig and must provide an
    `export` method that takes a rich Console and handles exporting them appropriately.
    """

    def __init__(self, exporter_config: "ExporterConfig") -> None: ...

    async def export(self, console: "Console") -> None: ...


@runtime_checkable
class CustomDatasetLoaderProtocol(Protocol):
    """Protocol for custom dataset loaders that load dataset from a file and convert it to a list of Conversation objects."""

    @classmethod
    def can_load(
        cls, data: dict[str, Any] | None = None, filename: "str | Path | None" = None
    ) -> bool:
        """Check if this loader can handle the given data format.

        Args:
            data: Optional dictionary representing a single line from the JSONL file.
                  None indicates path-based detection only (e.g., for directories).
            filename: Optional path to the input file/directory for path-based detection

        Returns:
            True if this loader can handle the data format, False otherwise
        """
        ...

    @classmethod
    def get_preferred_sampling_strategy(cls) -> "DatasetSamplingStrategy":
        """Get the preferred dataset sampling strategy for this loader.

        Returns:
            DatasetSamplingStrategy: The preferred sampling strategy
        """
        ...

    def load_dataset(self) -> dict[str, list["CustomDatasetT"]]: ...

    def convert_to_conversations(
        self, custom_data: dict[str, list["CustomDatasetT"]]
    ) -> list[Conversation]: ...


@runtime_checkable
class DataExporterProtocol(Protocol):
    """
    Protocol for data exporters.
    Any class implementing this protocol will be provided an ExporterConfig and must provide an
    `export` method that handles exporting the data appropriately.
    """

    def __init__(self, exporter_config: "ExporterConfig") -> None: ...

    def get_export_info(self) -> "FileExportInfo": ...

    async def export(self) -> None: ...


@runtime_checkable
class DatasetSamplingStrategyProtocol(Protocol):
    """
    Protocol for dataset sampling strategies.
    Any class implementing this protocol will be provided a list of conversation ids, and must
    provide a `next_conversation_id` method that returns the next conversation id, ensuring reproducibility.
    """

    def __init__(self, conversation_ids: list[str], **kwargs) -> None: ...
    def next_conversation_id(self) -> str: ...


@runtime_checkable
class EndpointProtocol(Protocol):
    """Protocol for an endpoint."""

    def __init__(self, model_endpoint: "ModelEndpointInfo", **kwargs) -> None: ...

    @classmethod
    def metadata(cls) -> "EndpointMetadata": ...

    def format_payload(self, request_info: RequestInfo) -> RequestOutputT: ...
    def extract_response_data(self, record: RequestRecord) -> list[ParsedResponse]: ...
    def get_endpoint_headers(self, request_info: RequestInfo) -> dict[str, str]: ...
    def get_endpoint_params(self, request_info: RequestInfo) -> dict[str, str]: ...


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


@runtime_checkable
class HooksProtocol(Protocol):
    """Protocol for hooks methods provided by the HooksMixin."""

    def get_hooks(self, *hook_types: HookType, reversed: bool = False) -> list[Hook]:
        """Get the hooks for the given hook type(s), optionally reversed."""
        ...

    async def run_hooks(
        self, *hook_types: HookType, reversed: bool = False, **kwargs
    ) -> None:
        """Run the hooks for the given hook type, waiting for each hook to complete before running the next one.
        If reversed is True, the hooks will be run in reverse order. This is useful for stop/cleanup starting with
        the children and ending with the parent.
        """
        ...


@runtime_checkable
class ServiceManagerProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for a service manager that manages the running of services using the specific ServiceRunType.
    Abstracts away the details of service deployment and management.
    see :class:`aiperf.controller.base_service_manager.BaseServiceManager` for more details.
    """

    def __init__(
        self,
        required_services: dict[ServiceTypeT, int],
        service_config: "ServiceConfig",
        user_config: "UserConfig",
        log_queue: "multiprocessing.Queue | None" = None,
    ): ...

    required_services: dict[ServiceTypeT, int]
    service_map: dict[ServiceTypeT, list[ServiceRunInfo]]
    service_id_map: dict[str, ServiceRunInfo]

    async def run_service(
        self, service_type: ServiceTypeT, num_replicas: int = 1
    ) -> None: ...

    async def run_services(self, service_types: dict[ServiceTypeT, int]) -> None: ...
    async def run_required_services(self) -> None: ...
    async def shutdown_all_services(self) -> list[BaseException | None]: ...
    async def kill_all_services(self) -> list[BaseException | None]: ...
    async def stop_service(
        self, service_type: ServiceTypeT, service_id: str | None = None
    ) -> list[BaseException | None]: ...
    async def stop_services_by_type(
        self, service_types: list[ServiceTypeT]
    ) -> list[BaseException | None]: ...
    async def wait_for_all_services_registration(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.REGISTRATION_TIMEOUT,
    ) -> None: ...

    async def wait_for_all_services_start(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = Environment.SERVICE.START_TIMEOUT,
    ) -> None: ...


@runtime_checkable
class ServiceProtocol(MessageBusClientProtocol, Protocol):
    """Protocol for a service. Essentially a MessageBusClientProtocol with a service_type and service_id attributes."""

    def __init__(
        self,
        user_config: "UserConfig",
        service_config: "ServiceConfig",
        service_id: str | None = None,
        **kwargs,
    ) -> None: ...

    service_type: ServiceTypeT
    service_id: str


@runtime_checkable
class RecordProcessorProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for a record processor that processes the incoming records and returns the results of the post processing."""

    async def process_record(
        self, record: ParsedResponseRecord, metadata: MetricRecordMetadata
    ) -> "MetricRecordDict": ...


@runtime_checkable
class ResultsProcessorProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for a results processor that processes the results of multiple
    record processors, and provides the ability to summarize the results."""

    async def process_result(self, record_data: "MetricRecordsData") -> None: ...

    async def summarize(self) -> list["MetricResult"]: ...


@runtime_checkable
class TelemetryResultsProcessorProtocol(Protocol):
    """Protocol for telemetry results processors that handle TelemetryRecord objects.

    This protocol is separate from ResultsProcessorProtocol because telemetry data
    has fundamentally different structure (hierarchical with metadata) compared
    to inference metrics (flat key-value pairs).
    """

    async def process_telemetry_record(self, record: TelemetryRecord) -> None:
        """Process individual telemetry record with rich metadata.

        Args:
            record: TelemetryRecord containing GPU metrics and hierarchical metadata
        """
        ...

    async def summarize(self) -> list["MetricResult"]:
        """Generate MetricResult list with hierarchical tags for telemetry data.

        Returns:
            List of MetricResult objects with hierarchical tags that preserve
            dcgm_url -> gpu_uuid grouping structure for dashboard filtering.
        """
        ...


@runtime_checkable
class RequestRateGeneratorProtocol(Protocol):
    """Protocol for a request rate generator that generates the next interval for a request rate."""

    def __init__(self, config: "TimingManagerConfig") -> None: ...

    def next_interval(self) -> float: ...


@runtime_checkable
class TransportProtocol(AIPerfLifecycleProtocol, Protocol):
    """Protocol for a transport that sends requests to an inference server."""

    def __init__(self, **kwargs) -> None: ...

    @classmethod
    def metadata(cls) -> "TransportMetadata": ...

    def get_transport_headers(self, request_info: RequestInfo) -> dict[str, str]: ...

    def build_headers(self, request_info: RequestInfo) -> dict[str, str]: ...

    def build_url(self, request_info: RequestInfo) -> str: ...

    def get_url(self, request_info: RequestInfo) -> str: ...

    async def send_request(
        self, request_info: RequestInfo, payload: RequestInputT
    ) -> RequestRecord: ...
