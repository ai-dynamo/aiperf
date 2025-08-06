# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import multiprocessing
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from aiperf.common.constants import (
    DEFAULT_COMMS_REQUEST_TIMEOUT,
    DEFAULT_SERVICE_REGISTRATION_TIMEOUT,
    DEFAULT_SERVICE_START_TIMEOUT,
)
from aiperf.common.enums import (
    CommClientType,
    LifecycleState,
)
from aiperf.common.hooks import Hook, HookType
from aiperf.common.models import (
    ParsedResponseRecord,
    RequestRecord,
    ResponseData,
    ServiceRunInfo,
    Turn,
)
from aiperf.common.tokenizer import Tokenizer
from aiperf.common.types import (
    CommAddressType,
    MessageCallbackMapT,
    MessageOutputT,
    MessageT,
    MessageTypeT,
    MetricTagT,
    ModelEndpointInfoT,
    RequestInputT,
    RequestOutputT,
    ServiceTypeT,
)

if TYPE_CHECKING:
    from aiperf.common.config import ServiceConfig, UserConfig
    from aiperf.common.enums.metric_enums import MetricValueTypeT
    from aiperf.common.models.record_models import MetricResult
    from aiperf.metrics.metric_dicts import MetricRecordDict


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
        timeout: float = DEFAULT_COMMS_REQUEST_TIMEOUT,
    ) -> MessageOutputT: ...

    async def request_async(
        self,
        message: MessageT,
        callback: Callable[[MessageOutputT], Coroutine[Any, Any, None]],
    ) -> None: ...


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
class DataExporterProtocol(Protocol):
    """
    Protocol for data exporters.
    Any class implementing this protocol must provide an `export` method
    that takes a list of Record objects and handles exporting them appropriately.
    """

    async def export(self) -> None: ...


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
class InferenceClientProtocol(Protocol):
    """Protocol for an inference server client.

    This protocol defines the methods that must be implemented by any inference server client
    implementation that is compatible with the AIPerf framework.
    """

    def __init__(self, model_endpoint: ModelEndpointInfoT) -> None:
        """Create a new inference server client based on the provided configuration."""
        ...

    async def initialize(self) -> None:
        """Initialize the inference server client in an asynchronous context."""
        ...

    async def send_request(
        self,
        model_endpoint: ModelEndpointInfoT,
        payload: RequestInputT,
    ) -> RequestRecord:
        """Send a request to the inference server.

        This method is used to send a request to the inference server.

        Args:
            model_endpoint: The endpoint to send the request to.
            payload: The payload to send to the inference server.
        Returns:
            The raw response from the inference server.
        """
        ...

    async def close(self) -> None:
        """Close the client."""
        ...


@runtime_checkable
class ResponseExtractorProtocol(Protocol):
    """Protocol for a response extractor that extracts the response data from a raw inference server
    response and converts it to a list of ResponseData objects."""

    async def extract_response_data(
        self, record: RequestRecord, tokenizer: Tokenizer | None
    ) -> list[ResponseData]:
        """Extract the response data from a raw inference server response and convert it to a list of ResponseData objects."""
        ...


@runtime_checkable
class RequestConverterProtocol(Protocol):
    """Protocol for a request converter that converts a raw request to a formatted request for the inference server."""

    async def format_payload(
        self, model_endpoint: ModelEndpointInfoT, turn: Turn
    ) -> RequestOutputT:
        """Format the turn for the inference server."""
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
        timeout_seconds: float = DEFAULT_SERVICE_REGISTRATION_TIMEOUT,
    ) -> None: ...

    async def wait_for_all_services_start(
        self,
        stop_event: asyncio.Event,
        timeout_seconds: float = DEFAULT_SERVICE_START_TIMEOUT,
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
class RecordProcessorProtocol(Protocol):
    """Protocol for a record processor that processes the incoming records and returns the results of the post processing."""

    async def process_record(
        self, record: ParsedResponseRecord
    ) -> "MetricRecordDict": ...


@runtime_checkable
class ResultsProcessorProtocol(Protocol):
    """Protocol for a results processor that processes the results of multiple
    record processors, and provides the ability to summarize the results."""

    async def process_result(
        self, result: dict[MetricTagT, "MetricValueTypeT"]
    ) -> None: ...

    async def summarize(self) -> list["MetricResult"]: ...
