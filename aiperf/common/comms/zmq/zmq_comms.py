# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import errno
import glob
import logging
import os
from abc import ABC
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, cast

import zmq.asyncio

from aiperf.common.comms.base import BaseCommunication
from aiperf.common.comms.client_enums import (
    ClientType,
    PubClientType,
    PullClientType,
    PushClientType,
    RepClientType,
    ReqClientType,
    SubClientType,
)
from aiperf.common.comms.zmq.clients import (
    ZMQClient,
    ZMQDealerReqClient,
    ZMQPubClient,
    ZMQPullClient,
    ZMQPushClient,
    ZMQRouterRepClient,
    ZMQSubClient,
)
from aiperf.common.config import (
    BaseZMQCommunicationConfig,
    ZMQInprocConfig,
    ZMQIPCConfig,
    ZMQTCPTransportConfig,
)
from aiperf.common.enums import CommunicationBackend, MessageType, Topic
from aiperf.common.exceptions import (
    CommunicationError,
    CommunicationErrorReason,
)
from aiperf.common.factories import CommunicationFactory
from aiperf.common.messages import Message

logger = logging.getLogger(__name__)


class BaseZMQCommunication(BaseCommunication, ABC):
    """ZeroMQ-based implementation of the Communication interface.

    Uses ZeroMQ for publish/subscribe and request/reply patterns to
    facilitate communication between AIPerf components.
    """

    def __init__(
        self,
        config: BaseZMQCommunicationConfig | None = None,
    ) -> None:
        """Initialize ZMQ communication.

        Args:
            config: ZMQCommunicationConfig object with configuration parameters
        """
        self.stop_event: asyncio.Event = asyncio.Event()
        self.initialized_event: asyncio.Event = asyncio.Event()
        self.config = config or ZMQIPCConfig()

        self._context: zmq.asyncio.Context | None = None
        self.clients: dict[ClientType, ZMQClient] = {}

        logger.debug(
            "ZMQ communication using protocol: %s",
            type(self.config).__name__,
        )

    @property
    def context(self) -> zmq.asyncio.Context:
        """Get the ZeroMQ context.

        Returns:
            ZeroMQ context
        """
        if not self._context:
            raise CommunicationError(
                CommunicationErrorReason.INITIALIZATION_ERROR,
                "Communication channels are not initialized",
            )
        return self._context

    @property
    def is_initialized(self) -> bool:
        """Check if communication channels are initialized.

        Returns:
            True if communication channels are initialized, False otherwise
        """
        return self.initialized_event.is_set()

    @property
    def is_shutdown(self) -> bool:
        """Check if communication channels are shutdown.

        Returns:
            True if communication channels are shutdown, False otherwise
        """
        return self.stop_event.is_set()

    async def initialize(self) -> None:
        """Initialize communication channels.

        Returns:
            True if initialization was successful, False otherwise
        """
        if self.is_initialized:
            return

        # Increase the number of IO threads to 2
        self._context = zmq.asyncio.Context(io_threads=2)
        self.initialized_event.set()

    async def shutdown(self) -> None:
        """Gracefully shutdown communication channels.

        This method will wait for all clients to shutdown before shutting down
        the context.

        Returns:
            True if shutdown was successful, False otherwise
        """
        if self.is_shutdown:
            return

        try:
            if not self.stop_event.is_set():
                self.stop_event.set()

            await asyncio.gather(
                *(client.shutdown() for client in self.clients.values())
            )

            if self.context:
                self.context.term()

            self._context = None

        except asyncio.CancelledError:
            pass

        except Exception as e:
            raise CommunicationError(
                CommunicationErrorReason.SHUTDOWN_ERROR,
                "Failed to shutdown ZMQ communication",
            ) from e

        finally:
            self.clients = {}
            self._context = None

    def _ensure_initialized(self) -> None:
        """Ensure the communication channels are initialized.

        Raises:
            CommunicationError: If the communication channels are not initialized
                or shutdown
            asyncio.CancelledError: If the communication channels are shutdown
        """
        if not self.is_initialized:
            raise CommunicationError(
                CommunicationErrorReason.INITIALIZATION_ERROR,
                "Communication channels are not initialized",
            )
        if self.is_shutdown:
            raise asyncio.CancelledError()

    def _create_pub_client(self, client_type: PubClientType) -> ZMQPubClient:
        """Create a ZMQ publisher client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQPubClient instance

        Raises:
            CommunicationError: If the client type is invalid
        """
        match client_type:
            case PubClientType.CONTROLLER:
                return ZMQPubClient(
                    self.context,
                    self.config.controller_pub_sub_address,
                    bind=True,
                )

            case PubClientType.COMPONENT:
                return ZMQPubClient(
                    self.context,
                    self.config.component_pub_sub_address,
                    bind=False,
                )

            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"Invalid client type: {client_type}",
                )

    def _create_sub_client(self, client_type: SubClientType) -> ZMQSubClient:
        """Create a ZMQ subscriber client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQSubClient instance

        Raises:
            CommunicationError: If the client type is invalid
        """
        match client_type:
            case SubClientType.CONTROLLER:
                return ZMQSubClient(
                    self.context,
                    self.config.controller_pub_sub_address,
                    bind=False,
                )

            case SubClientType.COMPONENT:
                return ZMQSubClient(
                    self.context,
                    self.config.component_pub_sub_address,
                    bind=True,
                )

            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"Invalid client type: {client_type}",
                )

    def _create_push_client(self, client_type: PushClientType) -> ZMQPushClient:
        """Create a ZMQ push client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQPushClient instance

        Raises:
            CommunicationError: If the client type is invalid
        """
        match client_type:
            case PushClientType.INFERENCE_RESULTS:
                return ZMQPushClient(
                    self.context,
                    self.config.inference_push_pull_address,
                    bind=False,  # Workers are the pushers
                    socket_ops={zmq.SNDHWM: 0},  # Unlimited send queue
                )

            case PushClientType.CREDIT_DROP:
                return ZMQPushClient(
                    self.context,
                    self.config.credit_drop_address,
                    bind=True,
                )

            case PushClientType.CREDIT_RETURN:
                return ZMQPushClient(
                    self.context,
                    self.config.credit_return_address,
                    bind=False,
                )

            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"Invalid client type: {client_type}",
                )

    def _create_pull_client(self, client_type: PullClientType) -> ZMQPullClient:
        """Create a ZMQ pull client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQPullClient instance

        Raises:
            CommunicationError: If the client type is invalid
        """
        match client_type:
            case PullClientType.INFERENCE_RESULTS:
                return ZMQPullClient(
                    self.context,
                    self.config.inference_push_pull_address,
                    bind=True,  # Records manager is the pull
                    socket_ops={
                        zmq.RCVBUF: 1024 * 1024 * 32,  # 1GB OS buffer
                        zmq.RCVHWM: 0,  # Unlimited ZMQ queue
                    },
                )

            case PullClientType.CREDIT_DROP:
                return ZMQPullClient(
                    self.context,
                    self.config.credit_drop_address,
                    bind=False,
                )

            case PullClientType.CREDIT_RETURN:
                return ZMQPullClient(
                    self.context,
                    self.config.credit_return_address,
                    bind=True,
                )

            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"Invalid client type: {client_type}",
                )

    def _create_req_client(self, client_type: ReqClientType) -> ZMQDealerReqClient:
        """Create a ZMQ request client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQReqClient instance

        Raises:
            CommunicationError: If the client type is invalid
        """
        match client_type:
            case ReqClientType.CONVERSATION_DATA:
                return ZMQDealerReqClient(
                    self.context,
                    self.config.conversation_data_address,
                    bind=False,  # Worker manager is the request
                )
            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"Invalid client type: {client_type}",
                )

    def _create_rep_client(self, client_type: RepClientType) -> ZMQRouterRepClient:
        """Create a ZMQ reply client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQRepClient instance

        Raises:
            CommunicationError: If the client type is invalid
        """
        match client_type:
            case RepClientType.CONVERSATION_DATA:
                return ZMQRouterRepClient(
                    self.context,
                    self.config.conversation_data_address,
                    bind=True,  # Data manager is the reply
                )

            case _:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"Invalid client type: {client_type}",
                )

    async def create_clients(self, *types: ClientType) -> None:
        """Create and initialize ZMQ clients based on the client types.

        Args:
            types: List of ClientType enums indicating the types of clients to
            create and initialize

        Raises:
            CommunicationError: If the clients were not created
                successfully
        """

        for client_type in types:
            if client_type in self.clients:
                continue

            if isinstance(client_type, PubClientType):
                client = self._create_pub_client(client_type)

            elif isinstance(client_type, SubClientType):
                client = self._create_sub_client(client_type)

            elif isinstance(client_type, PushClientType):
                client = self._create_push_client(client_type)

            elif isinstance(client_type, PullClientType):
                client = self._create_pull_client(client_type)

            elif isinstance(client_type, ReqClientType):
                client = self._create_req_client(client_type)

            elif isinstance(client_type, RepClientType):
                client = self._create_rep_client(client_type)

            else:
                raise CommunicationError(
                    CommunicationErrorReason.CLIENT_NOT_FOUND,
                    f"Invalid client type: {client_type}",
                )

            await client.initialize()

            self.clients[client_type] = client

    async def publish(self, topic: Topic, message: Message) -> None:
        """Publish a message to a topic. If the client type is not found, it will
        be created.

        Args:
            topic: The topic to publish the message to
            message: The message to publish

        Raises:
            CommunicationError: If the message was not published successfully
        """

        self._ensure_initialized()
        client_type = PubClientType.from_topic(topic)

        if client_type not in self.clients:
            logger.debug(
                "Client type %r not found for pub topic %r, creating client",
                client_type,
                topic,
            )
            await self.create_clients(client_type)

        try:
            await cast(ZMQPubClient, self.clients[client_type]).publish(topic, message)
        except Exception as e:
            logger.error(
                "Exception publishing message to topic: %s, message: %s, error: %s",
                topic,
                message,
                e,
            )
            raise CommunicationError(
                CommunicationErrorReason.PUBLISH_ERROR,
                f"Failed to publish message to topic: {topic}, message: {message}, error: {e}",
            ) from e

    async def subscribe(
        self,
        topic: Topic,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to a topic. If the proper ZMQ client type is not found, it
        will be created.

        Args:
            topic: The topic to subscribe to
            callback: The callback to call when a message is received

        Raises:
            CommunicationError: If there was an error subscribing to the
                topic, or if the communication channels are not initialized
                or shutdown
        """

        self._ensure_initialized()

        client_type = SubClientType.from_topic(topic)

        if client_type not in self.clients:
            logger.debug(
                "Client type %r not found for sub topic %r, creating client",
                client_type,
                topic,
            )
            await self.create_clients(client_type)

        try:
            await cast(ZMQSubClient, self.clients[client_type]).subscribe(
                topic, callback
            )
        except Exception as e:
            logger.error(f"Exception subscribing to topic: {e}")
            raise CommunicationError(
                CommunicationErrorReason.SUBSCRIBE_ERROR,
                f"Failed to subscribe to topic: {topic}, error: {e}",
            ) from e

    async def request(
        self,
        topic: Topic,
        message: Message,
    ) -> Message:
        """Request a message from a target. If the proper ZMQ client type is not
        found, it will be created.

        Args:
            topic: The topic to request from
            message: The message to request

        Returns:
            The response from the target

        Raises:
            CommunicationError: If there was an error requesting from the
                target, or if the communication channels are not initialized
                or shutdown
        """

        self._ensure_initialized()

        client_type = ReqClientType.from_topic(topic)

        if client_type not in self.clients:
            logger.debug(
                "Client type %r not found for req topic %r, creating client",
                client_type,
                topic,
            )
            await self.create_clients(client_type)

        try:
            return await cast(ZMQDealerReqClient, self.clients[client_type]).request(
                message
            )
        except Exception as e:
            logger.error(f"Exception requesting from {topic}: {e}")
            raise CommunicationError(
                CommunicationErrorReason.REQUEST_ERROR,
                f"Failed to request from topic: {topic}, error: {e}",
            ) from e

    async def register_request_handler(
        self,
        service_id: str,
        topic: Topic,
        message_type: MessageType,
        handler: Callable[[Message], Coroutine[Any, Any, Message | None]],
    ) -> None:
        """Register a request handler for a topic.

        Args:
            service_id: The service ID to register the handler for
            topic: The topic to register the handler for
            message_type: The message type to register the handler for
            handler: The handler to register
        """

        self._ensure_initialized()

        client_type = RepClientType.from_topic(topic)

        if client_type not in self.clients:
            logger.debug(
                "Client type %r not found for req topic %r, creating client",
                client_type,
                topic,
            )
            await self.create_clients(client_type)

        try:
            cast(
                ZMQRouterRepClient, self.clients[client_type]
            ).register_request_handler(service_id, message_type, handler)
        except Exception as e:
            logger.error(f"Exception registering request handler for {topic}: {e}")
            raise CommunicationError(
                CommunicationErrorReason.REQUEST_ERROR,
                f"Failed to register request handler for topic: {topic}, error: {e}",
            ) from e

    async def push(self, topic: Topic, message: Message) -> None:
        """Push a message to a topic. If the proper ZMQ client type is not found,
        it will be created.

        Args:
            topic: The topic to push the message to
            message: The message to push

        Raises:
            CommunicationError: If there was an error pushing the message, or if the
                communication channels are not initialized or shutdown
        """

        self._ensure_initialized()

        client_type = PushClientType.from_topic(topic)

        if client_type not in self.clients:
            logger.debug(
                "Client type %r not found for push, creating client",
                client_type,
            )
            await self.create_clients(client_type)

        try:
            await cast(ZMQPushClient, self.clients[client_type]).push(message)
        except Exception as e:
            logger.error(f"Exception pushing data: {e}")
            raise CommunicationError(
                CommunicationErrorReason.PUSH_ERROR,
                f"Failed to push data to topic: {topic}, error: {e}",
            ) from e

    async def register_pull_callback(
        self,
        message_type: MessageType,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a callback for a pull client.

        Args:
            message_type: The message type to register the callback for
            callback: The callback to register

        Raises:
            CommunicationError: If there was an error registering the callback, or if
                the communication channels are not initialized
        """

        logger.debug(f"Pulling data for {message_type}")

        self._ensure_initialized()

        client_type = PullClientType.from_message_type(message_type)

        if client_type not in self.clients:
            logger.debug(
                "Client type %r not found for pull, creating client",
                client_type,
            )
            await self.create_clients(client_type)

        # Only adds to the callback list, does not block, and does not raise an exception
        await cast(ZMQPullClient, self.clients[client_type]).register_pull_callback(
            message_type, callback
        )


@CommunicationFactory.register(CommunicationBackend.ZMQ_TCP)
class ZMQTCPCommunication(BaseZMQCommunication):
    """ZeroMQ-based implementation of the Communication interface using TCP transport."""

    def __init__(self, config: ZMQTCPTransportConfig | None = None) -> None:
        """Initialize ZMQ TCP communication.

        Args:
            config: ZMQTCPTransportConfig object with configuration parameters
        """
        super().__init__(config or ZMQTCPTransportConfig())


@CommunicationFactory.register(CommunicationBackend.ZMQ_IPC)
class ZMQIPCCommunication(BaseZMQCommunication):
    """ZeroMQ-based implementation of the Communication interface using IPC transport."""

    def __init__(self, config: ZMQIPCConfig | None = None) -> None:
        """Initialize ZMQ IPC communication.

        Args:
            config: ZMQIPCConfig object with configuration parameters
        """
        super().__init__(config or ZMQIPCConfig())

    async def initialize(self) -> None:
        """Initialize communication channels.

        This method will create the IPC socket directory if needed.

        Raises:
            CommunicationError: If the communication channels are not initialized
                or shutdown
        """
        await super().initialize()
        self._setup_ipc_directory()

    async def shutdown(self) -> None:
        """Gracefully shutdown communication channels.

        This method will wait for all clients to shutdown before shutting down
        the context.

        Raises:
            CommunicationError: If there was an error shutting down the communication
                channels
        """
        await super().shutdown()
        self._cleanup_ipc_sockets()

    def _setup_ipc_directory(self) -> None:
        """Create IPC socket directory if using IPC transport."""
        self._ipc_socket_dir = Path(self.config.path)
        self._ipc_socket_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created IPC socket directory: {self._ipc_socket_dir}")

    def _cleanup_ipc_sockets(self) -> None:
        """Clean up IPC socket files."""
        if self._ipc_socket_dir and self._ipc_socket_dir.exists():
            # Remove all .ipc files in the directory
            ipc_files = glob.glob(str(self._ipc_socket_dir / "*.ipc"))
            for ipc_file in ipc_files:
                try:
                    if os.path.exists(ipc_file):
                        os.unlink(ipc_file)
                        logger.debug(f"Removed IPC socket file: {ipc_file}")
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        logger.warning(
                            "Failed to remove IPC socket file %s: %s",
                            ipc_file,
                            e,
                        )


@CommunicationFactory.register(CommunicationBackend.ZMQ_INPROC)
class ZMQInprocCommunication(ZMQIPCCommunication):
    """ZeroMQ-based implementation of the Communication interface using in-process
    transport. Note that communications between workers is still done over IPC sockets,
    which is why this class inherits from ZMQIPCCommunication."""

    def __init__(self, config: ZMQInprocConfig | None = None) -> None:
        """Initialize ZMQ in-process communication.

        Args:
            config: ZMQInprocConfig object with configuration parameters
        """
        super().__init__(config or ZMQInprocConfig())
