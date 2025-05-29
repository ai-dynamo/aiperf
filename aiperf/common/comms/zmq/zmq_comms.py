# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import logging
import uuid
from collections.abc import Callable, Coroutine
from typing import Any

import zmq.asyncio

from aiperf.common.comms import CommunicationFactory
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
from aiperf.common.comms.zmq.clients import ZMQClient
from aiperf.common.comms.zmq.clients.pub import ZMQPubClient
from aiperf.common.comms.zmq.clients.pull import ZMQPullClient
from aiperf.common.comms.zmq.clients.push import ZMQPushClient
from aiperf.common.comms.zmq.clients.rep import ZMQRepClient
from aiperf.common.comms.zmq.clients.req import ZMQReqClient
from aiperf.common.comms.zmq.clients.sub import ZMQSubClient
from aiperf.common.enums import CommunicationBackend, TopicType
from aiperf.common.exceptions import (
    CommunicationClientCreationError,
    CommunicationNotInitializedError,
    CommunicationPublishError,
    CommunicationPushError,
    CommunicationRequestError,
    CommunicationResponseError,
    CommunicationShutdownError,
    CommunicationSubscribeError,
)
from aiperf.common.models import Message, ZMQCommunicationConfig

logger = logging.getLogger(__name__)


@CommunicationFactory.register(CommunicationBackend.ZMQ_TCP)
class ZMQCommunication(BaseCommunication):
    """ZeroMQ-based implementation of the Communication interface.

    Uses ZeroMQ for publish/subscribe and request/reply patterns to
    facilitate communication between AIPerf components.
    """

    def __init__(
        self,
        config: ZMQCommunicationConfig | None = None,
    ) -> None:
        """Initialize ZMQ communication.

        Args:
            config: ZMQCommunicationConfig object with configuration parameters
        """
        self.stop_event: asyncio.Event = asyncio.Event()
        self.initialized_event: asyncio.Event = asyncio.Event()
        self.config = config or ZMQCommunicationConfig()

        # Generate client_id if not provided
        if not self.config.client_id:
            self.config.client_id = f"client_{uuid.uuid4().hex[:8]}"

        self._context: zmq.asyncio.Context | None = None
        self.clients: dict[ClientType, ZMQClient] = {}

        logger.debug(
            "ZMQ communication using protocol: %s with client ID: %s",
            type(self.config.protocol_config).__name__,
            self.config.client_id,
        )

    @property
    def context(self) -> zmq.asyncio.Context:
        """Get the ZeroMQ context.

        Returns:
            ZeroMQ context
        """
        if not self._context:
            raise CommunicationNotInitializedError()
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

        self._context = zmq.asyncio.Context()
        self.initialized_event.set()

    async def shutdown(self) -> None:
        """Gracefully shutdown communication channels.

        This method will wait for all clients to shutdown before shutting down
        the context.

        Returns:
            True if shutdown was successful, False otherwise
        """
        if self.is_shutdown:
            logger.debug("ZMQ communication already shutdown")
            return

        try:
            if not self.stop_event.is_set():
                self.stop_event.set()

            logger.debug(
                f"Shutting down ZMQ communication for client {self.config.client_id}"
            )
            await asyncio.gather(
                *(client.shutdown() for client in self.clients.values())
            )

            if self.context:
                self.context.term()

            self._context = None
            logger.debug("ZMQ communication shutdown successfully")

        except Exception as e:
            logger.error(f"Exception shutting down ZMQ communication: {e}")
            raise CommunicationShutdownError(
                "Failed to shutdown ZMQ communication"
            ) from e

        finally:
            self.clients = {}
            self._context = None

    def _ensure_initialized(self) -> None:
        """Ensure the communication channels are initialized.

        Raises:
            CommunicationNotInitializedError: If the communication channels are not
                initialized
            CommunicationShutdownError: If the communication channels are shutdown
        """
        if not self.is_initialized:
            raise CommunicationNotInitializedError()
        if self.is_shutdown:
            raise CommunicationShutdownError()

    def _create_pub_client(self, client_type: PubClientType) -> ZMQPubClient:
        """Create a ZMQ publisher client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQPubClient instance

        Raises:
            CommunicationClientCreationError: If the client type is invalid
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
                raise CommunicationClientCreationError(
                    f"Invalid client type: {client_type}"
                )

    def _create_sub_client(self, client_type: SubClientType) -> ZMQSubClient:
        """Create a ZMQ subscriber client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQSubClient instance

        Raises:
            CommunicationClientCreationError: If the client type is invalid
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
                raise CommunicationClientCreationError(
                    f"Invalid client type: {client_type}"
                )

    def _create_push_client(self, client_type: PushClientType) -> ZMQPushClient:
        """Create a ZMQ push client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQPushClient instance

        Raises:
            CommunicationClientCreationError: If the client type is invalid
        """
        match client_type:
            case PushClientType.INFERENCE_RESULTS:
                return ZMQPushClient(
                    self.context,
                    self.config.inference_push_pull_address,
                    bind=False,  # Workers are the pushers
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

            case PushClientType.RECORDS:
                return ZMQPushClient(
                    self.context,
                    self.config.records_address,
                    bind=False,
                )

            case _:
                raise CommunicationClientCreationError(
                    f"Invalid client type: {client_type}"
                )

    def _create_pull_client(self, client_type: PullClientType) -> ZMQPullClient:
        """Create a ZMQ pull client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQPullClient instance

        Raises:
            CommunicationClientCreationError: If the client type is invalid
        """
        match client_type:
            case PullClientType.INFERENCE_RESULTS:
                return ZMQPullClient(
                    self.context,
                    self.config.inference_push_pull_address,
                    bind=True,  # Records manager is the pull
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

            case PushClientType.RECORDS:
                return ZMQPullClient(
                    self.context,
                    self.config.records_address,
                    bind=True,
                )

            case _:
                raise CommunicationClientCreationError(
                    f"Invalid client type: {client_type}"
                )

    def _create_req_client(self, client_type: ReqClientType) -> ZMQReqClient:
        """Create a ZMQ request client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQReqClient instance

        Raises:
            CommunicationClientCreationError: If the client type is invalid
        """
        match client_type:
            case ReqClientType.CONVERSATION_DATA:
                return ZMQReqClient(
                    self.context,
                    self.config.conversation_data_address,
                    bind=False,  # Worker manager is the request
                )
            case _:
                raise CommunicationClientCreationError(
                    f"Invalid client type: {client_type}"
                )

    def _create_rep_client(self, client_type: RepClientType) -> ZMQRepClient:
        """Create a ZMQ reply client based on the client type.

        Args:
            client_type: The type of client to create

        Returns:
            ZMQRepClient instance

        Raises:
            CommunicationClientCreationError: If the client type is invalid
        """
        match client_type:
            case RepClientType.CONVERSATION_DATA:
                return ZMQRepClient(
                    self.context,
                    self.config.conversation_data_address,
                    bind=True,  # Data manager is the reply
                )

            case _:
                raise CommunicationClientCreationError(
                    f"Invalid client type: {client_type}"
                )

    async def create_clients(self, *types: ClientType) -> None:
        """Create and initialize ZMQ clients based on the client types.

        Args:
            types: List of ClientType enums indicating the types of clients to
            create and initialize

        Raises:
            CommunicationClientCreationError: If the clients were not created
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
                raise CommunicationClientCreationError(
                    f"Invalid client type: {client_type}"
                )

            await client.initialize()

            self.clients[client_type] = client

    async def publish(self, topic: TopicType, message: Message) -> None:
        """Publish a message to a topic. If the client type is not found, it will
        be created.

        Args:
            topic: The topic to publish the message to
            message: The message to publish

        Raises:
            CommunicationPublishError: If the message was not published successfully
        """

        self._ensure_initialized()
        client_type = PubClientType.from_topic(topic)

        if client_type not in self.clients:
            logger.warning(
                "Client type %s not found for pub topic %s, creating client",
                client_type,
                topic,
            )
            await self.create_clients(client_type)

        try:
            await self.clients[client_type].publish(topic, message)
        except Exception as e:
            logger.error(
                "Exception publishing message to topic: %s, message: %s, error: %s",
                topic,
                message,
                e,
            )
            raise CommunicationPublishError() from e

    async def subscribe(
        self,
        topic: TopicType,
        callback: Callable[[Message], Coroutine[Any, Any, None]],
    ) -> None:
        """Subscribe to a topic. If the proper ZMQ client type is not found, it
        will be created.

        Args:
            topic: The topic to subscribe to
            callback: The callback to call when a message is received

        Raises:
            CommunicationSubscribeError: If there was an error subscribing to the
                topic
            CommunicationNotInitializedError: If the communication channels are not
                initialized
            CommunicationShutdownError: If the communication channels are shutdown
        """

        self._ensure_initialized()

        client_type = SubClientType.from_topic(topic)

        if client_type not in self.clients:
            logger.debug(
                "Client type %s not found for sub topic %s, creating client",
                client_type,
                topic,
            )
            await self.create_clients(client_type)

        try:
            await self.clients[client_type].subscribe(topic, callback)
        except Exception as e:
            logger.error(f"Exception subscribing to topic: {e}")
            raise CommunicationSubscribeError() from e

    async def request(
        self,
        target: str,
        request_data: Message,
        timeout: float = 5.0,
    ) -> Message:
        """Request a message from a target. If the proper ZMQ client type is not
        found, it will be created.

        Args:
            target: The target to request from
            request_data: The data to request
            timeout: The timeout for the request

        Returns:
            The response from the target

        Raises:
            CommunicationRequestError: If there was an error requesting from the
                target
            CommunicationNotInitializedError: If the communication channels are not
                initialized
            CommunicationShutdownError: If the communication channels are shutdown
        """

        self._ensure_initialized()

        client_type = ReqClientType.from_topic(target)

        if client_type not in self.clients:
            logger.warning(
                "Client type %s not found for req topic %s, creating client",
                client_type,
                target,
            )
            await self.create_clients(client_type)

        try:
            return await self.clients[client_type].request(
                target, request_data, timeout
            )
        except Exception as e:
            logger.error(f"Exception requesting from {target}: {e}")
            raise CommunicationRequestError() from e

    async def respond(self, target: str, response: Message) -> None:
        """Respond to a request. If the proper ZMQ client type is not found, it
        will be created.

        Args:
            target: The target to respond to
            response: The response to send

        Raises:
            CommunicationRespondError: If there was an error responding to the
                target
            CommunicationNotInitializedError: If the communication channels are not
                initialized
            CommunicationShutdownError: If the communication channels are shutdown
        """

        self._ensure_initialized()

        client_type = RepClientType.from_topic(target)

        if client_type not in self.clients:
            logger.warning(
                "Client type %s not found for rep topic %s, creating client",
                client_type,
                target,
            )
            await self.create_clients(client_type)

        try:
            await self.clients[client_type].respond(target, response)
        except Exception as e:
            logger.error(f"Exception responding to {target}: {e}")
            raise CommunicationResponseError() from e

    async def push(self, topic: TopicType, message: Message) -> None:
        """Push a message to a topic. If the proper ZMQ client type is not found,
        it will be created.

        Args:
            topic: The topic to push the message to
            message: The message to push

        Raises:
            CommunicationPushError: If there was an error pushing the message
            CommunicationNotInitializedError: If the communication channels are not
                initialized
            CommunicationShutdownError: If the communication channels are shutdown
        """

        self._ensure_initialized()

        client_type = PushClientType.from_topic(topic)

        if client_type not in self.clients:
            logger.warning(
                "Client type %s not found for push, creating client",
                client_type,
            )
            await self.create_clients(client_type)

        try:
            await self.clients[client_type].push(message)
        except Exception as e:
            logger.error(f"Exception pushing data: {e}")
            raise CommunicationPushError() from e

    async def pull(
        self,
        topic: TopicType,
        callback: Callable[[Message], None],
    ) -> None:
        """Pull a message from a topic. If the proper ZMQ client type is not found,
        it will be created.

        Args:
            topic: The topic to pull the message from
            callback: The callback to call when a message is received

        Raises:
            CommunicationPullError: If there was an error pulling the message
            CommunicationNotInitializedError: If the communication channels are not
                initialized
            CommunicationShutdownError: If the communication channels are shutdown
        """

        logger.debug(f"Pulling data from {topic}")

        self._ensure_initialized()

        client_type = PullClientType.from_topic(topic)

        if client_type not in self.clients:
            logger.warning(
                "Client type %s not found for pull, creating client",
                client_type,
            )
            await self.create_clients(client_type)

        # Only adds to the callback list, does not block, and does not raise an exception
        await self.clients[client_type].pull(topic, callback)
