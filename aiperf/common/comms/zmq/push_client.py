# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import zmq.asyncio

from aiperf.common.comms.zmq.zmq_base_client import BaseZMQClient
from aiperf.common.decorators import implements_protocol
from aiperf.common.enums import CommClientType
from aiperf.common.exceptions import CommunicationError
from aiperf.common.factories import CommunicationClientFactory
from aiperf.common.messages import Message
from aiperf.common.protocols import PushClientProtocol

MAX_PUSH_RETRIES = 2
"""Maximum number of retries for pushing a message."""

RETRY_DELAY_INTERVAL_SEC = 0.1
"""The interval to wait before retrying to push a message."""


@implements_protocol(PushClientProtocol)
@CommunicationClientFactory.register(CommClientType.PUSH)
class ZMQPushClient(BaseZMQClient):
    """
    ZMQ PUSH socket client for sending work to PULL sockets.

    The PUSH socket sends messages to PULL sockets in a pipeline pattern,
    distributing work fairly among available PULL workers.

    ASCII Diagram:
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │    PUSH     │      │    PULL     │      │    PULL     │
    │ (Producer)  │      │ (Worker 1)  │      │ (Worker 2)  │
    │             │      └─────────────┘      └─────────────┘
    │   Tasks:    │             ▲                     ▲
    │   - Task A  │─────────────┘                     │
    │   - Task B  │───────────────────────────────────┘
    │   - Task C  │─────────────┐
    │   - Task D  │             ▼
    └─────────────┘      ┌─────────────┐
                         │    PULL     │
                         │ (Worker 3)  │
                         └─────────────┘

    Usage Pattern:
    - Round-robin distribution of work tasks (One-to-Many)
    - Each message delivered to exactly one worker
    - Pipeline pattern for distributed processing
    - Automatic load balancing across available workers

    PUSH/PULL is a One-to-Many communication pattern. If you need Many-to-Many,
    use a ZMQ Proxy as well. see :class:`ZMQPushPullProxy` for more details.
    """

    def __init__(
        self,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize the ZMQ Push client class.

        Args:
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(zmq.SocketType.PUSH, address, bind, socket_ops, **kwargs)

    async def _push_message(
        self,
        message: Message,
        retry_count: int = 0,
        max_retries: int = MAX_PUSH_RETRIES,
    ) -> None:
        """Push a message to the socket. Will retry up to max_retries times.

        Args:
            message: Message to be sent must be a Message object
            retry_count: Current retry count
            max_retries: Maximum number of times to retry pushing the message
        """
        try:
            data_json = message.model_dump_json()
            await self.socket.send_string(data_json)
            self.trace(lambda msg=data_json: f"Pushed json data: {msg}")
        except (asyncio.CancelledError, zmq.ContextTerminated):
            self.debug("Push client cancelled or context terminated")
            return
        except zmq.Again as e:
            self.debug("Push client timed out")
            if retry_count >= max_retries:
                raise CommunicationError(
                    f"Failed to push data after {retry_count} retries: {e}",
                ) from e

            await asyncio.sleep(RETRY_DELAY_INTERVAL_SEC)
            return await self._push_message(message, retry_count + 1, max_retries)
        except Exception as e:
            raise CommunicationError(f"Failed to push data: {e}") from e

    async def push(self, message: Message) -> None:
        """Push data to a target. The message will be routed automatically
        based on the message type.

        Args:
            message: Message to be sent must be a Message object
        """
        await self._check_initialized()

        await self._push_message(message)
