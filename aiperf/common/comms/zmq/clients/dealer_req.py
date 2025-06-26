# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import uuid

import zmq.asyncio

from aiperf.common.comms.zmq.clients.base import BaseZMQClient
from aiperf.common.exceptions import CommunicationError, CommunicationErrorReason
from aiperf.common.messages import Message


class ZMQDealerReqClient(BaseZMQClient):
    def __init__(
        self,
        context: zmq.asyncio.Context,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
    ) -> None:
        """
        Initialize the ZMQ Dealer (Req) client class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(context, zmq.SocketType.DEALER, address, bind, socket_ops)

    async def request(
        self,
        request_data: Message,
    ) -> Message:
        """Send a request and wait for a response.

        Args:
            request_data: Request data (must be a Message object)

        Returns:
            ResponseData object
        """
        self._ensure_initialized()

        # Generate request ID if not provided
        if not request_data.request_id:
            request_data.request_id = uuid.uuid4().hex

        request_json = request_data.model_dump_json()

        try:
            # Send request
            await self._socket.send_string(request_json)

            # Wait for response with timeout
            response_json = await self._socket.recv_string()
            response = Message.from_json(response_json)
            return response

        except Exception as e:
            raise CommunicationError(
                CommunicationErrorReason.REQUEST_ERROR,
                f"Exception sending request: {e.__class__.__name__} {e}",
            ) from e
