#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import asyncio
import logging
import uuid

import zmq.asyncio
from zmq import SocketType

from aiperf.common.comms.zmq.clients.base import BaseZMQClient
from aiperf.common.decorators import aiperf_task, on_cleanup
from aiperf.common.exceptions import (
    CommunicationRequestError,
)
from aiperf.common.models.message import BaseMessage, Message
from aiperf.common.models.payload import ErrorPayload

logger = logging.getLogger(__name__)


class ZMQReqClient(BaseZMQClient):
    def __init__(
        self,
        context: zmq.asyncio.Context,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
    ) -> None:
        """
        Initialize the ZMQ Req class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(context, SocketType.REQ, address, bind, socket_ops)
        self._response_futures = {}

    @aiperf_task
    async def _process_messages(self) -> None:
        """Process incoming response messages in the background.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for messages from the socket and handle them.
        """
        while not self.is_shutdown:
            try:
                if not self.is_initialized:
                    await self.initialized_event.wait()

                response_json = await self.socket.recv_string()
                await self._handle_response(response_json)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Exception processing messages: {e}")
                await asyncio.sleep(0.1)

    async def _handle_response(self, response_json: str) -> None:
        """Handle a response message.

        Args:
            response_json: The JSON response string
        """
        try:
            response = BaseMessage.model_validate_json(response_json)
            request_id = response.request_id

            if request_id in self._response_futures:
                future = self._response_futures[request_id]
                if not future.done():
                    future.set_result(response_json)
            else:
                logger.warning(
                    f"Received response for unknown request ID: {request_id}"
                )
        except Exception as e:
            logger.error(f"Exception handling response: {e}")
            raise CommunicationRequestError("Exception handling response") from e

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up any pending futures."""
        # Resolve any pending futures with errors
        for request_id, future in self._response_futures.items():
            if not future.done():
                error_response = BaseMessage(
                    request_id=request_id,
                    payload=ErrorPayload(
                        error_message="Socket was shut down",
                    ),
                )
                future.set_result(error_response.model_dump_json())

        self._response_futures.clear()

    async def request(
        self,
        target: str,
        request_data: Message,
        timeout: float = 5.0,
    ) -> Message:
        """Send a request and wait for a response.

        Args:
            target: Target component to send request to
            request_data: Request data (must be a RequestData instance)
            timeout: Timeout in seconds

        Returns:
            ResponseData object
        """
        self._ensure_initialized()

        try:
            # Set target if not already set
            if not request_data.target:
                request_data.target = target

            # Ensure client_id is set
            if not request_data.client_id:
                request_data.client_id = self.client_id

            # Generate request ID if not provided
            if not request_data.request_id:
                request_data.request_id = uuid.uuid4().hex

            # Serialize request
            request_json = request_data.model_dump_json()

            # Create future for response
            future = asyncio.Future()
            self._response_futures[request_data.request_id] = future

            # Send request
            await self.socket.send_string(request_json)

            # Wait for response with timeout
            try:
                response_json = await asyncio.wait_for(future, timeout)
                response = BaseMessage.model_validate_json(response_json)
                return response

            except asyncio.TimeoutError as e:
                logger.error(
                    f"Timeout waiting for response to request {request_data.request_id}"
                )
                raise CommunicationRequestError(
                    f"Timeout waiting for response to request {request_data.request_id}"
                ) from e

            finally:
                # Clean up future
                self._response_futures.pop(request_data.request_id, None)

        except Exception as e:
            logger.error(f"Exception sending request to {target}: {e}")
            raise CommunicationRequestError(
                f"Exception sending request to {target}: {e}"
            ) from e
