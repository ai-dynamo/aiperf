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

import zmq.asyncio
from zmq import SocketType

from aiperf.common.comms.zmq.clients.base import BaseZMQClient
from aiperf.common.exceptions import (
    CommunicationResponseError,
)
from aiperf.common.hooks import aiperf_task, on_cleanup
from aiperf.common.models import BaseMessage, Message

logger = logging.getLogger(__name__)


class ZMQRepClient(BaseZMQClient):
    def __init__(
        self,
        context: zmq.asyncio.Context,
        address: str,
        bind: bool,
        socket_ops: dict | None = None,
    ) -> None:
        """
        Initialize the ZMQ REP class.

        Args:
            context (zmq.asyncio.Context): The ZMQ context.
            address (str): The address to bind or connect to.
            bind (bool): Whether to bind or connect the socket.
            socket_ops (dict, optional): Additional socket options to set.
        """
        super().__init__(context, SocketType.REP, address, bind, socket_ops)

        self._response_futures = {}
        self._response_data = {}

    @on_cleanup
    async def _cleanup(self) -> None:
        """Clean up any pending futures."""
        # Resolve any pending futures with errors
        for _, future in self._response_futures.items():
            if not future.done():
                future.set_exception(ConnectionError("Socket was shut down"))

        self._response_futures.clear()
        self._response_data.clear()

    async def wait_for_request(self, timeout: float | None = None) -> Message:
        """Wait for a request to arrive.

        Args:
            timeout: Timeout in seconds or None for no timeout

        Returns:
            Message or Exception if request was not received successfully

        Raises:
            CommunicationNotInitializedError: If the client is not initialized
            CommunicationResponseError: If the request was not received successfully
        """
        self._ensure_initialized()

        try:
            # Create a future for the next request
            request_id = "next_request"  # Special ID for the next request
            future = asyncio.Future()
            self._response_futures[request_id] = future

            try:
                # Wait for the request with optional timeout
                if timeout is not None:
                    request_json = await asyncio.wait_for(future, timeout)
                else:
                    request_json = await future

                # Parse the request
                request = BaseMessage.model_validate_json(request_json)
                return request

            except asyncio.TimeoutError as e:
                logger.debug("Timeout waiting for request")
                raise CommunicationResponseError("Timeout waiting for request") from e

            except Exception as e:
                logger.error(f"Exception waiting for request: {e}")
                raise CommunicationResponseError("Exception waiting for request") from e

            finally:
                # Clean up future
                self._response_futures.pop(request_id, None)

        except Exception as e:
            logger.error(f"Exception waiting for request: {e}")
            raise CommunicationResponseError("Exception waiting for request") from e

    async def respond(self, target: str, response: Message) -> None:
        """Send a response to a request.

        Args:
            target: Target component to send response to
            response: Response message (must be a Message instance)

        Raises:
            CommunicationNotInitializedError: If the client is not initialized
            CommunicationResponseError: If the response was not sent successfully
        """
        self._ensure_initialized()

        try:
            # Serialize response using Pydantic's built-in method
            response_json = response.model_dump_json()

            # Send response
            await self.socket.send_string(response_json)

        except Exception as e:
            logger.error(f"Exception sending response to {target}: {e}")
            raise CommunicationResponseError(
                f"Exception sending response to {target}"
            ) from e

    @aiperf_task
    async def _rep_receiver(self) -> None:
        """Background task for receiving requests and sending responses.

        This method is a coroutine that will run indefinitely until the client is
        shutdown. It will wait for requests from the socket and send responses.
        """
        while not self.is_shutdown:
            try:
                if not self.is_initialized:
                    await self.initialized_event.wait()

                # Receive request
                request_json = await self.socket.recv_string()

                # Parse JSON to create RequestData object
                request = BaseMessage.model_validate_json(request_json)
                request_id = request.request_id

                # Store request data
                self._response_data[request_id] = request

                # Check for special "next_request" future
                if "next_request" in self._response_futures:
                    future = self._response_futures.pop("next_request")
                    if not future.done():
                        future.set_result(request_json)
                # Resolve future if it exists for the specific request ID
                elif request_id in self._response_futures:
                    future = self._response_futures[request_id]
                    if not future.done():
                        future.set_result(request_json)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Exception receiving request: {e}")
                await asyncio.sleep(0.1)
