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
from aiperf.common.enums import StrEnum


# Message-related enums
class MessageType(StrEnum):
    """The various types of messages that can be sent between services.

    The message type is used to determine what Pydantic model the payload maps to.
    The mappings between message types and payload types are defined in the
    payload definitions.
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
    """A message sent by a component service to the system controller to
    report an error."""

    CREDIT_DROP = "credit_drop"
    """A message sent by the Timing Manager service to allocate credits
    for a worker."""

    CREDIT_RETURN = "credit_return"
    """A message sent by the Worker services to return credits to the credit pool."""

    DATA = "data"
    """A message containing data. This is TBD."""


class CommandType(StrEnum):
    """List of commands that the SystemController can send to component services."""

    START = "start"
    """A command to start the service. The service should have already
    been configured."""

    STOP = "stop"
    """A command to stop the service."""

    CONFIGURE = "configure"
    """A command to configure the service with the configuration present
    in the payload."""
