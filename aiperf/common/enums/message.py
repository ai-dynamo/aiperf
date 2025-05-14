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
from aiperf.common.enums.base import StrEnum


# Message-related enums
class MessageType(StrEnum):
    """Types of messages exchanged between services."""

    UNKNOWN = "unknown"
    REGISTRATION = "registration"
    HEARTBEAT = "heartbeat"
    COMMAND = "command"
    RESPONSE = "response"
    STATUS = "status"
    DATA = "data"
    ERROR = "error"
    CONVERSATION = "conversation"
    RESULT = "result"
    WORKER_REQUEST = "worker_request"
    WORKER_RESPONSE = "worker_response"
    CREDIT_DROP = "credit_drop"
    CREDIT_RETURN = "credit_return"


class CommandType(StrEnum):
    """Commands that can be sent to services."""

    START = "start"
    STOP = "stop"
    CONFIGURE = "configure"
    STATUS = "status"
