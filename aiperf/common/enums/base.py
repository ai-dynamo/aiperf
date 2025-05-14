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
from enum import Enum

__all__ = [
    "ServiceState",
    "MessageType",
    "CommandType",
    "Topic",
    "CommBackend",
    "ServiceRunType",
]


class ServiceState(Enum):
    """Enum representing the possible states of a service."""

    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    READY = "ready"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class MessageType(Enum):
    """Enum representing the types of messages that can be exchanged between services."""

    REGISTRATION = "registration"
    HEARTBEAT = "heartbeat"
    COMMAND = "command"
    RESPONSE = "response"
    STATUS = "status"
    DATA = "data"
    ERROR = "error"
    CREDIT = "credit"


class CommandType(Enum):
    """Enum representing the types of commands that can be sent to services."""

    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    CONFIGURE = "configure"
    PROFILE = "profile"
    SHUTDOWN = "shutdown"
    STATUS = "status"
    HEALTH_CHECK = "health_check"


class Topic(Enum):
    """Enum representing the different topics for communication between services."""

    REGISTRATION = "registration"
    COMMAND = "command"
    RESPONSE = "response"
    DATA = "data"
    STATUS = "status"
    HEARTBEAT = "heartbeat"


class CommBackend(Enum):
    """Enum representing the different communication backends."""

    ZMQ = "zmq"
    MEMORY = "memory"


class ServiceRunType(Enum):
    """Enum representing the different ways to run a service."""

    ASYNC = "async"
    MULTIPROCESSING = "process"
    KUBERNETES = "k8s"
