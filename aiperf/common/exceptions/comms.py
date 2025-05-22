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

from aiperf.common.exceptions.base import AIPerfError


class CommunicationError(AIPerfError):
    """Base class for all communication exceptions."""

    message: str = "Communication exception occurred"


class CommunicationNotInitializedError(CommunicationError):
    """Exception raised when communication channels are not initialized."""

    message: str = "Communication channels are not initialized"


class CommunicationInitializationError(CommunicationError):
    """Exception raised when communication channels fail to initialize."""

    message: str = "Failed to initialize communication channels"


class CommunicationPublishError(CommunicationError):
    """Exception raised when communication channels fail to publish a message."""

    message: str = "Failed to publish message"


class CommunicationShutdownError(CommunicationError):
    """Exception raised when communication channels fail to shutdown."""

    message: str = "Failed to shutdown communication channels"


class CommunicationSubscribeError(CommunicationError):
    """Exception raised when communication channels fail to subscribe to a topic."""

    message: str = "Failed to subscribe to a topic"


class CommunicationPullError(CommunicationError):
    """Exception raised when communication channels fail to pull a message from
    a topic."""

    message: str = "Failed to pull a message from a topic"


class CommunicationPushError(CommunicationError):
    """Exception raised when communication channels fail to push a message to
    a topic."""

    message: str = "Failed to push a message to a topic"


class CommunicationRequestError(CommunicationError):
    """Exception raised when communication channels fail to send a request."""

    message: str = "Failed to send a request"


class CommunicationResponseError(CommunicationError):
    """Exception raised when communication channels fail to receive a response."""

    message: str = "Failed to receive a response"


class CommunicationClientCreationError(CommunicationError):
    """Exception raised when communication channels fail to create a client."""

    message: str = "Failed to create a client"


class CommunicationClientNotFoundError(CommunicationError):
    """Exception raised when a communication client is not found."""

    message: str = "Communication client not found"


class CommunicationCreateError(CommunicationError):
    """Exception raised when communication channels fail to create a client."""

    message: str = "Failed to create a communication client"


class CommunicationTypeUnknownError(CommunicationError):
    """Exception raised when the communication type is unknown."""

    message: str = "Communication type is unknown"


class CommunicationTypeAlreadyRegisteredError(CommunicationError):
    """Exception raised when the communication type is already registered."""

    message: str = "Communication type is already registered"
