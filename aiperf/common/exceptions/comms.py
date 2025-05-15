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

from aiperf.common.exceptions.base import AIPerfException


class CommunicationNotInitializedError(AIPerfException):
    """Exception raised when communication channels are not initialized."""

    message: str = "Communication channels are not initialized"


class CommunicationInitializationError(AIPerfException):
    """Exception raised when communication channels fail to initialize."""

    message: str = "Communication channels failed to initialize"


class CommunicationPublishError(AIPerfException):
    """Exception raised when communication channels fail to publish a response."""

    message: str = "Communication channels failed to publish a response"


class CommunicationShutdownError(AIPerfException):
    """Exception raised when communication channels fail to shutdown."""

    message: str = "Communication channels failed to shutdown"


class CommunicationSubscribeError(AIPerfException):
    """Exception raised when communication channels fail to subscribe to a topic."""

    message: str = "Communication channels failed to subscribe to a topic"


class CommunicationPullError(AIPerfException):
    """Exception raised when communication channels fail to pull a response from
    a topic."""

    message: str = "Communication channels failed to pull a response from a topic"


class CommunicationPushError(AIPerfException):
    """Exception raised when communication channels fail to push a response to
    a topic."""

    message: str = "Communication channels failed to push a response to a topic"


class CommunicationRequestError(AIPerfException):
    """Exception raised when communication channels fail to send a request."""

    message: str = "Communication channels failed to send a request"


class CommunicationResponseError(AIPerfException):
    """Exception raised when communication channels fail to receive a response."""

    message: str = "Communication channels failed to receive a response"
