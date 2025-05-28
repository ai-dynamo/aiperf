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

################################################################################
# Base Exceptions
################################################################################


class AIPerfError(Exception):
    """Base class for all exceptions raised by AIPerf."""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {super().__str__()}"


class AIPerfMultiError(AIPerfError):
    """Exception raised when running multiple tasks and one or more fail."""

    def __init__(self, exceptions) -> None:
        super().__init__(f"{','.join([str(e) for e in exceptions])}")
        self.exceptions = exceptions


################################################################################
# Communication Exceptions
################################################################################


class CommunicationError(AIPerfError):
    """Base class for all communication exceptions."""


class CommunicationNotInitializedError(CommunicationError):
    """Exception raised when communication channels are not initialized."""


class CommunicationInitializationError(CommunicationError):
    """Exception raised when communication channels fail to initialize."""


class CommunicationPublishError(CommunicationError):
    """Exception raised when communication channels fail to publish a message."""


class CommunicationShutdownError(CommunicationError):
    """Exception raised when communication channels fail to shutdown."""


class CommunicationSubscribeError(CommunicationError):
    """Exception raised when communication channels fail to subscribe to a topic."""


class CommunicationPullError(CommunicationError):
    """Exception raised when communication channels fail to pull a message from
    a topic."""


class CommunicationPushError(CommunicationError):
    """Exception raised when communication channels fail to push a message to
    a topic."""


class CommunicationRequestError(CommunicationError):
    """Exception raised when communication channels fail to send a request."""


class CommunicationResponseError(CommunicationError):
    """Exception raised when communication channels fail to receive a response."""


class CommunicationClientCreationError(CommunicationError):
    """Exception raised when communication channels fail to create a client."""


class CommunicationClientNotFoundError(CommunicationError):
    """Exception raised when a communication client is not found."""


class CommunicationCreateError(CommunicationError):
    """Exception raised when communication channels fail to create a client."""


class CommunicationTypeUnknownError(CommunicationError):
    """Exception raised when the communication type is unknown."""


class CommunicationTypeAlreadyRegisteredError(CommunicationError):
    """Exception raised when the communication type is already registered."""


################################################################################
# Configuration Exceptions
################################################################################


class ConfigError(AIPerfError):
    """Base class for all exceptions raised by configuration errors."""


class ConfigLoadError(ConfigError):
    """Exception raised for configuration load errors."""


class ConfigParseError(ConfigError):
    """Exception raised for configuration parse errors."""


class ConfigValidationError(ConfigError):
    """Exception raised for configuration validation errors."""


################################################################################
# Dataset Generator Exceptions
################################################################################


class GeneratorError(AIPerfError):
    """Base class for all exceptions raised by data generator modules."""


class GeneratorInitializationError(GeneratorError):
    """Exception raised for data generator initialization errors."""


class GeneratorConfigurationError(GeneratorError):
    """Exception raised for data generator configuration errors."""


################################################################################
# Service Exceptions
################################################################################


class ServiceError(AIPerfError):
    """Base class for all exceptions raised by services."""

    # TODO: possibly have the base exception class accept the service information
    #       and add it to the pre-defined messages for each exception


class ServiceInitializationError(ServiceError):
    """Exception raised for service initialization errors."""


class ServiceStartError(ServiceError):
    """Exception raised for service start errors."""


class ServiceStopError(ServiceError):
    """Exception raised for service stop errors."""


class ServiceCleanupError(ServiceError):
    """Exception raised for service cleanup errors."""


class ServiceMessageProcessingError(ServiceError):
    """Exception raised for service message processing errors."""


class ServiceRegistrationError(ServiceError):
    """Exception raised for service registration errors."""


class ServiceStatusError(ServiceError):
    """Exception raised for service status errors."""


class ServiceRunError(ServiceError):
    """Exception raised for service run errors."""


class ServiceConfigureError(ServiceError):
    """Exception raised for service configure errors."""


class ServiceHeartbeatError(ServiceError):
    """Exception raised for service heartbeat errors."""


################################################################################
# Tokenizer Exceptions
################################################################################


class TokenizerError(AIPerfError):
    """Base class for tokenizer exceptions."""


class TokenizerInitializationError(TokenizerError):
    """Exception raised for errors during tokenizer initialization."""
