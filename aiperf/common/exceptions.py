# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from aiperf.common.enums import ServiceType

################################################################################
# Base Exceptions
################################################################################


class AIPerfError(Exception):
    """Base class for all exceptions raised by AIPerf."""

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {super().__str__()}"


class AIPerfMultiError(AIPerfError):
    """Exception raised when running multiple tasks and one or more fail."""

    def __init__(self, message: str, exceptions: list[Exception]) -> None:
        super().__init__(f"{message}: {','.join([str(e) for e in exceptions])}")
        self.exceptions = exceptions


class CommunicationError(AIPerfError):
    """Generic communication error."""


class ProxyError(AIPerfError):
    """Exception raised when a proxy encounters an error."""


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

    def __init__(
        self,
        message: str,
        service_type: ServiceType,
        service_id: str,
    ) -> None:
        super().__init__(
            f"{message} for service of type {service_type} with id {service_id}"
        )
        self.service_type = service_type
        self.service_id = service_id


################################################################################
# Tokenizer Exceptions
################################################################################


class TokenizerError(AIPerfError):
    """Base class for tokenizer exceptions."""


class TokenizerInitializationError(TokenizerError):
    """Exception raised for errors during tokenizer initialization."""


################################################################################
# Hook Exceptions
################################################################################


class UnsupportedHookError(AIPerfError):
    """Exception raised when a hook is defined on a class that does not support it."""


################################################################################
# Factory Exceptions
################################################################################


class FactoryCreationError(AIPerfError):
    """Exception raised when a factory encounters an error while creating a class."""


################################################################################
# Metric Exceptions
################################################################################


class MetricTypeError(AIPerfError):
    """Exception raised when a metric type encounters an error while creating a class."""
