from aiperf.common.exceptions.base import AIPerfException


class ConfigurationException(AIPerfException):
    """Base class for all exceptions raised by configuration errors."""

    pass


class ConfigurationLoadException(ConfigurationException):
    """Exception raised for configuration load errors."""

    pass


class ConfigurationParseException(ConfigurationException):
    """Exception raised for configuration parse errors."""

    pass


class ConfigurationValidationException(ConfigurationException):
    """Exception raised for configuration validation errors."""

    pass
