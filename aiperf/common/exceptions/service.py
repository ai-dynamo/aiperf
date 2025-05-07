from aiperf.common.exceptions.base import AIPerfException


class ServiceException(AIPerfException):
    """Base class for all exceptions raised by services."""

    pass


class ServiceInitializationException(ServiceException):
    """Exception raised for service initialization errors."""

    pass


class ServiceStartException(ServiceException):
    """Exception raised for service start errors."""

    pass


class ServiceStopException(ServiceException):
    """Exception raised for service stop errors."""

    pass


class ServiceCleanupException(ServiceException):
    """Exception raised for service cleanup errors."""

    pass


class ServiceMessageProcessingException(ServiceException):
    """Exception raised for service message processing errors."""

    pass


class ServiceRegistrationException(ServiceException):
    """Exception raised for service registration errors."""

    pass


class ServiceStatusException(ServiceException):
    """Exception raised for service status errors."""

    pass
