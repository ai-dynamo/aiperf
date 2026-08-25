# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiperf.common.types import ServiceTypeT


class AIPerfError(Exception):
    """Base class for all exceptions raised by AIPerf."""

    def raw_str(self) -> str:
        """Return the raw string representation of the exception."""
        return super().__str__()

    def __str__(self) -> str:
        """Return the string representation of the exception with the class name."""
        return super().__str__()


class AIPerfMultiError(AIPerfError):
    """Exception raised when running multiple tasks and one or more fail."""

    def __init__(self, message: str | None, exceptions: list[Exception]) -> None:
        self.exceptions = exceptions

        err_strings = [
            e.raw_str() if isinstance(e, AIPerfError) else str(e) for e in exceptions
        ]
        if message:
            super().__init__(f"{message}: {','.join(err_strings)}")
        else:
            super().__init__(",".join(err_strings))


class HookError(AIPerfError):
    """Exception raised when a hook encounters an error."""

    def __init__(self, hook_class_name: str, hook_func_name: str, e: Exception) -> None:
        self.hook_class_name = hook_class_name
        self.hook_func_name = hook_func_name
        self.exception = e
        super().__init__(f"{hook_class_name}.{hook_func_name}: {e}")


class ServiceError(AIPerfError):
    """Generic service error."""

    def __init__(
        self,
        message: str,
        service_type: "ServiceTypeT",
        service_id: str,
    ) -> None:
        super().__init__(
            f"{message} for service of type {service_type} with id {service_id}"
        )
        self.service_type = service_type
        self.service_id = service_id


class LifecycleOperationError(AIPerfError):
    """Exception raised when a lifecycle operation fails and the lifecycle should stop gracefully."""

    def __init__(
        self,
        operation: str,
        original_exception: Exception | None,
        lifecycle_id: str,
    ) -> None:
        self.operation = operation
        self.original_exception = original_exception
        self.lifecycle_id = lifecycle_id
        super().__init__(
            str(original_exception)
            if original_exception
            else f"Failed to perform operation '{operation}'"
        )


class CommunicationError(AIPerfError):
    """Generic communication error."""


class ConfigurationError(AIPerfError):
    """Exception raised when something fails to configure, or there is a configuration error."""


class ConsoleExporterDisabled(AIPerfError):
    """Raised when initializing a console exporter to indicate to the caller that it is disabled and should not be used."""


class DataExporterDisabled(AIPerfError):
    """Raised when initializing a data exporter to indicate to the caller that it is disabled and should not be used."""


class DatasetError(AIPerfError):
    """Generic dataset error."""


class DatasetLoaderError(AIPerfError):
    """Generic dataset loader error."""


class DatasetGeneratorError(AIPerfError):
    """Generic dataset generator error."""


class IncompatibleMetricsEndpointError(AIPerfError):
    """Raised when an HTTP metrics endpoint returns content that cannot be
    interpreted as Prometheus exposition format (e.g. a JSON body, or text
    that fails the Prometheus parser). Indicates a structural mismatch the
    collector cannot recover from by retrying — the affected collector should
    auto-disable rather than spam parse errors at the configured interval.

    A representative trigger is the TensorRT-LLM ``/metrics`` endpoint, which
    serves an iteration-stats JSON array (``application/json``) at the same
    path Prometheus scrapers expect.
    """


class InitializationError(AIPerfError):
    """Exception raised when something fails to initialize."""


class InferenceClientError(AIPerfError):
    """Exception raised when a inference client encounters an error."""


class InvalidInferenceResultError(AIPerfError):
    """Exception raised when an inference result is invalid."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
        self.notes = []

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def __str__(self) -> str:
        return f"{self.message}: {', '.join(self.notes)}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.__str__()}')"


class InvalidOperationError(AIPerfError):
    """Exception raised when an operation is invalid."""


class InvalidPayloadError(InferenceClientError):
    """Exception raised when a inference client receives an invalid payload."""


class InvalidStateError(AIPerfError):
    """Exception raised when something is in an invalid state."""


class MemoryMapDatasetError(AIPerfError):
    """Base exception for memory-mapped dataset errors."""


class MemoryMapSerializationError(MemoryMapDatasetError):
    """Exception raised when serialization/deserialization of mmap data fails."""


class MemoryMapFileOperationError(MemoryMapDatasetError):
    """Exception raised when file operations on mmap files fail."""


class MetricTypeError(AIPerfError):
    """Exception raised when a metric type encounters an error while creating a class."""


class MetricUnitError(AIPerfError):
    """Exception raised when trying to convert a metric to or from a unit that is does not support it."""


class NotFoundError(AIPerfError):
    """Exception raised when something is not found or not available."""


class NotInitializedError(AIPerfError):
    """Exception raised when something that should be initialized is not."""


class NoMetricValue(AIPerfError):
    """Raised when a metric value is not available."""


class PluginNotFoundError(AIPerfError):
    """Exception raised when a plugin is not found. This is used to indicate that a plugin is not found when trying to get a plugin class or metadata."""


class PluginDisabled(AIPerfError):
    """Raised when initializing an accumulator or stream exporter to indicate it is disabled and should not be loaded."""


class PostProcessorDisabled(PluginDisabled):
    """Raised when initializing a post processor to indicate to the caller that it is disabled and should not be used."""


class ProxyError(AIPerfError):
    """Exception raised when a proxy encounters an error."""


class ServiceProcessDiedError(AIPerfError):
    """Raised when a managed service subprocess exits before it was asked to stop.

    Names the service, its type, and (when the OS reported one) its exit code
    or terminating signal, so the controller can distinguish a crash or OOM
    kill from a clean shutdown without re-reading process tables.

    Example:
        >>> raise ServiceProcessDiedError(
        ...     service_id="worker_0_a1b2", service_type="worker", exit_code=-9
        ... )
        Traceback (most recent call last):
        aiperf.common.exceptions.ServiceProcessDiedError: Service process
        'worker_0_a1b2' (worker) died unexpectedly with exit code -9 (likely
        killed by signal 9, e.g. an out-of-memory kill or an external SIGKILL)
    """

    def __init__(
        self,
        *,
        service_id: str,
        service_type: "ServiceTypeT",
        exit_code: int | None = None,
    ) -> None:
        self.service_id = service_id
        self.service_type = service_type
        self.exit_code = exit_code
        if exit_code is None:
            cause = (
                "no exit code was reported, which usually means the process was "
                "reaped elsewhere or the handle was closed before it was waited on"
            )
        elif exit_code < 0:
            cause = (
                f"likely killed by signal {-exit_code}, e.g. an out-of-memory kill "
                f"or an external SIGKILL"
            )
        else:
            cause = (
                "likely an unhandled exception during startup or a failed "
                "dependency connection; check the service's own log output"
            )
        detail = "" if exit_code is None else f" with exit code {exit_code}"
        super().__init__(
            f"Service process {service_id!r} ({service_type}) died unexpectedly"
            f"{detail} ({cause})"
        )


class ServiceRegistrationTimeoutError(AIPerfError, TimeoutError):
    """Raised when not every expected service registered before the startup deadline.

    Carries the registered/expected counts and the per-service-type shortfall
    so the operator can report exactly which pods never came up, rather than a
    bare "timed out". Also a ``TimeoutError`` so callers that only care about
    the timeout class can catch it generically.

    Example:
        >>> raise ServiceRegistrationTimeoutError(
        ...     registered=6, expected=8, timeout_sec=120.0, missing={"worker": 2}
        ... )
        Traceback (most recent call last):
        aiperf.common.exceptions.ServiceRegistrationTimeoutError: Only 6 of 8
        services registered within 120.0s; still missing: worker x2 (likely a
        service that crashed on startup, a pod that never scheduled, or a
        message-bus address the service could not reach)
    """

    def __init__(
        self,
        *,
        registered: int,
        expected: int,
        timeout_sec: float | None,
        missing: dict[str, int],
        context: str | None = None,
    ) -> None:
        self.registered = registered
        self.expected = expected
        self.timeout_sec = timeout_sec
        self.missing = missing
        self.context = context
        missing_detail = (
            ", ".join(f"{name} x{count}" for name, count in sorted(missing.items()))
            or "unknown service types"
        )
        # A caller waiting on specific service IDs knows names the aggregate
        # counts cannot express ("ghost"), so its own description leads.
        prefix = f"{context}. " if context else ""
        window = "with no timeout" if timeout_sec is None else f"within {timeout_sec}s"
        super().__init__(
            f"{prefix}Only {registered} of {expected} services registered "
            f"{window}; still missing: {missing_detail} (likely a service "
            f"that crashed on startup, a pod that never scheduled, or a "
            f"message-bus address the service could not reach)"
        )


class ShutdownError(AIPerfError):
    """Exception raised when a service encounters an error while shutting down."""


class SSEResponseError(AIPerfError):
    """Exception raised when a SSE response contains an error."""

    def __init__(self, message: str, error_code: int = 500) -> None:
        self.error_code = error_code
        super().__init__(message)


class TokenizerError(AIPerfError):
    """Exception raised when a tokenizer fails to load or encounters an error."""

    def __init__(self, message: str, tokenizer_name: str | None = None) -> None:
        super().__init__(message)
        self.tokenizer_name = tokenizer_name


class UnsupportedHookError(AIPerfError):
    """Exception raised when a hook is defined on a class that does not have any base classes that provide that hook type."""


class ValidationError(AIPerfError):
    """Exception raised when something fails validation."""
