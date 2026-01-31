# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
from collections.abc import Callable

from aiperf.common import aiperf_logger
from aiperf.common.aiperf_logger import (
    _CRITICAL,
    _DEBUG,
    _ERROR,
    _INFO,
    _NOTICE,
    _SUCCESS,
    _TRACE,
    _WARNING,
    AIPerfLogger,
)
from aiperf.common.mixins.base_mixin import BaseMixin


class AIPerfLoggerMixin(BaseMixin):
    """Mixin to provide lazy evaluated logging for f-strings.

    This mixin provides a logger with lazy evaluation support for f-strings,
    and direct log functions for all standard and custom logging levels.

    see :class:`AIPerfLogger` for more details.

    Usage:
        class MyClass(AIPerfLoggerMixin):
            def __init__(self):
                super().__init__()
                self.trace(lambda: f"Processing {item} of {count} ({item / count * 100}% complete)")
                self.info("Simple string message")
                self.debug(lambda i=i: f"Binding loop variable: {i}")
                self.warning("Warning message: %s", "legacy support")
                self.success("Benchmark completed successfully")
                self.notice("Warmup has completed")
                self.exception(f"Direct f-string usage: {e}")
    """

    def __init__(self, logger_name: str | None = None, **kwargs) -> None:
        self.logger = AIPerfLogger(logger_name or self.__class__.__name__)
        self._log = self.logger._log
        self.is_enabled_for = self.logger._logger.isEnabledFor
        # Directly set the trace_or_debug method to the logger's trace_or_debug method to avoid
        # the overhead of the extra call stack.
        self.trace_or_debug = self.logger.trace_or_debug
        super().__init__(**kwargs)

    @property
    def is_debug_enabled(self) -> bool:
        return self.is_enabled_for(_DEBUG)

    @property
    def is_trace_enabled(self) -> bool:
        return self.is_enabled_for(_TRACE)

    def log(
        self, level: int, message: str | Callable[..., str], *args, **kwargs
    ) -> None:
        """Log a message at a specified level with lazy evaluation."""
        if self.is_enabled_for(level):
            self._log(level, message, *args, **kwargs)

    def trace(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a trace message with lazy evaluation."""
        if self.is_enabled_for(_TRACE):
            self._log(_TRACE, message, *args, **kwargs)

    def debug(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a debug message with lazy evaluation."""
        if self.is_enabled_for(_DEBUG):
            self._log(_DEBUG, message, *args, **kwargs)

    def info(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log an info message with lazy evaluation."""
        if self.is_enabled_for(_INFO):
            self._log(_INFO, message, *args, **kwargs)

    def notice(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a notice message with lazy evaluation."""
        if self.is_enabled_for(_NOTICE):
            self._log(_NOTICE, message, *args, **kwargs)

    def warning(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a warning message with lazy evaluation."""
        if self.is_enabled_for(_WARNING):
            self._log(_WARNING, message, *args, **kwargs)

    def success(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a success message with lazy evaluation."""
        if self.is_enabled_for(_SUCCESS):
            self._log(_SUCCESS, message, *args, **kwargs)

    def error(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log an error message with lazy evaluation."""
        if self.is_enabled_for(_ERROR):
            self._log(_ERROR, message, *args, **kwargs)

    def exception(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log an exception message with lazy evaluation."""
        if self.is_enabled_for(_ERROR):
            self._log(_ERROR, message, *args, exc_info=True, **kwargs)

    def critical(self, message: str | Callable[..., str], *args, **kwargs) -> None:
        """Log a critical message with lazy evaluation."""
        if self.is_enabled_for(_CRITICAL):
            self._log(_CRITICAL, message, *args, **kwargs)


# Add this file to the list of ignored files to avoid this file from being the source of the log messages
# in the AIPerfLogger class to skip it when determining the caller.
# NOTE: Using similar logic to logging._srcfile
_srcfile = os.path.normcase(AIPerfLoggerMixin.info.__code__.co_filename)
aiperf_logger._ignored_files.append(_srcfile)
