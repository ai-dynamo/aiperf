# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

from rich.console import Console, ConsoleRenderable, Group
from rich.highlighter import ReprHighlighter
from rich.logging import RichHandler
from rich.text import Text
from rich.traceback import Traceback

from aiperf.common.aiperf_logger import _DEBUG, _TRACE, AIPerfLogger
from aiperf.common.environment import Environment
from aiperf.common.utils import is_tty
from aiperf.config import ServiceDefaults
from aiperf.config.defaults import OutputDefaults
from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, ServiceType, UIType

if TYPE_CHECKING:
    from aiperf.config import BenchmarkConfig

LogQueue: TypeAlias = multiprocessing.Queue

_logger = AIPerfLogger(__name__)
_global_log_queue: LogQueue | None = None
_log_queue_lock = threading.Lock()

_LOG_LEVEL_STYLES = {
    "TRACE": "dim",
    "DEBUG": "dim",
    "INFO": "cyan",
    "NOTICE": "blue",
    "WARNING": "yellow",
    "SUCCESS": "green",
    "ERROR": "red",
    "CRITICAL": "bold red",
}

_BASIC_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s (%(filename)s:%(lineno)d)"
)
_BASIC_DATE_FORMAT = "%H:%M:%S"


def _create_basic_handler(level: str | int) -> logging.StreamHandler:
    """Create a basic non-rich StreamHandler for non-TTY environments.

    Uses sys.stdout to match CustomRichHandler's Console() which also defaults to stdout.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(_BASIC_LOG_FORMAT, datefmt=_BASIC_DATE_FORMAT)
    )
    return handler


def get_global_log_queue() -> LogQueue:
    """Get the global log queue. Will create a new queue if it doesn't exist.

    Thread-safe singleton pattern using double-checked locking.
    """
    global _global_log_queue
    if _global_log_queue is None:
        with _log_queue_lock:
            if _global_log_queue is None:
                from aiperf.common.subprocess_manager import get_mp_context

                _global_log_queue = get_mp_context().Queue(
                    maxsize=Environment.LOGGING.QUEUE_MAXSIZE
                )
    return _global_log_queue


async def cleanup_global_log_queue() -> None:
    """Clean up the global log queue to prevent semaphore leaks.

    This should be called during shutdown to properly close and join the queue,
    which releases the internal semaphores used by multiprocessing.Queue.
    Thread-safe.
    """
    global _global_log_queue
    with _log_queue_lock:
        if _global_log_queue is not None:
            try:
                _global_log_queue.close()
                await asyncio.wait_for(
                    asyncio.to_thread(_global_log_queue.join_thread), timeout=1.0
                )
                _logger.debug("Cleaned up global log queue")
            except (OSError, asyncio.TimeoutError, ValueError) as e:
                _logger.debug(f"Error cleaning up log queue: {e}")
            finally:
                from aiperf.common.resource_tracker import unregister_queue_semaphores

                unregister_queue_semaphores(_global_log_queue)
                _global_log_queue = None


def _is_service_in_types(service_id: str, service_types: set[ServiceType]) -> bool:
    """Check if a service is in a set of services."""
    for service_type in service_types:
        # for cases of service_id being "worker_xxxxxx" and service_type being "worker",
        # we want to set the log level to debug
        if (
            service_id == service_type
            or service_id.startswith(f"{service_type}_")
            and service_id
            != f"{service_type}_manager"  # for worker vs worker_manager, etc.
        ):
            return True

        # Check if the provided logger name is the same as the service's class name
        ServiceClass = plugins.get_class(PluginType.SERVICE, service_type)
        if ServiceClass.__name__ == service_id:
            return True
    return False


def _is_running_in_kubernetes() -> bool:
    """Check if we're running inside a Kubernetes pod."""
    return os.environ.get("KUBERNETES_SERVICE_HOST") is not None


def _should_use_structured_logging() -> bool:
    """Check if structured (JSON) logging should be used.

    Returns True when AIPERF_STRUCTURED_LOGS env var is set.
    """
    return os.environ.get("AIPERF_STRUCTURED_LOGS") is not None


def _should_use_plain_logging() -> bool:
    """Determine if we should use plain text logging instead of Rich.

    Returns True when:
    - Running in Kubernetes (no TTY, log aggregation friendly)
    - No TTY attached (CI/CD, piped output)
    - AIPERF_PLAIN_LOGS env var is set

    Note: Structured logging takes precedence if enabled.
    """
    import sys

    if _should_use_structured_logging():
        return False  # Structured logging takes precedence
    if os.environ.get("AIPERF_PLAIN_LOGS"):
        return True
    if _is_running_in_kubernetes():
        return True
    return not sys.stdout.isatty()


class StructuredLogHandler(logging.Handler):
    """JSON structured logging handler for log aggregation systems.

    Produces newline-delimited JSON (NDJSON) output compatible with:
    - CloudWatch Logs Insights
    - Elasticsearch/OpenSearch
    - Loki
    - Datadog
    - Splunk

    Enable by setting AIPERF_STRUCTURED_LOGS=1 environment variable.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record as JSON."""
        import orjson

        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "source": {
                    "file": record.filename,
                    "line": record.lineno,
                    "function": record.funcName,
                },
            }

            # Add exception info if present
            if record.exc_info and record.exc_info[0] is not None:
                import traceback

                log_entry["exception"] = {
                    "type": record.exc_info[0].__name__,
                    "message": str(record.exc_info[1]),
                    "traceback": traceback.format_exception(*record.exc_info),
                }

            # Add extra fields if present (for custom context)
            if hasattr(record, "extra"):
                log_entry["extra"] = record.extra

            print(orjson.dumps(log_entry).decode(), flush=True)
        except Exception:  # noqa: BLE001 - logging.Handler.emit contract: delegate any failure to handleError
            self.handleError(record)


def setup_child_process_logging(
    log_queue: LogQueue | None = None,
    service_id: str | None = None,
    config: BenchmarkConfig | None = None,
) -> None:
    """Set up logging for a child process to send logs to the main process.

    This should be called early in child process initialization.

    Args:
        log_queue: The multiprocessing queue to send logs to. If None, tries to get the global queue.
        service_id: The ID of the service to log under. If None, logs will be under the process name.
        config: The BenchmarkConfig used to determine log level and log folder.
    """
    root_logger = logging.getLogger()
    level = ServiceDefaults.LOG_LEVEL.upper()
    if config:
        level = config.log_level.upper()

        if service_id:
            # If the service is in the trace or debug services, set the level to trace or debug
            if Environment.DEV.TRACE_SERVICES and _is_service_in_types(
                service_id, Environment.DEV.TRACE_SERVICES
            ):
                level = _TRACE
            elif Environment.DEV.DEBUG_SERVICES and _is_service_in_types(
                service_id, Environment.DEV.DEBUG_SERVICES
            ):
                level = _DEBUG

    # Set the root logger level to ensure logs are passed to handlers
    root_logger.setLevel(level)

    # Remove all existing handlers to avoid duplicate logs
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    if log_queue is not None and config and config.ui_type == UIType.DASHBOARD:
        # For dashboard UI, we want to log to the queue, so it can be displayed in the UI
        # log viewer, instead of the console directly.
        queue_handler = MultiProcessLogHandler(log_queue, service_id)
        queue_handler.setLevel(level)
        root_logger.addHandler(queue_handler)
    elif _should_use_structured_logging():
        # Structured JSON logging for log aggregation (CloudWatch, Loki, etc.)
        # Enable with AIPERF_STRUCTURED_LOGS=1
        structured_handler = StructuredLogHandler()
        structured_handler.setLevel(level)
        root_logger.addHandler(structured_handler)
    elif is_tty():
        # For TTY environments, set up custom rich logging to the console
        rich_handler = CustomRichHandler(
            rich_tracebacks=True,
            show_path=False,
            console=Console(),
            show_time=False,
            show_level=False,
            tracebacks_show_locals=False,
        )
        rich_handler.setLevel(level)
        root_logger.addHandler(rich_handler)
    else:
        # For non-TTY environments, use basic logging without rich formatting
        root_logger.addHandler(_create_basic_handler(level))

    if config and config.artifacts.dir:
        file_handler = create_file_handler(
            config.artifacts.dir / OutputDefaults.LOG_FOLDER, level
        )
        root_logger.addHandler(file_handler)


def setup_subprocess_logging(level: str | int) -> None:
    """Lightweight logging setup for ephemeral subprocesses (e.g. ProcessPoolExecutor workers)."""
    logging.basicConfig(level=level, force=True)


# TODO: Integrate with the subprocess logging instead of being separate
def setup_rich_logging(config: BenchmarkConfig) -> None:
    """Set up rich logging with appropriate configuration. Falls back to basic logging for non-TTY."""
    # Set logging level for the root logger (affects all loggers)
    level = config.log_level.upper()
    logging.root.setLevel(level)

    if is_tty():
        console_handler = CustomRichHandler(
            rich_tracebacks=True,
            show_path=False,
            console=Console(),
            show_time=False,
            show_level=False,
            tracebacks_show_locals=False,
        )
    else:
        console_handler = _create_basic_handler(level)
    logging.root.addHandler(console_handler)

    # Enable file logging for services
    # TODO: Use config to determine if file logging is enabled and the folder path.
    log_folder = config.artifacts.dir / OutputDefaults.LOG_FOLDER
    log_folder.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_folder / OutputDefaults.LOG_FILE)
    file_handler.setLevel(level)
    file_handler.formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.root.addHandler(file_handler)

    _logger.debug(lambda: f"Logging initialized with level: {level}")


def create_file_handler(
    log_folder: Path,
    level: str | int,
) -> logging.FileHandler:
    """Configure a file handler for logging."""

    log_folder.mkdir(parents=True, exist_ok=True)
    log_file_path = log_folder / OutputDefaults.LOG_FILE

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return file_handler


class CustomRichHandler(RichHandler):
    """Custom RichHandler that formats logs with the logger name right-aligned at the end."""

    DEFAULT_WIDTH = 120
    MAX_MESSAGE_LENGTH = 2000

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.highlighter = ReprHighlighter()

    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Traceback | None,
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        """Render log for display with file:line at end, using character-level wrapping."""
        timestamp = f"{time.strftime('%H:%M:%S', time.localtime(record.created))}.{int(record.msecs):03d}"
        level_style = _LOG_LEVEL_STYLES.get(record.levelname, "white")
        message = record.getMessage()[: self.MAX_MESSAGE_LENGTH]
        log_suffix = f"({record.filename}:{record.lineno})"

        # Calculate widths
        console_width = self.console.size.width if self.console else self.DEFAULT_WIDTH
        target_width = max(console_width - 2, 40)

        prefix = f"{timestamp} {record.levelname:<8} "
        prefix_len = len(prefix)
        content_width = target_width - prefix_len

        # Combine message and suffix into one string for character-level wrapping
        full_content = f"{message} {log_suffix}"
        suffix_start_pos = (
            len(message) + 1
        )  # Position where suffix starts in full_content

        # Only indent continuation lines on wide consoles (90+)
        indent_continuations = console_width >= 90
        # Continuation lines get full width when not indented
        continuation_width = content_width if indent_continuations else target_width

        # Manual character-level wrapping
        lines = []
        remaining = full_content
        is_first_line = True
        while remaining:
            line_width = content_width if is_first_line else continuation_width
            if len(remaining) <= line_width:
                lines.append(remaining)
                break
            lines.append(remaining[:line_width])
            remaining = remaining[line_width:]
            is_first_line = False

        # Build output with proper styling
        parts = []
        char_pos = 0  # Track position in full_content
        for i, line in enumerate(lines):
            if i > 0:
                parts.append(Text("\n"))
                if indent_continuations:
                    parts.append(Text(" " * prefix_len))
            else:
                parts.append(Text(f"{timestamp} ", style="log.time"))
                parts.append(Text(f"{record.levelname:<8} ", style=level_style))

            line_end_pos = char_pos + len(line)

            # Determine how much of this line is message vs suffix
            if char_pos >= suffix_start_pos:
                # Entire line is suffix
                parts.append(Text(line, style="dim italic"))
            elif line_end_pos <= suffix_start_pos:
                # Entire line is message
                parts.append(self.highlighter(Text(line)))
            else:
                # Line contains both message and suffix
                msg_chars = suffix_start_pos - char_pos
                parts.append(self.highlighter(Text(line[:msg_chars])))
                parts.append(Text(line[msg_chars:], style="dim italic"))

            char_pos = line_end_pos

        formatted_log = Text.assemble(*parts)
        formatted_log.no_wrap = True  # Prevent Rich from re-wrapping

        if traceback:
            return Group(formatted_log, traceback)

        return formatted_log

    def emit(self, record: logging.LogRecord) -> None:
        """Emit with soft_wrap=False to prevent Rich from adding word-based wrapping."""
        message = self.format(record)
        traceback = None
        if (
            self.rich_tracebacks
            and record.exc_info
            and record.exc_info != (None, None, None)
        ):
            traceback = Traceback.from_exception(*record.exc_info)
        message_renderable = self.render_message(record, message)
        log_renderable = self.render(
            record=record, traceback=traceback, message_renderable=message_renderable
        )
        self.console.print(log_renderable, soft_wrap=False)


class MultiProcessLogHandler(RichHandler):
    """Custom logging handler that forwards log records to a multiprocessing queue."""

    def __init__(self, log_queue: LogQueue, service_id: str | None = None) -> None:
        super().__init__()
        self.log_queue = log_queue
        self.service_id = service_id
        self._proc_name = multiprocessing.current_process().name
        self._proc_id = multiprocessing.current_process().pid

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the queue."""
        try:
            # Create a serializable log data structure
            log_data = {
                "name": record.name,
                "levelname": record.levelname,
                "levelno": record.levelno,
                "msg": record.getMessage(),
                "created": record.created,
                "filename": record.filename,
                "lineno": record.lineno,
                "process_name": self._proc_name,
                "process_id": self._proc_id,
                "service_id": self.service_id,
            }
            self.log_queue.put_nowait(log_data)
        except queue.Full:
            # Drop logs if queue is full to prevent blocking. Do not log to prevent recursion.
            pass
        except Exception:  # noqa: BLE001, S110 - log handler emit path must never re-raise or recurse
            # Do not log to prevent recursion
            pass
