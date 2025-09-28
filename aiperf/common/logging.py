# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import multiprocessing
import re
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console, ConsoleRenderable
from rich.logging import RichHandler
from rich.text import Text
from rich.traceback import Traceback

from aiperf.common.aiperf_logger import _DEBUG, _TRACE, AIPerfLogger
from aiperf.common.config import ServiceConfig, ServiceDefaults, UserConfig
from aiperf.common.config.config_defaults import OutputDefaults
from aiperf.common.enums import ServiceType
from aiperf.common.factories import ServiceFactory

_LOG_LEVEL_STYLES = {
    "TRACE": "dim",
    "DEBUG": "dim",
    "INFO": "cyan",
    "NOTICE": "blue",
    "WARNING": "yellow",
    "SUCCESS": "green",
    "ERROR": "red",
    "CRITICAL": "red",
}

# Regex pattern for parsing structured log format from subprocess output
# Format: created|levelno|levelname|name|process_name|process_id|service_id|pathname|lineno|msg
SUBPROCESS_LOG_PATTERN = re.compile(
    r"(?P<created>\d+\.\d+)\|(?P<levelno>\d+)\|(?P<levelname>\w+)\|(?P<name>[^|]+)\|(?P<process_name>[^|]+)\|(?P<process_id>\d+)\|(?P<service_id>[^|]*)\|(?P<pathname>[^|]*)\|(?P<lineno>\d+)\|(?P<msg>.*)"
)

_logger = AIPerfLogger(__name__)


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
        if ServiceFactory.get_class_from_type(service_type).__name__ == service_id:
            return True
    return False


def setup_logging(
    service_id: str | None = None,
    service_config: ServiceConfig | None = None,
    user_config: UserConfig | None = None,
    use_structured_subprocess_format: bool = True,
) -> None:
    """Set up logging for a service.

    This should be called early in service initialization.

    Args:
        service_id: The ID of the service to log under. If None, logs will be under the process name.
        service_config: The service configuration used to determine the log level.
        user_config: The user configuration used to determine the log folder.
        use_structured_subprocess_format: If True, use structured pipe-delimited format for subprocess parsing.
    """
    root_logger = logging.getLogger()
    level = ServiceDefaults.LOG_LEVEL.upper()
    if service_config:
        level = service_config.log_level.upper()

        if service_id:
            # If the service is in the trace or debug services, set the level to trace or debug
            if service_config.developer.trace_services and _is_service_in_types(
                service_id, service_config.developer.trace_services
            ):
                level = _TRACE
            elif service_config.developer.debug_services and _is_service_in_types(
                service_id, service_config.developer.debug_services
            ):
                level = _DEBUG

    # Set the root logger level to ensure logs are passed to handlers
    root_logger.setLevel(level)

    # Remove all existing handlers to avoid duplicate logs
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    if use_structured_subprocess_format:
        # Use structured format for subprocess output that will be parsed by parent process
        structured_handler = StructuredSubprocessLogHandler(service_id)
        structured_handler.setLevel(level)
        root_logger.addHandler(structured_handler)
    else:
        # For all other cases, set up rich logging to the console
        rich_handler = CustomRichHandler(
            rich_tracebacks=True,
            show_path=True,
            console=Console(),
            show_time=True,
            show_level=True,
            tracebacks_show_locals=False,
            log_time_format="%H:%M:%S.%f",
            omit_repeated_times=False,
        )
        rich_handler.setLevel(level)
        root_logger.addHandler(rich_handler)

    if user_config and user_config.output.artifact_directory:
        file_handler = create_file_handler(
            user_config.output.artifact_directory / OutputDefaults.LOG_FOLDER, level
        )
        root_logger.addHandler(file_handler)


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
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return file_handler


class StructuredSubprocessLogHandler(logging.Handler):
    """Custom logging handler that outputs structured log format for subprocess parsing."""

    def __init__(self, service_id: str | None = None) -> None:
        super().__init__()
        self.service_id = service_id
        self.process_id = multiprocessing.current_process().pid
        self.process_name = multiprocessing.current_process().name

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record in structured pipe-delimited format."""
        try:
            # Format: created|levelno|levelname|name|process_name|process_id|service_id|pathname|lineno|msg
            pathname = getattr(record, "pathname", "")
            lineno = getattr(record, "lineno", 0)

            structured_log = f"{record.created}|{record.levelno}|{record.levelname}|{record.name}|{self.process_name}|{self.process_id}|{self.service_id or ''}|{pathname}|{lineno}|{record.getMessage()}"
            print(structured_log, flush=True)
        except Exception:
            # Do not log to prevent recursion
            pass


class CustomRichHandler(RichHandler):
    """Custom RichHandler that formats log records in our own format."""

    def render(
        self,
        *,
        record: logging.LogRecord,
        traceback: Traceback | None,
        message_renderable: ConsoleRenderable,
    ) -> ConsoleRenderable:
        """Render log for display.

        Args:
            record (LogRecord): logging Record.
            traceback (Optional[Traceback]): Traceback instance or None for no Traceback.
            message_renderable (ConsoleRenderable): Renderable (typically Text) containing log message contents.

        Returns:
            ConsoleRenderable: Renderable to display log.
        """
        timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]  # fmt: skip
        level_style = _LOG_LEVEL_STYLES.get(record.levelname, "white")

        # wrapped_text = textwrap.fill(
        #     record.getMessage(),
        #     width=self.console.size.width - 22,
        #     subsequent_indent=" " * 22,
        # )

        formatted_log = Text.assemble(
            Text.from_markup(f"[dim]{timestamp}[/dim] "),
            Text.from_markup(
                f"[bold][{level_style}]{record.levelname:>7}[/{level_style}][/bold] "
            ),
            Text.from_markup(f"[bold]{record.processName}[/bold] "),
            Text.from_markup(f"[bold]{record.name}[/bold] "),
            self.highlighter(Text(record.getMessage())),
        )

        return formatted_log


def parse_subprocess_log_line(line: str) -> logging.LogRecord | None:
    """Parse a structured log line from subprocess output.

    Args:
        line: The log line to parse

    Returns:
        LogRecord with parsed log data or None if parsing fails
    """
    match = SUBPROCESS_LOG_PATTERN.match(line)
    if not match:
        return None
    # Create a LogRecord directly from parsed data
    record = logging.LogRecord(
        name=match.group("name"),
        level=int(match.group("levelno")),
        pathname=match.group("pathname"),
        lineno=int(match.group("lineno")),
        msg=match.group("msg"),
        args=(),
        exc_info=None,
        func=None,
        sinfo=None,
    )

    # Set additional attributes from subprocess
    record.created = float(match.group("created"))
    record.msecs = (record.created % 1) * 1000
    record.processName = match.group("process_name")
    record.process = int(match.group("process_id"))
    record.levelname = match.group("levelname")

    # Store service_id as custom attribute
    record.service_id = match.group("service_id")

    return record


def handle_subprocess_log_line(line: str, fallback_service_id: str) -> None:
    """Handle a subprocess log line by parsing and forwarding to appropriate logger.

    This function handles both structured and unstructured log lines from subprocesses,
    parsing them and forwarding directly to the appropriate logger without creating
    temporary objects.

    Args:
        line: The log line from subprocess output
        fallback_service_id: Service ID to use if line is not structured
    """
    # Try structured parsing first
    parsed_record = parse_subprocess_log_line(line)

    if parsed_record:
        original_logger = logging.getLogger(parsed_record.name)
        if original_logger.isEnabledFor(parsed_record.levelno):
            original_logger.handle(parsed_record)
    else:
        fallback_logger = logging.getLogger(fallback_service_id)
        if fallback_logger.isEnabledFor(logging.WARNING):
            record = logging.LogRecord(
                name=fallback_service_id,
                level=logging.WARNING,
                pathname="<subprocess>",
                lineno=0,
                msg=line,
                args=(),
                exc_info=None,
                func=None,
                sinfo=None,
            )
            record.created = time.time()
            record.msecs = (record.created % 1) * 1000
            record.levelname = "WARNING"

            fallback_logger.handle(record)
