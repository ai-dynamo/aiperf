# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import multiprocessing
import queue
from functools import lru_cache
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from aiperf.common.aiperf_logger import _DEBUG, _TRACE, AIPerfLogger
from aiperf.common.config import ServiceConfig, ServiceDefaults, UserConfig
from aiperf.common.enums import ServiceType

LOG_QUEUE_MAXSIZE = 1000


logger = AIPerfLogger(__name__)


@lru_cache(maxsize=1)
def get_global_log_queue() -> multiprocessing.Queue:
    """Get the global log queue. Will create a new queue if it doesn't exist."""
    return multiprocessing.Queue(maxsize=LOG_QUEUE_MAXSIZE)


def _is_service_in_types(service_id: str, service_types: set[ServiceType]) -> bool:
    """Check if a service is in a set of services."""
    for service_type in service_types:
        # for cases of service_id being "worker_xxxxxx" and service_type being "worker",
        # we want to set the log level to debug
        if (
            service_id == service_type.value
            or service_id.startswith(f"{service_type.value}_")
            and service_id
            != f"{service_type.value}_manager"  # for worker vs worker_manager, etc.
        ):
            return True
    return False


def setup_child_process_logging(
    log_queue: "multiprocessing.Queue | None" = None,
    service_id: str | None = None,
    service_config: ServiceConfig | None = None,
    user_config: UserConfig | None = None,
) -> None:
    """Set up logging for a child process to send logs to the main process.

    This should be called early in child process initialization.

    Args:
        log_queue: The multiprocessing queue to send logs to. If None, tries to get the global queue.
        service_id: The ID of the service to log under. If None, logs will be under the process name.
        service_config: The service configuration used to determine the log level.
        user_config: The user configuration used to determine the log folder.
    """
    root_logger = logging.getLogger()
    level = ServiceDefaults.LOG_LEVEL.upper()
    if service_config:
        level = service_config.log_level.upper()

        if service_id:
            # If the service is in the trace or debug services, set the level to trace or debug
            if service_config.trace_services and _is_service_in_types(
                service_id, service_config.trace_services
            ):
                level = _TRACE
            elif service_config.debug_services and _is_service_in_types(
                service_id, service_config.debug_services
            ):
                level = _DEBUG

    # Set the root logger level to ensure logs are passed to handlers
    root_logger.setLevel(level)

    # Remove all existing handlers to avoid duplicate logs
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)

    if log_queue is not None:
        # Set up handler for child process
        queue_handler = MultiProcessLogHandler(log_queue, service_id)
        queue_handler.setLevel(level)
        root_logger.addHandler(queue_handler)

    if service_config and service_config.disable_ui:
        # Set up rich logging to the console
        rich_handler = RichHandler(
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
            user_config.output.artifact_directory / "logs", level
        )
        root_logger.addHandler(file_handler)


# TODO: Integrate with the subprocess logging instead of being separate
def setup_rich_logging(user_config: UserConfig, service_config: ServiceConfig) -> None:
    """Set up rich logging with appropriate configuration."""
    # Set logging level for the root logger (affects all loggers)
    level = service_config.log_level.upper()
    logging.root.setLevel(level)

    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_path=True,
        console=Console(),
        show_time=True,
        show_level=True,
        tracebacks_show_locals=False,
        log_time_format="%H:%M:%S.%f",
        omit_repeated_times=False,
    )
    logging.root.addHandler(rich_handler)

    # Enable file logging for services
    # TODO: Use config to determine if file logging is enabled and the folder path.
    log_folder = user_config.output.artifact_directory / "logs"
    log_folder.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_folder / "aiperf.log")
    file_handler.setLevel(level)
    file_handler.formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.root.addHandler(file_handler)

    logger.debug(lambda: f"Logging initialized with level: {level}")


def create_file_handler(
    log_folder: Path,
    level: str | int,
) -> logging.FileHandler:
    """Configure a file handler for logging."""

    log_folder.mkdir(parents=True, exist_ok=True)
    log_file_path = log_folder / "aiperf.log"

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return file_handler


class MultiProcessLogHandler(RichHandler):
    """Custom logging handler that forwards log records to a multiprocessing queue."""

    def __init__(
        self, log_queue: multiprocessing.Queue, service_id: str | None = None
    ) -> None:
        super().__init__()
        self.log_queue = log_queue
        self.service_id = service_id

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
                "process_name": multiprocessing.current_process().name,
                "process_id": multiprocessing.current_process().pid,
                "service_id": self.service_id,
            }
            self.log_queue.put_nowait(log_data)
        except queue.Full:
            # Drop logs if queue is full to prevent blocking
            pass
        except Exception:
            # Do not log to prevent recursion
            pass
