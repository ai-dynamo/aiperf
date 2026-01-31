# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Logging configuration for the plot command.

This module provides logging setup specific to the plot functionality,
separate from the main AIPerf benchmark logging. Logs are written to
the output directory alongside generated visualizations.
"""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from aiperf.common.aiperf_logger import AIPerfLogger
from aiperf.plot.constants import PLOT_LOG_FILE

_logger = AIPerfLogger(__name__)


def setup_console_only_logging(log_level: str = "INFO") -> None:
    """
    Set up console-only logging for plot operations.

    This is a fallback mode used when the output directory is not available
    or cannot be created. Configures logging to output only to console via
    RichHandler without a file handler.

    Args:
        log_level: Logging level (e.g., "DEBUG", "INFO", "WARNING"). Defaults to "INFO".
    """
    root_logger = logging.getLogger()
    level = log_level.upper()
    root_logger.setLevel(level)

    for existing_handler in root_logger.handlers[:]:
        existing_handler.close()
        root_logger.removeHandler(existing_handler)

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


def setup_plot_logging(output_dir: Path, log_level: str = "INFO") -> None:
    """
    Set up logging for the plot command.

    Configures logging to output to both console (via RichHandler) and a log
    file in the output directory. This function can be called multiple times
    safely as it clears existing handlers before adding new ones.

    Console output shows WARNING and above by default, or DEBUG and above when
    log_level is DEBUG. File output always captures all logs at the specified level.

    Args:
        output_dir: Directory where plot outputs (and logs) will be saved.
        log_level: Logging level (e.g., "DEBUG", "INFO", "WARNING"). Defaults to "INFO".
    """
    root_logger = logging.getLogger()

    level = log_level.upper()
    root_logger.setLevel(level)

    for existing_handler in root_logger.handlers[:]:
        existing_handler.close()
        root_logger.removeHandler(existing_handler)

    # Console handler: WARNING+ by default, DEBUG+ when log_level is DEBUG
    console_level = level if level == "DEBUG" else "WARNING"
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
    rich_handler.setLevel(console_level)
    root_logger.addHandler(rich_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = output_dir / PLOT_LOG_FILE

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root_logger.addHandler(file_handler)

    _logger.info(f"Plot logging initialized with level: {level}")
    _logger.info(f"Log file: {log_file_path}")
