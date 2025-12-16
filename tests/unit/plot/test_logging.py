# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for plot logging configuration.

This module tests the logging setup functions for console-only and file-based
logging configurations.
"""

import logging

import pytest
from rich.logging import RichHandler

from aiperf.plot.constants import PLOT_LOG_FILE
from aiperf.plot.logging import setup_console_only_logging, setup_plot_logging


class TestSetupConsoleOnlyLogging:
    """Tests for setup_console_only_logging function."""

    def test_console_only_logging_default_level(self):
        """Verifies INFO level set by default."""
        setup_console_only_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    @pytest.mark.parametrize("level", ["DEBUG", "WARNING", "ERROR"])
    def test_console_only_logging_custom_level(self, level):
        """Tests DEBUG, WARNING, ERROR levels."""
        setup_console_only_logging(log_level=level)

        root_logger = logging.getLogger()
        assert root_logger.level == getattr(logging, level)

    def test_console_only_logging_handler_cleanup(self):
        """Removes existing handlers before adding new."""
        root_logger = logging.getLogger()
        existing_handler = logging.StreamHandler()
        root_logger.addHandler(existing_handler)

        setup_console_only_logging()

        assert len(root_logger.handlers) == 1
        assert isinstance(root_logger.handlers[0], RichHandler)

    def test_console_only_logging_rich_handler_configuration(self):
        """Verifies RichHandler settings."""
        setup_console_only_logging()

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        assert isinstance(handler, RichHandler)
        assert handler.level == logging.INFO

    def test_console_only_logging_uppercase_conversion(self):
        """Converts 'info' to 'INFO'."""
        setup_console_only_logging(log_level="info")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_console_only_logging_multiple_calls(self):
        """Safely handles multiple setup calls."""
        setup_console_only_logging()
        setup_console_only_logging()

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 1

    def test_console_only_logging_handler_level_matches_root(self):
        """Handler level matches root logger level."""
        setup_console_only_logging(log_level="DEBUG")

        root_logger = logging.getLogger()
        handler = root_logger.handlers[0]

        assert root_logger.level == logging.DEBUG
        assert handler.level == logging.DEBUG


class TestSetupPlotLogging:
    """Tests for setup_plot_logging function."""

    def test_setup_plot_logging_default_level(self, tmp_path):
        """Verifies INFO level set by default."""
        setup_plot_logging(tmp_path)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    @pytest.mark.parametrize("level", ["DEBUG", "WARNING", "ERROR"])
    def test_setup_plot_logging_custom_level(self, tmp_path, level):
        """Tests DEBUG, WARNING, ERROR levels."""
        setup_plot_logging(tmp_path, log_level=level)

        root_logger = logging.getLogger()
        assert root_logger.level == getattr(logging, level)

    def test_setup_plot_logging_console_level_logic(self, tmp_path):
        """DEBUG shows all, others show WARNING+."""
        setup_plot_logging(tmp_path, log_level="DEBUG")
        root_logger = logging.getLogger()
        rich_handler = next(
            h for h in root_logger.handlers if isinstance(h, RichHandler)
        )
        assert rich_handler.level == logging.DEBUG

        setup_plot_logging(tmp_path, log_level="INFO")
        root_logger = logging.getLogger()
        rich_handler = next(
            h for h in root_logger.handlers if isinstance(h, RichHandler)
        )
        assert rich_handler.level == logging.WARNING

        setup_plot_logging(tmp_path, log_level="WARNING")
        root_logger = logging.getLogger()
        rich_handler = next(
            h for h in root_logger.handlers if isinstance(h, RichHandler)
        )
        assert rich_handler.level == logging.WARNING

    def test_setup_plot_logging_creates_output_directory(self, tmp_path):
        """Creates output_dir with parents=True."""
        nested_dir = tmp_path / "level1" / "level2" / "output"

        setup_plot_logging(nested_dir)

        assert nested_dir.exists()
        assert nested_dir.is_dir()

    def test_setup_plot_logging_log_file_creation(self, tmp_path):
        """Creates log file at correct path."""
        setup_plot_logging(tmp_path)

        log_file = tmp_path / PLOT_LOG_FILE
        assert log_file.exists()
        assert log_file.is_file()

    def test_setup_plot_logging_file_handler_encoding(self, tmp_path):
        """Verifies utf-8 encoding."""
        setup_plot_logging(tmp_path)

        root_logger = logging.getLogger()
        file_handler = next(
            h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
        )
        assert file_handler.encoding == "utf-8"

    def test_setup_plot_logging_file_handler_level(self, tmp_path):
        """File handler uses specified level."""
        setup_plot_logging(tmp_path, log_level="DEBUG")

        root_logger = logging.getLogger()
        file_handler = next(
            h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
        )
        assert file_handler.level == logging.DEBUG

    def test_setup_plot_logging_handler_cleanup(self, tmp_path):
        """Removes existing handlers before adding new."""
        root_logger = logging.getLogger()
        existing_handler = logging.StreamHandler()
        root_logger.addHandler(existing_handler)

        setup_plot_logging(tmp_path)

        assert len(root_logger.handlers) == 2
        assert any(isinstance(h, RichHandler) for h in root_logger.handlers)
        assert any(isinstance(h, logging.FileHandler) for h in root_logger.handlers)

    def test_setup_plot_logging_multiple_calls(self, tmp_path):
        """Safely handles multiple setup calls."""
        setup_plot_logging(tmp_path)
        setup_plot_logging(tmp_path)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 2

    def test_setup_plot_logging_log_file_created(self, tmp_path):
        """Verifies log file is created at correct location."""
        setup_plot_logging(tmp_path)

        log_file_path = tmp_path / PLOT_LOG_FILE
        assert log_file_path.exists()
        assert log_file_path.is_file()

    def test_setup_plot_logging_uppercase_conversion(self, tmp_path):
        """Converts 'info' to 'INFO'."""
        setup_plot_logging(tmp_path, log_level="info")

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_setup_plot_logging_handler_count(self, tmp_path):
        """Ensures exactly 2 handlers: RichHandler and FileHandler."""
        setup_plot_logging(tmp_path)

        root_logger = logging.getLogger()
        assert len(root_logger.handlers) == 2
        assert sum(isinstance(h, RichHandler) for h in root_logger.handlers) == 1
        assert (
            sum(isinstance(h, logging.FileHandler) for h in root_logger.handlers) == 1
        )
