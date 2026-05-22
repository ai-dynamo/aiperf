# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
from unittest.mock import MagicMock

import pytest

from aiperf.common.logging import (
    CustomRichHandler,
    MultiProcessLogHandler,
    _create_basic_handler,
    setup_child_process_logging,
    setup_rich_logging,
)
from aiperf.plugin.enums import UIType


@pytest.fixture()
def clean_root_logger():
    """Remove all handlers from root logger before and after each test."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    yield
    for h in root.handlers[:]:
        root.removeHandler(h)


@pytest.fixture()
def mock_run(tmp_path):
    """Minimal mock BenchmarkRun exposing the cfg paths setup_*_logging reads.

    Mirrors the v2 BenchmarkRun shape: ``cfg.logging.level``, ``cfg.runtime.ui``,
    and ``cfg.artifacts.dir``. Defaults to INFO + DASHBOARD; tests adjust as needed.
    """
    run = MagicMock()
    run.cfg.logging.level = "INFO"
    run.cfg.runtime.ui = UIType.DASHBOARD
    run.cfg.artifacts.dir = tmp_path
    return run


# ---------------------------------------------------------------------------
# _create_basic_handler
# ---------------------------------------------------------------------------
class TestCreateBasicHandler:
    """Tests for the _create_basic_handler factory function."""

    def test_returns_stream_handler(self):
        """Should return a logging.StreamHandler, not a RichHandler."""
        handler = _create_basic_handler("INFO")
        assert isinstance(handler, logging.StreamHandler)
        assert not isinstance(handler, CustomRichHandler)

    @pytest.mark.parametrize(
        "level_str,level_int",
        [
            ("DEBUG", logging.DEBUG),
            ("INFO", logging.INFO),
            ("WARNING", logging.WARNING),
            ("ERROR", logging.ERROR),
        ],
    )
    def test_sets_correct_log_level(self, level_str, level_int):
        """Should set the handler level to match the requested level."""
        handler = _create_basic_handler(level_str)
        assert handler.level == level_int

    def test_formatter_uses_expected_format(self):
        """Should configure the formatter with the expected format string and date format."""
        handler = _create_basic_handler("INFO")
        fmt = handler.formatter
        assert fmt is not None
        assert "%(msecs)03d" in fmt._fmt
        assert "%(levelname)" in fmt._fmt
        assert "%(filename)s" in fmt._fmt
        assert fmt.datefmt == "%H:%M:%S"


# ---------------------------------------------------------------------------
# setup_rich_logging
# ---------------------------------------------------------------------------
class TestSetupRichLogging:
    """Tests for setup_rich_logging TTY-aware handler selection."""

    @pytest.fixture(autouse=True)
    def _clean(self, clean_root_logger):
        """Ensure clean root logger for every test."""

    def test_tty_uses_custom_rich_handler(self, monkeypatch, mock_run):
        """When is_tty() is True, root logger should get a CustomRichHandler."""
        monkeypatch.setattr("aiperf.common.logging.is_tty", lambda: True)

        setup_rich_logging(mock_run)

        root = logging.getLogger()
        rich_handlers = [h for h in root.handlers if isinstance(h, CustomRichHandler)]
        assert len(rich_handlers) == 1

    def test_non_tty_uses_basic_stream_handler(self, monkeypatch, mock_run):
        """When is_tty() is False, root logger should get a basic StreamHandler."""
        monkeypatch.setattr("aiperf.common.logging.is_tty", lambda: False)

        setup_rich_logging(mock_run)

        root = logging.getLogger()
        console_handlers = [
            h for h in root.handlers if type(h) is logging.StreamHandler
        ]
        assert len(console_handlers) == 1
        assert not isinstance(console_handlers[0], CustomRichHandler)

    @pytest.mark.parametrize("tty", [True, False])
    def test_file_handler_always_added(self, monkeypatch, mock_run, tty):
        """A FileHandler should always be added regardless of TTY state."""
        monkeypatch.setattr("aiperf.common.logging.is_tty", lambda: tty)

        setup_rich_logging(mock_run)

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1


# ---------------------------------------------------------------------------
# setup_child_process_logging
# ---------------------------------------------------------------------------
class TestSetupChildProcessLogging:
    """Tests for setup_child_process_logging TTY-aware handler selection."""

    @pytest.fixture(autouse=True)
    def _clean(self, clean_root_logger):
        """Ensure clean root logger for every test."""

    def test_dashboard_with_queue_uses_multiprocess_handler(
        self, monkeypatch, mock_run
    ):
        """Dashboard UI with a log queue should use MultiProcessLogHandler."""
        mock_run.cfg.runtime.ui = UIType.DASHBOARD
        mock_queue = MagicMock()

        # TTY state should be irrelevant when dashboard + queue
        monkeypatch.setattr("aiperf.common.logging.is_tty", lambda: True)

        setup_child_process_logging(
            log_queue=mock_queue,
            service_id="worker_1",
            run=mock_run,
        )

        root = logging.getLogger()
        mp_handlers = [
            h for h in root.handlers if isinstance(h, MultiProcessLogHandler)
        ]
        assert len(mp_handlers) == 1

    def test_non_dashboard_tty_uses_custom_rich_handler(self, monkeypatch, mock_run):
        """Non-dashboard UI in a TTY should use CustomRichHandler."""
        mock_run.cfg.runtime.ui = UIType.SIMPLE
        monkeypatch.setattr("aiperf.common.logging.is_tty", lambda: True)

        setup_child_process_logging(
            log_queue=None,
            service_id="worker_1",
            run=mock_run,
        )

        root = logging.getLogger()
        rich_handlers = [h for h in root.handlers if isinstance(h, CustomRichHandler)]
        assert len(rich_handlers) == 1

    def test_non_dashboard_non_tty_uses_basic_handler(self, monkeypatch, mock_run):
        """Non-dashboard UI in a non-TTY should use a basic StreamHandler."""
        mock_run.cfg.runtime.ui = UIType.NONE
        monkeypatch.setattr("aiperf.common.logging.is_tty", lambda: False)

        setup_child_process_logging(
            log_queue=None,
            service_id="worker_1",
            run=mock_run,
        )

        root = logging.getLogger()
        console_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(
                h, CustomRichHandler | MultiProcessLogHandler | logging.FileHandler
            )
        ]
        assert len(console_handlers) == 1

    def test_dashboard_without_queue_and_tty_uses_rich_handler(
        self, monkeypatch, mock_run
    ):
        """Dashboard UI without a log queue in a TTY should fall through to CustomRichHandler."""
        mock_run.cfg.runtime.ui = UIType.DASHBOARD
        monkeypatch.setattr("aiperf.common.logging.is_tty", lambda: True)

        setup_child_process_logging(
            log_queue=None,
            service_id="worker_1",
            run=mock_run,
        )

        root = logging.getLogger()
        rich_handlers = [h for h in root.handlers if isinstance(h, CustomRichHandler)]
        assert len(rich_handlers) == 1

    def test_dashboard_without_queue_and_non_tty_uses_basic_handler(
        self, monkeypatch, mock_run
    ):
        """Dashboard UI without a log queue in a non-TTY should fall through to basic handler."""
        mock_run.cfg.runtime.ui = UIType.DASHBOARD
        monkeypatch.setattr("aiperf.common.logging.is_tty", lambda: False)

        setup_child_process_logging(
            log_queue=None,
            service_id="worker_1",
            run=mock_run,
        )

        root = logging.getLogger()
        console_handlers = [
            h
            for h in root.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(
                h, CustomRichHandler | MultiProcessLogHandler | logging.FileHandler
            )
        ]
        assert len(console_handlers) == 1

    def test_file_handler_added_when_run_provided(self, monkeypatch, mock_run):
        """File handler should be added when a BenchmarkRun is provided."""
        mock_run.cfg.runtime.ui = UIType.NONE
        monkeypatch.setattr("aiperf.common.logging.is_tty", lambda: False)

        setup_child_process_logging(
            log_queue=None,
            service_id="worker_1",
            run=mock_run,
        )

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_no_file_handler_when_no_run(self, monkeypatch):
        """No file handler should be added when run is None."""
        monkeypatch.setattr("aiperf.common.logging.is_tty", lambda: False)

        setup_child_process_logging(
            log_queue=None,
            service_id="worker_1",
            run=None,
        )

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0
