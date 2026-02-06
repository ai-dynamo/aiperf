# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import logging
import re
import time

import pytest

from aiperf.common.logging import (
    _BASIC_DATE_FORMAT,
    _BASIC_LOG_FORMAT,
    CustomRichHandler,
    _create_basic_handler,
)
from aiperf.common.utils import is_tty


class TestIsTTY:
    """Tests for the is_tty utility function."""

    def test_returns_true_when_stdout_is_tty(self, monkeypatch):
        """Should return True when stdout.isatty() returns True."""
        mock_stdout = io.StringIO()
        mock_stdout.isatty = lambda: True
        monkeypatch.setattr("sys.stdout", mock_stdout)
        assert is_tty() is True

    def test_returns_false_when_stdout_is_not_tty(self, monkeypatch):
        """Should return False when stdout.isatty() returns False."""
        mock_stdout = io.StringIO()
        mock_stdout.isatty = lambda: False
        monkeypatch.setattr("sys.stdout", mock_stdout)
        assert is_tty() is False

    def test_returns_false_when_stdout_is_none(self, monkeypatch):
        """Should return False when stdout is None."""
        monkeypatch.setattr("sys.stdout", None)
        assert is_tty() is False

    def test_returns_false_when_stdout_lacks_isatty(self, monkeypatch):
        """Should return False when stdout has no isatty attribute (e.g. TeeStream)."""

        class FakeStream:
            pass

        monkeypatch.setattr("sys.stdout", FakeStream())
        assert is_tty() is False


class TestBasicHandlerFormat:
    """Tests for the basic (non-TTY) log handler millisecond formatting."""

    def test_basic_handler_includes_milliseconds(self):
        """Should include milliseconds in the basic handler output."""
        handler = _create_basic_handler("DEBUG")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="hello",
            args=None,
            exc_info=None,
        )
        output = handler.format(record)
        timestamp = output.split(" ")[0]
        # Expect HH:MM:SS.mmm format
        assert re.match(r"\d{2}:\d{2}:\d{2}\.\d{3}", timestamp)

    def test_basic_handler_sets_level(self):
        """Should set the handler level correctly."""
        handler = _create_basic_handler("WARNING")
        assert handler.level == logging.WARNING

    def test_basic_format_string_contains_msecs(self):
        """The format string should contain msecs placeholder."""
        assert "%(msecs)03d" in _BASIC_LOG_FORMAT

    def test_basic_date_format_excludes_microseconds(self):
        """The date format should not contain %f since we use %(msecs)03d instead."""
        assert "%f" not in _BASIC_DATE_FORMAT


class TestCustomRichHandlerTimestamp:
    """Tests for the CustomRichHandler timestamp formatting."""

    @pytest.fixture()
    def log_record(self) -> logging.LogRecord:
        """Create a log record with a known timestamp."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="test message",
            args=None,
            exc_info=None,
        )
        return record

    def test_timestamp_format_matches_expected_pattern(self, log_record):
        """The rendered timestamp should be HH:MM:SS.mmm."""
        expected_time = time.strftime("%H:%M:%S", time.localtime(log_record.created))
        expected_ms = f"{int(log_record.msecs):03d}"
        expected = f"{expected_time}.{expected_ms}"

        handler = CustomRichHandler(show_time=False, show_level=False, show_path=False)
        renderable = handler.render(
            record=log_record, traceback=None, message_renderable=None
        )
        rendered_text = str(renderable)
        assert expected in rendered_text

    def test_timestamp_milliseconds_are_three_digits(self, log_record):
        """Milliseconds should always be zero-padded to 3 digits."""
        log_record.msecs = 5.0
        handler = CustomRichHandler(show_time=False, show_level=False, show_path=False)
        renderable = handler.render(
            record=log_record, traceback=None, message_renderable=None
        )
        rendered_text = str(renderable)
        assert re.search(r"\d{2}:\d{2}:\d{2}\.005", rendered_text)


class TestFileHandlerFormat:
    """Tests for file handler millisecond formatting."""

    def test_file_handler_format_includes_milliseconds(self, tmp_path):
        """File handler output should include milliseconds."""
        from aiperf.common.logging import create_file_handler

        handler = create_file_handler(tmp_path, "DEBUG")
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="file log test",
            args=None,
            exc_info=None,
        )
        output = handler.format(record)
        # Expect YYYY-MM-DD HH:MM:SS.mmm format in the output
        assert re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}", output)
