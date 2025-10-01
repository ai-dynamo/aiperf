# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for logging tests."""

import logging
import time
from pathlib import Path

import pytest


@pytest.fixture
def cleanup_logging():
    """Fixture to clean up logging configuration after tests."""
    yield
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


@pytest.fixture
def create_log_record():
    """Factory fixture to create log records with custom parameters."""

    def _create(
        name: str = "test_logger",
        level: int = logging.INFO,
        pathname: str = "/test/path.py",
        lineno: int = 42,
        msg: str = "Test message",
        func: str = "test_func",
    ) -> logging.LogRecord:
        record = logging.LogRecord(
            name=name,
            level=level,
            pathname=pathname,
            lineno=lineno,
            msg=msg,
            args=(),
            exc_info=None,
            func=func,
            sinfo=None,
        )
        record.created = time.time()
        return record

    return _create


@pytest.fixture
def create_log_data():
    """Factory fixture to create structured log data dictionaries."""

    def _create(
        name: str = "test_logger",
        levelno: int = logging.INFO,
        levelname: str = "INFO",
        msg: str = "Test message",
        service_id: str = "test_service",
        pathname: str = "/test/path.py",
        lineno: int = 100,
        process_name: str = "TestProcess",
        process_id: int = 12345,
        **kwargs,
    ) -> dict:
        log_data = {
            "created": time.time(),
            "levelno": levelno,
            "levelname": levelname,
            "name": name,
            "process_name": process_name,
            "process_id": process_id,
            "service_id": service_id,
            "pathname": pathname,
            "lineno": lineno,
            "msg": msg,
        }
        log_data.update(kwargs)
        return log_data

    return _create


@pytest.fixture
def temp_log_folder(tmp_path: Path) -> Path:
    """Fixture providing a temporary directory for log files."""
    log_folder = tmp_path / "logs"
    return log_folder
