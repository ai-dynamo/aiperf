# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for macOS-specific terminal corruption fixes in cli_runner.py."""

import multiprocessing
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.config import BenchmarkConfig
from aiperf.plugin.enums import UIType

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="macOS-specific tests; rely on `fcntl` and Darwin platform check",
)

_MINIMAL_CONFIG = {
    "models": ["test-model"],
    "endpoint": {
        "urls": ["http://localhost:8000/v1/chat/completions"],
        "wait_for_model_timeout": 0,
    },
    "datasets": [
        {
            "name": "default",
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    ],
    "phases": [
        {
            "name": "profiling",
            "type": "concurrency",
            "requests": 100,
            "concurrency": 1,
        }
    ],
}


def _make_config(ui: UIType) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate({**_MINIMAL_CONFIG, "runtime": {"ui": ui}})


class TestMacOSTerminalFixes:
    """Test the macOS-specific terminal corruption fixes in cli_runner.py."""

    def test_spawn_method_set_on_macos_dashboard(
        self,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
    ):
        from aiperf.cli_runner._process_setup import (
            _configure_multiprocessing_start_method,
        )

        _configure_multiprocessing_start_method(using_dashboard=True)

        mock_multiprocessing_set_start_method.assert_called_once_with(
            "spawn", force=True
        )

    def test_spawn_method_not_set_on_linux(
        self,
        mock_platform_linux: Mock,
        mock_multiprocessing_set_start_method: Mock,
    ):
        from aiperf.cli_runner._process_setup import (
            _configure_multiprocessing_start_method,
        )

        _configure_multiprocessing_start_method(using_dashboard=True)

        mock_multiprocessing_set_start_method.assert_not_called()

    def test_spawn_method_not_set_for_simple_ui(
        self,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
    ):
        from aiperf.cli_runner._process_setup import (
            _configure_multiprocessing_start_method,
        )

        _configure_multiprocessing_start_method(using_dashboard=False)

        mock_multiprocessing_set_start_method.assert_not_called()

    @patch("fcntl.fcntl")
    def test_fd_cloexec_not_set_on_linux(
        self,
        mock_fcntl: Mock,
        mock_platform_linux: Mock,
        mock_get_global_log_queue: Mock,
    ):
        from aiperf.cli_runner._process_setup import _setup_ui_queues

        mock_get_global_log_queue.return_value = MagicMock(spec=multiprocessing.Queue)

        _setup_ui_queues(
            using_dashboard=True,
            run=MagicMock(cfg=_make_config(UIType.DASHBOARD)),
            logger=MagicMock(),
        )

        mock_fcntl.assert_not_called()

    def test_runtime_error_in_set_start_method_is_handled(
        self,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
    ):
        from aiperf.cli_runner._process_setup import (
            _configure_multiprocessing_start_method,
        )

        mock_multiprocessing_set_start_method.side_effect = RuntimeError(
            "context already set"
        )

        _configure_multiprocessing_start_method(using_dashboard=True)

        mock_multiprocessing_set_start_method.assert_called_once()

    def test_log_queue_created_before_ui_on_dashboard(
        self,
        mock_platform_darwin: Mock,
        mock_get_global_log_queue: Mock,
    ):
        from aiperf.cli_runner._process_setup import _setup_ui_queues

        mock_queue = MagicMock(spec=multiprocessing.Queue)
        mock_get_global_log_queue.return_value = mock_queue

        log_queue = _setup_ui_queues(
            using_dashboard=True,
            run=MagicMock(cfg=_make_config(UIType.DASHBOARD)),
            logger=MagicMock(),
        )

        mock_get_global_log_queue.assert_called_once()
        assert log_queue == mock_queue
