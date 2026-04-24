# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for macOS-specific terminal corruption fixes in cli_runner.py"""

import multiprocessing
from unittest.mock import MagicMock, Mock, patch

import pytest

from aiperf.cli_runner import run_benchmark
from aiperf.config import AIPerfConfig, BenchmarkPlan
from aiperf.config.loader import build_benchmark_plan
from aiperf.plugin.enums import UIType

_BASE = dict(
    models=["test-model"],
    endpoint={"urls": ["http://localhost:8000/v1/chat/completions"]},
    datasets={
        "default": {
            "type": "synthetic",
            "entries": 100,
            "prompts": {"isl": 128, "osl": 64},
        }
    },
    phases={"default": {"type": "concurrency", "requests": 10, "concurrency": 1}},
)


@pytest.fixture(autouse=True)
def _mock_tokenizer_validation(mock_get_global_error_queue):
    """Prevent real HF network calls and forkserver init during CLI runner tests."""
    with patch(
        "aiperf.common.tokenizer_validator.validate_tokenizer_early",
        return_value={"test-model": "test-model"},
    ):
        yield


class TestMacOSTerminalFixes:
    """Test the macOS-specific terminal corruption fixes in cli_runner.py"""

    @pytest.fixture(autouse=True)
    def _mock_os_exit(self):
        """_run_single_benchmark terminates with os._exit to bypass Python teardown;
        neutralize it so test assertions can run."""
        with patch("os._exit"):
            yield

    @pytest.fixture
    def config_dashboard(self) -> BenchmarkPlan:
        """Create a BenchmarkPlan with Dashboard UI type."""
        return build_benchmark_plan(
            AIPerfConfig(**_BASE, runtime={"ui": UIType.DASHBOARD})
        )

    @pytest.fixture
    def config_simple(self) -> BenchmarkPlan:
        """Create a BenchmarkPlan with Simple UI type."""
        return build_benchmark_plan(
            AIPerfConfig(**_BASE, runtime={"ui": UIType.SIMPLE})
        )

    def test_spawn_method_set_on_macos_dashboard(
        self,
        config_dashboard: BenchmarkPlan,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that spawn method is set when on macOS with Dashboard UI."""
        run_benchmark(config_dashboard)

        mock_multiprocessing_set_start_method.assert_called_once_with(
            "spawn", force=True
        )

    def test_no_start_method_set_on_linux(
        self,
        config_dashboard: BenchmarkPlan,
        mock_platform_linux: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that spawn method is NOT set on Linux."""
        run_benchmark(config_dashboard)

        mock_multiprocessing_set_start_method.assert_not_called()

    def test_no_start_method_set_for_simple_ui_on_macos(
        self,
        config_simple: BenchmarkPlan,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that spawn method is NOT set when not using Dashboard UI on macOS."""
        run_benchmark(config_simple)

        mock_multiprocessing_set_start_method.assert_not_called()

    @patch("fcntl.fcntl")
    def test_fd_cloexec_not_set_on_linux(
        self,
        mock_fcntl: Mock,
        config_dashboard: BenchmarkPlan,
        mock_platform_linux: Mock,
        mock_bootstrap_and_run_service: Mock,
        mock_get_global_log_queue: Mock,
    ):
        """Test that FD_CLOEXEC is NOT set on Linux."""
        mock_get_global_log_queue.return_value = MagicMock(spec=multiprocessing.Queue)

        run_benchmark(config_dashboard)

        mock_fcntl.assert_not_called()

    def test_runtime_error_in_set_start_method_is_handled(
        self,
        config_dashboard: BenchmarkPlan,
        mock_platform_darwin: Mock,
        mock_multiprocessing_set_start_method: Mock,
        mock_bootstrap_and_run_service: Mock,
    ):
        """Test that RuntimeError when setting start method is gracefully handled."""
        mock_multiprocessing_set_start_method.side_effect = RuntimeError(
            "context already set"
        )

        run_benchmark(config_dashboard)

        mock_multiprocessing_set_start_method.assert_called_once()

    def test_log_queue_created_before_ui_on_dashboard(
        self,
        config_dashboard: BenchmarkPlan,
        mock_platform_darwin: Mock,
        mock_bootstrap_and_run_service: Mock,
        mock_get_global_log_queue: Mock,
    ):
        """Test that log_queue is created early when using Dashboard UI."""
        mock_queue = MagicMock(spec=multiprocessing.Queue)
        mock_get_global_log_queue.return_value = mock_queue

        run_benchmark(config_dashboard)

        mock_get_global_log_queue.assert_called_once()

        mock_bootstrap_and_run_service.assert_called_once()
        call_kwargs = mock_bootstrap_and_run_service.call_args.kwargs
        assert "log_queue" in call_kwargs
        assert call_kwargs["log_queue"] == mock_queue
