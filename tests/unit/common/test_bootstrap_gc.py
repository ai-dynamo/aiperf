# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for garbage collection disabling in bootstrap.py"""

import pytest

from aiperf.common.bootstrap import bootstrap_and_run_service
from aiperf.common.config import ServiceConfig, UserConfig
from tests.unit.common.conftest import MockGC


class TestBootstrapGarbageCollection:
    """Test the garbage collection disabling in bootstrap.py"""

    @pytest.fixture(autouse=True)
    def setup_bootstrap_mocks(
        self,
        mock_psutil_process,
        mock_setup_child_process_logging,
        register_dummy_services,
    ):
        """Combine common bootstrap mocks that are used but not called in tests."""
        pass

    def test_gc_disabled_for_worker_service(
        self,
        service_config_no_uvloop: ServiceConfig,
        user_config: UserConfig,
        mock_log_queue,
        mock_gc: MockGC,
    ):
        """Test that GC is disabled for Worker service."""
        bootstrap_and_run_service(
            "test_worker",
            service_config=service_config_no_uvloop,
            user_config=user_config,
            log_queue=mock_log_queue,
            service_id="test_worker",
        )

        # Verify GC was disabled
        assert mock_gc.collect.call_count == 3  # Called 3 times in a loop
        mock_gc.freeze.assert_called_once()
        mock_gc.set_threshold.assert_called_once_with(0)
        mock_gc.disable.assert_called_once()

    def test_gc_disabled_for_timing_manager_service(
        self,
        service_config_no_uvloop: ServiceConfig,
        user_config: UserConfig,
        mock_log_queue,
        mock_gc: MockGC,
    ):
        """Test that GC is disabled for TimingManager service."""
        bootstrap_and_run_service(
            "test_timing_manager",
            service_config=service_config_no_uvloop,
            user_config=user_config,
            log_queue=mock_log_queue,
            service_id="test_timing_manager",
        )

        # Verify GC was disabled
        assert mock_gc.collect.call_count == 3  # Called 3 times in a loop
        mock_gc.freeze.assert_called_once()
        mock_gc.set_threshold.assert_called_once_with(0)
        mock_gc.disable.assert_called_once()

    def test_gc_not_disabled_for_other_services(
        self,
        service_config_no_uvloop: ServiceConfig,
        user_config: UserConfig,
        mock_log_queue,
        mock_gc: MockGC,
    ):
        """Test that GC is NOT disabled for services other than Worker and TimingManager."""
        bootstrap_and_run_service(
            "test_dummy",
            service_config=service_config_no_uvloop,
            user_config=user_config,
            log_queue=mock_log_queue,
            service_id="test_dummy",
        )

        # Verify GC was NOT disabled
        mock_gc.collect.assert_not_called()
        mock_gc.freeze.assert_not_called()
        mock_gc.set_threshold.assert_not_called()
        mock_gc.disable.assert_not_called()

    def test_gc_operations_occur_in_correct_order(
        self,
        service_config_no_uvloop: ServiceConfig,
        user_config: UserConfig,
        mock_log_queue,
        mock_gc: MockGC,
    ):
        """Test that GC operations occur in the correct order: collect -> freeze -> set_threshold -> disable."""
        bootstrap_and_run_service(
            "test_worker",
            service_config=service_config_no_uvloop,
            user_config=user_config,
            log_queue=mock_log_queue,
            service_id="test_worker",
        )

        # Verify order: collect (3x), freeze, set_threshold, disable
        expected = [
            "collect",
            "collect",
            "collect",
            "freeze",
            "set_threshold",
            "disable",
        ]
        assert mock_gc.call_order == expected
