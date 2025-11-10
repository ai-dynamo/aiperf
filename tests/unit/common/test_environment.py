# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from pytest import param

from aiperf.common.environment import _ServiceSettings


class TestServiceSettingsUvloopWindows:
    """Test suite for automatic uvloop disabling on Windows."""

    @pytest.mark.parametrize(
        "platform_name,expected_disable_uvloop",
        [
            param("Windows", True, id="windows_auto_disabled"),
            param("Linux", False, id="linux_enabled"),
            param("Darwin", False, id="macos_enabled"),
        ],
    )
    @patch("aiperf.common.environment.platform.system")
    def test_platform_uvloop_detection(
        self, mock_platform, platform_name, expected_disable_uvloop
    ):
        """Test that uvloop is automatically disabled on Windows and enabled elsewhere."""
        mock_platform.return_value = platform_name

        settings = _ServiceSettings()

        assert settings.DISABLE_UVLOOP is expected_disable_uvloop

    @pytest.mark.parametrize(
        "platform_name,manual_setting,expected_result",
        [
            param("Windows", False, True, id="windows_override_attempt"),
            param("Windows", True, True, id="windows_manual_disable"),
            param("Linux", True, True, id="linux_manual_disable"),
            param("Linux", False, False, id="linux_default_enabled"),
            param("Darwin", True, True, id="macos_manual_disable"),
            param("Darwin", False, False, id="macos_default_enabled"),
        ],
    )
    @patch("aiperf.common.environment.platform.system")
    def test_manual_uvloop_settings(
        self, mock_platform, platform_name, manual_setting, expected_result
    ):
        """Test manual DISABLE_UVLOOP settings across platforms."""
        mock_platform.return_value = platform_name

        settings = _ServiceSettings(DISABLE_UVLOOP=manual_setting)

        assert settings.DISABLE_UVLOOP is expected_result
