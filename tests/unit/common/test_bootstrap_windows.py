# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Windows-specific event loop policy fix in bootstrap.py.

Bug 5: pyzmq's async sockets call loop.add_reader() / add_writer(), which the
default Windows ProactorEventLoop does not implement. AIPerf forces the
SelectorEventLoopPolicy on Windows before the loop is created. These tests
mock IS_WINDOWS to exercise both branches from non-Windows CI.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

from aiperf.common.bootstrap import _configure_event_loop_policy_for_platform


class TestConfigureEventLoopPolicyForPlatform:
    """Verify the platform-conditional event-loop-policy setup."""

    def test_sets_event_loop_policy_when_windows(self) -> None:
        """On Windows, asyncio.set_event_loop_policy must be called once
        with a WindowsSelectorEventLoopPolicy instance."""
        # WindowsSelectorEventLoopPolicy doesn't exist on non-Windows Python,
        # so patch with create=True so the attribute lookup succeeds.
        with (
            patch("aiperf.common.bootstrap.IS_WINDOWS", True),
            patch.object(
                asyncio,
                "WindowsSelectorEventLoopPolicy",
                create=True,
            ) as mock_policy_cls,
            patch("asyncio.set_event_loop_policy") as mock_set_policy,
        ):
            _configure_event_loop_policy_for_platform()

        mock_policy_cls.assert_called_once_with()
        mock_set_policy.assert_called_once_with(mock_policy_cls.return_value)

    def test_does_not_touch_event_loop_policy_when_not_windows(self) -> None:
        """On non-Windows platforms the helper is a no-op — leave the
        platform default in place (uvloop, or asyncio default)."""
        with (
            patch("aiperf.common.bootstrap.IS_WINDOWS", False),
            patch("asyncio.set_event_loop_policy") as mock_set_policy,
        ):
            _configure_event_loop_policy_for_platform()

        mock_set_policy.assert_not_called()
