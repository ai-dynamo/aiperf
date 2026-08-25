# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform-conditional bootstrap branches read the shared IS_MACOS constant."""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from pytest import param

from aiperf.cli_runner._process_setup import _setup_ui_queues


@pytest.mark.parametrize(
    "is_macos,expect_cloexec",
    [
        param(True, True, id="macos"),
        param(False, False, id="not_macos"),
    ],
)  # fmt: skip
def test_setup_ui_queues_dashboard_branches_on_is_macos_constant(
    is_macos: bool, expect_cloexec: bool
) -> None:
    """The FD_CLOEXEC mitigation is driven by aiperf.common.constants.IS_MACOS."""
    with (
        patch("aiperf.common.constants.IS_MACOS", is_macos),
        patch(
            "aiperf.cli_runner._process_setup._set_fd_cloexec_on_terminal"
        ) as mock_cloexec,
        patch("aiperf.common.logging.get_global_log_queue", return_value=Mock()),
    ):
        _setup_ui_queues(using_dashboard=True, run=Mock(), logger=Mock())

    assert mock_cloexec.called is expect_cloexec
