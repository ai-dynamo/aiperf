# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys

_IMPORT_CODE = """
import sys
sys.modules["termios"] = None
import aiperf.common.constants as constants
constants.IS_WINDOWS = True
import aiperf.ui.dashboard.aiperf_dashboard_ui
"""


def test_dashboard_ui_import_without_termios_succeeds() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_CODE],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
