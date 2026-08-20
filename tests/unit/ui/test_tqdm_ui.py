# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

from aiperf.config import BenchmarkRun
from aiperf.ui.tqdm_ui import TQDMProgressUI


def test_phase_progress_hook_accepts_named_phase_payload() -> None:
    """The hook runner dispatches named-phase progress with ``phase_stats=``."""
    ui = object.__new__(TQDMProgressUI)
    ui._create_or_update_requests_bar = MagicMock()
    phase_stats = MagicMock(
        phase="profiling",
        phase_name="load",
        phase_kind="profiling",
        timeout_triggered=False,
    )

    ui._on_phase_progress(phase_stats=phase_stats)

    ui._create_or_update_requests_bar.assert_called_once()
    assert ui._create_or_update_requests_bar.call_args.args[:3] == (
        "Load",
        "green",
        phase_stats,
    )


def test_ui_does_not_own_controller_communication_lifecycle(
    benchmark_run: BenchmarkRun,
) -> None:
    """Stopping a UI must leave the controller-owned shared bus running."""
    comms = MagicMock()

    with patch(
        "aiperf.common.mixins.communication_mixin.plugins.get_class",
        return_value=lambda **_: comms,
    ):
        ui = TQDMProgressUI(run=benchmark_run)

    assert ui.comms is comms
    assert comms not in ui._children
