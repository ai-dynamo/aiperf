# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ignore_trace_delays`` must not strip timestamps under fixed schedule.

The flag's own contract says so: "No effect under fixed-schedule (timestamps
drive that mode before they could be ignored)". Honouring it matters because
fixed_schedule has nothing else to schedule on -- emitting None timestamps
fails in PhaseOrchestrator on the first turn, long after config resolution has
already accepted the run.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from tests.unit.dataset.loader.conftest import make_weka_run


def _loader(tmp_path, *, ignore_trace_delays: bool, fixed_schedule: bool):
    trace = tmp_path / "t.json"
    trace.write_text("{}", encoding="utf-8")
    run = make_weka_run(
        ignore_trace_delays=ignore_trace_delays,
        # Passing an offset is what makes the helper emit a fixed_schedule phase.
        fixed_schedule_start_offset=0 if fixed_schedule else None,
    )
    return WekaTraceLoader(filename=str(trace), run=run)


@pytest.mark.parametrize(
    "ignore,fixed,expected",
    [
        param(True, True, False, id="fixed-schedule-wins-over-ignore"),
        param(True, False, True, id="ignore-applies-without-fixed-schedule"),
        param(False, True, False, id="not-requested-under-fixed-schedule"),
        param(False, False, False, id="not-requested"),
    ],
)  # fmt: skip
def test_delays_are_only_stripped_when_the_phase_can_afford_it(
    tmp_path, ignore: bool, fixed: bool, expected: bool
) -> None:
    loader = _loader(tmp_path, ignore_trace_delays=ignore, fixed_schedule=fixed)
    assert loader._ignore_trace_delays_effective is expected
