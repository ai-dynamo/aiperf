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
from tests.unit.dataset.loader.test_weka_trace import (
    _stub_prompt_generator_for_reconstructor,
)

BLOCK = 64


def _write_trace(tmp_path) -> str:
    import json

    trace = {
        "id": "demo",
        "models": ["test-model"],
        "block_size": BLOCK,
        "hash_id_scope": "local",
        "tool_tokens": 0,
        "system_tokens": 0,
        "requests": [
            {
                "t": t,
                "type": "n",
                "model": "test-model",
                "in": len(ids) * BLOCK,
                "out": 16,
                "hash_ids": ids,
                "stop": "end_turn",
            }
            for t, ids in [(0.0, [1, 2]), (5.0, [1, 2, 3])]
        ],
    }
    path = tmp_path / "trace_0001.json"
    path.write_text(json.dumps(trace), encoding="utf-8")
    return str(path)


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


def test_fixed_schedule_keeps_the_recorded_timestamps_on_emitted_turns(
    tmp_path,
) -> None:
    """The behavioural half: assert what the orchestrator actually consumes.

    Asserting only on ``_ignore_trace_delays_effective`` pins the computation,
    not the use of it -- recomputing the right value and then passing the raw
    flag to reconstruction leaves that assertion green while every emitted
    timestamp goes back to ``None``, which is the exact regression this guards.
    Fixed schedule has nothing else to schedule on, so a ``None`` here fails
    far downstream in PhaseOrchestrator, long after resolution accepted the run.
    """
    run = make_weka_run(ignore_trace_delays=True, fixed_schedule_start_offset=0)
    loader = WekaTraceLoader(filename=_write_trace(tmp_path), run=run)
    _stub_prompt_generator_for_reconstructor(loader)

    conversations = loader.convert_to_conversations(loader.load_dataset())
    turns = [turn for conv in conversations for turn in conv.turns]

    assert turns, "fixture produced no turns"
    assert all(turn.timestamp is not None for turn in turns), (
        "fixed schedule must keep the recorded timestamps even with "
        "--ignore-trace-delays set"
    )
