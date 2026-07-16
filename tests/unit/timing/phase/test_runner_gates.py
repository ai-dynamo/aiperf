# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit-level test: PhaseRunner accepts a PhaseGateClient via constructor."""

import inspect

from aiperf.timing.phase.runner import PhaseRunner


def test_phase_runner_accepts_phase_gate_kwarg() -> None:
    """PhaseRunner.__init__ accepts a `phase_gate` kwarg (None-acceptable default)."""
    params = inspect.signature(PhaseRunner.__init__).parameters
    assert "phase_gate" in params
    # default should be None (kwarg, not required)
    assert params["phase_gate"].default is None
