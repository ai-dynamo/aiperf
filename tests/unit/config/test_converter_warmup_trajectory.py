# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for --trajectory-start-min/max-ratio (mid-conversation warmup seeding).

Covers the CLI->warmup-phase conversion in ``_converter_warmup.build_warmup``
and the min<=max constraint enforced on the phase config.
"""

from __future__ import annotations

import pytest

from aiperf.config.flags._converter_warmup import build_warmup
from aiperf.config.flags.cli_config import CLIConfig
from aiperf.config.phases import ConcurrencyPhase


def _make_user(loadgen: CLIConfig) -> CLIConfig:
    endpoint = CLIConfig(url="http://localhost:8000/test", model_names=["test-model"])
    return CLIConfig(
        **endpoint.model_dump(exclude_unset=True),
        **loadgen.model_dump(exclude_unset=True),
    )


class TestWarmupTrajectorySeeding:
    def test_max_ratio_with_trigger_resolves(self):
        loadgen = CLIConfig(warmup_duration=10.0, trajectory_start_max_ratio=0.7)
        warmup = build_warmup(_make_user(loadgen))
        assert warmup is not None
        assert warmup["trajectory_start_max_ratio"] == 0.7
        # min not set -> left to the phase default (0.0), absent from the dict
        assert "trajectory_start_min_ratio" not in warmup

    def test_min_and_max_resolve(self):
        loadgen = CLIConfig(
            warmup_duration=10.0,
            trajectory_start_min_ratio=0.3,
            trajectory_start_max_ratio=0.7,
        )
        warmup = build_warmup(_make_user(loadgen))
        assert warmup["trajectory_start_min_ratio"] == 0.3
        assert warmup["trajectory_start_max_ratio"] == 0.7

    def test_trajectory_without_warmup_trigger_raises(self):
        loadgen = CLIConfig(trajectory_start_max_ratio=0.5)
        with pytest.raises(ValueError, match="without any warmup trigger"):
            build_warmup(_make_user(loadgen))

    def test_no_trajectory_flags_omits_keys(self):
        loadgen = CLIConfig(warmup_duration=10.0)
        warmup = build_warmup(_make_user(loadgen))
        assert "trajectory_start_min_ratio" not in warmup
        assert "trajectory_start_max_ratio" not in warmup


class TestPhaseTrajectoryValidation:
    def _phase(self, lo: float, hi: float) -> ConcurrencyPhase:
        return ConcurrencyPhase(
            name="warmup",
            type="concurrency",
            concurrency=4,
            duration=10,
            trajectory_start_min_ratio=lo,
            trajectory_start_max_ratio=hi,
        )

    def test_min_greater_than_max_raises(self):
        with pytest.raises(ValueError, match="must be <="):
            self._phase(0.7, 0.3)

    def test_valid_range_ok(self):
        phase = self._phase(0.3, 0.7)
        assert phase.trajectory_start_min_ratio == 0.3
        assert phase.trajectory_start_max_ratio == 0.7

    def test_default_disabled_ok(self):
        phase = ConcurrencyPhase(
            name="warmup", type="concurrency", concurrency=4, duration=10
        )
        assert phase.trajectory_start_min_ratio == 0.0
        assert phase.trajectory_start_max_ratio == 0.0
