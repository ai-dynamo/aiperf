# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the concurrency schedule expansion and writer output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import orjson
import pytest
from pydantic import ValidationError

from aiperf.dataset.agentic_code_gen.models import (
    ConcurrencyScheduleAnchor,
    ConcurrencyScheduleConfig,
    SessionDistributionConfig,
)
from aiperf.dataset.agentic_code_gen.schedule import expand_schedule
from aiperf.dataset.agentic_code_gen.session_synthesizer import SessionSynthesizer
from aiperf.dataset.agentic_code_gen.writer import write_dataset


def _cfg(**kwargs) -> ConcurrencyScheduleConfig:
    defaults = {
        "interpolation": "linear",
        "tick_sec": 1.0,
        "noise_sigma": 0.0,
        "anchors": [
            ConcurrencyScheduleAnchor(time_sec=0, concurrency=10),
            ConcurrencyScheduleAnchor(time_sec=10, concurrency=20),
        ],
    }
    defaults.update(kwargs)
    return ConcurrencyScheduleConfig(**defaults)


class TestExpandSchedule:
    def test_linear_ramp_matches_endpoints(self) -> None:
        rng = np.random.default_rng(0)
        ticks = expand_schedule(_cfg(), rng)
        assert ticks[0] == (0.0, 10)
        assert ticks[-1][1] == 20
        assert len(ticks) == 11

    def test_linear_interpolates_intermediate_values(self) -> None:
        rng = np.random.default_rng(0)
        ticks = expand_schedule(_cfg(), rng)
        values = [c for _, c in ticks]
        assert values == [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    def test_step_interpolation_holds_previous_value(self) -> None:
        rng = np.random.default_rng(0)
        ticks = expand_schedule(_cfg(interpolation="step"), rng)
        values = [c for _, c in ticks]
        assert values[:-1] == [10] * 10
        assert values[-1] == 20

    def test_hold_then_ramp(self) -> None:
        rng = np.random.default_rng(0)
        cfg = _cfg(
            tick_sec=100.0,
            anchors=[
                ConcurrencyScheduleAnchor(time_sec=0, concurrency=10),
                ConcurrencyScheduleAnchor(time_sec=600, concurrency=10),
                ConcurrencyScheduleAnchor(time_sec=900, concurrency=20),
            ],
        )
        ticks = expand_schedule(cfg, rng)
        values_before_ramp = [c for t, c in ticks if t <= 600]
        values_after_600 = [c for t, c in ticks if t > 600]
        assert set(values_before_ramp) == {10}
        assert values_after_600[-1] == 20
        assert all(v >= 10 for v in values_after_600)

    def test_same_seed_produces_identical_ticks(self) -> None:
        cfg = _cfg(noise_sigma=0.2)
        ticks_a = expand_schedule(cfg, np.random.default_rng(42))
        ticks_b = expand_schedule(cfg, np.random.default_rng(42))
        assert ticks_a == ticks_b

    def test_different_seeds_produce_different_noisy_ticks(self) -> None:
        cfg = _cfg(noise_sigma=0.3)
        ticks_a = expand_schedule(cfg, np.random.default_rng(1))
        ticks_b = expand_schedule(cfg, np.random.default_rng(2))
        assert ticks_a != ticks_b

    def test_noise_shifts_distribution_around_target(self) -> None:
        flat_cfg = _cfg(
            tick_sec=1.0,
            noise_sigma=0.1,
            anchors=[
                ConcurrencyScheduleAnchor(time_sec=0, concurrency=100),
                ConcurrencyScheduleAnchor(time_sec=2000, concurrency=100),
            ],
        )
        ticks = expand_schedule(flat_cfg, np.random.default_rng(42))
        values = np.array([c for _, c in ticks], dtype=float)
        assert 95 < values.mean() < 105
        assert 8 < values.std() < 14

    def test_concurrency_clamped_to_at_least_one(self) -> None:
        cfg = _cfg(
            noise_sigma=2.0,
            anchors=[
                ConcurrencyScheduleAnchor(time_sec=0, concurrency=1),
                ConcurrencyScheduleAnchor(time_sec=100, concurrency=1),
            ],
        )
        ticks = expand_schedule(cfg, np.random.default_rng(42))
        assert all(c >= 1 for _, c in ticks)


class TestScheduleConfigValidation:
    def test_rejects_first_anchor_not_at_zero(self) -> None:
        with pytest.raises(ValidationError, match="first anchor must be at time_sec=0"):
            ConcurrencyScheduleConfig(
                anchors=[
                    ConcurrencyScheduleAnchor(time_sec=5, concurrency=10),
                    ConcurrencyScheduleAnchor(time_sec=10, concurrency=20),
                ],
            )

    def test_rejects_non_monotonic_anchors(self) -> None:
        with pytest.raises(ValidationError, match="anchors must be strictly monotonic"):
            ConcurrencyScheduleConfig(
                anchors=[
                    ConcurrencyScheduleAnchor(time_sec=0, concurrency=10),
                    ConcurrencyScheduleAnchor(time_sec=10, concurrency=20),
                    ConcurrencyScheduleAnchor(time_sec=5, concurrency=15),
                ],
            )

    def test_rejects_duplicate_timestamps(self) -> None:
        with pytest.raises(ValidationError, match="anchors must be strictly monotonic"):
            ConcurrencyScheduleConfig(
                anchors=[
                    ConcurrencyScheduleAnchor(time_sec=0, concurrency=10),
                    ConcurrencyScheduleAnchor(time_sec=10, concurrency=20),
                    ConcurrencyScheduleAnchor(time_sec=10, concurrency=30),
                ],
            )

    def test_rejects_single_anchor(self) -> None:
        with pytest.raises(ValidationError):
            ConcurrencyScheduleConfig(
                anchors=[ConcurrencyScheduleAnchor(time_sec=0, concurrency=10)],
            )


class TestScheduleWriterIntegration:
    """The writer only emits schedule.jsonl when concurrency_schedule is set."""

    def test_no_schedule_when_not_configured(
        self, tmp_path: Path, coding_config: SessionDistributionConfig
    ) -> None:
        synth = SessionSynthesizer(coding_config, seed=42)
        sessions = synth.synthesize_sessions(3)
        _, _, _, schedule_path = write_dataset(
            sessions, tmp_path / "run", coding_config, seed=42
        )
        assert schedule_path is None
        assert not (tmp_path / "run" / "schedule.jsonl").exists()

    def test_schedule_written_when_configured(
        self, tmp_path: Path, coding_config: SessionDistributionConfig
    ) -> None:
        config = coding_config.model_copy(
            update={
                "concurrency_schedule": ConcurrencyScheduleConfig(
                    tick_sec=5.0,
                    anchors=[
                        ConcurrencyScheduleAnchor(time_sec=0, concurrency=10),
                        ConcurrencyScheduleAnchor(time_sec=30, concurrency=20),
                    ],
                )
            }
        )
        synth = SessionSynthesizer(config, seed=42)
        sessions = synth.synthesize_sessions(3)
        _, _, _, schedule_path = write_dataset(
            sessions, tmp_path / "run", config, seed=42
        )
        assert schedule_path is not None
        assert schedule_path.name == "schedule.jsonl"

        lines = [
            orjson.loads(line)
            for line in schedule_path.read_bytes().splitlines()
            if line.strip()
        ]
        assert all(set(row.keys()) == {"time_sec", "concurrency"} for row in lines)
        assert lines[0] == {"time_sec": 0.0, "concurrency": 10}
        assert lines[-1]["concurrency"] == 20

    def test_schedule_respects_synth_seed(
        self, tmp_path: Path, coding_config: SessionDistributionConfig
    ) -> None:
        """Same seed → identical schedule.jsonl even when noise is applied."""
        config = coding_config.model_copy(
            update={
                "concurrency_schedule": ConcurrencyScheduleConfig(
                    tick_sec=1.0,
                    noise_sigma=0.3,
                    anchors=[
                        ConcurrencyScheduleAnchor(time_sec=0, concurrency=50),
                        ConcurrencyScheduleAnchor(time_sec=10, concurrency=50),
                    ],
                )
            }
        )
        synth = SessionSynthesizer(config, seed=7)
        sessions = synth.synthesize_sessions(1)
        _, _, _, schedule_path_a = write_dataset(
            sessions, tmp_path / "run_a", config, seed=7
        )
        _, _, _, schedule_path_b = write_dataset(
            sessions, tmp_path / "run_b", config, seed=7
        )
        assert schedule_path_a.read_bytes() == schedule_path_b.read_bytes()
