# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the concurrency schedule loader."""

from __future__ import annotations

from pathlib import Path

import orjson
import pytest

from aiperf.dataset.loader.schedule import ScheduleTick, load_schedule


def _write_schedule(path: Path, rows: list[dict]) -> Path:
    path.write_bytes(b"\n".join(orjson.dumps(row) for row in rows) + b"\n")
    return path


class TestLoadSchedule:
    def test_round_trip_basic(self, tmp_path: Path) -> None:
        path = _write_schedule(
            tmp_path / "schedule.jsonl",
            [
                {"time_sec": 0.0, "concurrency": 10},
                {"time_sec": 5.0, "concurrency": 15},
                {"time_sec": 10.0, "concurrency": 20},
            ],
        )
        ticks = load_schedule(path)
        assert ticks == [
            ScheduleTick(time_sec=0.0, concurrency=10),
            ScheduleTick(time_sec=5.0, concurrency=15),
            ScheduleTick(time_sec=10.0, concurrency=20),
        ]

    def test_ignores_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "schedule.jsonl"
        path.write_bytes(
            b'{"time_sec":0,"concurrency":10}\n\n{"time_sec":5,"concurrency":20}\n'
        )
        ticks = load_schedule(path)
        assert len(ticks) == 2

    def test_rejects_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "schedule.jsonl"
        path.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            load_schedule(path)

    def test_rejects_non_zero_start(self, tmp_path: Path) -> None:
        path = _write_schedule(
            tmp_path / "schedule.jsonl",
            [
                {"time_sec": 5.0, "concurrency": 10},
                {"time_sec": 10.0, "concurrency": 20},
            ],
        )
        with pytest.raises(ValueError, match="time_sec=0"):
            load_schedule(path)

    def test_rejects_non_monotonic_ticks(self, tmp_path: Path) -> None:
        path = _write_schedule(
            tmp_path / "schedule.jsonl",
            [
                {"time_sec": 0.0, "concurrency": 10},
                {"time_sec": 10.0, "concurrency": 15},
                {"time_sec": 5.0, "concurrency": 20},
            ],
        )
        with pytest.raises(ValueError, match="strictly monotonic"):
            load_schedule(path)

    def test_rejects_duplicate_timestamps(self, tmp_path: Path) -> None:
        path = _write_schedule(
            tmp_path / "schedule.jsonl",
            [
                {"time_sec": 0.0, "concurrency": 10},
                {"time_sec": 5.0, "concurrency": 15},
                {"time_sec": 5.0, "concurrency": 20},
            ],
        )
        with pytest.raises(ValueError, match="strictly monotonic"):
            load_schedule(path)

    def test_rejects_non_positive_concurrency(self, tmp_path: Path) -> None:
        path = _write_schedule(
            tmp_path / "schedule.jsonl",
            [
                {"time_sec": 0.0, "concurrency": 0},
                {"time_sec": 5.0, "concurrency": 10},
            ],
        )
        with pytest.raises(Exception, match="greater than or equal to 1"):
            load_schedule(path)
