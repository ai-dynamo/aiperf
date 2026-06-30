# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from pydantic import ValidationError

from aiperf.config.phases import PoissonPhase, UserCentricPhase
from aiperf.config.rate_series import RateSeriesConfig, read_rate_series_json
from aiperf.plugin.enums import PhaseType


def _write_json(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_rate_series_config_loads_json_path(tmp_path: Path) -> None:
    json_path = _write_json(
        tmp_path / "rate.json",
        '{"points":[{"time_s":0,"qps":1},{"time_s":60,"qps":7},{"time_s":120,"qps":40}]}',
    )

    config = RateSeriesConfig(path=str(json_path))

    assert config.initial_qps == 1.0
    assert [(point.time_s, point.qps) for point in config.points] == [
        (0.0, 1.0),
        (60.0, 7.0),
        (120.0, 40.0),
    ]


def test_rate_series_config_canonicalizes_loaded_path(tmp_path: Path) -> None:
    json_path = _write_json(
        tmp_path / "rate.json",
        '{"points":[{"time_s":0,"qps":1},{"time_s":60,"qps":7}]}',
    )

    config = RateSeriesConfig(path=str(json_path))
    round_tripped = RateSeriesConfig.model_validate(config.model_dump())

    assert config.path is None
    assert round_tripped.path is None
    assert [(point.time_s, point.qps) for point in round_tripped.points] == [
        (0.0, 1.0),
        (60.0, 7.0),
    ]


def test_rate_series_config_accepts_inline_points() -> None:
    config = RateSeriesConfig(
        points=[{"time_s": 0, "qps": 10}, {"time_s": 5, "qps": 20}]
    )

    assert config.initial_qps == 10.0
    assert config.points[1].qps == 20.0


def test_read_rate_series_json_accepts_top_level_points_array(tmp_path: Path) -> None:
    json_path = _write_json(
        tmp_path / "rate.json",
        '[{"time_s":0,"qps":1},{"time_s":60,"qps":7}]',
    )

    points = read_rate_series_json(str(json_path))

    assert [(point.time_s, point.qps) for point in points] == [(0.0, 1.0), (60.0, 7.0)]


def test_read_rate_series_json_rejects_wrong_shape(tmp_path: Path) -> None:
    json_path = _write_json(tmp_path / "bad.json", '{"time_s":0,"qps":1}')

    with pytest.raises(ValueError, match="top-level key: points"):
        read_rate_series_json(str(json_path))


def test_read_rate_series_json_rejects_non_increasing_times(tmp_path: Path) -> None:
    json_path = _write_json(
        tmp_path / "bad.json",
        '{"points":[{"time_s":0,"qps":1},{"time_s":0,"qps":7}]}',
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        read_rate_series_json(str(json_path))


def test_read_rate_series_json_preserves_point_validation_cause(
    tmp_path: Path,
) -> None:
    json_path = _write_json(
        tmp_path / "bad.json",
        '{"points":[{"time_s":0,"qps":1},{"time_s":5,"qps":0}]}',
    )

    with pytest.raises(ValueError) as exc:
        read_rate_series_json(str(json_path))

    message = str(exc.value)
    assert "Invalid request-rate series point 1" in message
    assert "greater than 0" in message


def test_rate_series_config_rejects_single_point() -> None:
    with pytest.raises(ValidationError, match="at least two points"):
        RateSeriesConfig(points=[{"time_s": 0, "qps": 10}])


def test_rate_phase_allows_rate_series_without_scalar_rate() -> None:
    phase = PoissonPhase(
        name="profiling",
        type=PhaseType.POISSON,
        requests=10,
        rate_series={"points": [{"time_s": 0, "qps": 5}, {"time_s": 10, "qps": 15}]},
    )

    assert phase.rate is None
    assert phase.rate_series is not None
    assert phase.rate_series.initial_qps == 5.0


def test_rate_phase_allows_rate_ramp_with_rate_series() -> None:
    phase = PoissonPhase(
        name="profiling",
        type=PhaseType.POISSON,
        requests=10,
        rate_ramp=60,
        rate_series={"points": [{"time_s": 0, "qps": 5}, {"time_s": 10, "qps": 15}]},
    )

    assert phase.rate_ramp is not None
    assert phase.rate_series is not None


def test_user_centric_phase_rejects_rate_series() -> None:
    with pytest.raises(ValidationError, match="user-centric phases"):
        UserCentricPhase(
            name="profiling",
            type=PhaseType.USER_CENTRIC,
            requests=10,
            users=2,
            rate_series={
                "points": [{"time_s": 0, "qps": 5}, {"time_s": 10, "qps": 15}]
            },
        )
