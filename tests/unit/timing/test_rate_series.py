# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib

import pytest

from aiperf.config.rate_series import RateSeriesConfig
from aiperf.timing.rate_series import RateSeriesController
from tests.harness.time_traveler import TimeTraveler


def series_config() -> RateSeriesConfig:
    return RateSeriesConfig(
        points=[
            {"time_s": 10.0, "qps": 5.0},
            {"time_s": 20.0, "qps": 15.0},
            {"time_s": 30.0, "qps": 10.0},
        ]
    )


class TestRateSeriesController:
    def test_value_at_interpolates_and_holds_edges(self) -> None:
        controller = RateSeriesController(
            setter=lambda value: None,
            config=series_config(),
            update_interval=0.1,
        )

        assert controller.value_at(0.0) == pytest.approx(5.0)
        assert controller.value_at(10.0) == pytest.approx(5.0)
        assert controller.value_at(15.0) == pytest.approx(10.0)
        assert controller.value_at(20.0) == pytest.approx(15.0)
        assert controller.value_at(25.0) == pytest.approx(12.5)
        assert controller.value_at(40.0) == pytest.approx(10.0)

    @pytest.mark.asyncio
    async def test_start_sets_initial_value(self, time_traveler: TimeTraveler) -> None:
        values: list[float] = []
        controller = RateSeriesController(
            setter=values.append,
            config=series_config(),
            update_interval=0.5,
        )

        task = controller.start()
        await time_traveler.sleep(0.1)

        assert values == [5.0]
        controller.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_start_waits_for_start_delay(
        self, time_traveler: TimeTraveler
    ) -> None:
        values: list[float] = []
        controller = RateSeriesController(
            setter=values.append,
            config=series_config(),
            update_interval=0.5,
            start_delay=2.0,
        )

        task = controller.start()
        await time_traveler.sleep(1.0)
        assert values == []

        await time_traveler.sleep(1.0)
        assert values == [5.0]

        controller.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_start_stops_after_final_point(
        self, time_traveler: TimeTraveler
    ) -> None:
        values: list[float] = []
        controller = RateSeriesController(
            setter=values.append,
            config=RateSeriesConfig(
                points=[{"time_s": 0.0, "qps": 5.0}, {"time_s": 1.0, "qps": 10.0}]
            ),
            update_interval=0.25,
        )

        task = controller.start()
        await time_traveler.sleep(1.25)
        await task

        assert controller.is_running is False
        assert values[-1] == 10.0

    @pytest.mark.asyncio
    async def test_start_logs_setter_failures(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        def fail_setter(value: float) -> None:
            raise RuntimeError(f"cannot set {value}")

        controller = RateSeriesController(
            setter=fail_setter,
            config=series_config(),
            update_interval=0.5,
        )

        task = controller.start()
        await task

        assert "Request-rate series update failed" in caplog.text
