# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import contextlib
from unittest.mock import MagicMock

import pytest

from aiperf.timing.ramping import RampConfig, Ramper, RampType


def lin(s: float, t: float, d: float, step: float | None = None) -> RampConfig:
    return RampConfig(
        ramp_type=RampType.LINEAR, start=s, target=t, duration_sec=d, step_size=step
    )


def exp(s: float, t: float, d: float, e: float = 2.0) -> RampConfig:
    return RampConfig(
        ramp_type=RampType.EXPONENTIAL, start=s, target=t, duration_sec=d, exponent=e
    )


def cont(s: float, t: float, d: float, interval: float) -> RampConfig:
    return RampConfig(
        ramp_type=RampType.LINEAR,
        start=s,
        target=t,
        duration_sec=d,
        update_interval=interval,
    )


class TestRamper:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cfg,expected",
        [  # fmt: skip
            (lin(10, 10, 1.0), [10]),
            (lin(1, 5, 0.1), [1, 2, 3, 4, 5]),
            (lin(5, 1, 0.1), [5, 4, 3, 2, 1]),
            (lin(50, 50, 1.0), [50]),
            (lin(1, 100, 0.1, step=25), [1, 26, 51, 76, 100]),
        ],
    )
    async def test_linear_sequences(self, time_traveler, cfg, expected):
        vals: list[float] = []
        await Ramper(setter=vals.append, config=cfg).start()
        assert vals == expected

    @pytest.mark.asyncio
    async def test_exponential(self, time_traveler):
        vals: list[float] = []
        await Ramper(setter=vals.append, config=exp(1, 100, 1.0)).start()
        assert vals[0] == 1 and vals[-1] == 100 and len(vals) == 100
        for i, v in enumerate(vals):
            assert v == i + 1

    @pytest.mark.asyncio
    async def test_large_ramp(self, time_traveler):
        cnt = 0

        def counter(v: float) -> None:
            nonlocal cnt
            cnt += 1

        await Ramper(setter=counter, config=lin(1, 1000, 0.1, step=100)).start()
        assert cnt == 11

    @pytest.mark.asyncio
    async def test_very_short_duration(self, time_traveler):
        vals: list[float] = []
        await Ramper(setter=vals.append, config=lin(1, 5, 0.001)).start()
        assert vals == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_setter_exception(self, time_traveler):
        def fail(v: float) -> None:
            if v > 2:
                raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await Ramper(setter=fail, config=lin(1, 5, 0.1)).start()

    @pytest.mark.asyncio
    async def test_restart_with_new_ramper(self, time_traveler):
        vals: list[float] = []
        await Ramper(setter=vals.append, config=lin(1, 3, 0.05)).start()
        assert vals == [1, 2, 3]
        vals.clear()
        await Ramper(setter=vals.append, config=lin(10, 12, 0.05)).start()
        assert vals == [10, 11, 12]


class TestRamperStop:
    @pytest.mark.asyncio
    async def test_stop_stays_at_current(self, time_traveler):
        vals: list[float] = []
        r = Ramper(setter=vals.append, config=lin(1, 100, 10.0))
        task = r.start()
        await time_traveler.sleep(0.01)
        assert r.is_running
        r.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert vals[-1] != 100 and vals[-1] >= 1

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, time_traveler):
        r = Ramper(setter=MagicMock(), config=lin(1, 10, 0.1))
        await r.start()
        r.stop()
        r.stop()
        r.stop()

    @pytest.mark.asyncio
    async def test_stop_before_start(self):
        r = Ramper(setter=MagicMock(), config=lin(1, 10, 0.1))
        r.stop()


class TestRamperIsRunning:
    @pytest.mark.asyncio
    async def test_not_running_before_start(self):
        assert not Ramper(setter=MagicMock(), config=lin(1, 10, 0.1)).is_running

    @pytest.mark.asyncio
    async def test_running_during_ramp(self, time_traveler):
        r = Ramper(setter=MagicMock(), config=lin(1, 100, 10.0))
        task = r.start()
        await time_traveler.sleep(0.01)
        assert r.is_running
        r.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_not_running_after_completion(self, time_traveler):
        r = Ramper(setter=MagicMock(), config=lin(1, 5, 0.05))
        await r.start()
        assert not r.is_running


class TestRamperContinuous:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "s,t,first,last",
        [  # fmt: skip
            (10, 100, 10.0, 100.0),
            (1, 100, 1.0, 100.0),
            (1.5, 5.5, 1.5, 5.5),
            (100, 1, 100.0, 1.0),
        ],
    )
    async def test_start_and_target(self, time_traveler, s, t, first, last):
        vals: list[float] = []
        await Ramper(setter=vals.append, config=cont(s, t, 1.0, 0.2)).start()
        assert vals[0] == first and vals[-1] == last

    @pytest.mark.asyncio
    async def test_interpolation(self, time_traveler):
        vals: list[float] = []
        await Ramper(setter=vals.append, config=cont(1, 100, 10.0, 2.0)).start()
        assert vals[0] == 1.0 and vals[-1] == 100.0 and len(vals) >= 5

    @pytest.mark.asyncio
    async def test_update_interval_frequency(self, time_traveler):
        vals: list[float] = []
        await Ramper(setter=vals.append, config=cont(1, 10, 0.5, 0.1)).start()
        assert len(vals) >= 5

    @pytest.mark.asyncio
    async def test_stop_stays_at_current(self, time_traveler):
        vals: list[float] = []
        r = Ramper(setter=vals.append, config=cont(1, 100, 10.0, 0.5))
        task = r.start()
        await time_traveler.sleep(0.01)
        assert r.is_running
        r.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert vals[-1] != 100.0 and vals[-1] >= 1.0

    @pytest.mark.asyncio
    async def test_float_bounds(self, time_traveler):
        vals: list[float] = []
        await Ramper(setter=vals.append, config=cont(1.5, 5.5, 1.0, 0.2)).start()
        for v in vals:
            assert 1.5 <= v <= 5.5

    @pytest.mark.asyncio
    async def test_ramp_down_decreasing(self, time_traveler):
        vals: list[float] = []
        await Ramper(setter=vals.append, config=cont(100, 1, 1.0, 0.2)).start()
        for i in range(1, len(vals)):
            assert vals[i] <= vals[i - 1]
