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
    async def test_exponential_produces_all_steps(self, time_traveler):
        """Exponential strategy steps by 1, just with non-linear timing."""
        vals: list[float] = []
        await Ramper(setter=vals.append, config=exp(1, 100, 1.0)).start()
        assert vals == list(range(1, 101))

    @pytest.mark.asyncio
    async def test_step_size_controls_increments(self, time_traveler):
        """Custom step_size produces fewer calls with larger jumps."""
        vals: list[float] = []
        await Ramper(setter=vals.append, config=lin(1, 1000, 0.1, step=100)).start()
        assert vals == [1, 101, 201, 301, 401, 501, 601, 701, 801, 901, 1000]

    @pytest.mark.asyncio
    async def test_setter_exception_propagates(self, time_traveler):
        def fail(v: float) -> None:
            if v > 2:
                raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await Ramper(setter=fail, config=lin(1, 5, 0.1)).start()


class TestRamperStop:
    @pytest.mark.asyncio
    async def test_stop_cancels_ramp_mid_progress(self, time_traveler):
        """Stopping mid-ramp keeps partial progress, does not reach target."""
        vals: list[float] = []
        r = Ramper(setter=vals.append, config=lin(1, 100, 10.0))
        task = r.start()
        await time_traveler.sleep(0.01)
        r.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert 1 <= vals[-1] < 100

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, time_traveler):
        """Multiple stop() calls are safe (no-op after first)."""
        r = Ramper(setter=MagicMock(), config=lin(1, 10, 0.1))
        await r.start()
        r.stop()
        r.stop()
        r.stop()

    @pytest.mark.asyncio
    async def test_stop_before_start_is_safe(self):
        """Calling stop() before start() does not raise."""
        r = Ramper(setter=MagicMock(), config=lin(1, 10, 0.1))
        r.stop()


class TestRamperIsRunning:
    @pytest.mark.asyncio
    async def test_is_running_lifecycle(self, time_traveler):
        """is_running reflects task state: False->True->False."""
        r = Ramper(setter=MagicMock(), config=lin(1, 100, 10.0))
        assert not r.is_running

        task = r.start()
        await time_traveler.sleep(0.01)
        assert r.is_running

        r.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert not r.is_running

    @pytest.mark.asyncio
    async def test_is_running_false_after_completion(self, time_traveler):
        """is_running is False after ramp completes naturally."""
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
    async def test_reaches_start_and_target(self, time_traveler, s, t, first, last):
        """Continuous mode starts at start value and ends at target."""
        vals: list[float] = []
        await Ramper(setter=vals.append, config=cont(s, t, 1.0, 0.2)).start()
        assert vals[0] == first and vals[-1] == last

    @pytest.mark.asyncio
    async def test_produces_intermediate_values(self, time_traveler):
        """Continuous mode calls setter multiple times with interpolated values."""
        vals: list[float] = []
        await Ramper(setter=vals.append, config=cont(1, 100, 10.0, 2.0)).start()
        assert vals[0] == 1.0 and vals[-1] == 100.0
        assert len(vals) >= 5
        # Values should be monotonically increasing
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]

    @pytest.mark.asyncio
    async def test_stop_keeps_partial_progress(self, time_traveler):
        """Stopping continuous ramp mid-way keeps current value."""
        vals: list[float] = []
        r = Ramper(setter=vals.append, config=cont(1, 100, 10.0, 0.5))
        task = r.start()
        await time_traveler.sleep(0.01)
        r.stop()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert 1.0 <= vals[-1] < 100.0

    @pytest.mark.asyncio
    async def test_values_stay_within_bounds(self, time_traveler):
        """All intermediate values stay within [start, target] range."""
        vals: list[float] = []
        await Ramper(setter=vals.append, config=cont(1.5, 5.5, 1.0, 0.2)).start()
        for v in vals:
            assert 1.5 <= v <= 5.5
