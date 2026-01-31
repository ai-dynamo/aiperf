# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import ValidationError

from aiperf.plugin import plugins
from aiperf.plugin.enums import PluginType, RampType
from aiperf.timing.ramping import (
    BaseRampStrategy as RampStrategy,
)
from aiperf.timing.ramping import (
    ExponentialStrategy,
    LinearStrategy,
    PoissonStrategy,
    RampConfig,
)


def cfg(t: RampType, s: float, tg: float, d: float, **kw) -> RampConfig:
    return RampConfig(ramp_type=t, start=s, target=tg, duration_sec=d, **kw)


def lin(s: float, t: float, d: float, step: float | None = None) -> RampConfig:
    return cfg(RampType.LINEAR, s, t, d, step_size=step)


def exp(s: float, t: float, d: float, e: float = 2.0) -> RampConfig:
    return cfg(RampType.EXPONENTIAL, s, t, d, exponent=e)


def poi(s: float, t: float, d: float) -> RampConfig:
    return cfg(RampType.POISSON, s, t, d)


class TestLinearStrategy:
    @pytest.mark.parametrize(
        "start,target,current,exp_next",
        [(1, 100, 100, None), (1, 100, 1, 2), (100, 1, 100, 99), (50, 50, 50, None)],
    )
    def test_next_step(
        self, start: int, target: int, current: int, exp_next: int | None
    ) -> None:
        s = LinearStrategy(lin(start, target, 10.0))
        assert isinstance(s, RampStrategy)
        r = s.next_step(current, elapsed_sec=0.0)
        if exp_next is None:
            assert r is None
        else:
            assert r is not None and r[1] == exp_next

    def test_start_target_properties(self) -> None:
        s = LinearStrategy(lin(5, 50, 10.0))
        assert s.start == 5 and s.target == 50

    @pytest.mark.parametrize(
        "start,target,dur,exp_int",
        [(1, 100, 9.9, 9.9 / 99), (100, 1, 9.9, 9.9 / 99), (1, 500, 1.0, 1.0 / 499)],
    )
    def test_interval(
        self, start: int, target: int, dur: float, exp_int: float
    ) -> None:
        r = LinearStrategy(lin(start, target, dur)).next_step(start, elapsed_sec=0.0)
        assert r is not None and abs(r[0] - exp_int) < 1e-6

    def test_timing_self_corrects(self) -> None:
        s = LinearStrategy(lin(1, 100, 10.0))
        r1 = s.next_step(1, elapsed_sec=0.0)
        assert r1 is not None and abs(r1[0] - 10.0 / 99) < 1e-4
        r2 = s.next_step(50, elapsed_sec=5.0)
        assert r2 is not None and abs(r2[0] - (10.0 * 50 / 99 - 5.0)) < 1e-4

    def test_small_ramp(self) -> None:
        r = LinearStrategy(lin(1, 2, 1.0)).next_step(1, elapsed_sec=0.0)
        assert r == (1.0, 2)

    def test_full_ramp(self) -> None:
        s, cur, vals = LinearStrategy(lin(1, 10, 9.0)), 1, [1]
        while (r := s.next_step(cur, elapsed_sec=0.0)) is not None:
            cur = r[1]
            vals.append(cur)
        assert vals == list(range(1, 11))

    @pytest.mark.parametrize(
        "start,target,cur,exp_next",
        [(1, 100, 1, 11), (100, 1, 100, 90), (1, 100, 95, 100), (100, 1, 5, 1)],
    )
    def test_step_size(self, start: int, target: int, cur: int, exp_next: int) -> None:
        r = LinearStrategy(lin(start, target, 10.0, step=10)).next_step(
            cur, elapsed_sec=0.0
        )
        assert r is not None and r[1] == exp_next

    def test_step_size_timing(self) -> None:
        r = LinearStrategy(lin(1, 100, 10.0, step=10)).next_step(1, elapsed_sec=0.0)
        assert r is not None and r[1] == 11 and abs(r[0] - 10.0 * 10 / 99) < 1e-4

    def test_step_size_self_corrects(self) -> None:
        s = LinearStrategy(lin(1, 100, 10.0, step=10))
        r1 = s.next_step(1, elapsed_sec=0.0)
        assert r1 is not None and abs(r1[0] - 10.0 * 10 / 99) < 1e-4
        r2 = s.next_step(51, elapsed_sec=5.0)
        assert (
            r2 is not None
            and r2[1] == 61
            and abs(r2[0] - (10.0 * 60 / 99 - 5.0)) < 1e-4
        )

    def test_step_size_full_ramp(self) -> None:
        s, cur, vals = LinearStrategy(lin(1, 100, 4.0, step=25)), 1, [1]
        while (r := s.next_step(cur, elapsed_sec=0.0)) is not None:
            cur = r[1]
            vals.append(cur)
        assert vals == [1, 26, 51, 76, 100]


class TestExponentialStrategy:
    @pytest.mark.parametrize("e", [1.0, 0.5])
    def test_invalid_exponent(self, e: float) -> None:
        with pytest.raises(ValidationError, match="greater than 1"):
            exp(1, 100, 1.0, e)

    @pytest.mark.parametrize("cur,none", [(100, True), (150, True), (1, False)])
    def test_at_or_above_target(self, cur: int, none: bool) -> None:
        s = ExponentialStrategy(exp(1, 100, 1.0))
        assert isinstance(s, RampStrategy)
        assert (s.next_step(cur, elapsed_sec=0.0) is None) == none

    def test_increments_by_one(self) -> None:
        s = ExponentialStrategy(exp(1, 100, 1.0))
        r1 = s.next_step(1, elapsed_sec=0.0)
        assert r1 is not None and r1[1] == 2
        r2 = s.next_step(50, elapsed_sec=0.5)
        assert r2 is not None and r2[1] == 51

    def test_delays_decrease_with_ease_in(self) -> None:
        """Test exponential ease-in: first delay longest, delays decrease, last shortest."""
        s, delays, cur, elapsed = ExponentialStrategy(exp(1, 100, 1.0)), [], 1, 0.0
        for _ in range(99):
            r = s.next_step(cur, elapsed_sec=elapsed)
            if r is None:
                break
            delays.append(r[0])
            elapsed += r[0]
            cur = r[1]
        # First delay is longest (ease-in starts slow)
        assert delays[0] > 0.09
        # Delays monotonically decrease
        for i in range(1, len(delays)):
            assert delays[i] <= delays[i - 1] + 0.001
        # Last delay is shortest
        assert delays[-1] < 0.02

    def test_higher_exponent_slower_start(self) -> None:
        r_low = ExponentialStrategy(exp(1, 100, 1.0, 2.0)).next_step(1, elapsed_sec=0.0)
        r_high = ExponentialStrategy(exp(1, 100, 1.0, 3.0)).next_step(
            1, elapsed_sec=0.0
        )
        assert r_low is not None and r_high is not None and r_high[0] > r_low[0]

    def test_full_ramp(self) -> None:
        s, cur, elapsed, vals = ExponentialStrategy(exp(1, 100, 1.0)), 1, 0.0, [1]
        while cur < 100:
            r = s.next_step(cur, elapsed_sec=elapsed)
            if r is None:
                break
            elapsed += r[0]
            cur = r[1]
            vals.append(cur)
        assert vals == list(range(1, 101)) and abs(elapsed - 1.0) < 0.01

    def test_ramp_down(self) -> None:
        """Test exponential ramp down: all values, timing, and decreasing delays."""
        s, cur, elapsed, vals, delays = (
            ExponentialStrategy(exp(100, 1, 1.0)),
            100,
            0.0,
            [100],
            [],
        )
        while cur > 1:
            r = s.next_step(cur, elapsed_sec=elapsed)
            if r is None:
                break
            delays.append(r[0])
            elapsed += r[0]
            cur = r[1]
            vals.append(cur)
        # All values from 100 down to 1
        assert vals == list(range(100, 0, -1))
        # Total time matches duration
        assert abs(elapsed - 1.0) < 0.01
        # Delays decrease (ease-in applies to ramp down too)
        for i in range(1, len(delays)):
            assert delays[i] <= delays[i - 1] + 0.001

    def test_returns_none_below_target_down(self) -> None:
        assert (
            ExponentialStrategy(exp(100, 1, 1.0)).next_step(0, elapsed_sec=0.5) is None
        )


class TestPoissonStrategy:
    def test_properties(self) -> None:
        s = PoissonStrategy(poi(5, 50, 10.0))
        assert isinstance(s, RampStrategy) and s.start == 5 and s.target == 50

    def test_returns_none_when_complete(self) -> None:
        s = PoissonStrategy(poi(1, 3, 1.0))
        assert s.next_step(1, elapsed_sec=0.0) is not None
        assert s.next_step(2, elapsed_sec=0.5) is not None
        assert s.next_step(3, elapsed_sec=1.0) is None

    @pytest.mark.parametrize(
        "start,target,cur,check",
        [(1, 100, 1, lambda v: 1 < v <= 100), (100, 1, 100, lambda v: 1 <= v < 100)],
    )
    def test_ramp_direction(
        self, start: int, target: int, cur: int, check: callable
    ) -> None:
        r = PoissonStrategy(poi(start, target, 10.0)).next_step(cur, elapsed_sec=0.0)
        assert r is not None and check(r[1])

    def test_full_ramp(self) -> None:
        s, cur, vals = PoissonStrategy(poi(1, 10, 9.0)), 1, [1]
        while (r := s.next_step(cur, elapsed_sec=0.0)) is not None:
            cur = r[1]
            vals.append(cur)
        for i in range(1, len(vals)):
            assert vals[i] >= vals[i - 1]
        assert vals[-1] == 10

    def test_total_time(self) -> None:
        s, cur, elapsed, total = PoissonStrategy(poi(1, 100, 10.0)), 1, 0.0, 0.0
        while cur < 100:
            r = s.next_step(cur, elapsed_sec=elapsed)
            if r is None:
                break
            total += r[0]
            elapsed += r[0]
            cur = r[1]
        assert abs(total - 10.0) < 0.001

    def test_variable_intervals(self) -> None:
        s, delays, cur, elapsed = PoissonStrategy(poi(1, 20, 10.0)), [], 1, 0.0
        for _ in range(10):
            r = s.next_step(cur, elapsed_sec=elapsed)
            if r is None:
                break
            delays.append(r[0])
            elapsed += r[0]
            cur = r[1]
        assert len(set(round(d, 6) for d in delays)) > 1

    def test_deterministic(self) -> None:
        t1 = PoissonStrategy(poi(1, 10, 5.0))._event_times
        t2 = PoissonStrategy(poi(1, 10, 5.0))._event_times
        assert len(t1) == len(t2) and all(
            abs(a - b) < 1e-10 for a, b in zip(t1, t2, strict=True)
        )

    def test_already_at_target(self) -> None:
        assert PoissonStrategy(poi(50, 50, 10.0)).next_step(50, elapsed_sec=0.0) is None

    @pytest.mark.parametrize(
        "start,target,mono,bound",
        [
            (1.0, 10.7, lambda a, b: a <= b, lambda v, t: v <= t),
            (10.7, 1.0, lambda a, b: a >= b, lambda v, t: v >= t),
        ],  # fmt: skip
    )
    def test_fractional(
        self, start: float, target: float, mono: callable, bound: callable
    ) -> None:
        s = PoissonStrategy(poi(start, target, 5.0))
        assert len(s._event_times) >= 1
        for i in range(1, len(s._values)):
            assert mono(s._values[i - 1], s._values[i])
        assert s._values[0] == start and bound(s._values[-1], target)


class TestValueAt:
    @pytest.mark.parametrize(
        "start,target,elapsed,exp",
        [(10, 100, 0.0, 10.0), (1, 101, 5.0, 51.0), (100, 1, 5.0, 50.5)],
    )
    def test_linear(self, start: int, target: int, elapsed: float, exp: float) -> None:
        v = LinearStrategy(lin(start, target, 10.0)).value_at(elapsed)
        assert v is not None and abs(v - exp) < 0.01

    @pytest.mark.parametrize("elapsed", [10.0, 15.0])
    def test_linear_none_at_completion(self, elapsed: float) -> None:
        assert LinearStrategy(lin(1, 100, 10.0)).value_at(elapsed) is None

    def test_exponential_ease_in_curve(self) -> None:
        """Test exponential value_at: slow early (below linear), accelerates later."""
        s = ExponentialStrategy(exp(1, 101, 10.0))
        # At 50% time, value is well below 50% progress due to ease-in
        v_mid = s.value_at(5.0)
        assert v_mid is not None and v_mid < 51.0 and abs(v_mid - 26.0) < 0.1
        # At 80% time, value has accelerated past midpoint
        v_late = s.value_at(8.0)
        assert v_late is not None and abs(v_late - 65.0) < 0.1

    def test_linear_step_size_interpolates(self) -> None:
        v = LinearStrategy(lin(1, 101, 10.0, step=25)).value_at(5.0)
        assert v is not None and abs(v - 51.0) < 0.01

    @pytest.mark.parametrize("elapsed", [0.0, 5.0])
    def test_none_for_zero_range(self, elapsed: float) -> None:
        assert LinearStrategy(lin(50, 50, 10.0)).value_at(elapsed) is None

    @pytest.mark.parametrize("elapsed", [0.001, 0.01])
    def test_very_small_duration(self, elapsed: float) -> None:
        assert LinearStrategy(lin(1, 100, 0.001)).value_at(elapsed) is None

    def test_higher_exponent_slower(self) -> None:
        v2 = ExponentialStrategy(exp(1, 101, 10.0, 2.0)).value_at(5.0)
        v3 = ExponentialStrategy(exp(1, 101, 10.0, 3.0)).value_at(5.0)
        assert v2 is not None and v3 is not None and v3 < v2


class TestPoissonValueAt:
    def test_start(self) -> None:
        v = PoissonStrategy(poi(10, 100, 10.0)).value_at(0.0)
        assert v is not None and v == 10.0

    def test_step_function(self) -> None:
        s = PoissonStrategy(poi(1, 10, 9.0))
        if s._event_times:
            t0 = s._event_times[0]
            if t0 - 0.001 > 0:
                assert s.value_at(t0 - 0.001) == s._values[0]
            if t0 + 0.001 < 9.0:
                assert s.value_at(t0 + 0.001) == s._values[1]

    def test_near_end(self) -> None:
        v = PoissonStrategy(poi(1, 100, 10.0)).value_at(9.999)
        assert v is not None and 1 <= v <= 100

    def test_ramp_down(self) -> None:
        s = PoissonStrategy(poi(100, 1, 10.0))
        assert s.value_at(0.0) == 100
        v = s.value_at(5.0)
        assert v is not None and v < 100

    def test_consistent_with_next_step(self) -> None:
        s = PoissonStrategy(poi(1, 10, 5.0))
        traj, elapsed, cur = [(0.0, 1.0)], 0.0, 1
        events = list(s._event_times)
        while (r := s.next_step(cur, elapsed_sec=elapsed)) is not None:
            elapsed += r[0]
            cur = r[1]
            traj.append((elapsed, float(cur)))
        s2 = PoissonStrategy(poi(1, 10, 5.0))
        for i, t in enumerate(events):
            v = s2.value_at(t + 0.0001)
            if v is not None:
                assert v == traj[i + 1][1]


class TestEdgeCasesAndFactory:
    @pytest.mark.parametrize(
        "strategy",
        [
            LinearStrategy(lin(1, 1_000_000, 100.0)),
            LinearStrategy(lin(1, 1_000_000, 100.0, step=10)),
            ExponentialStrategy(exp(1, 1_000_000, 100.0)),
            PoissonStrategy(poi(1, 1_000, 100.0)),
        ],
    )
    def test_large_values(self, strategy: RampStrategy) -> None:
        r = strategy.next_step(1, elapsed_sec=0.0)
        assert r is not None and r[1] > 1 and r[0] > 0

    @pytest.mark.parametrize("strategy", [LinearStrategy(lin(1, 100, 0.001)), LinearStrategy(lin(1, 100, 0.001, step=10))])  # fmt: skip
    def test_small_duration(self, strategy: RampStrategy) -> None:
        r = strategy.next_step(1, elapsed_sec=0.0)
        assert r is not None and r[0] <= 0.001 and r[1] > 1

    def test_poisson_small_duration(self) -> None:
        s = PoissonStrategy(poi(1, 100, 0.001))
        r = s.next_step(1, elapsed_sec=0.0)
        if r is None:
            assert len(s._event_times) == 0
        else:
            assert r[0] >= 0

    @pytest.mark.parametrize(
        "config,cls",
        [
            (lin(1, 100, 10.0), LinearStrategy),
            (lin(1, 100, 10.0, step=10), LinearStrategy),
            (exp(1, 100, 10.0), ExponentialStrategy),
            (poi(1, 100, 10.0), PoissonStrategy),
        ],
    )
    def test_plugin(self, config: RampConfig, cls: type) -> None:
        RampClass = plugins.get_class(PluginType.RAMP, config.ramp_type)
        s = RampClass(config=config)
        assert isinstance(s, cls)
        if config.ramp_type != RampType.EXPONENTIAL:
            assert s.start == 1 and s.target == 100
