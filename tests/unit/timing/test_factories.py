# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from aiperf.plugin import plugins
from aiperf.plugin.enums import ArrivalPattern, PluginType, RampType, TimingMode
from aiperf.timing.intervals import IntervalGeneratorConfig
from aiperf.timing.ramping import RampConfig
from aiperf.timing.strategies.core import TimingStrategyProtocol
from tests.unit.timing.conftest import make_phase_config


@dataclass
class MockConvSource:
    convs: list[Any] = field(default_factory=list)

    def next_conversation(self):
        return None if not self.convs else self.convs.pop(0)


@dataclass
class MockStopChk:
    can_send: bool = True
    can_start: bool = True

    def can_send_any_turn(self) -> bool:
        return self.can_send

    def can_start_new_session(self) -> bool:
        return self.can_start


@dataclass
class MockCredIssuer:
    issued: list = field(default_factory=list)

    async def issue_credit(self, **kw) -> None:
        self.issued.append(kw)


@dataclass
class MockLifecycle:
    is_complete: bool = False
    is_sending_complete: bool = False
    started_at_perf_ns: int = 0

    def start(self) -> None:
        pass

    def mark_sending_complete(self) -> None:
        self.is_sending_complete = True

    def mark_complete(self) -> None:
        self.is_complete = True

    def cancel(self) -> None:
        pass


@dataclass
class MockSched:
    tasks: list = field(default_factory=list)

    def schedule_later(self, delay: float, coro) -> None:
        self.tasks.append((delay, coro))

    def cancel_all(self) -> None:
        self.tasks.clear()


@pytest.fixture
def ts_deps():
    return {
        "conversation_source": MockConvSource(),
        "scheduler": MockSched(),
        "stop_checker": MockStopChk(),
        "credit_issuer": MockCredIssuer(),
        "lifecycle": MockLifecycle(),
    }


def mk_int_cfg(pattern, rate=10.0, smooth=None):
    return IntervalGeneratorConfig(
        arrival_pattern=pattern, request_rate=rate, arrival_smoothness=smooth
    )


def mk_ramp_cfg(rtype, start=1.0, target=10.0, dur=5.0, exp=None, step=None):
    return RampConfig(
        ramp_type=rtype,
        start=start,
        target=target,
        duration_sec=dur,
        exponent=exp,
        step_size=step,
    )


def create_interval_generator(cfg: IntervalGeneratorConfig):
    """Helper to create interval generator via plugin system."""
    GeneratorClass = plugins.get_class(PluginType.ARRIVAL_PATTERN, cfg.arrival_pattern)
    return GeneratorClass(config=cfg)


def create_ramp_strategy(cfg: RampConfig):
    """Helper to create ramp strategy via plugin system."""
    RampClass = plugins.get_class(PluginType.RAMP, cfg.ramp_type)
    return RampClass(config=cfg)


def create_timing_strategy(timing_mode: TimingMode, config, **deps):
    """Helper to create timing strategy via plugin system."""
    StrategyClass = plugins.get_class(PluginType.TIMING_STRATEGY, timing_mode)
    return StrategyClass(config=config, **deps)


class TestIntervalGeneratorPlugin:
    @pytest.mark.parametrize("rate,exp_int", [(10.0, 0.1), (100.0, 0.01), (1.0, 1.0)])
    def test_constant_interval(self, rate, exp_int):
        g = create_interval_generator(mk_int_cfg(ArrivalPattern.CONSTANT, rate))
        assert g.next_interval() == pytest.approx(exp_int)
        assert g.rate == rate

    def test_poisson_varies(self):
        g = create_interval_generator(mk_int_cfg(ArrivalPattern.POISSON))
        ints = [g.next_interval() for _ in range(10)]
        assert g.rate == 10.0
        assert all(i > 0 for i in ints)
        assert len(set(ints)) > 1

    def test_gamma_smoothness(self):
        g = create_interval_generator(mk_int_cfg(ArrivalPattern.GAMMA, smooth=2.0))
        ints = [g.next_interval() for _ in range(10)]
        assert g.rate == 10.0 and all(i > 0 for i in ints)
        assert hasattr(g, "smoothness") and g.smoothness == 2.0

    def test_burst_zero(self):
        g = create_interval_generator(
            mk_int_cfg(ArrivalPattern.CONCURRENCY_BURST, rate=None)
        )
        assert g.next_interval() == 0 and g.rate == 0.0

    @pytest.mark.parametrize(
        "pat",
        [ArrivalPattern.CONSTANT, ArrivalPattern.POISSON, ArrivalPattern.GAMMA],
    )
    def test_rate_required(self, pat):
        with pytest.raises(ValueError):
            create_interval_generator(mk_int_cfg(pat, rate=None))

    @pytest.mark.parametrize(
        "pat",
        [ArrivalPattern.CONSTANT, ArrivalPattern.POISSON, ArrivalPattern.GAMMA],
    )
    @pytest.mark.parametrize("bad_rate", [0.0, -1.0])
    def test_rate_positive(self, pat, bad_rate):
        with pytest.raises(ValueError):
            create_interval_generator(mk_int_cfg(pat, rate=bad_rate))

    def test_constant_rate_update(self):
        g = create_interval_generator(mk_int_cfg(ArrivalPattern.CONSTANT))
        assert g.next_interval() == pytest.approx(0.1)
        g.set_rate(20.0)
        assert g.rate == 20.0 and g.next_interval() == pytest.approx(0.05)

    def test_poisson_rate_update(self):
        g = create_interval_generator(mk_int_cfg(ArrivalPattern.POISSON))
        g.set_rate(100.0)
        assert g.rate == 100.0
        avg = sum(g.next_interval() for _ in range(100)) / 100
        assert avg < 0.1

    def test_gamma_rate_update(self):
        g = create_interval_generator(mk_int_cfg(ArrivalPattern.GAMMA, smooth=2.0))
        g.set_rate(50.0)
        assert g.rate == 50.0

    def test_burst_ignores_set_rate(self):
        g = create_interval_generator(
            mk_int_cfg(ArrivalPattern.CONCURRENCY_BURST, rate=None)
        )
        g.set_rate(100.0)
        assert g.next_interval() == 0 and g.rate == 0.0

    @pytest.mark.parametrize(
        "pat",
        [ArrivalPattern.CONSTANT, ArrivalPattern.POISSON, ArrivalPattern.GAMMA],
    )
    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_set_rate_validates(self, pat, bad):
        g = create_interval_generator(
            mk_int_cfg(pat, smooth=2.0 if pat == ArrivalPattern.GAMMA else None)
        )
        with pytest.raises(ValueError, match="must be > 0"):
            g.set_rate(bad)


class TestRampStrategyPlugin:
    def test_creates_linear(self):
        s = create_ramp_strategy(mk_ramp_cfg(RampType.LINEAR))
        assert s.start == 1.0 and s.target == 10.0

    def test_creates_exponential(self):
        s = create_ramp_strategy(mk_ramp_cfg(RampType.EXPONENTIAL, exp=2.0))
        assert s.start == 1.0 and s.target == 10.0

    @pytest.mark.parametrize("start,target,dir", [(1.0, 10.0, "inc"), (10.0, 1.0, "dec"), (5.0, 5.0, "const")])  # fmt: skip
    def test_linear_directions(self, start, target, dir):
        s = create_ramp_strategy(
            mk_ramp_cfg(RampType.LINEAR, start=start, target=target)
        )
        assert s.start == start and s.target == target
        r = s.next_step(start, 0.0)
        if dir == "const":
            assert r is None
        else:
            assert r is not None
            d, nv = r
            assert d >= 0
            assert (nv > start) if dir == "inc" else (nv < start)

    def test_linear_discrete_step(self):
        s = create_ramp_strategy(mk_ramp_cfg(RampType.LINEAR))
        r = s.next_step(1.0, 0.0)
        assert r is not None
        d, nv = r
        assert nv == 2.0 and d >= 0

    def test_linear_custom_step(self):
        s = create_ramp_strategy(mk_ramp_cfg(RampType.LINEAR, step=3.0))
        r = s.next_step(1.0, 0.0)
        assert r is not None and r[1] == 4.0

    def test_linear_value_at(self):
        s = create_ramp_strategy(mk_ramp_cfg(RampType.LINEAR))
        assert s.value_at(0.0) == pytest.approx(1.0)
        assert s.value_at(2.5) == pytest.approx(5.5)
        assert s.value_at(5.0) is None

    def test_exponential_ease_in(self):
        s = create_ramp_strategy(mk_ramp_cfg(RampType.EXPONENTIAL, exp=2.0))
        assert s.value_at(2.5) == pytest.approx(3.25)

    @pytest.mark.parametrize("bad_exp", [1.0, 0.5, 0.0, -1.0])
    def test_exponential_validates_exponent(self, bad_exp):
        with pytest.raises(ValidationError, match="greater than 1"):
            mk_ramp_cfg(RampType.EXPONENTIAL, exp=bad_exp)

    def test_ramp_completion(self):
        s = create_ramp_strategy(mk_ramp_cfg(RampType.LINEAR))
        assert s.next_step(10.0, 0.0) is None
        assert s.value_at(10.0) is None


class TestTimingStrategyPlugin:
    @pytest.mark.parametrize(
        "mode,extra",
        [
            (
                TimingMode.REQUEST_RATE,
                {"request_rate": 10.0, "arrival_pattern": ArrivalPattern.POISSON},
            ),
            (TimingMode.FIXED_SCHEDULE, {}),
            (TimingMode.USER_CENTRIC_RATE, {"request_rate": 10.0, "num_users": 5}),
        ],  # fmt: skip
    )
    def test_creates_strategy(self, mode, extra, ts_deps):
        cfg = make_phase_config(timing_mode=mode, request_count=100, **extra)
        s = create_timing_strategy(timing_mode=mode, config=cfg, **ts_deps)
        assert isinstance(s, TimingStrategyProtocol)

    def test_missing_deps_error(self):
        cfg = make_phase_config(
            timing_mode=TimingMode.REQUEST_RATE,
            request_rate=10.0,
            arrival_pattern=ArrivalPattern.POISSON,
        )
        with pytest.raises(TypeError):
            create_timing_strategy(timing_mode=TimingMode.REQUEST_RATE, config=cfg)

    def test_unregistered_type_error(self, ts_deps):
        from aiperf.plugin.types import TypeNotFoundError

        cfg = make_phase_config(timing_mode=TimingMode.REQUEST_RATE)
        with pytest.raises((KeyError, ValueError, TypeNotFoundError)):
            create_timing_strategy(
                timing_mode="not_real",  # type: ignore
                config=cfg,
                **ts_deps,
            )


class TestPluginRegistry:
    def test_interval_all_patterns(self):
        entries = list(plugins.iter_entries(PluginType.ARRIVAL_PATTERN))
        entry_names = [e.name for e in entries]
        for p in ArrivalPattern:
            assert p in entry_names

    def test_ramp_all_types(self):
        entries = list(plugins.iter_entries(PluginType.RAMP))
        entry_names = [e.name for e in entries]
        for t in RampType:
            assert t in entry_names

    def test_timing_all_modes(self):
        entries = list(plugins.iter_entries(PluginType.TIMING_STRATEGY))
        entry_names = [e.name for e in entries]
        for m in TimingMode:
            assert m in entry_names


class TestPluginIntegration:
    @pytest.mark.parametrize(
        "pat", [ArrivalPattern.CONSTANT, ArrivalPattern.POISSON, ArrivalPattern.GAMMA]
    )
    def test_interval_gen_valid_intervals(self, pat):
        cfg = mk_int_cfg(
            pat, rate=10.0, smooth=1.0 if pat == ArrivalPattern.GAMMA else None
        )
        g = create_interval_generator(cfg)
        total = sum(g.next_interval() for _ in range(100))
        avg = total / 100
        assert 0.02 <= avg <= 0.8

    @pytest.mark.parametrize("rtype", [RampType.LINEAR, RampType.EXPONENTIAL])
    def test_ramp_reaches_target(self, rtype):
        cfg = mk_ramp_cfg(rtype, exp=2.0 if rtype == RampType.EXPONENTIAL else None)
        s = create_ramp_strategy(cfg)
        cur, elapsed, steps = s.start, 0.0, 0
        while steps < 1000:
            r = s.next_step(cur, elapsed)
            if r is None:
                break
            elapsed += r[0]
            cur = r[1]
            steps += 1
        assert cur == s.target

    def test_gamma_smoothness_variance(self):
        bursty = create_interval_generator(
            mk_int_cfg(ArrivalPattern.GAMMA, smooth=0.25)
        )
        smooth = create_interval_generator(
            mk_int_cfg(ArrivalPattern.GAMMA, smooth=10.0)
        )
        bi = [bursty.next_interval() for _ in range(2000)]
        si = [smooth.next_interval() for _ in range(2000)]

        def var(d):
            m = sum(d) / len(d)
            return sum((x - m) ** 2 for x in d) / len(d)

        assert var(bi) > var(si)
