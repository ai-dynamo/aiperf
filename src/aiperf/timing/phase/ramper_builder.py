# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build concurrency and rate rampers for a credit phase."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from aiperf.common.environment import Environment
from aiperf.timing.ramping import Ramper, RampType, TimingRampConfig
from aiperf.timing.strategies.core import RateSettableProtocol

if TYPE_CHECKING:
    from aiperf.timing.concurrency import ConcurrencyManager
    from aiperf.timing.config import CreditPhaseConfig
    from aiperf.timing.strategies.core import TimingStrategyProtocol


def build_rampers(
    *,
    config: CreditPhaseConfig,
    strategy: TimingStrategyProtocol,
    concurrency_manager: ConcurrencyManager,
    info: Callable[[str], None],
    warning: Callable[[str], None],
) -> list[Ramper]:
    """Create rampers for concurrency and rate if ramp durations are configured.

    Concurrency rampers use stepped mode (discrete integer steps), starting at 1.
    Rate rampers use continuous mode (smooth float interpolation), starting at a
    rate proportional to target (to avoid issues when target < 1 QPS).
    """
    rampers: list[Ramper] = []
    session = _build_session_concurrency_ramper(config, concurrency_manager, info)
    if session is not None:
        rampers.append(session)
    prefill = _build_prefill_concurrency_ramper(config, concurrency_manager, info)
    if prefill is not None:
        rampers.append(prefill)
    rate = _build_rate_ramper(config, strategy, info, warning)
    if rate is not None:
        rampers.append(rate)
    return rampers


def _build_session_concurrency_ramper(
    config: CreditPhaseConfig,
    concurrency_manager: ConcurrencyManager,
    info: Callable[[str], None],
) -> Ramper | None:
    if not (config.concurrency_ramp_duration_sec and config.concurrency):
        return None
    info(
        f"Starting session concurrency ramp: 1 → {config.concurrency} "
        f"over {config.concurrency_ramp_duration_sec}s"
    )
    ramp_config = TimingRampConfig(
        ramp_type=RampType.LINEAR,
        start=1,
        target=config.concurrency,
        duration_sec=config.concurrency_ramp_duration_sec,
    )

    def setter(limit: float) -> None:
        return concurrency_manager.set_session_limit(config.phase, int(limit))

    return Ramper(setter=setter, config=ramp_config)


def _build_prefill_concurrency_ramper(
    config: CreditPhaseConfig,
    concurrency_manager: ConcurrencyManager,
    info: Callable[[str], None],
) -> Ramper | None:
    if not (
        config.prefill_concurrency_ramp_duration_sec and config.prefill_concurrency
    ):
        return None
    info(
        f"Starting prefill concurrency ramp: 1 → {config.prefill_concurrency} "
        f"over {config.prefill_concurrency_ramp_duration_sec}s"
    )
    ramp_config = TimingRampConfig(
        ramp_type=RampType.LINEAR,
        start=1,
        target=config.prefill_concurrency,
        duration_sec=config.prefill_concurrency_ramp_duration_sec,
    )

    def setter(limit: float) -> None:
        return concurrency_manager.set_prefill_limit(config.phase, int(limit))

    return Ramper(setter=setter, config=ramp_config)


def _build_rate_ramper(
    config: CreditPhaseConfig,
    strategy: TimingStrategyProtocol,
    info: Callable[[str], None],
    warning: Callable[[str], None],
) -> Ramper | None:
    if not (config.request_rate_ramp_duration_sec and config.request_rate):
        return None
    # Start at one linear increment (proportional to target, not fixed 1 QPS).
    # This avoids awkward cases where target < 1 QPS would actually increase.
    update_interval = Environment.TIMING.RATE_RAMP_UPDATE_INTERVAL
    start_rate = config.request_rate * (
        update_interval / config.request_rate_ramp_duration_sec
    )
    info(
        f"Starting request rate ramp: {start_rate:.2f} → {config.request_rate} QPS "
        f"over {config.request_rate_ramp_duration_sec}s"
    )
    ramp_config = TimingRampConfig(
        ramp_type=RampType.LINEAR,
        start=start_rate,
        target=config.request_rate,
        duration_sec=config.request_rate_ramp_duration_sec,
        update_interval=update_interval,
    )
    if not isinstance(strategy, RateSettableProtocol):
        warning(
            f"Strategy {strategy.__class__.__name__} does not implement RateSettableProtocol. "
            "Request rate will be fixed at the target value."
        )
        return None
    return Ramper(setter=strategy.set_request_rate, config=ramp_config)
