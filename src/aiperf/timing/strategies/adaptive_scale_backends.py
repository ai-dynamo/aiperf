# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adaptive scale control backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from aiperf.plugin.enums import ArrivalPattern
from aiperf.timing.adaptive_config import AdaptiveControlVariable
from aiperf.timing.concurrency import PhaseRuntimeKey


class AdaptiveControlBackend(Protocol):
    """Uniform setter interface for one adaptive control variable."""

    variable: AdaptiveControlVariable
    minimum: float
    maximum: float

    @property
    def current(self) -> float: ...

    def set(self, value: float) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class _BaseControlBackend:
    variable: AdaptiveControlVariable
    minimum: float
    maximum: float
    _current: float

    @property
    def current(self) -> float:
        return self._current

    def _clamp(self, value: float) -> float:
        return max(self.minimum, min(value, self.maximum))

    def snapshot(self) -> dict[str, Any]:
        return {
            "target_value": self.current,
            "actual_value": self.current,
        }


class SessionConcurrencyControlBackend(_BaseControlBackend):
    def __init__(
        self, *, concurrency_manager, phase: PhaseRuntimeKey, minimum: int, maximum: int
    ):
        super().__init__("concurrency", minimum, maximum, minimum)
        self._concurrency_manager = concurrency_manager
        self._phase = phase

    def set(self, value: float) -> None:
        self._current = int(self._clamp(value))
        self._concurrency_manager.set_session_limit(self._phase, int(self._current))


class PrefillConcurrencyControlBackend(_BaseControlBackend):
    def __init__(
        self, *, concurrency_manager, phase: PhaseRuntimeKey, minimum: int, maximum: int
    ):
        super().__init__("prefill_concurrency", minimum, maximum, minimum)
        self._concurrency_manager = concurrency_manager
        self._phase = phase

    def set(self, value: float) -> None:
        self._current = int(self._clamp(value))
        self._concurrency_manager.set_prefill_limit(self._phase, int(self._current))


class RequestRateControlBackend(_BaseControlBackend):
    def __init__(self, *, rate_setter, minimum: float, maximum: float):
        super().__init__("request_rate", minimum, maximum, minimum)
        self._rate_setter = rate_setter

    def set(self, value: float) -> None:
        self._current = self._clamp(value)
        self._rate_setter(self._current)


class UsersControlBackend(_BaseControlBackend):
    """Target-user backend backed by optional strategy hooks."""

    def __init__(self, *, strategy: Any, minimum: int, maximum: int):
        super().__init__("users", minimum, maximum, minimum)
        self._strategy = strategy

    def set(self, value: float) -> None:
        self._current = int(self._clamp(value))
        setter = getattr(self._strategy, "set_target_users", None)
        if setter is None:
            raise ValueError("adaptive users requires a user-centric adaptive strategy")
        setter(int(self._current))

    def snapshot(self) -> dict[str, Any]:
        snapshotter = getattr(self._strategy, "user_control_snapshot", None)
        if snapshotter is None:
            return super().snapshot()
        data = dict(snapshotter())
        data.setdefault("target_value", self.current)
        return data


def _require_int(value: float | None, name: str) -> int:
    if value is None or value < 1 or int(value) != value:
        raise ValueError(f"{name} must be an integer >= 1, got {value!r}")
    return int(value)


def _explicit_or_configured(
    explicit: float | None, configured: float | None
) -> float | None:
    return explicit if explicit is not None else configured


def _validate_bounds(minimum: float, maximum: float, variable: str) -> None:
    if maximum <= minimum:
        raise ValueError(f"adaptive {variable} max must be > min")


def build_adaptive_control_backend(
    *, strategy: Any, concurrency_manager, config
) -> AdaptiveControlBackend:
    """Build the backend that mutates one adaptive control variable."""
    builders = {
        "concurrency": _build_concurrency_backend,
        "prefill_concurrency": _build_prefill_concurrency_backend,
        "request_rate": _build_request_rate_backend,
        "users": _build_users_backend,
    }
    try:
        builder = builders[config.adaptive_control_variable]
    except KeyError as exc:
        raise ValueError(
            f"unsupported adaptive control variable {config.adaptive_control_variable!r}"
        ) from exc
    return builder(
        strategy=strategy, concurrency_manager=concurrency_manager, config=config
    )


def _build_concurrency_backend(*, strategy: Any, concurrency_manager, config):
    max_value = _require_int(
        _explicit_or_configured(config.adaptive_control_max, config.concurrency),
        "adaptive concurrency max",
    )
    min_value = _require_int(config.adaptive_control_min, "adaptive concurrency min")
    _validate_bounds(min_value, max_value, "concurrency")
    return SessionConcurrencyControlBackend(
        concurrency_manager=concurrency_manager,
        phase=config.phase_index if config.phase_index is not None else config.phase,
        minimum=min_value,
        maximum=max_value,
    )


def _build_prefill_concurrency_backend(*, strategy: Any, concurrency_manager, config):
    if config.prefill_concurrency is None and config.adaptive_control_max is None:
        raise ValueError(
            "adaptive prefill_concurrency requires prefill_concurrency or control.max"
        )
    if config.concurrency is None:
        raise ValueError(
            "adaptive prefill_concurrency requires a session concurrency cap"
        )
    max_value = _require_int(
        _explicit_or_configured(
            config.adaptive_control_max,
            config.prefill_concurrency,
        ),
        "adaptive prefill_concurrency max",
    )
    if max_value > config.concurrency:
        raise ValueError("adaptive prefill_concurrency max must be <= concurrency")
    min_value = _require_int(
        config.adaptive_control_min,
        "adaptive prefill_concurrency min",
    )
    _validate_bounds(min_value, max_value, "prefill_concurrency")
    return PrefillConcurrencyControlBackend(
        concurrency_manager=concurrency_manager,
        phase=config.phase_index if config.phase_index is not None else config.phase,
        minimum=min_value,
        maximum=max_value,
    )


def _build_request_rate_backend(*, strategy: Any, concurrency_manager, config):
    if config.arrival_pattern == ArrivalPattern.CONCURRENCY_BURST:
        raise ValueError("adaptive request_rate requires a rate-controlled phase")
    max_value = _explicit_or_configured(
        config.adaptive_control_max,
        config.request_rate,
    )
    if max_value is None:
        raise ValueError("adaptive request_rate requires request_rate or control.max")
    min_value = float(config.adaptive_control_min)
    max_float = float(max_value)
    _validate_bounds(min_value, max_float, "request_rate")
    return RequestRateControlBackend(
        rate_setter=strategy.set_request_rate,
        minimum=min_value,
        maximum=max_float,
    )


def _build_users_backend(*, strategy: Any, concurrency_manager, config):
    max_value = _require_int(
        _explicit_or_configured(config.adaptive_control_max, config.num_users),
        "adaptive users max",
    )
    if config.num_users is None:
        raise ValueError("adaptive users requires user-centric num_users")
    min_value = _require_int(config.adaptive_control_min, "adaptive users min")
    _validate_bounds(min_value, max_value, "users")
    return UsersControlBackend(
        strategy=strategy,
        minimum=min_value,
        maximum=max_value,
    )
