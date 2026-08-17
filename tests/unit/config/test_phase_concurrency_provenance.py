# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``concurrency`` provenance on phase configs survives the sweep round trip.

``ConcurrencyPhase.concurrency`` defaults to a positive ``1``, so its VALUE
cannot distinguish an inherited ceiling from one the operator asked for, and
``model_fields_set`` is wiped by the sweep orchestrator's dump -> revalidate
boundary. These tests pin the persisted flag that carries the intent instead.
"""

from __future__ import annotations

from typing import Any

import pydantic
import pytest
from pytest import param

from aiperf.config.flags import CLIConfig
from aiperf.config.flags._converter_profiling import build_profiling
from aiperf.config.phases import PhaseConfig

_PHASE_ADAPTER = pydantic.TypeAdapter(PhaseConfig)


def _phase(**overrides: Any) -> Any:
    return _PHASE_ADAPTER.validate_python(
        {"name": "profiling", "type": "concurrency", **overrides}
    )


def _round_trip(phase: Any) -> Any:
    """Mirror the sweep orchestrator's dump -> subprocess validate boundary."""
    return _PHASE_ADAPTER.validate_python(
        phase.model_dump(mode="json", exclude_none=True)
    )


def test_defaulted_concurrency_is_not_explicit() -> None:
    phase = _phase()
    assert phase.concurrency == 1
    assert phase.concurrency_explicitly_set is False
    assert phase._concurrency_explicitly_set is False


def test_explicit_concurrency_is_explicit_even_at_the_default_value() -> None:
    phase = _phase(concurrency=1)
    assert phase.concurrency_explicitly_set is True


def test_defaulted_concurrency_survives_round_trip_as_not_explicit() -> None:
    phase = _round_trip(_phase())
    # model_fields_set is uninformative after the round trip; the flag is not.
    assert "concurrency" in phase.model_fields_set
    assert phase.concurrency_explicitly_set is False


def test_explicit_concurrency_survives_round_trip_as_explicit() -> None:
    phase = _round_trip(_phase(concurrency=4))
    assert phase.concurrency_explicitly_set is True
    assert phase.concurrency == 4


def test_flag_survives_repeated_round_trips() -> None:
    phase = _phase()
    for _ in range(3):
        phase = _round_trip(phase)
    assert phase.concurrency_explicitly_set is False


@pytest.mark.parametrize(
    "alias",
    [
        param("concurrency_explicitly_set", id="field-name"),
        param("_concurrency_explicitly_set", id="underscore-alias"),
    ],
)  # fmt: skip
def test_incoming_flag_wins_over_fields_set(alias: str) -> None:
    """An incoming key carries the ORIGINAL intent across the boundary."""
    phase = _PHASE_ADAPTER.validate_python(
        {"name": "profiling", "type": "concurrency", "concurrency": 8, alias: False}
    )
    assert phase.concurrency_explicitly_set is False


def _cli_phase(**cli_kwargs: Any) -> Any:
    cli = CLIConfig(url="http://localhost:8000/v1", model_names=["m"], **cli_kwargs)
    return _PHASE_ADAPTER.validate_python({"name": "profiling", **build_profiling(cli)})


def test_bare_cli_produces_inherited_concurrency() -> None:
    """No ``--concurrency``: the production phase carries 1 but is NOT explicit."""
    phase = _cli_phase()
    assert phase.concurrency == 1
    assert phase.concurrency_explicitly_set is False


def test_cli_concurrency_flag_is_explicit() -> None:
    phase = _cli_phase(concurrency=1)
    assert phase.concurrency == 1
    assert phase.concurrency_explicitly_set is True


def test_credit_phase_config_carries_the_flag() -> None:
    """``_build_profiling_config`` copies provenance onto the CreditPhaseConfig."""
    from aiperf.timing.config import _build_profiling_config
    from aiperf.timing.request_cancellation import RequestCancellationConfig

    def _credit(phase: Any) -> Any:
        return _build_profiling_config(
            phase,
            default_cancellation=RequestCancellationConfig(),
            phase_index=0,
            profiling_index=0,
            is_graph=True,
        )

    assert _credit(_cli_phase()).concurrency_explicitly_set is False
    assert _credit(_cli_phase(concurrency=4)).concurrency_explicitly_set is True
