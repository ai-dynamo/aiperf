# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scenario explicit-set provenance must survive the sweep orchestrator boundary.

The sweep orchestrator serializes each ``BenchmarkRun`` with
``model_dump(mode="json", exclude_none=True)`` (``orchestrator/local_executor.py``)
and the subprocess rebuilds it with ``BenchmarkRun.model_validate``
(``orchestrator/subprocess_runner.py``). ``model_fields_set`` does not survive
that round trip -- every dumped key comes back marked "set" -- so any provenance
flag recomputed from ``model_fields_set`` after the boundary reads a defaulted
field as explicitly authored, and the scenario resolver raises instead of
auto-filling. These tests pin both directions across that boundary.
"""

from __future__ import annotations

from typing import Any

import orjson
import pytest

from aiperf.common.enums import CacheBustTarget
from aiperf.common.scenario import apply_scenario
from aiperf.config.resolution.plan import BenchmarkRun

from .test_scenario_validator import _build_run


def _round_trip(run: BenchmarkRun) -> BenchmarkRun:
    """Mirror the orchestrator's dump/validate boundary exactly."""
    payload = orjson.loads(orjson.dumps(run.model_dump(mode="json", exclude_none=True)))
    return BenchmarkRun.model_validate(payload)


def _cache_bust_dataset(target: str | None) -> dict[str, Any]:
    dataset: dict[str, Any] = {
        "name": "main",
        "type": "public",
        "dataset": "semianalysis_cc_traces_weka_with_subagents",
    }
    if target is not None:
        dataset["cache_bust"] = {"target": target}
    return dataset


@pytest.mark.parametrize("round_trip", [False, True], ids=["fresh", "round_trip"])
def test_unset_streaming_is_auto_filled_by_scenario(round_trip: bool) -> None:
    """A user who never passed --streaming gets the scenario auto-fill, not a lock error."""
    run = _build_run(
        streaming=None,
        extra={"ignore_eos": True},
        dataset=_cache_bust_dataset(CacheBustTarget.FIRST_TURN_PREFIX.value),
    )
    if round_trip:
        run = _round_trip(run)

    assert run.cfg.endpoint.streaming is False
    outcome = apply_scenario(run)

    assert [v.flag for v in outcome.violations] == []
    assert "streaming" in outcome.applied_locks
    assert run.cfg.endpoint.streaming is True


@pytest.mark.parametrize("round_trip", [False, True], ids=["fresh", "round_trip"])
def test_explicit_no_streaming_still_violates_after_round_trip(
    round_trip: bool,
) -> None:
    """An explicit --no-streaming must remain a violation on both sides of the boundary."""
    run = _build_run(
        streaming=False,
        extra={"ignore_eos": True},
        dataset=_cache_bust_dataset(CacheBustTarget.FIRST_TURN_PREFIX.value),
        unsafe_override=True,
    )
    if round_trip:
        run = _round_trip(run)

    outcome = apply_scenario(run)

    assert "--streaming" in [v.flag for v in outcome.violations]
    assert "streaming" not in outcome.applied_locks


@pytest.mark.parametrize("round_trip", [False, True], ids=["fresh", "round_trip"])
def test_unset_cache_bust_target_is_auto_filled_by_scenario(round_trip: bool) -> None:
    """An untouched cache_bust.target auto-fills from the scenario after a round trip."""
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset=_cache_bust_dataset(None),
    )
    if round_trip:
        run = _round_trip(run)

    outcome = apply_scenario(run)

    assert [v.flag for v in outcome.violations] == []
    assert "cache_bust" in outcome.applied_locks
    assert run.cfg.get_cache_bust_target() == CacheBustTarget.FIRST_TURN_PREFIX


@pytest.mark.parametrize("round_trip", [False, True], ids=["fresh", "round_trip"])
def test_explicit_conflicting_cache_bust_target_still_violates(
    round_trip: bool,
) -> None:
    """An explicitly authored conflicting target stays a violation after a round trip."""
    run = _build_run(
        streaming=True,
        extra={"ignore_eos": True},
        dataset=_cache_bust_dataset(CacheBustTarget.NONE.value),
        unsafe_override=True,
    )
    if round_trip:
        run = _round_trip(run)

    outcome = apply_scenario(run)

    assert "--cache-bust" in [v.flag for v in outcome.violations]
    assert "cache_bust" not in outcome.applied_locks
