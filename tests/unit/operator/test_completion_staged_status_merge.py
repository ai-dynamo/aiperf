# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``_merge_staged_status``.

Completion stages its terminal status on a second ``StatusBuilder`` built over
the same stored ``status``, then merges it into the live patch. ``staged_sb``
regenerates ``conditions`` from the CR as it exists on the apiserver, so it
knows nothing about conditions the caller staged earlier in the same tick (the
``WorkersReady`` condition ``_update_worker_counts`` writes just before the
monitor calls into ``handle_completion``, for instance). A plain
``dict.update`` replaced the whole ``conditions`` key and silently dropped
those; the merge must be per condition type.
"""

from __future__ import annotations

from typing import Any

import kopf

from aiperf.operator.handlers.completion import _merge_staged_status
from aiperf.operator.status import StatusBuilder


def _condition(cond_type: str, status: str = "True") -> dict[str, Any]:
    return {"type": cond_type, "status": status, "reason": cond_type}


def _target(conditions: list[dict[str, Any]] | None) -> StatusBuilder:
    patch = kopf.Patch()
    if conditions is not None:
        patch.status["conditions"] = conditions
    return StatusBuilder(patch, {})


def test_merge_preserves_conditions_staged_earlier_in_the_tick() -> None:
    target = _target([_condition("WorkersReady")])

    _merge_staged_status(
        target,
        {"phase": "Completed", "conditions": [_condition("ResultsAvailable")]},
    )

    assert target._patch.status["phase"] == "Completed"
    assert [c["type"] for c in target._patch.status["conditions"]] == [
        "WorkersReady",
        "ResultsAvailable",
    ]


def test_merge_lets_staged_condition_win_on_the_same_type() -> None:
    target = _target([_condition("ResultsAvailable", status="False")])

    _merge_staged_status(target, {"conditions": [_condition("ResultsAvailable")]})

    assert target._patch.status["conditions"] == [_condition("ResultsAvailable")]


def test_merge_without_existing_conditions_takes_the_staged_list() -> None:
    target = _target(None)
    staged = [_condition("ResultsAvailable")]

    _merge_staged_status(target, {"conditions": staged})

    assert target._patch.status["conditions"] == staged


def test_merge_without_staged_conditions_keeps_existing() -> None:
    target = _target([_condition("WorkersReady")])

    _merge_staged_status(target, {"phase": "Completed"})

    assert target._patch.status["conditions"] == [_condition("WorkersReady")]
    assert target._patch.status["phase"] == "Completed"
