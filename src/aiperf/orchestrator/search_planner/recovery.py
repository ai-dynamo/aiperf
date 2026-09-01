# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic reconstruction of adaptive search planners."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import orjson

from aiperf.orchestrator.models import RunResult

if TYPE_CHECKING:
    from aiperf.orchestrator.search_planner.base import SearchPlanner

SEARCH_CHECKPOINT_FILENAME = "search_checkpoint.json"


def restore_planner_history(
    planner: SearchPlanner,
    history: list[tuple[dict[str, Any], list[RunResult]]],
) -> None:
    """Replay durable terminal results into a freshly constructed planner."""
    for expected_values, results in history:
        proposal = planner.ask()
        if proposal is None:
            raise ValueError("adaptive planner converged before restoring its history")
        _, variation = proposal
        if variation.values != expected_values:
            raise ValueError(
                "adaptive planner proposed different values while restoring its history"
            )
        planner.tell(variation, results)


def write_search_checkpoint(
    base_dir: Path,
    history: list[tuple[dict[str, Any], list[RunResult]]],
) -> None:
    """Atomically persist the planner inputs needed after a pod restart."""
    payload = {
        "iterations": [
            {
                "variation_values": values,
                "results": [result.model_dump(mode="json") for result in results],
            }
            for values, results in history
        ]
    }
    destination = base_dir / SEARCH_CHECKPOINT_FILENAME
    temporary = destination.with_suffix(".tmp")
    temporary.write_bytes(orjson.dumps(payload))
    temporary.replace(destination)


def read_search_checkpoint(
    base_dir: Path,
) -> list[tuple[dict[str, Any], list[RunResult]]]:
    """Read the last complete adaptive-search checkpoint, if one exists."""
    path = base_dir / SEARCH_CHECKPOINT_FILENAME
    if not path.is_file():
        return []
    payload = orjson.loads(path.read_bytes())
    iterations = payload.get("iterations")
    if not isinstance(iterations, list):
        raise ValueError("adaptive search checkpoint has no iterations list")
    history: list[tuple[dict[str, Any], list[RunResult]]] = []
    for iteration in iterations:
        values = iteration.get("variation_values")
        results = iteration.get("results")
        if not isinstance(values, dict) or not isinstance(results, list):
            raise ValueError("adaptive search checkpoint has an invalid iteration")
        history.append(
            (values, [RunResult.model_validate(result) for result in results])
        )
    return history
