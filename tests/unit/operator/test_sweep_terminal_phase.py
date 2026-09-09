# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal-phase resolution table for AIPerfSweep.

The sweep-controller's pre-fix logic collapsed any failed child to
``status.phase=Failed``, masking partial successes (e.g. 5/6 trials passed,
1 flaky child failed). The CRD enum has carried ``PartiallyFailed`` since
v1alpha1 was first written but no writer ever emitted it. This module
locks in the (completed, failed, max_failures) -> phase table.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.sweep_controller.main import resolve_terminal_phase


@pytest.mark.parametrize(
    ("completed", "failed", "max_failures", "expected"),
    [
        # No failures -> Succeeded regardless of max_failures.
        param(6, 0, 0, "Succeeded", id="all-succeeded-unbounded"),
        param(1, 0, 0, "Succeeded", id="single-success"),
        param(6, 0, 3, "Succeeded", id="all-succeeded-with-budget"),
        # The user-reported live regression: 5 succeeded + 1 failed used
        # to round to Failed; must be PartiallyFailed.
        param(5, 1, 0, "PartiallyFailed", id="five-of-six-passed"),
        param(1, 1, 0, "PartiallyFailed", id="half-and-half"),
        param(99, 1, 0, "PartiallyFailed", id="single-flaky-trial"),
        # All failed (no successful trial) -> Failed.
        param(0, 6, 0, "Failed", id="all-failed-unbounded"),
        param(0, 1, 0, "Failed", id="single-only-failed"),
        # Explicit budget exceeded -> Failed even if some succeeded.
        param(4, 2, 2, "Failed", id="budget-met-exact"),
        param(3, 5, 2, "Failed", id="budget-exceeded"),
        # Budget set but not exceeded -> PartiallyFailed (some succeeded).
        param(5, 1, 3, "PartiallyFailed", id="under-budget-some-failed"),
        param(8, 1, 5, "PartiallyFailed", id="under-budget-large-sweep"),
        # Edge: 0/0 (no children at all) -> Succeeded by definition (failed==0).
        # The orchestrator never calls aggregation_complete with an empty
        # all_results list, but this is the safe default if it ever does.
        param(0, 0, 0, "Succeeded", id="empty-results"),
    ],
)  # fmt: skip
def test_resolve_terminal_phase(
    completed: int, failed: int, max_failures: int, expected: str
) -> None:
    assert (
        resolve_terminal_phase(
            completed=completed, failed=failed, max_failures=max_failures
        )
        == expected
    )
