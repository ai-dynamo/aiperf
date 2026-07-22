# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guards the plan-driven orchestrator execution API."""

from __future__ import annotations

from aiperf.orchestrator.orchestrator import MultiRunOrchestrator


def test_orchestrator_execution_uses_plan_driven_api() -> None:
    """Strategy resolution stays in cli_runner and out of the orchestrator."""
    for obsolete_method in (
        "_resolve_strategy",
        "_execute",
        "_execute_loop",
        "_create_sweep_strategy",
        "_create_confidence_strategy",
    ):
        assert not hasattr(MultiRunOrchestrator, obsolete_method), (
            f"obsolete orchestrator method {obsolete_method!r} unexpectedly exists"
        )

    assert hasattr(MultiRunOrchestrator, "execute")

    from aiperf.cli_runner import _strategy

    assert hasattr(_strategy, "build_strategy")
