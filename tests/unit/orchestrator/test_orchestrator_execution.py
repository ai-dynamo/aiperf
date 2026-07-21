# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supersession marker for the v1 orchestrator execution-method tests.

The v1 suite exercised the agentx orchestrator's strategy auto-detection and
generic execution loop, all driven off a v1 ``UserConfig``:

- ``_resolve_strategy(config)`` mapping ``loadgen.concurrency`` (scalar vs list)
  + ``loadgen.num_profile_runs`` to a ``ParameterSweepStrategy`` /
  ``FixedTrialsStrategy`` / ``SweepConfidenceStrategy`` (and raising for a
  single run);
- ``_execute(config, strategy)`` choosing the strategy's own ``execute()``
  (Option B) or falling back to ``_execute_loop``;
- ``_execute_loop(config, strategy)`` with ``tag_result`` / ``get_run_label`` /
  ``get_run_path`` / ``collect_failed_values`` / cooldown via ``time.sleep`` and
  "sweep values failed" warnings;
- ``SweepConfidenceStrategy.execute()`` iteration order + metadata tagging +
  cooldowns for repeated vs independent modes;
- the ``_create_sweep_strategy`` / ``_create_confidence_strategy`` factories.

main's #1035 removed ALL of these from the orchestrator. The v2
``MultiRunOrchestrator`` takes a pre-built ``BenchmarkPlan`` (the CLI runner
already resolved variations + strategy at plan-build time) and a ``RunExecutor``;
it has no ``_resolve_strategy`` / ``_execute`` / ``_execute_loop`` /
``_create_*_strategy``, no ``UserConfig``, and no in-orchestrator ``time.sleep``
(cooldown reads ``plan.sweep.cooldown_seconds``). The iteration order /
metadata-tagging / cooldown concerns these tests covered are now expressed
through ``BenchmarkPlan`` + ``SweepMode`` and verified on v2:

| v1 test concern                                       | v2 coverage |
| ----------------------------------------------------- | ----------- |
| _resolve_strategy auto-detect from config             | tests/unit/orchestrator/test_strategies.py + cli_runner._strategy.build_strategy |
| _execute Option-B vs generic loop                     | tests/unit/orchestrator/test_multi_run_orchestrator.py (execute -> _execute_repeated/_independent) |
| _execute_loop tag_result / failed-value warnings      | test_multi_run_orchestrator.py (_stamp_variation_metadata, failure threshold) |
| repeated vs independent iteration order               | test_multi_run_orchestrator.py (SweepMode REPEATED/INDEPENDENT order asserts) |
| repeated/independent cooldown application             | test_multi_run_orchestrator.py (plan.sweep cooldown) |
| _create_sweep / _create_confidence factories          | cli_runner._strategy.build_strategy + test_strategies.py |

This module keeps a single guard so the supersession is self-verifying: if any
of the v1 execution methods re-appear on the orchestrator, the guard fails and
forces a real re-port of the suite mapped above.
"""

from __future__ import annotations

from aiperf.orchestrator.orchestrator import MultiRunOrchestrator


def test_v1_orchestrator_execution_methods_are_gone() -> None:
    """The v1 strategy-resolution + generic-loop methods must not exist on v2.

    Guards the supersession documented in this module's docstring. On v2 the
    orchestrator consumes a pre-resolved ``BenchmarkPlan`` + ``RunExecutor``;
    strategy resolution moved to ``aiperf.cli_runner._strategy.build_strategy``.
    If any v1 method re-appears, the v1 execution model was re-introduced and its
    tests must be properly re-ported (see the table in this module's docstring)
    rather than left as this stub.
    """
    for v1_method in (
        "_resolve_strategy",
        "_execute",
        "_execute_loop",
        "_create_sweep_strategy",
        "_create_confidence_strategy",
    ):
        assert not hasattr(MultiRunOrchestrator, v1_method), (
            f"v1 orchestrator execution method {v1_method!r} re-appeared on v2 "
            "MultiRunOrchestrator -- re-port the v1 suite (see module docstring)."
        )

    # The v2 plan-driven entry point must exist (catches a refactor that drops
    # the real execute path while leaving this stub stale).
    assert hasattr(MultiRunOrchestrator, "execute")

    # Strategy resolution re-homed onto the cli_runner strategy builder.
    from aiperf.cli_runner import _strategy

    assert hasattr(_strategy, "build_strategy")
