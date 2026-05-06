# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drain-observer mixin for ``BranchOrchestrator``.

Closes the concurrency-race window where the orchestrator's final drain
step (``dispatch_join_turn`` returning False under cap, last descendant
decrement, all-children-rolled-back) lands AFTER the last
``on_credit_return`` callback's deferred ``_maybe_signal_dag_completion``
check. Without this hook ``all_credits_returned_event`` is never set and
the phase runner waits forever.

Wired by ``CreditCallbackHandler.set_branch_orchestrator``; fired from
``_handle_child_done``, ``_handle_child_errored_fail_fast``, and
``_drain_vestigial_gates`` after their state mutations complete.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class BranchOrchestratorDrainMixin:
    """Drain-observer plumbing: PhaseRunner registers a callback here so it
    learns when the orchestrator transitions to ``has_pending_branch_work() is False``.
    See module docstring for the full credit-return / cleanup semantics.
    """

    _drain_observer: Callable[[], None] | None = None

    def set_drain_observer(self, observer: Callable[[], None] | None) -> None:
        """Register/detach the sync drain observer."""
        self._drain_observer = observer

    def _notify_drain(self) -> None:
        """Fire the registered drain observer (no-op if unset)."""
        observer = self._drain_observer
        if observer is None:
            return
        try:
            observer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("drain observer raised: %s", exc)
