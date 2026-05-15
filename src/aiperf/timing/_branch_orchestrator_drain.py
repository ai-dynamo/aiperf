# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drain-observer + abort-observer mixin for ``BranchOrchestrator``.

The drain observer closes the concurrency-race window where the
orchestrator's final drain step (``dispatch_join_turn`` returning False
under cap, last descendant decrement, all-children-rolled-back) lands
AFTER the last ``on_credit_return`` callback's deferred
``_maybe_signal_dag_completion`` check. Without this hook
``all_credits_returned_event`` is never set and the phase runner waits
forever.

The abort observer fires under ``AIPERF_DAG_FAIL_FAST=true`` after
``_handle_child_errored_fail_fast`` finalizes the parent + orphan
sibling tear-down. The phase-side handler cancels every active phase
lifecycle so the strategy loop stops issuing new wire credits, honoring
the docs' "abort the whole run on first DAG child error" contract.

Both are wired by ``CreditCallbackHandler.set_branch_orchestrator``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class BranchOrchestratorDrainMixin:
    """Drain + abort observer plumbing.

    PhaseRunner / CreditCallbackHandler register callbacks here so they
    learn (a) when the orchestrator transitions to
    ``has_pending_branch_work() is False`` (drain), and (b) when
    ``AIPERF_DAG_FAIL_FAST`` triggers a whole-run abort (abort). See
    module docstring for the full credit-return / cleanup semantics.
    """

    _drain_observer: Callable[[], None] | None = None
    _abort_observer: Callable[[], None] | None = None

    def set_drain_observer(self, observer: Callable[[], None] | None) -> None:
        """Register/detach the sync drain observer."""
        self._drain_observer = observer

    def set_abort_observer(self, observer: Callable[[], None] | None) -> None:
        """Register/detach the sync abort observer (fired on FAIL_FAST)."""
        self._abort_observer = observer

    def _notify_drain(self) -> None:
        """Fire the registered drain observer (no-op if unset)."""
        observer = self._drain_observer
        if observer is None:
            return
        try:
            observer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("drain observer raised: %s", exc)

    def _notify_abort(self) -> None:
        """Fire the registered abort observer (no-op if unset)."""
        observer = self._abort_observer
        if observer is None:
            return
        try:
            observer()
        except Exception as exc:  # noqa: BLE001
            logger.warning("abort observer raised: %s", exc)
