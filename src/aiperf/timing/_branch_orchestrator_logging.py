# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic logging mixin for ``BranchOrchestrator``.

Cleanup-time stats and leak-detection logs. Split from
``branch_orchestrator.py`` to keep that file under the per-file ergonomics
cap.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class BranchOrchestratorLoggingMixin:
    """Cleanup-time stats logging: emits one structured ``BranchStats`` line
    when the orchestrator drains. See module docstring.
    """

    def _log_stats(self) -> None:
        s = self.stats
        logger.info(
            "BranchOrchestrator stats: spawned=%d completed=%d errored=%d suspended=%d "
            "resumed=%d parents_failed_due_to_child_error=%d joins_suppressed=%d",
            s.children_spawned, s.children_completed, s.children_errored,
            s.parents_suspended, s.parents_resumed,
            s.parents_failed_due_to_child_error, s.joins_suppressed,
        )  # fmt: skip

    def _log_leaks(self) -> None:
        leaked = self._iter_pending_joins()
        if not (leaked or self._child_to_join or self._descendant_counts):
            return
        logger.warning(
            "BranchOrchestrator leaked state at cleanup: "
            "%d active_joins, %d future_joins, %d tracked children, "
            "%d parents with descendants",
            len(self._active_joins),
            sum(len(g) for g in self._future_joins.values()),
            len(self._child_to_join),
            len(self._descendant_counts),
        )
        now_ns = time.monotonic_ns()
        for parent_corr, pending in leaked:
            age_ms = (now_ns - pending.created_at_ns) / 1_000_000
            logger.warning(
                "Abandoned pending join for parent %s "
                "(outstanding=%d, gated_turn_index=%s, age_ms=%.0f)",
                parent_corr,
                pending.total_outstanding,
                pending.gated_turn_index,
                age_ms,
            )
