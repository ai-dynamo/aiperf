# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seamless phase handoff coordination for credit callbacks."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from aiperf.common.enums import CreditPhase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aiperf.credit.callback_handler import PhaseCallbackContext
    from aiperf.credit.structs import Credit


class PhaseHandoffCoordinator:
    """Routes non-final source-phase returns into the next phase."""

    def __init__(self) -> None:
        self._targets: dict[CreditPhase, CreditPhase] = {}
        self._pending: dict[CreditPhase, list[Credit]] = defaultdict(list)

    def start(self, source_phase: CreditPhase, target_phase: CreditPhase) -> None:
        if source_phase != target_phase:
            self._targets[source_phase] = target_phase

    def clear(self, source_phase: CreditPhase) -> None:
        self._targets.pop(source_phase, None)

    def target_for(
        self, credit: Credit, handler: PhaseCallbackContext
    ) -> CreditPhase | None:
        target_phase = self._targets.get(credit.phase)
        if target_phase is None:
            return None
        if credit.is_final_turn or credit.agent_depth > 0:
            return None
        if handler.lifecycle.is_complete:
            return None
        return target_phase

    async def drain(
        self,
        target_phase: CreditPhase,
        handlers: Mapping[CreditPhase, PhaseCallbackContext],
    ) -> None:
        pending = self._pending.pop(target_phase, [])
        target = handlers.get(target_phase)
        if target is None:
            self._pending[target_phase].extend(pending)
            return
        for credit in pending:
            await self._dispatch(credit, target)

    async def dispatch_or_queue(
        self,
        credit: Credit,
        target_phase: CreditPhase,
        handlers: Mapping[CreditPhase, PhaseCallbackContext],
    ) -> None:
        target = handlers.get(target_phase)
        if (
            target is None
            or not target.lifecycle.is_started
            or target.lifecycle.is_complete
        ):
            self._pending[target_phase].append(credit)
            return
        await self._dispatch(credit, target)

    async def _dispatch(self, credit: Credit, target: PhaseCallbackContext) -> None:
        if not target.stop_checker.can_send_any_turn():
            target.concurrency_manager.release_session_slot(credit.phase)
            return
        await target.strategy.handle_phase_handoff(credit)
