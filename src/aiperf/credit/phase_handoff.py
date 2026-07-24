# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seamless phase handoff coordination for credit callbacks."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from aiperf.common.phase import PhaseRuntimeKey, phase_runtime_key

if TYPE_CHECKING:
    from collections.abc import Mapping

    from aiperf.credit.callback_handler import PhaseCallbackContext
    from aiperf.credit.structs import Credit


class PhaseHandoffCoordinator:
    """Route live root-session returns into the next phase."""

    def __init__(self) -> None:
        self._targets: dict[PhaseRuntimeKey, PhaseRuntimeKey] = {}
        self._pending: dict[PhaseRuntimeKey, list[Credit]] = defaultdict(list)

    def start(
        self, source_phase: PhaseRuntimeKey, target_phase: PhaseRuntimeKey
    ) -> None:
        if source_phase != target_phase:
            self._targets[source_phase] = target_phase

    def clear(self, source_phase: PhaseRuntimeKey) -> None:
        self._targets.pop(source_phase, None)

    def target_for(
        self, credit: Credit, handler: PhaseCallbackContext
    ) -> PhaseRuntimeKey | None:
        source_phase = phase_runtime_key(credit.phase, credit.phase_index)
        target_phase = self._targets.get(source_phase)
        if target_phase is None or not handler.allow_session_handoff:
            return None
        if (
            credit.is_final_turn
            or credit.agent_depth > 0
            or credit.parent_correlation_id is not None
        ):
            return None
        if not handler.lifecycle.is_sending_complete or handler.lifecycle.is_complete:
            return None
        return target_phase

    async def drain(
        self,
        target_phase: PhaseRuntimeKey,
        handlers: Mapping[PhaseRuntimeKey, PhaseCallbackContext],
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
        target_phase: PhaseRuntimeKey,
        handlers: Mapping[PhaseRuntimeKey, PhaseCallbackContext],
    ) -> None:
        target = handlers.get(target_phase)
        if target is None or not target.lifecycle.is_started:
            self._pending[target_phase].append(credit)
            return
        await self._dispatch(credit, target)

    @staticmethod
    async def _dispatch(credit: Credit, target: PhaseCallbackContext) -> None:
        source_phase = phase_runtime_key(credit.phase, credit.phase_index)
        if target.lifecycle.is_complete or not target.stop_checker.can_send_any_turn():
            target.concurrency_manager.release_session_slot(source_phase)
            return
        await target.strategy.handle_phase_handoff(credit)
