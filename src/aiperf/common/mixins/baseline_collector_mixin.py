# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixin for services that take phase-boundary baseline readings."""

from __future__ import annotations

from abc import abstractmethod

from aiperf.common.enums import MessageType
from aiperf.common.hooks import on_message
from aiperf.common.messages import PhaseBaselineRequestMessage


class BaselineCollectorMixin:
    """Mix into a BaseComponentService to join best-effort baseline capture."""

    @abstractmethod
    async def collect_baseline(self, message: PhaseBaselineRequestMessage) -> None:
        """Take a single point-in-time baseline reading for the request."""

    @on_message(MessageType.PHASE_BASELINE_REQUEST)
    async def _on_phase_baseline_request(
        self, message: PhaseBaselineRequestMessage
    ) -> None:
        try:
            await self.collect_baseline(message)
        except Exception as exc:  # per-collector fault tolerance
            self.warning(
                "Baseline capture failed for "
                f"phase {message.phase_name!r} {message.kind}: "
                f"{type(exc).__name__}: {exc}"
            )
