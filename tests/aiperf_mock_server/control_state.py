# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Countable state for mock-server control-plane admin routes."""

from __future__ import annotations


class ControlState:
    def __init__(self) -> None:
        self.profiler_starts = 0
        self.profiler_stops = 0
        self.reset_count = 0
        self.events: list[str] = []

    def reset(self) -> None:
        self.profiler_starts = 0
        self.profiler_stops = 0
        self.reset_count = 0
        self.events.clear()

    def record(self, event: str) -> None:
        self.events.append(event)


control_state = ControlState()
