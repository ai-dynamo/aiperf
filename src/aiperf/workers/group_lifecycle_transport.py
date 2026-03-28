# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport contract for worker-group lifecycle fanout."""

from __future__ import annotations

from typing import Protocol


class GroupLifecycleTransport(Protocol):
    """Protocol for sending lifecycle commands to group children."""

    async def fanout_command(self, child_ids: list[str], command: str) -> None:
        """Send the same lifecycle command to the supplied children."""
