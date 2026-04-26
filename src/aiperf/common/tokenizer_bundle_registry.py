# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry of HF tokenizer snapshot directories warmed by the controller.

Populated by ``tokenizer_validator.validate_tokenizers_eager`` as each per-tokenizer
warmer process completes. Read by ``TokenizerRouter`` to serve tar+zstd bundles.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class _Entry:
    """Per-tokenizer registration: snapshot path + readiness event."""

    snapshot_dir: Path | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)


class TokenizerBundleRegistry:
    """Maps tokenizer names to their resolved on-disk snapshot directories."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def register_pending(self, name: str) -> None:
        """Reserve a slot for ``name`` if not already present."""
        with self._lock:
            self._entries.setdefault(name, _Entry())

    def mark_ready(self, name: str, snapshot_dir: Path) -> None:
        """Record the resolved snapshot directory and unblock waiters."""
        with self._lock:
            entry = self._entries.setdefault(name, _Entry())
            entry.snapshot_dir = snapshot_dir
            entry.ready.set()

    def get(self, name: str) -> tuple[Path | None, asyncio.Event] | None:
        """Return (snapshot_dir, ready_event) or ``None`` if unknown."""
        with self._lock:
            entry = self._entries.get(name)
        if entry is None:
            return None
        return entry.snapshot_dir, entry.ready
