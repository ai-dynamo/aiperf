# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NDJSON renderer for aiperf kube watch.

Emits one JSON line per snapshot to an output stream. Designed for AI agent
consumption — each line is a complete, self-contained state snapshot that
an agent can parse and reason about without additional commands.
"""

from __future__ import annotations

import sys
from typing import IO, TYPE_CHECKING

import orjson

if TYPE_CHECKING:
    from aiperf.kubernetes.watch_models import WatchSnapshot


class JsonRenderer:
    """Renders WatchSnapshot as NDJSON (one JSON object per line)."""

    def __init__(self, output: IO[str] | None = None) -> None:
        self._output = output or sys.stdout

    def render(self, snapshot: WatchSnapshot) -> None:
        """Serialize snapshot to JSON and write as a single line."""
        data = snapshot.to_dict()
        line = orjson.dumps(data, option=orjson.OPT_UTC_Z).decode()
        self._output.write(line)
        self._output.write("\n")
        self._output.flush()

    def start(self) -> None:
        """No-op for JSON mode (no setup needed)."""

    def stop(self) -> None:
        """No-op for JSON mode (no teardown needed)."""
