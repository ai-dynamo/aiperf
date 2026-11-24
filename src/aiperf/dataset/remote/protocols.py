# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protocol for the ``dataset_source_resolver`` plugin category."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from aiperf.config.dataset.remote_ref import RemoteRef
    from aiperf.dataset.remote.models import ProbeResult, ResolvedSource


@runtime_checkable
class SourceResolverProtocol(Protocol):
    """Fetch one URI scheme's dataset references into the local cache."""

    async def probe(self, ref: RemoteRef) -> ProbeResult:
        """Check existence and access, pin the version, and read any manifest."""
        ...

    async def read_prefix(self, probe: ProbeResult, nbytes: int) -> tuple[bytes, bool]:
        """Return up to ``nbytes`` leading bytes and whether they are the whole file."""
        ...

    async def materialize(self, probe: ProbeResult) -> ResolvedSource:
        """Download the probed version into the cache."""
        ...

    async def fetch_file(self, probe: ProbeResult, relpath: str) -> Path:
        """Download one file of a probed folder into the cache; return its local path."""
        ...
