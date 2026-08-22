# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest-gated result index for the operator API."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunRecord:
    """One indexed run and its readiness-gated declared artifact set."""

    status: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] | None = None


class ResultsIndex:
    """In-memory index backed by caller-selected immutable result roots."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._runs: dict[str, RunRecord] = {}

    def update_status(self, run_id: str, status: dict[str, Any]) -> None:
        """Store the last reconciled CR status for a run."""
        self._runs.setdefault(run_id, RunRecord()).status = status

    def publish_manifest(self, run_id: str, manifest: dict[str, Any]) -> None:
        """Publish an already-validated readiness manifest as the sole artifact authority."""
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError("results manifest must contain artifacts")
        self._runs.setdefault(run_id, RunRecord()).manifest = manifest

    def ready_manifest(self, run_id: str) -> dict[str, Any] | None:
        """Return the published manifest, or None before publication."""
        record = self._runs.get(run_id)
        return record.manifest if record else None

    def artifact(self, run_id: str, name: str) -> tuple[bytes, str] | None:
        """Read only an artifact explicitly declared in the ready manifest."""
        manifest = self.ready_manifest(run_id)
        if manifest is None:
            return None
        entry = next(
            (item for item in manifest["artifacts"] if item.get("path") == name), None
        )
        if entry is None:
            return None
        path = (self._root / run_id / name).resolve()
        if self._root not in path.parents:
            return None
        try:
            data = path.read_bytes()
        except OSError:
            return None
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            return None
        return data, entry.get("contentType", "application/octet-stream")

    def stats(self) -> dict[str, int]:
        """Return bounded summary counts with no artifact content."""
        return {
            "runs": len(self._runs),
            "readyRuns": sum(
                record.manifest is not None for record in self._runs.values()
            ),
        }

    def rebuild(self) -> None:
        """Discard ephemeral index entries; reconciler repopulates them from CRs."""
        self._runs.clear()
