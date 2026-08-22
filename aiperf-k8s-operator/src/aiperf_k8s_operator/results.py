# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manifest-gated result index for the operator API."""

from __future__ import annotations

import hashlib
import json
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
        """Rebuild readiness state from private result roots without trusting markers."""
        self._runs.clear()
        if not self._root.is_dir():
            return
        for run_dir in self._root.iterdir():
            if not run_dir.is_dir() or run_dir.is_symlink():
                continue
            manifest_path = run_dir / "results-manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not self._is_ready_manifest(run_dir.name, manifest):
                continue
            self.publish_manifest(run_dir.name, manifest)

    @staticmethod
    def _is_ready_manifest(run_id: str, manifest: Any) -> bool:
        """Accept only a complete v1 manifest addressed to its containing run directory."""
        if not isinstance(manifest, dict):
            return False
        if manifest.get("contractVersion") != "native-k8s/v1":
            return False
        if manifest.get("runId") != run_id or manifest.get("ready") is not True:
            return False
        if not isinstance(manifest.get("artifactRoot"), str):
            return False
        if not isinstance(manifest.get("wasCancelled"), bool):
            return False
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, list):
            return False
        return all(
            isinstance(artifact, dict)
            and isinstance(artifact.get("path"), str)
            and isinstance(artifact.get("sha256"), str)
            and len(artifact["sha256"]) == 64
            and isinstance(artifact.get("bytes"), int)
            and artifact["bytes"] >= 0
            and isinstance(artifact.get("contentType"), str)
            for artifact in artifacts
        )
