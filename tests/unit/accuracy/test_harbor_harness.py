# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor-owned resolution tests for every accepted dataset source form."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("harbor", reason="requires agentic-accuracy worker lock")

from harbor.models.job.config import DatasetConfig
from harbor.models.trial.config import TaskConfig
from harbor.registry.client.factory import RegistryClientFactory

from aiperf.accuracy.harbor import HarborHarness


def _task(path: Path) -> TaskConfig:
    return TaskConfig(path=path)


@pytest.mark.asyncio
async def test_hub_package_resolves_to_content_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    digest = "a" * 64
    seen: list[DatasetConfig] = []

    async def get_task_configs(self: DatasetConfig, *_args: Any) -> list[TaskConfig]:
        seen.append(self)
        self.ref = f"sha256:{digest}"
        return [_task(tmp_path / "matplotlib__matplotlib-14623")]

    monkeypatch.setattr(DatasetConfig, "get_task_configs", get_task_configs)
    harness = await HarborHarness.create(
        "swe-bench/swe-bench-verified@latest",
        "fixture-model",
        {"max_episodes": 1},
    )
    try:
        assert seen[0].name == "swe-bench/swe-bench-verified"
        assert seen[0].ref == f"sha256:{digest}"
        assert harness.identity["dataset"] == {
            "provider": "Harbor Hub package registry",
            "benchmark": "swe-bench/swe-bench-verified",
            "repository": "swe-bench/swe-bench-verified",
            "revision": f"sha256:{digest}",
            "evaluation_splits": ["tasks"],
        }
    finally:
        await harness.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("authored", ["bfcl", "bfcl@1.0"])
async def test_legacy_registry_resolves_version_before_tasks(
    authored: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    metadata_requests: list[str] = []
    task_versions: list[str | None] = []

    class Client:
        async def get_dataset_metadata(self, requested: str) -> Any:
            metadata_requests.append(requested)
            return SimpleNamespace(version="1.0")

    async def get_task_configs(self: DatasetConfig, *_args: Any) -> list[TaskConfig]:
        task_versions.append(self.version)
        return [_task(tmp_path / "bfcl-live-simple-22-5-0")]

    monkeypatch.setattr(
        RegistryClientFactory, "create", staticmethod(lambda **_kwargs: Client())
    )
    monkeypatch.setattr(DatasetConfig, "get_task_configs", get_task_configs)
    harness = await HarborHarness.create(authored, "fixture-model", {})
    try:
        assert metadata_requests == [authored]
        assert task_versions == ["1.0"]
        assert harness.identity["dataset"] == {
            "provider": "Harbor legacy registry",
            "benchmark": "bfcl",
            "repository": "bfcl",
            "revision": "1.0",
            "evaluation_splits": ["tasks"],
        }
        assert harness.episodes[0].source == "bfcl"
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_local_directory_records_content_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "tasks"
    dataset.mkdir()
    (dataset / "manifest.txt").write_text("canonical fixture\n")

    async def get_task_configs(self: DatasetConfig, *_args: Any) -> list[TaskConfig]:
        assert self.path == dataset.resolve()
        return [_task(dataset / "task-one")]

    monkeypatch.setattr(DatasetConfig, "get_task_configs", get_task_configs)
    harness = await HarborHarness.create(dataset.as_posix(), "fixture-model", {})
    try:
        identity = harness.identity["dataset"]
        assert identity["provider"] == "Harbor local task directory"
        assert identity["benchmark"] == dataset.resolve().as_posix()
        assert identity["revision"].startswith("sha256:")
        assert len(identity["revision"]) == len("sha256:") + 64
    finally:
        await harness.close()


@pytest.mark.asyncio
async def test_empty_authored_revision_is_rejected() -> None:
    with pytest.raises(ValueError, match="revision after '@' must not be empty"):
        await HarborHarness.create("bfcl@", "fixture-model", {})
