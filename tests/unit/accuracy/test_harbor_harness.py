# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor-owned resolution tests for every accepted dataset source form."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("harbor", reason="requires agentic-accuracy worker lock")

from harbor.models.job.config import DatasetConfig
from harbor.models.trial.config import TaskConfig
from harbor.registry.client.factory import RegistryClientFactory
from harbor.trial.trial import Trial

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


@pytest.mark.asyncio
async def test_inference_gateway_is_episode_scoped_and_host_environment_restored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def get_task_configs(self: DatasetConfig, *_args: Any) -> list[TaskConfig]:
        self.ref = f"sha256:{'f' * 64}"
        return [_task(tmp_path / "task-one")]

    monkeypatch.setattr(DatasetConfig, "get_task_configs", get_task_configs)
    monkeypatch.setenv("OPENAI_API_KEY", "original-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://original.invalid/v1")
    harness = await HarborHarness.create(
        "fixture/tasks@locked",
        "fixture-model",
        {
            "inference_gateway": {
                "base_url": "http://10.0.0.99:43123",
                "api_key": "run-secret",
            }
        },
    )
    episode_id = harness.episodes[0].episode_id
    try:
        assert os.environ["OPENAI_API_KEY"] == "run-secret"
        assert os.environ["OPENAI_BASE_URL"] == "http://10.0.0.99:43123"
        encoded = episode_id.replace(":", "%3A")
        assert harness._inference_environment(episode_id, "environment") == {
            "OPENAI_API_KEY": "run-secret",
            "OPENAI_BASE_URL": (
                f"http://10.0.0.99:43123/episodes/{encoded}/environment/v1"
            ),
        }
        assert harness._inference_environment(episode_id, "verifier")[
            "OPENAI_BASE_URL"
        ].endswith("/verifier/v1")
    finally:
        await harness.close()
    assert os.environ["OPENAI_API_KEY"] == "original-key"
    assert os.environ["OPENAI_BASE_URL"] == "https://original.invalid/v1"


@pytest.mark.asyncio
async def test_trial_environment_and_verifier_receive_distinct_callback_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def get_task_configs(self: DatasetConfig, *_args: Any) -> list[TaskConfig]:
        self.ref = f"sha256:{'e' * 64}"
        return [_task(tmp_path / "task-one")]

    captured: list[Any] = []

    class FakeTrial:
        async def run(self) -> Any:
            return SimpleNamespace(
                exception_info=None,
                verifier_result=SimpleNamespace(rewards={"reward": 1.0}),
                compute_token_cost_totals=lambda: (0, 0, 0, 0.0),
            )

    async def create(_cls: type[Trial], config: Any) -> FakeTrial:
        captured.append(config)
        return FakeTrial()

    monkeypatch.setattr(DatasetConfig, "get_task_configs", get_task_configs)
    monkeypatch.setattr(Trial, "create", classmethod(create))
    harness = await HarborHarness.create(
        "fixture/tasks@locked",
        "fixture-model",
        {
            "output_dir": tmp_path.as_posix(),
            "inference_gateway": {
                "base_url": "http://10.0.0.99:43123",
                "api_key": "run-secret",
            },
        },
    )
    episode_id = harness.episodes[0].episode_id
    encoded = episode_id.replace(":", "%3A")
    try:
        await harness.start_episodes([episode_id])
        events = await harness.poll_events(1, 1_000)
        assert events[0].episode_result is not None
        assert events[0].episode_result.outcome == "completed"
        assert captured[0].environment.env == {
            "OPENAI_API_KEY": "run-secret",
            "OPENAI_BASE_URL": (
                f"http://10.0.0.99:43123/episodes/{encoded}/environment/v1"
            ),
        }
        assert captured[0].verifier.env == {
            "OPENAI_API_KEY": "run-secret",
            "OPENAI_BASE_URL": (
                f"http://10.0.0.99:43123/episodes/{encoded}/verifier/v1"
            ),
        }
    finally:
        await harness.close()
