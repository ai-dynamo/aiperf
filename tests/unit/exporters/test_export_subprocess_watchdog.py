# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-run export subprocesses must arm the parent-death watchdog with their export timeout.

If the CLI dies without running its exit handlers (SIGKILL, OOM, crash), the
spawned export child would otherwise keep the inherited stdout/stderr open
until its MLflow / W&B HTTP calls give up.
"""

import logging
import sys
import threading
from types import ModuleType
from typing import Any

import pytest
from pytest import param

from aiperf.common import environment as env_module
from aiperf.exporters import mlflow_export_subprocess, wandb_export_subprocess


class _RecordingWatchdog:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []
        self.finished = threading.Event()

    def __call__(
        self, name: str, deadline_seconds: float, logger: logging.Logger
    ) -> threading.Event:
        self.calls.append((name, deadline_seconds))
        return self.finished


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list[str | None] = []

    def put(self, item: str | None) -> None:
        self.items.append(item)


def _install_fake_exporter(
    monkeypatch: pytest.MonkeyPatch,
    exporter_path: str,
    watchdog: _RecordingWatchdog,
    *,
    fail: bool,
) -> None:
    class _FakeExporter:
        def __init__(self, exporter_config: Any) -> None:
            pass

        def _export_sync(self) -> None:
            # The guard must already be armed while the export blocks.
            assert watchdog.calls, "watchdog not armed before the export ran"
            assert not watchdog.finished.is_set()
            if fail:
                raise RuntimeError("tracking server unreachable")

    monkeypatch.setattr(exporter_path, _FakeExporter)


@pytest.mark.parametrize(
    "module, exporter_path, env_section, name",
    [
        param(
            mlflow_export_subprocess,
            "aiperf.exporters.mlflow_data_exporter.MLflowDataExporter",
            "MLFLOW",
            "MLflow export",
            id="mlflow",
        ),
        param(
            wandb_export_subprocess,
            "aiperf.exporters.wandb_data_exporter.WandbDataExporter",
            "WANDB",
            "W&B export",
            id="wandb",
        ),
    ],
)  # fmt: skip
@pytest.mark.parametrize("fail", [param(False, id="ok"), param(True, id="raises")])  # fmt: skip
def test_run_export_in_subprocess_arms_watchdog_with_export_timeout(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    exporter_path: str,
    env_section: str,
    name: str,
    fail: bool,
) -> None:
    monkeypatch.setattr(
        getattr(env_module.Environment, env_section), "EXPORT_TIMEOUT_SECONDS", 42.0
    )
    watchdog = _RecordingWatchdog()
    monkeypatch.setattr(module, "start_parent_death_watchdog", watchdog)
    _install_fake_exporter(monkeypatch, exporter_path, watchdog, fail=fail)
    queue = _FakeQueue()

    module.run_export_in_subprocess(exporter_config=object(), result_queue=queue)

    assert watchdog.calls == [(name, 42.0)]
    assert watchdog.finished.is_set()
    assert len(queue.items) == 1
    assert (queue.items[0] is not None) == fail


@pytest.mark.parametrize(
    "module, exporter_module",
    [
        param(mlflow_export_subprocess, "aiperf.exporters.mlflow_data_exporter", id="mlflow"),
        param(wandb_export_subprocess, "aiperf.exporters.wandb_data_exporter", id="wandb"),
    ],
)  # fmt: skip
def test_run_export_in_subprocess_import_failure_reported_on_queue(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    exporter_module: str,
) -> None:
    watchdog = _RecordingWatchdog()
    monkeypatch.setattr(module, "start_parent_death_watchdog", watchdog)
    # A None entry in sys.modules makes the lazy import raise ImportError.
    monkeypatch.setitem(sys.modules, exporter_module, None)
    queue = _FakeQueue()

    module.run_export_in_subprocess(exporter_config=object(), result_queue=queue)

    assert len(queue.items) == 1
    assert queue.items[0] is not None
    assert "Error(" in queue.items[0] and exporter_module in queue.items[0]
    assert watchdog.finished.is_set()
