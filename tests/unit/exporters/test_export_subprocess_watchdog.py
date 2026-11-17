# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Post-run export subprocesses must arm the parent-death watchdog with their export timeout.

If the CLI dies without running its exit handlers (SIGKILL, OOM, crash), the
spawned export child would otherwise keep the inherited stdout/stderr open
until its MLflow / W&B HTTP calls give up.
"""

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pytest import param

from aiperf.common import environment as env_module
from aiperf.common.constants import IS_WINDOWS
from aiperf.exporters import mlflow_export_subprocess, wandb_export_subprocess


class _RecordingWatchdog:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def __call__(
        self, name: str, deadline_seconds: float, logger: logging.Logger
    ) -> None:
        self.calls.append((name, deadline_seconds))


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


# The CLI spawns the real export entrypoint with a fake exporter that returns at
# once but leaves teardown work that never finishes, then SIGKILLs itself once
# that teardown has started. The test reads the inherited stdout: EOF arrives
# only once the export child is gone.
_TEARDOWN_HANG_PROG = """
import atexit, importlib, multiprocessing as mp, os, signal, sys, threading, time, types


def child(entry_module, exporter_module, exporter_cls, hang, marker, queue):
    def _hang():
        open(marker, "w").close()
        time.sleep(120)  # stands in for an SDK teardown call that never returns

    class _Exporter:
        def __init__(self, exporter_config):
            pass

        def _export_sync(self):
            if hang == "atexit":
                atexit.register(_hang)
            else:
                # The main thread only counts as finished once run() has
                # returned and threading._shutdown() has started, which then
                # blocks joining this thread (3.12 and 3.13 order the other
                # exit steps differently, so gate on nothing else).
                main = threading.main_thread()
                threading.Thread(target=lambda: (main.join(), _hang())).start()

    fake = types.ModuleType(exporter_module)
    setattr(fake, exporter_cls, _Exporter)
    sys.modules[exporter_module] = fake
    importlib.import_module(entry_module).run_export_in_subprocess(None, queue)


if __name__ == "__main__":
    marker = sys.argv[5]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=child, args=(*sys.argv[1:6], queue), daemon=True)
    proc.start()
    deadline = time.monotonic() + 60
    while not os.path.exists(marker):
        if time.monotonic() > deadline:
            sys.exit("export child never reached teardown")
        time.sleep(0.05)
    print(proc.pid, flush=True)
    os.kill(os.getpid(), signal.SIGKILL)
"""


@pytest.mark.skipif(IS_WINDOWS, reason="signal.SIGKILL is unavailable on Windows")
@pytest.mark.parametrize(
    "entry_module, exporter_module, exporter_cls, env_var, name",
    [
        param(
            "aiperf.exporters.mlflow_export_subprocess",
            "aiperf.exporters.mlflow_data_exporter",
            "MLflowDataExporter",
            "AIPERF_MLFLOW_EXPORT_TIMEOUT_SECONDS",
            "MLflow export",
            id="mlflow",
        ),
        param(
            "aiperf.exporters.wandb_export_subprocess",
            "aiperf.exporters.wandb_data_exporter",
            "WandbDataExporter",
            "AIPERF_WANDB_EXPORT_TIMEOUT_SECONDS",
            "W&B export",
            id="wandb",
        ),
    ],
)  # fmt: skip
@pytest.mark.parametrize(
    "hang",
    [
        param("atexit", id="atexit-hook"),
        param("thread", id="non-daemon-thread"),
    ],
)  # fmt: skip
def test_export_child_teardown_hang_after_cli_sigkill_still_releases_pipe(
    tmp_path: Path,
    entry_module: str,
    exporter_module: str,
    exporter_cls: str,
    env_var: str,
    name: str,
    hang: str,
) -> None:
    prog = tmp_path / "teardown_hang.py"
    prog.write_text(_TEARDOWN_HANG_PROG)
    marker = tmp_path / "teardown-started"
    cli = subprocess.Popen(
        [
            sys.executable,
            str(prog),
            entry_module,
            exporter_module,
            exporter_cls,
            hang,
            str(marker),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, env_var: "1"},
    )
    child_pid = None
    try:
        first = cli.stdout.readline().strip()
        assert first.isdigit(), f"CLI stand-in failed: {first}{cli.stdout.read()}"
        child_pid = int(first)
        rest: list[str] = []
        reader = threading.Thread(target=lambda: rest.append(cli.stdout.read()))
        reader.start()
        reader.join(timeout=20.0)

        assert not reader.is_alive(), (
            "pipe never reached EOF: the export child's teardown outlived the CLI"
        )
        assert f"{name} cleanup exceeded" in rest[0]
    finally:
        if cli.poll() is None:
            cli.kill()
        cli.wait(timeout=10)
        if child_pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
