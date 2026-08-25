# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

from aiperf.rust_shims.__main__ import main


def test_unknown_shim_is_rejected_without_importing_registered_modules(
    monkeypatch,
) -> None:
    imported: list[str] = []

    def record_import(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(main=lambda _arguments: 0)

    monkeypatch.setattr(importlib, "import_module", record_import)

    assert main(["unknown"]) == 2
    assert imported == []


def test_launcher_defers_import_and_forwards_remaining_arguments(monkeypatch) -> None:
    captured: list[list[str]] = []
    module = SimpleNamespace(main=lambda arguments: captured.append(arguments) or 7)
    imported: list[str] = []

    def record_import(name: str) -> SimpleNamespace:
        imported.append(name)
        return module

    monkeypatch.setattr(importlib, "import_module", record_import)

    assert main(["live-streaming", "--wire"]) == 7
    assert imported == ["aiperf.rust_shims.live_streaming_worker"]
    assert captured == [["--wire"]]


def test_launcher_dispatches_each_registered_shim(monkeypatch) -> None:
    imported: list[str] = []

    def record_import(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(main=lambda _arguments: 0)

    monkeypatch.setattr(importlib, "import_module", record_import)

    assert main(["slurm-generate"]) == 0
    assert imported == ["aiperf.rust_shims.slurm.generate"]


def test_launcher_rejects_missing_name() -> None:
    assert main([]) == 2


def test_launcher_converts_module_errors_to_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(RuntimeError("unavailable")),
    )

    assert main(["live-streaming"]) == 1


def test_launcher_rejects_invalid_shim_main(monkeypatch) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(main="not-callable"),
    )

    assert main(["live-streaming"]) == 1


def test_launcher_rejects_non_integer_result(monkeypatch) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(main=lambda _arguments: "invalid"),
    )

    assert main(["live-streaming"]) == 1


def test_package_import_does_not_load_shims(monkeypatch) -> None:
    monkeypatch.delitem(
        sys.modules, "aiperf.rust_shims.live_streaming_worker", raising=False
    )
    monkeypatch.delitem(sys.modules, "aiperf.rust_shims.slurm.generate", raising=False)

    importlib.reload(importlib.import_module("aiperf.rust_shims"))

    assert "aiperf.rust_shims.live_streaming_worker" not in sys.modules
    assert "aiperf.rust_shims.slurm.generate" not in sys.modules
