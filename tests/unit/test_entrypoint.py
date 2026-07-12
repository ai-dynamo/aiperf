# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contract for the low-overhead AIPerf process entry point."""

from __future__ import annotations

import sys

import pytest

from aiperf import entrypoint


def test_dynosim_run_forwards_raw_arguments_without_loading_general_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authored = ["trace with spaces.jsonl", "--replay-mode=online", "--x", "a,b"]
    observed: list[list[str]] = []

    def canonical_main(arguments: list[str]) -> int:
        observed.append(arguments)
        return 0

    monkeypatch.setattr(
        entrypoint,
        "_import_dynamo_symbol",
        lambda module, symbol: canonical_main
        if (module, symbol) == ("dynamo.replay.main", "main")
        else pytest.fail(f"unexpected import {(module, symbol)}"),
    )

    assert entrypoint.main(["dynosim", "run", *authored]) == 0
    assert observed == [authored]
    assert "aiperf.cli" not in sys.modules


def test_dynosim_mocker_restores_process_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authored = ["--engine-type", "trtllm", "--event-plane=zmq"]
    previous = ["pytest", "sentinel"]
    observed: list[list[str]] = []
    monkeypatch.setattr(sys, "argv", previous)

    def canonical_main() -> None:
        observed.append(list(sys.argv))

    monkeypatch.setattr(
        entrypoint,
        "_import_dynamo_symbol",
        lambda module, symbol: canonical_main
        if (module, symbol) == ("dynamo.mocker.main", "main")
        else pytest.fail(f"unexpected import {(module, symbol)}"),
    )

    assert entrypoint.main(["dynosim", "mocker", *authored]) is None
    assert observed == [["aiperf dynosim mocker", *authored]]
    assert sys.argv is previous


@pytest.mark.parametrize(
    ("operation", "module"),
    [("run", "dynamo.replay"), ("mocker", "dynamo.mocker")],
)
def test_real_cli_replaces_shim_with_canonical_dynamo_process(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    module: str,
) -> None:
    authored = ["trace with spaces.jsonl", "--flag=a,b"]
    observed: list[tuple[str, list[str]]] = []

    class ExecCalled(Exception):
        pass

    def fake_execv(executable: str, arguments: list[str]) -> None:
        observed.append((executable, arguments))
        raise ExecCalled

    monkeypatch.setattr(entrypoint.os, "execv", fake_execv)
    monkeypatch.setattr(
        sys,
        "argv",
        ["aiperf", "dynosim", operation, *authored],
    )

    with pytest.raises(ExecCalled):
        entrypoint.main()

    assert observed == [
        (
            sys.executable,
            [sys.executable, "-m", module, *authored],
        )
    ]
