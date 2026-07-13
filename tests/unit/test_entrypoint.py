# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavioral contract for the AIPerf process entry point."""

from __future__ import annotations

import pytest

from aiperf import entrypoint


def test_main_forwards_explicit_arguments_to_cli_app(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aiperf.cli

    observed: list[object] = []

    def fake_app(arguments: object = None) -> int:
        observed.append(arguments)
        return 0

    monkeypatch.setattr(aiperf.cli, "app", fake_app)

    assert entrypoint.main(["profile", "--config", "bench.yaml"]) == 0
    assert observed == [["profile", "--config", "bench.yaml"]]


def test_main_delegates_argv_to_cli_app_when_no_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aiperf.cli

    observed: list[object] = []

    def fake_app(arguments: object = None) -> int:
        observed.append(arguments)
        return 7

    monkeypatch.setattr(aiperf.cli, "app", fake_app)

    # No explicit arguments -> Cyclopts reads sys.argv itself (called with none).
    assert entrypoint.main() == 7
    assert observed == [None]
