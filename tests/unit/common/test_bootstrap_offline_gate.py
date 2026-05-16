# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HF offline-mode gate in bootstrap.py: controller-pod containers opt out;
worker pods and local mode default to offline."""

from __future__ import annotations

import multiprocessing
from unittest.mock import patch

import pytest

from aiperf.common import bootstrap


def _run_gate(env_vars: dict[str, str], parent_present: bool) -> dict[str, str]:
    """Drive _configure_child_process with a stubbed env + parent presence."""
    captured = {}

    def fake_signal(*_a, **_kw) -> None:
        return None

    with (
        patch.object(
            multiprocessing,
            "parent_process",
            return_value=object() if parent_present else None,
        ),
        patch("signal.signal", side_effect=fake_signal),
        patch.dict("os.environ", env_vars, clear=False),
    ):
        bootstrap._configure_child_process()
        captured["HF_HUB_OFFLINE"] = __import__("os").environ.get("HF_HUB_OFFLINE", "")
        captured["TRANSFORMERS_OFFLINE"] = __import__("os").environ.get(
            "TRANSFORMERS_OFFLINE", ""
        )
    return captured


@pytest.fixture(autouse=True)
def _scrub_offline_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    yield


def test_local_mode_enables_offline() -> None:
    captured = _run_gate({}, parent_present=True)
    assert captured["HF_HUB_OFFLINE"] == "1"
    assert captured["TRANSFORMERS_OFFLINE"] == "1"


def test_worker_pod_enables_offline() -> None:
    # Worker pod: AIPERF_JOB_ID set, AIPERF_CONTROLLER_POD unset.
    captured = _run_gate({"AIPERF_JOB_ID": "job-7"}, parent_present=True)
    assert captured["HF_HUB_OFFLINE"] == "1"
    assert captured["TRANSFORMERS_OFFLINE"] == "1"


def test_controller_pod_skips_offline() -> None:
    captured = _run_gate(
        {"AIPERF_JOB_ID": "job-7", "AIPERF_CONTROLLER_POD": "1"},
        parent_present=True,
    )
    assert captured["HF_HUB_OFFLINE"] == ""
    assert captured["TRANSFORMERS_OFFLINE"] == ""


def test_main_process_skips_signal_and_gate() -> None:
    captured = _run_gate({"AIPERF_CONTROLLER_POD": "1"}, parent_present=False)
    assert captured["HF_HUB_OFFLINE"] == ""
    assert captured["TRANSFORMERS_OFFLINE"] == ""
