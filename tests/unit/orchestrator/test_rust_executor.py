# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Rust-runner capability negotiation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import orjson
import pytest

from aiperf.orchestrator import rust_executor


def _completed(payload: object, *, returncode: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["aiperf-runner", "--capabilities"],
        returncode=returncode,
        stdout=orjson.dumps(payload) + b"\n",
        stderr=b"runner diagnostic" if returncode else b"",
    )


def test_capabilities_accept_matching_protocol_and_report_schema(monkeypatch) -> None:
    response = {
        "event": "runner_capabilities",
        "protocol_versions": [1],
        "report_schema_version": "2.0",
        "dataset_types": ["synthetic"],
        "phase_types": ["concurrency"],
        "runner_version": "0.0.0",
    }
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(response))

    assert rust_executor._load_capabilities(Path("runner")) == response


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("protocol_versions", [2], "does not support protocol 1"),
        ("report_schema_version", "3.0", "report schema '3.0' is incompatible"),
        ("event", "something_else", "unknown capability response"),
    ],
)
def test_capabilities_reject_incompatible_runner(
    monkeypatch, field: str, value: object, match: str
) -> None:
    response = {
        "event": "runner_capabilities",
        "protocol_versions": [1],
        "report_schema_version": "2.0",
    }
    response[field] = value
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: _completed(response))

    with pytest.raises((RuntimeError, ValueError), match=match):
        rust_executor._load_capabilities(Path("runner"))


def test_capabilities_surface_process_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: _completed({}, returncode=2),
    )

    with pytest.raises(RuntimeError, match="exit 2.*runner diagnostic"):
        rust_executor._load_capabilities(Path("runner"))
