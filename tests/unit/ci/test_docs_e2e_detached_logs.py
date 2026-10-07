# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A detached server's logs must still reach the job log.

``docker run -d`` returns as soon as the container is created, so the setup
process -- the only handle the setup log monitor can read -- exits having
printed a container id and nothing else. Seven documented server groups start
that way. Without a follower attached, an engine that dies on boot (CUDA OOM,
a rejected flag, a missing weight file) surfaces only as "health check failed,
return code 1" with no diagnostic output at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests/ci/test_docs_end_to_end"))

from test_runner import EndToEndTestRunner  # noqa: E402


def _runner_with(monkeypatch, running: set[str]) -> tuple:
    runner = EndToEndTestRunner()
    followed: list[tuple[str, str]] = []
    monkeypatch.setattr(runner, "_known_container_ids", lambda: running)
    monkeypatch.setattr(
        runner,
        "_stream_container_logs",
        lambda cid, name: followed.append((cid, name)),
    )
    return runner, followed


def test_containers_started_by_setup_are_followed(monkeypatch) -> None:
    runner, followed = _runner_with(monkeypatch, {"aiperf", "vllm", "otel"})
    for thread in runner._follow_detached_containers({"aiperf"}, "otel-mlflow-openai"):
        thread.join(timeout=5)

    assert sorted(cid for cid, _ in followed) == ["otel", "vllm"]
    assert {name for _, name in followed} == {"otel-mlflow-openai"}


def test_containers_predating_setup_are_not_followed(monkeypatch) -> None:
    """The AIPerf container itself is already running; it is not the server."""
    runner, followed = _runner_with(monkeypatch, {"aiperf"})
    started = runner._follow_detached_containers({"aiperf"}, "vllm-default-openai")

    assert started == []
    assert followed == []
