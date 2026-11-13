# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A detached server's logs must still reach the job log.

``docker run -d`` returns as soon as the container is created, so the setup
process -- the only handle the setup log monitor can read -- exits having
printed a container id and nothing else. Without a follower attached, an
engine that dies on boot (CUDA OOM, a rejected flag, a missing weight file)
surfaces only as "health check failed, return code 1" with no diagnostic
output at all.

The mirror case matters just as much: every documented setup block today runs
``docker run`` in the foreground, whose output already arrives through the
setup pipe. Following those as well prints the whole boot twice under one
``SERVER[name]`` prefix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from test_runner import EndToEndTestRunner


def _runner_with(
    monkeypatch, running: set[str], attached: set[str] = frozenset()
) -> EndToEndTestRunner:
    from test_runner import EndToEndTestRunner

    runner = EndToEndTestRunner()
    followed: list[tuple[str, str]] = []
    monkeypatch.setattr(runner, "_known_container_ids", lambda: running)
    monkeypatch.setattr(runner, "_is_attached_to_setup", lambda cid: cid in attached)
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


def test_a_foreground_container_is_not_followed_twice(monkeypatch) -> None:
    """Its output already arrives through the setup process's stdout."""
    runner, followed = _runner_with(monkeypatch, {"aiperf", "vllm"}, attached={"vllm"})
    started = runner._follow_detached_containers({"aiperf"}, "vllm-default-openai")

    assert started == []
    assert followed == []


def test_a_foreground_container_does_not_mask_a_detached_sibling(monkeypatch) -> None:
    """A setup block may start both; only the silent one needs a follower."""
    runner, followed = _runner_with(
        monkeypatch, {"aiperf", "vllm", "otel"}, attached={"vllm"}
    )
    for thread in runner._follow_detached_containers({"aiperf"}, "otel-mlflow-openai"):
        thread.join(timeout=5)

    assert [cid for cid, _ in followed] == ["otel"]


def test_an_unanswerable_inspect_still_follows(monkeypatch) -> None:
    """A duplicated log beats a missing boot traceback, so docker failure follows."""
    from test_runner import EndToEndTestRunner

    runner = EndToEndTestRunner()
    monkeypatch.setattr(
        "test_runner.subprocess.run",
        lambda *a, **k: _Failed(),
    )
    assert runner._is_attached_to_setup("deadbeef") is False


class _Failed:
    returncode = 1
    stdout = ""
    stderr = "No such object"
