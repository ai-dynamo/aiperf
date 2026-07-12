# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hermetic proof of the stateful worker protocol used by agent harnesses."""

from __future__ import annotations

from typing import Any

import pytest

from aiperf.accuracy import worker as worker_module
from aiperf.accuracy.agentic import (
    AgenticEpisode,
    AgenticEpisodeResult,
    AgenticEvent,
    AgenticHarness,
    AgenticModelCall,
    AgenticModelResult,
    EventQueue,
)
from aiperf.accuracy.worker import AccuracyWorker, _dispatch


class FixtureAgenticHarness(AgenticHarness):
    """Two-episode harness whose environment transition is deterministic."""

    def __init__(self) -> None:
        self._events = EventQueue()
        self._episodes = [
            AgenticEpisode("episode-1", "swebench.task-1", "fixture/swebench"),
            AgenticEpisode("episode-2", "terminal.task-2", "fixture/terminal"),
        ]
        self.started: list[str] = []
        self.submitted: list[AgenticModelResult] = []
        self.results: dict[str, AgenticEpisodeResult] = {}
        self.closed = False

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "harness": "fixture",
            "harness_version": "1",
            "harness_source_sha256": "a" * 64,
            "dataset": {
                "provider": "fixture",
                "benchmark": "fixture/agentic",
                "repository": "fixture/agentic",
                "revision": "b" * 64,
                "evaluation_splits": ["tasks"],
            },
            "agent": "fixture-agent",
            "agent_version": "1",
            "environment": "fixture",
            "verifier": "fixture verifier",
            "episode_count": 2,
            "primary_reward": "reward",
        }

    @property
    def episodes(self) -> list[AgenticEpisode]:
        return list(self._episodes)

    async def start_episodes(self, episode_ids: list[str]) -> None:
        self.started.extend(episode_ids)
        for episode_id in episode_ids:
            await self._events.put(
                AgenticEvent.call(
                    AgenticModelCall(
                        episode_id=episode_id,
                        call_id=f"{episode_id}:call:00000000",
                        turn_index=0,
                        prompt="model-safe instruction",
                        messages=[
                            {"role": "user", "content": "model-safe instruction"}
                        ],
                        generation={
                            "max_tokens": 64,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "stop": [],
                        },
                    )
                )
            )

    async def poll_events(self, limit: int, wait_ms: int) -> list[AgenticEvent]:
        return await self._events.poll(limit, wait_ms)

    async def submit_model_results(self, items: list[AgenticModelResult]) -> None:
        self.submitted.extend(items)
        for item in items:
            result = AgenticEpisodeResult(
                episode_id=item.episode_id,
                task=(
                    "swebench.task-1"
                    if item.episode_id == "episode-1"
                    else "terminal.task-2"
                ),
                outcome="completed",
                rewards={"reward": 1.0 if item.response == "fixed" else 0.0},
                primary_reward="reward",
                duration_seconds=1.25,
                model_calls=1,
                prompt_tokens=item.prompt_tokens,
                completion_tokens=item.completion_tokens,
            )
            self.results[item.episode_id] = result
            await self._events.put(AgenticEvent.completed(result))

    async def cancel_episodes(self, episode_ids: list[str]) -> None:
        for episode_id in episode_ids:
            result = AgenticEpisodeResult(
                episode_id=episode_id,
                task="cancelled",
                outcome="cancelled",
                rewards={},
                primary_reward=None,
                duration_seconds=0.0,
                model_calls=0,
            )
            self.results[episode_id] = result
            await self._events.put(AgenticEvent.completed(result))

    async def finish(self) -> list[AgenticEpisodeResult]:
        return [self.results[episode.episode_id] for episode in self._episodes]

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fixture_harness(monkeypatch: pytest.MonkeyPatch) -> FixtureAgenticHarness:
    harness = FixtureAgenticHarness()

    async def create(_dataset: str, _model: str, _config: Any) -> AgenticHarness:
        return harness

    monkeypatch.setattr(worker_module, "_verify_agentic_environment", lambda: None)
    monkeypatch.setattr(worker_module, "_create_agentic_harness", create)
    return harness


@pytest.mark.asyncio
async def test_agentic_protocol_round_trip_keeps_rust_as_inference_owner(
    fixture_harness: FixtureAgenticHarness,
) -> None:
    worker = AccuracyWorker()
    loaded = await worker.load_agentic(
        {
            "id": 1,
            "op": "load_agentic",
            "dataset": "fixture/agentic@sha256:locked",
            "model": "fixture-model",
            "config": {"task_concurrency": 2},
        }
    )
    assert loaded["episode_count"] == 2
    page = worker.next_episodes(0, 10)
    assert [item["episode_id"] for item in page["items"]] == [
        "episode-1",
        "episode-2",
    ]
    assert "instruction" not in page["items"][0]

    await worker.start_episodes(["episode-1", "episode-2"])
    calls = await worker.poll_agentic(10, 0)
    assert [event["kind"] for event in calls["events"]] == [
        "model_call",
        "model_call",
    ]
    assert calls["events"][0]["call"]["messages"] == [
        {"role": "user", "content": "model-safe instruction"}
    ]

    submitted = [
        {
            "episode_id": event["call"]["episode_id"],
            "call_id": event["call"]["call_id"],
            "status": "completed",
            "response": "fixed" if index == 0 else "not fixed",
            "prompt_tokens": 10,
            "completion_tokens": 2,
        }
        for index, event in enumerate(calls["events"])
    ]
    await worker.submit_model_results(submitted)
    terminal = await worker.poll_agentic(10, 0)
    assert [event["result"]["rewards"]["reward"] for event in terminal["events"]] == [
        1.0,
        0.0,
    ]
    finished = await worker.finish_agentic()
    assert [item["episode_id"] for item in finished["items"]] == [
        "episode-1",
        "episode-2",
    ]
    assert len(fixture_harness.submitted) == 2
    await worker.close()
    assert fixture_harness.closed is True


@pytest.mark.asyncio
async def test_agentic_dispatch_rejects_unknown_fields_before_harness_mutation(
    fixture_harness: FixtureAgenticHarness,
) -> None:
    worker = AccuracyWorker()
    response, _ = await _dispatch(
        worker,
        {
            "id": 1,
            "op": "load_agentic",
            "dataset": "fixture/agentic",
            "model": "fixture",
            "config": {},
        },
    )
    assert response["harness"] == "fixture"
    with pytest.raises(ValueError, match="unknown field"):
        await _dispatch(
            worker,
            {
                "id": 2,
                "op": "start_episodes",
                "episode_ids": ["episode-1"],
                "private_tests": ["must not cross"],
            },
        )
    assert fixture_harness.started == []


def test_model_result_schema_distinguishes_infrastructure_failure() -> None:
    failed = AgenticModelResult.from_wire(
        {
            "episode_id": "episode-1",
            "call_id": "call-1",
            "status": "failed",
            "response": "partial",
            "error_kind": "transport_error",
            "error_message": "connection reset",
        }
    )
    assert failed.status == "failed"
    assert failed.response == "partial"
    with pytest.raises(ValueError, match="requires error_kind"):
        AgenticModelResult.from_wire(
            {
                "episode_id": "episode-1",
                "call_id": "call-1",
                "status": "failed",
                "response": "",
            }
        )
    with pytest.raises(ValueError, match="unknown field"):
        AgenticModelResult.from_wire(
            {
                "episode_id": "episode-1",
                "call_id": "call-1",
                "status": "completed",
                "response": "ok",
                "ground_truth": "secret",
            }
        )
