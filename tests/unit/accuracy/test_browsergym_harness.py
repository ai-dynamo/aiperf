# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-release AgentLab/BrowserGym adapter tests."""

from __future__ import annotations

import os
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("agentlab", reason="requires BrowserGym agentic worker lock")
pytest.importorskip("browsergym", reason="requires BrowserGym agentic worker lock")

from agentlab.llm.llm_utils import Discussion, HumanMessage, SystemMessage
from bgym import Benchmark

from aiperf.accuracy.agentic import (
    AgenticEpisode,
    AgenticModelResult,
    EventQueue,
)
from aiperf.accuracy.browsergym import (
    AIPerfAgentLabChatModel,
    BrowserGymHarness,
    _canonical_sequential_env_args,
    _convert_summary,
    _parse_dataset,
    _start_daemon_thread,
    _validate_config,
)
from aiperf.accuracy.model_broker import ModelCallBroker


@pytest.mark.asyncio
async def test_agentlab_model_round_trips_multimodal_messages_through_rust() -> None:
    events = EventQueue()
    broker = ModelCallBroker(events)
    model = AIPerfAgentLabChatModel(
        broker=broker,
        episode_id="browser-episode",
        target_model="target-model",
        max_tokens=512,
        temperature=0.1,
    )
    messages = Discussion(
        [
            SystemMessage("Use the canonical browser action syntax."),
            HumanMessage(
                [
                    {"type": "text", "text": "Inspect the screenshot."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA=="},
                    },
                ]
            ),
        ]
    )
    pending = _start_daemon_thread(
        lambda: model(messages), name="test-agentlab-callback"
    )
    event = (await events.poll(1, 1_000))[0]
    assert event.model_call is not None
    call = event.model_call
    assert call.model == "target-model"
    assert call.prompt == "Inspect the screenshot."
    assert call.messages[1]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AA=="},
    }
    broker.submit(
        AgenticModelResult(
            episode_id="browser-episode",
            call_id=call.call_id,
            status="completed",
            response="<action>click('7')</action>",
            reasoning="button seven",
            prompt_tokens=31,
            completion_tokens=9,
            cached_tokens=4,
            response_id="response-7",
            finish_reason="stop",
            error_kind=None,
            error_message=None,
        )
    )
    response = await pending
    assert response == {
        "role": "assistant",
        "content": "<action>click('7')</action>",
        "log_probs": None,
    }
    assert broker.model_call_count("browser-episode") == 1
    broker.close()


@pytest.mark.asyncio
async def test_real_registry_freezes_one_pinned_miniwob_episode(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(Benchmark, "prepare_backends", lambda _self: None)
    harness = await BrowserGymHarness.create(
        "browsergym/miniwob@0.14.3",
        "target-model",
        {
            "environment": "browsergym",
            "output_dir": tmp_path.as_posix(),
            "max_episodes": 1,
            "task_concurrency": 1,
        },
    )
    try:
        identity = harness.identity
        assert identity["harness"] == "agentlab-browsergym"
        assert identity["harness_version"] == "agentlab-0.4.2+browsergym-0.14.3"
        assert len(identity["harness_source_sha256"]) == 64
        assert identity["dataset"]["benchmark"] == "browsergym/miniwob@0.14.3"
        assert identity["dataset"]["revision"].startswith("sha256:")
        assert identity["episode_count"] == 1
        assert harness.episodes[0].source == "browsergym/miniwob@0.14.3"
    finally:
        await harness.close()


def test_sequential_order_respects_browsergym_dependency_graph() -> None:
    env_args = [
        SimpleNamespace(task_name="child"),
        SimpleNamespace(task_name="independent"),
        SimpleNamespace(task_name="parent"),
    ]
    benchmark = SimpleNamespace(
        env_args_list=env_args,
        dependency_graph_over_tasks=lambda: {
            "child": ["parent"],
            "independent": [],
            "parent": [],
        },
    )
    ordered = _canonical_sequential_env_args(benchmark)
    assert [item.task_name for item in ordered] == [
        "independent",
        "parent",
        "child",
    ]


def test_browsergym_config_and_release_are_strict() -> None:
    config = _validate_config(
        {
            "environment": "browsergym",
            "task_concurrency": 1,
            "primary_reward": "reward",
        }
    )
    assert config["max_tokens"] == 4_096
    assert _parse_dataset("browsergym/webarena_verified") == (
        "browsergym/webarena_verified@0.14.3",
        "webarena_verified",
    )
    with pytest.raises(ValueError, match="task_concurrency=1"):
        _validate_config({"environment": "browsergym", "task_concurrency": 2})
    with pytest.raises(ValueError, match="revision must be '0.14.3'"):
        _parse_dataset("browsergym/miniwob@latest")


def test_summary_distinguishes_model_score_from_infrastructure_error() -> None:
    episode = AgenticEpisode("episode", "miniwob.click-test[seed=42]", "fixture")
    completed = _convert_summary(
        episode,
        {
            "cum_reward": 0.0,
            "cum_raw_reward": 0.0,
            "err_msg": None,
            "terminated": True,
            "truncated": False,
        },
        duration_seconds=1.0,
        model_calls=2,
        primary_reward="reward",
        artifact_path="artifact",
    )
    assert completed.outcome == "completed"
    assert completed.rewards == {"reward": 0.0, "raw_reward": 0.0}
    failed = _convert_summary(
        episode,
        {
            "cum_reward": 0.0,
            "err_msg": "Playwright browser failed to start",
        },
        duration_seconds=1.0,
        model_calls=0,
        primary_reward="reward",
        artifact_path="artifact",
    )
    assert failed.outcome == "infrastructure_error"
    assert failed.rewards == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_miniwob_environment_and_canonical_reward(tmp_path) -> None:
    if os.getenv("AIPERF_RUN_BROWSERGYM_CANARY") != "1":
        pytest.skip("set AIPERF_RUN_BROWSERGYM_CANARY=1 for the real browser canary")
    harness = await BrowserGymHarness.create(
        "browsergym/miniwob@0.14.3",
        "fixture-model",
        {
            "environment": "browsergym",
            "output_dir": tmp_path.as_posix(),
            "task_names": ["miniwob.click-test"],
            "max_episodes": 1,
            "task_concurrency": 1,
        },
    )
    episode = harness.episodes[0]
    terminal = None
    try:
        await harness.start_episodes([episode.episode_id])
        while terminal is None:
            for event in await harness.poll_events(10, 5_000):
                if event.model_call is not None:
                    match = re.search(
                        r"^\s*\[(\d+)\].*button",
                        event.model_call.prompt,
                        re.MULTILINE | re.IGNORECASE,
                    )
                    assert match is not None
                    await harness.submit_model_results(
                        [
                            AgenticModelResult(
                                episode_id=episode.episode_id,
                                call_id=event.model_call.call_id,
                                status="completed",
                                response=(
                                    "I'm clicking the button as requested.\n"
                                    f"<action>\nclick('{match.group(1)}')\n</action>"
                                ),
                                reasoning=None,
                                prompt_tokens=100,
                                completion_tokens=20,
                                cached_tokens=0,
                                response_id="miniwob-canary",
                                finish_reason="stop",
                                error_kind=None,
                                error_message=None,
                            )
                        ]
                    )
                elif event.episode_result is not None:
                    terminal = event.episode_result
        assert terminal.outcome == "completed"
        assert terminal.rewards["reward"] == 1.0
        assert (await harness.finish()) == [terminal]
    finally:
        await harness.close()
