# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-commit MCPMark Verified adapter tests."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

pytest.importorskip("src.evaluator", reason="requires MCPMark agentic worker lock")

from src.agents import mcpmark_agent as canonical_agent_module
from src.agents.mcpmark_agent import MCPMarkAgent
from src.results_reporter import TaskResult

from aiperf.accuracy.agentic import AgenticEpisode, AgenticModelResult, EventQueue
from aiperf.accuracy.mcpmark import (
    _CANONICAL_COMPACTION_DISABLED,
    _CANONICAL_MAX_TOKENS,
    _CANONICAL_MAX_TURNS,
    _CANONICAL_MCP_SERVER_IDENTITIES,
    _CANONICAL_TIMEOUT_SECONDS,
    _MCPMARK_COMMIT,
    _MCPMARK_SOURCE_SHA256,
    MCPMarkHarness,
    _convert_task_result,
    _model_api_key_variable,
    _parse_dataset,
    _require_mcpmark_environment,
    _rust_litellm_calls,
    _start_daemon_thread,
    _validate_config,
)
from aiperf.accuracy.model_broker import ModelCallBroker
from aiperf.accuracy.worker import AccuracyWorker


def _completed_result(
    *,
    episode_id: str,
    call_id: str,
    response: str,
    assistant_message: dict[str, Any],
    finish_reason: str,
) -> AgenticModelResult:
    return AgenticModelResult(
        episode_id=episode_id,
        call_id=call_id,
        status="completed",
        response=response,
        reasoning=None,
        prompt_tokens=31,
        completion_tokens=7,
        cached_tokens=0,
        response_id=f"response-{call_id}",
        finish_reason=finish_reason,
        error_kind=None,
        error_message=None,
        assistant_message=assistant_message,
    )


@pytest.mark.asyncio
async def test_real_registry_freezes_exact_verified_filesystem_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    environment_root = tmp_path / "environments"
    category_root = environment_root / "file_property"
    category_root.mkdir(parents=True)
    (category_root / "fixture.txt").write_text("canonical environment fixture")
    monkeypatch.setenv("FILESYSTEM_TEST_ROOT", environment_root.as_posix())
    _require_mcpmark_environment()
    harness = await MCPMarkHarness.create(
        f"mcpmark/filesystem/standard@{_MCPMARK_COMMIT}",
        "gpt-4o",
        {
            "environment": "filesystem",
            "output_dir": tmp_path.as_posix(),
            "task_names": ["file_property/size_classification"],
            "max_episodes": 1,
            "task_concurrency": 1,
        },
    )
    try:
        identity = harness.identity
        assert identity["harness"] == "mcpmark-verified"
        assert identity["harness_source_sha256"] == _MCPMARK_SOURCE_SHA256
        assert identity["dataset"]["benchmark"] == (
            f"mcpmark/filesystem@{_MCPMARK_COMMIT}"
        )
        assert identity["dataset"]["revision"].startswith(
            f"git:{_MCPMARK_COMMIT}+selection-sha256:"
        )
        assert "+environment-sha256:" in identity["dataset"]["revision"]
        assert identity["canonical_agent_config"] == {
            "agent_name": "mcpmark",
            "task_suite": "standard",
            "mcp_service": "filesystem",
            "mcp_server": _CANONICAL_MCP_SERVER_IDENTITIES["filesystem"],
            "max_turns": _CANONICAL_MAX_TURNS,
            "max_tokens": _CANONICAL_MAX_TOKENS,
            "temperature": 1.0,
            "compaction_token": _CANONICAL_COMPACTION_DISABLED,
            "enable_summarize": False,
            "parser": "openai_tool_calls",
            "reasoning_effort": "default",
            "timeout_seconds": _CANONICAL_TIMEOUT_SECONDS,
        }
        assert identity["episode_count"] == 1
        assert harness.episodes[0].task == "file_property/size_classification"
        assert harness.episodes[0].source == f"mcpmark/filesystem@{_MCPMARK_COMMIT}"
    finally:
        await harness.close()


def test_dataset_config_and_model_mapping_are_strict(tmp_path) -> None:
    assert _parse_dataset("mcpmark/filesystem") == (
        f"mcpmark/filesystem@{_MCPMARK_COMMIT}",
        "filesystem",
        "standard",
    )
    config = _validate_config(
        {
            "environment": "filesystem",
            "output_dir": tmp_path.as_posix(),
            "task_concurrency": 1,
            "max_turns": _CANONICAL_MAX_TURNS,
        },
        "filesystem",
    )
    assert "max_turns" not in config
    assert _model_api_key_variable("claude-sonnet-4") == "ANTHROPIC_API_KEY"
    assert _model_api_key_variable("locally-served-model") == "OPENAI_API_KEY"
    with pytest.raises(ValueError, match="revision must be"):
        _parse_dataset("mcpmark/filesystem@latest")
    with pytest.raises(ValueError, match="canonical max-turn limit"):
        _validate_config(
            {
                "environment": "filesystem",
                "output_dir": tmp_path.as_posix(),
                "task_concurrency": 1,
                "max_turns": 99,
            },
            "filesystem",
        )
    with pytest.raises(ValueError, match="task_concurrency=1"):
        _validate_config(
            {
                "environment": "filesystem",
                "output_dir": tmp_path.as_posix(),
                "task_concurrency": 2,
            },
            "filesystem",
        )


def test_worker_advertises_exact_mcpmark_provider_and_lock() -> None:
    hello = AccuracyWorker().hello(1)
    assert "agentic" in hello["capabilities"]
    assert "agentic_mcpmark" in hello["capabilities"]
    assert hello["packages"]["MCPMark"] == "0.0.1"
    assert hello["packages"]["litellm"] == "1.80.0"
    assert len(hello["dependency_lock_sha256"]) == 64


class _FakeMcpServer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {
            "content": [{"type": "text", "text": "canonical tool completed"}],
            "isError": False,
        }


@pytest.mark.asyncio
async def test_canonical_agent_tool_loop_round_trips_only_through_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenHttpClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise AssertionError("canonical MCPMark attempted direct model HTTP")

    monkeypatch.setattr(
        canonical_agent_module.httpx, "AsyncClient", ForbiddenHttpClient
    )
    events = EventQueue()
    broker = ModelCallBroker(events)
    episode_id = "mcpmark:test-episode"
    model_name = "target-served-by-rust"
    server = _FakeMcpServer()
    agent = MCPMarkAgent(
        litellm_input_model_name="openai/gpt-4o",
        api_key="not-used",
        base_url=None,
        mcp_service="filesystem",
        timeout=30,
        service_config={},
        reasoning_effort="default",
        compaction_token=_CANONICAL_COMPACTION_DISABLED,
    )
    functions = [
        {
            "name": "write_file",
            "description": "Write a file in the isolated task environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        }
    ]

    with _rust_litellm_calls(
        broker=broker, episode_id=episode_id, model_name=model_name
    ):
        pending = _start_daemon_thread(
            lambda: asyncio.run(
                agent._execute_litellm_tool_loop(  # noqa: SLF001
                    "Create the required file.", functions, server
                )
            ),
            name="test-mcpmark-canonical-loop",
        )
        first_event = (await events.poll(1, 1_000))[0]
        assert first_event.model_call is not None
        first = first_event.model_call
        assert first.model == model_name
        assert first.prompt == "Create the required file."
        assert first.messages == [
            {"role": "system", "content": MCPMarkAgent.SYSTEM_PROMPT},
            {"role": "user", "content": "Create the required file."},
        ]
        assert first.generation == {
            "max_tokens": _CANONICAL_MAX_TOKENS,
            "temperature": 1.0,
            "top_p": 1.0,
            "stop": [],
        }
        assert first.tools == [{"type": "function", "function": functions[0]}]
        assert first.tool_choice == "auto"
        assert first.extra_body == {"enforcer_mode": "on", "think_mode": "on"}
        broker.submit(
            _completed_result(
                episode_id=episode_id,
                call_id=first.call_id,
                response="",
                assistant_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "tool-1",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"path":"/tmp/result","content":"done"}',
                            },
                        }
                    ],
                },
                finish_reason="tool_calls",
            )
        )

        second_event = (await events.poll(1, 1_000))[0]
        assert second_event.model_call is not None
        second = second_event.model_call
        assert second.turn_index == 1
        assert second.messages[-1]["role"] == "tool"
        assert second.messages[-1]["tool_call_id"] == "tool-1"
        assert server.calls == [
            ("write_file", {"path": "/tmp/result", "content": "done"})
        ]
        broker.submit(
            _completed_result(
                episode_id=episode_id,
                call_id=second.call_id,
                response="Task completed",
                assistant_message={
                    "role": "assistant",
                    "content": "Task completed",
                },
                finish_reason="stop",
            )
        )
        result = await pending

    assert result["success"] is True
    assert result["turn_count"] == 2
    assert result["token_usage"] == {
        "input_tokens": 62,
        "output_tokens": 14,
        "total_tokens": 76,
        "reasoning_tokens": 0,
    }
    assert broker.model_call_count(episode_id) == 2
    broker.close()


def test_verifier_score_is_distinct_from_infrastructure_failure() -> None:
    episode = AgenticEpisode("episode", "file_property/size_classification", "fixture")
    failed_verification = TaskResult(
        task_name="file_property__size_classification",
        success=False,
        error_message="Max turns (100) exceeded",
        verification_error="task state did not match",
        verification_output="",
    )
    scored = _convert_task_result(
        episode,
        failed_verification,
        duration_seconds=1.0,
        model_calls=100,
        artifact_path="artifact",
    )
    assert scored.outcome == "completed"
    assert scored.rewards == {"pass": 0.0}

    verifier_crash = TaskResult(
        task_name="file_property__size_classification",
        success=False,
        verification_error="python verifier crashed",
        verification_output=None,
    )
    infrastructure = _convert_task_result(
        episode,
        verifier_crash,
        duration_seconds=1.0,
        model_calls=1,
        artifact_path="artifact",
    )
    assert infrastructure.outcome == "infrastructure_error"
    assert infrastructure.rewards == {}
    assert infrastructure.error_kind == "MCPMarkVerifierError"

    inference_failure = TaskResult(
        task_name="file_property__size_classification",
        success=False,
        error_message="AIPERF_RUST_INFERENCE:transport:connection reset",
        verification_error="task state did not match",
        verification_output="",
    )
    infrastructure = _convert_task_result(
        episode,
        inference_failure,
        duration_seconds=1.0,
        model_calls=3,
        artifact_path="artifact",
    )
    assert infrastructure.outcome == "infrastructure_error"
    assert infrastructure.rewards == {}
    assert infrastructure.error_kind == "RustInferenceError"
