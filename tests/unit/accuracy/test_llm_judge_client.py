# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import pytest
from pydantic import SecretStr

from aiperf.accuracy.graders._llm_judge import (
    JudgeError,
    LLMJudgeClient,
    backoff_seconds,
)
from aiperf.common.environment import Environment
from tests.harness.fake_judge import FakeJudge, JudgeReply

SECRET = "sk-super-secret-value"


@pytest.fixture
def configure_judge(monkeypatch):
    def _configure(url: str, **overrides: object) -> None:
        settings = {"JUDGE_URL": url, "JUDGE_MODEL": "judge-model", **overrides}
        for name, value in settings.items():
            monkeypatch.setattr(Environment.ACCURACY, name, value)

    return _configure


@pytest.mark.asyncio
class TestLLMJudgeClient:
    async def test_complete_sends_bearer_and_returns_content(
        self, configure_judge
    ) -> None:
        async with FakeJudge([JudgeReply(content="hello")]) as judge:
            configure_judge(judge.url, JUDGE_API_KEY=SecretStr(SECRET))
            client = LLMJudgeClient()
            try:
                assert await client.complete("sys", "user") == "hello"
            finally:
                await client.aclose()

        assert judge.headers[0]["Authorization"] == f"Bearer {SECRET}"
        body = judge.requests[0]
        assert body["model"] == "judge-model"
        assert body["temperature"] == 0
        assert body["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "user"},
        ]

    async def test_complete_without_api_key_sends_no_authorization(
        self, configure_judge
    ) -> None:
        async with FakeJudge([JudgeReply(content="x")]) as judge:
            configure_judge(judge.url, JUDGE_API_KEY=None)
            client = LLMJudgeClient()
            await client.complete("s", "u")
            await client.aclose()

        assert "Authorization" not in judge.headers[0]

    @pytest.mark.parametrize(
        "status,retryable",
        [
            pytest.param(500, True, id="server-error"),
            pytest.param(429, True, id="rate-limited"),
            pytest.param(401, False, id="unauthorized"),
            pytest.param(400, False, id="bad-request"),
        ],
    )  # fmt: skip
    async def test_complete_non_200_raises_judge_error(
        self, configure_judge, status: int, retryable: bool
    ) -> None:
        async with FakeJudge([JudgeReply(content="x", status=status)]) as judge:
            configure_judge(judge.url)
            client = LLMJudgeClient()
            with pytest.raises(JudgeError, match=f"HTTP {status}") as exc_info:
                await client.complete("s", "u")
            await client.aclose()

        assert exc_info.value.retryable is retryable

    @pytest.mark.parametrize(
        "raw_body",
        [
            pytest.param(b'{"choices": []}', id="empty-choices"),
            pytest.param(b"[1, 2]", id="list-body"),
            pytest.param(b"not json", id="not-json"),
            pytest.param(b'{"choices": [{"text": "x"}]}', id="no-message"),
        ],
    )  # fmt: skip
    async def test_complete_malformed_body_raises_retryable_judge_error(
        self, configure_judge, raw_body: bytes
    ) -> None:
        async with FakeJudge([JudgeReply(raw_body=raw_body)]) as judge:
            configure_judge(judge.url)
            client = LLMJudgeClient()
            with pytest.raises(JudgeError, match="malformed") as exc_info:
                await client.complete("s", "u")
            await client.aclose()

        assert exc_info.value.retryable is True

    async def test_complete_times_out(self, configure_judge) -> None:
        judge = FakeJudge([JudgeReply(content="late")], gate=asyncio.Event())
        async with judge:
            configure_judge(judge.url, JUDGE_TIMEOUT=0.2)
            client = LLMJudgeClient()
            with pytest.raises(asyncio.TimeoutError):
                await client.complete("s", "u")
            judge.gate.set()
            await client.aclose()

    async def test_semaphore_caps_in_flight_requests(self, configure_judge) -> None:
        judge = FakeJudge([JudgeReply(content="ok")], gate=asyncio.Event())
        async with judge:
            configure_judge(judge.url, JUDGE_CONCURRENCY=3)
            client = LLMJudgeClient()
            tasks = [
                asyncio.create_task(client.complete("s", str(i))) for i in range(10)
            ]
            await asyncio.wait_for(_spin_until(lambda: judge.peak_in_flight >= 3), 5)
            judge.gate.set()
            results = await asyncio.gather(*tasks)
            await client.aclose()

        assert results == ["ok"] * 10
        assert judge.peak_in_flight == 3

    async def test_aclose_without_session_is_noop(self, configure_judge) -> None:
        configure_judge("http://127.0.0.1:1/v1")
        client = LLMJudgeClient()
        await client.aclose()
        await client.aclose()

    async def test_secret_never_in_repr_or_logs(self, configure_judge, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        async with FakeJudge([JudgeReply(content="x", status=500)]) as judge:
            configure_judge(judge.url, JUDGE_API_KEY=SecretStr(SECRET))
            client = LLMJudgeClient()
            with pytest.raises(JudgeError) as exc_info:
                await client.complete("s", "u")
            await client.aclose()

        assert SECRET not in repr(client)
        assert SECRET not in str(exc_info.value)
        assert SECRET not in caplog.text
        assert SECRET not in repr(Environment.ACCURACY.JUDGE_API_KEY)


def test_llm_judge_client_unconfigured_raises_value_error(monkeypatch) -> None:
    monkeypatch.setattr(Environment.ACCURACY, "JUDGE_URL", None)
    with pytest.raises(ValueError, match="AIPERF_ACCURACY_JUDGE_URL"):
        LLMJudgeClient()


@pytest.mark.parametrize("attempt", [0, 1, 5, 20])
def test_backoff_seconds_capped_by_judge_timeout(monkeypatch, attempt: int) -> None:
    monkeypatch.setattr(Environment.ACCURACY, "JUDGE_TIMEOUT", 3.0)
    delay = backoff_seconds(attempt)
    assert 0 < delay <= 3.0


async def _spin_until(predicate: Callable[[], bool]) -> None:
    while not predicate():
        await asyncio.sleep(0)
