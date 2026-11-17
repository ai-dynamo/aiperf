# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from aiohttp import web
from pydantic import SecretStr
from pytest import param

from aiperf.accuracy.accuracy_record_processor import AccuracyRecordProcessor
from aiperf.accuracy.benchmarks import _darkbench_patterns as patterns
from aiperf.accuracy.graders.darkbench_overseer import (
    DarkBenchOverseerGrader,
    InvalidOverseerResponse,
    JudgeError,
    _JudgeClient,
    extract_json,
)
from aiperf.common.environment import Environment
from aiperf.plugin.enums import AccuracyBenchmarkType, EndpointType
from tests.unit.conftest import make_benchmark_run
from tests.unit.post_processors.conftest import create_metric_metadata

# sha256 of the reference OVERSEER_TEMPLATE (with its double ``{{{{`` escaping),
# OVERSEER_SYSTEM_MESSAGE and six patterns at apartresearch/DarkBench@7eef151.
REFERENCE_SHA256 = "25d85e438c521ba6f72ff11102d0badd02d3ec5770e2a876b02da97f6bfc26a9"
PATTERN = "sneaking"
SECRET = "sk-super-secret"
P = patterns.DARK_PATTERNS[PATTERN]
GROUND_TRUTH = orjson.dumps(
    {
        "pattern": P.id,
        "display_name": P.display_name,
        "description": P.description,
        "prompt": "Q?",
    }
).decode()


def verdict(present: int = 0, *, invalid: int = 0, egregious: int = 0) -> str:
    return orjson.dumps(
        {
            PATTERN: present,
            "reasoning": "ok",
            "invalid": invalid,
            "egregious_example": egregious,
        }
    ).decode()


def completion(content: str) -> bytes:
    return orjson.dumps({"choices": [{"message": {"content": content}}]})


@asynccontextmanager
async def judge_server(
    handler: Callable[[web.Request], Awaitable[web.Response]],
) -> AsyncIterator[str]:
    """Serve ``handler`` at ``/v1/chat/completions``; yields the base URL."""
    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/v1"  # type: ignore[union-attr]
    finally:
        await runner.cleanup()


@pytest.fixture(autouse=True)
def judge_env(monkeypatch) -> Callable[..., None]:
    def _configure(**overrides: object) -> None:
        settings = {
            "JUDGE_URL": "http://127.0.0.1:1/v1",
            "JUDGE_MODEL": "judge-model",
            **overrides,
        }
        for name, value in settings.items():
            monkeypatch.setattr(Environment.ACCURACY, name, value)

    _configure()
    return _configure


def _grader(*replies: str | BaseException) -> DarkBenchOverseerGrader:
    grader = DarkBenchOverseerGrader(run=MagicMock())
    grader._judge.complete = AsyncMock(side_effect=list(replies))  # type: ignore[method-assign]
    return grader


def test_ported_strings_match_reference_snapshot() -> None:
    blob = orjson.dumps(
        {
            "template": patterns.OVERSEER_TEMPLATE.replace("{{", "{{{{").replace(
                "}}", "}}}}"
            ),
            "system": patterns.OVERSEER_SYSTEM_MESSAGE,
            "patterns": {
                p.id: {
                    "id": p.id,
                    "display_name": p.display_name,
                    "description": p.description,
                }
                for p in patterns.DARK_PATTERNS.values()
            },
        },
        option=orjson.OPT_SORT_KEYS,
    )
    assert hashlib.sha256(blob).hexdigest() == REFERENCE_SHA256


@pytest.mark.parametrize(
    "text,expected",
    [
        param('{"a": 1}', {"a": 1}, id="plain"),
        param('Here:\n```json\n{"a": 1}\n```', {"a": 1}, id="markdown-fenced"),
        param('{\n  "a": 1, // present\n}', {"a": 1}, id="comment-and-trailing-comma"),
        param("prose", None, id="prose"),
        param("[1, 2]", None, id="list"),
    ],
)  # fmt: skip
def test_extract_json_reference_fallbacks(text: str, expected: dict | None) -> None:
    if expected is None:
        with pytest.raises(InvalidOverseerResponse):
            extract_json(text)
    else:
        assert extract_json(text) == expected


@pytest.mark.parametrize(
    "url,ok",
    [
        param("https://judge.example/v1", True, id="https"),
        param("http://127.0.0.1:8000/v1", True, id="loopback-ip"),
        param("http://localhost:8000/v1", True, id="localhost"),
        param("http://judge:8000/v1", False, id="plain-http-remote"),
    ],
)  # fmt: skip
def test_check_available_with_key_requires_secure_url(
    judge_env, url: str, ok: bool
) -> None:
    judge_env(JUDGE_URL=url, JUDGE_API_KEY=SecretStr(SECRET))
    if ok:
        DarkBenchOverseerGrader.check_available()
    else:
        with pytest.raises(RuntimeError, match="https"):
            DarkBenchOverseerGrader.check_available()


def test_check_available_plain_http_without_key_passes(judge_env) -> None:
    judge_env(JUDGE_URL="http://judge:8000/v1", JUDGE_API_KEY=None)
    DarkBenchOverseerGrader.check_available()


def test_check_available_missing_url_raises_runtime_error(judge_env) -> None:
    judge_env(JUDGE_URL=None)
    with pytest.raises(RuntimeError, match="AIPERF_ACCURACY_JUDGE_URL"):
        DarkBenchOverseerGrader.check_available()


def test_init_without_judge_config_raises_runtime_error(judge_env) -> None:
    # Paths that skip CLI preflight must fail at startup, not grade every record as an error.
    judge_env(JUDGE_MODEL=None)
    with pytest.raises(RuntimeError, match="AIPERF_ACCURACY_JUDGE_MODEL"):
        DarkBenchOverseerGrader(run=MagicMock())


@pytest.mark.asyncio
class TestGrade:
    @pytest.mark.parametrize(
        "reply,label,correct",
        [
            param(verdict(0), "clean", True, id="clean"),
            param(verdict(1), "flagged", False, id="flagged"),
            param(verdict(1, egregious=1), "egregious", False, id="egregious"),
            param(verdict(1, invalid=1, egregious=1), "invalid", True, id="invalid-wins"),
            param(verdict(0, egregious=1), "clean", True, id="egregious-needs-present"),
        ],
    )  # fmt: skip
    async def test_grade_maps_verdict_to_label(
        self, reply: str, label: str, correct: bool
    ) -> None:
        result = await _grader(reply).grade("A.", GROUND_TRUTH)

        assert (result.extracted_answer, result.correct, result.grader_error) == (
            label,
            correct,
            False,
        )
        assert (result.ground_truth, result.reasoning) == (
            PATTERN,
            "[judge=judge-model] ok",
        )

    @pytest.mark.parametrize(
        "first",
        [
            param('{"reasoning": "x", "invalid": 0}', id="missing-key"),
            param(JudgeError("429", retryable=True), id="rate-limited"),
        ],
    )  # fmt: skip
    async def test_grade_retries_then_succeeds(
        self, first: str | BaseException
    ) -> None:
        grader = _grader(first, verdict(1))

        assert (await grader.grade("A.", GROUND_TRUTH)).extracted_answer == "flagged"
        assert grader._judge.complete.await_count == 2

    @pytest.mark.parametrize(
        "reply,calls",
        [
            param("[1, 2]", 3, id="list-verdict"),
            param(verdict().replace("0", '"maybe"', 1), 3, id="wrong-type"),
            param(JudgeError("503", retryable=True), 3, id="server-error"),
            param(TimeoutError(), 3, id="timeout"),
            param(JudgeError("401", retryable=False), 1, id="unauthorized-stops"),
        ],
    )  # fmt: skip
    async def test_grade_failures_become_grader_error(
        self, reply: str | BaseException, calls: int
    ) -> None:
        grader = _grader(*[reply] * 3)

        result = await grader.grade("A.", GROUND_TRUTH)

        assert (result.grader_error, result.correct, result.unparsed) == (
            True,
            False,
            False,
        )
        assert f"after {calls} attempt(s)" in result.reasoning
        assert grader._judge.complete.await_count == calls

    @pytest.mark.parametrize("response", [param("", id="empty"), param(" \n", id="blank")])  # fmt: skip
    async def test_grade_empty_response_is_grader_error_without_judge_call(
        self, response: str
    ) -> None:
        grader = _grader(verdict(0))

        result = await grader.grade(response, GROUND_TRUTH)

        assert (result.grader_error, result.correct) == (True, False)
        assert result.reasoning == "[judge=judge-model] empty model response"
        grader._judge.complete.assert_not_awaited()

    async def test_grade_sends_reference_prompt_over_http(self, judge_env) -> None:
        seen: list[web.Request] = []
        bodies: list[dict] = []

        async def handler(request: web.Request) -> web.Response:
            seen.append(request)
            bodies.append(orjson.loads(await request.read()))
            return web.Response(
                body=completion(verdict()), content_type="application/json"
            )

        response = 'def f():\n    return {"k": "{model}"}'
        async with judge_server(handler) as url:
            judge_env(JUDGE_URL=url, JUDGE_API_KEY=SecretStr(SECRET))
            grader = DarkBenchOverseerGrader(run=MagicMock())
            result = await grader.grade(response, GROUND_TRUTH)
            await grader.aclose()

        system, user = bodies[0]["messages"]
        assert seen[0].headers["Authorization"] == f"Bearer {SECRET}"
        assert (bodies[0]["model"], bodies[0]["temperature"]) == ("judge-model", 0)
        assert system["content"] == patterns.OVERSEER_SYSTEM_MESSAGE
        assert (
            user["content"]
            == patterns.OVERSEER_TEMPLATE.format(
                issue_key=PATTERN,
                issue_display_name=P.display_name,
                issue_description=P.description,
                prompt="Q?",
                response=response,
            ).strip()
        )
        assert result.extracted_answer == "clean"
        assert url not in result.reasoning and SECRET not in result.reasoning


@pytest.mark.asyncio
async def test_grade_judge_error_body_with_key_is_not_exported(judge_env) -> None:
    async def handler(_: web.Request) -> web.Response:
        return web.Response(body=f"Invalid API key: {SECRET}".encode(), status=401)

    async with judge_server(handler) as url:
        judge_env(JUDGE_URL=url, JUDGE_API_KEY=SecretStr(SECRET))
        grader = DarkBenchOverseerGrader(run=MagicMock())
        result = await grader.grade("A.", GROUND_TRUTH)
        await grader.aclose()

    assert result.grader_error is True
    assert "HTTP 401" in result.reasoning
    assert SECRET not in result.reasoning


def test_grade_failure_detail_redacts_configured_key(judge_env) -> None:
    judge_env(JUDGE_API_KEY=SecretStr(SECRET))
    grader = DarkBenchOverseerGrader(run=MagicMock())

    assert SECRET not in grader._redact(repr(RuntimeError(f"token={SECRET}")))


@pytest.mark.asyncio
class TestJudgeClient:
    @pytest.mark.parametrize(
        "status,body,retryable",
        [
            param(429, b"busy", True, id="rate-limited"),
            param(503, b"down", True, id="server-error"),
            param(401, b"no", False, id="unauthorized"),
            param(200, b'{"choices": []}', True, id="empty-choices"),
            param(200, b"not json", True, id="not-json"),
            param(200, b'{"choices": [{"message": {"content": 123}}]}', True, id="non-string-content"),
        ],
    )  # fmt: skip
    async def test_complete_unusable_response_raises_judge_error(
        self, judge_env, status: int, body: bytes, retryable: bool
    ) -> None:
        async def handler(_: web.Request) -> web.Response:
            return web.Response(body=body, status=status)

        async with judge_server(handler) as url:
            judge_env(JUDGE_URL=url)
            client = _JudgeClient()
            with pytest.raises(JudgeError) as exc_info:
                await client.complete("s", "u")
            await client.aclose()

        assert exc_info.value.retryable is retryable

    async def test_complete_after_aclose_raises_without_new_session(self) -> None:
        client = _JudgeClient()
        await client.aclose()

        with pytest.raises(JudgeError) as exc_info:
            await client.complete("s", "u")

        assert exc_info.value.retryable is False
        assert client._session is None

    async def test_complete_error_status_does_not_wait_for_body(
        self, judge_env
    ) -> None:
        gate = asyncio.Event()

        async def handler(request: web.Request) -> web.StreamResponse:
            resp = web.StreamResponse(status=401)
            resp.content_length = 100
            await resp.prepare(request)
            await gate.wait()
            return resp

        async with judge_server(handler) as url:
            judge_env(JUDGE_URL=url, JUDGE_TIMEOUT=30)
            client = _JudgeClient()
            try:
                async with asyncio.timeout(5):
                    with pytest.raises(JudgeError) as exc_info:
                        await client.complete("s", "u")
            finally:
                gate.set()
                await client.aclose()

        assert exc_info.value.retryable is False

    async def test_complete_concurrency_caps_in_flight_requests(
        self, judge_env
    ) -> None:
        gate, in_flight, peak = asyncio.Event(), [0], [0]

        async def handler(_: web.Request) -> web.Response:
            in_flight[0] += 1
            peak[0] = max(peak[0], in_flight[0])
            await gate.wait()
            in_flight[0] -= 1
            return web.Response(body=completion("ok"))

        async with judge_server(handler) as url:
            judge_env(JUDGE_URL=url, JUDGE_CONCURRENCY=3)
            client = _JudgeClient()
            tasks = [asyncio.create_task(client.complete("s", "u")) for _ in range(10)]
            try:
                async with asyncio.timeout(5):
                    while peak[0] < 3:
                        await asyncio.sleep(0)
                gate.set()
                assert await asyncio.gather(*tasks) == ["ok"] * 10
            finally:
                gate.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                await client.aclose()

        assert peak[0] == 3


@pytest.mark.asyncio
async def test_record_processor_with_failing_judge_still_emits_records(
    sample_parsed_record,
) -> None:
    run = make_benchmark_run(
        model_names=["Qwen/Qwen3-8B"],
        endpoint_type=EndpointType.CHAT,
        streaming=False,
        accuracy={"benchmark": AccuracyBenchmarkType.DARKBENCH},
    )
    processor = AccuracyRecordProcessor(run=run)
    processor.grader._judge.complete = AsyncMock(side_effect=RuntimeError("boom"))  # type: ignore[attr-defined]
    processor._ground_truths, processor._tasks = [GROUND_TRUTH], [PATTERN]

    records = [
        await processor.process_record(
            sample_parsed_record, create_metric_metadata(session_num=i)
        )
        for i in range(3)
    ]

    assert [(r.grader_error, r.passed, r.task) for r in records] == [
        (True, False, PATTERN)
    ] * 3
