# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``aiperf.cli_commands.chat`` helpers -- URL resolution and
per-turn message assembly. The streaming/REPL I/O is covered by the
integration tests; the stats/metric logic lives in ``test_chat_stats``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import aiohttp
import orjson
import pytest
from pytest import param

from aiperf.cli_commands import chat as chat_mod
from aiperf.cli_commands.chat import (
    _build_turn_messages,
    _chat_completions_url,
    _consume_stream,
    _read_user_message,
    _run_turn,
    _settle,
)
from aiperf.common.exceptions import SSEResponseError


@pytest.mark.parametrize(
    "url,expected",
    [
        param("http://h:8000", "http://h:8000/v1/chat/completions", id="bare_host"),
        param("http://h:8000/v1", "http://h:8000/v1/chat/completions", id="v1_base"),
        param(
            "http://h:8000/v1/",
            "http://h:8000/v1/chat/completions",
            id="trailing_slash",
        ),
        param(
            "http://h:8000/v1/chat/completions",
            "http://h:8000/v1/chat/completions",
            id="full_path",
        ),
        param(
            "http://h:8000/openai",
            "http://h:8000/openai/v1/chat/completions",
            id="sub_path_base",
        ),
        param(
            "localhost:8000",
            "http://localhost:8000/v1/chat/completions",
            id="schemeless",
        ),
    ],
)  # fmt: skip
def test_chat_completions_url_resolves_expected(url: str, expected: str) -> None:
    assert _chat_completions_url(url) == expected


def test_build_turn_messages_with_history() -> None:
    system = [{"role": "system", "content": "be brief"}]
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _build_turn_messages(system, history, "next") == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "next"},
    ]


def test_build_turn_messages_no_history_is_stateless() -> None:
    # Empty history (the --no-history case) sends only system + new message.
    assert _build_turn_messages([], [], "solo") == [{"role": "user", "content": "solo"}]


class _FakeContent:
    """Minimal stand-in for ``aiohttp`` ``response.content`` (async byte iter)."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class _FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.content = _FakeContent(chunks)


@pytest.mark.asyncio
async def test_consume_stream_parses_content_reasoning_and_usage() -> None:
    resp = _FakeResponse(
        [
            # Opening role-only delta carries no content/reasoning -> skipped.
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"reasoning_content":"hmm"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":5}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )
    responses, output_parts, reasoning_parts, last_usage = await _consume_stream(resp)  # type: ignore[arg-type]
    assert output_parts == ["hi"]
    assert reasoning_parts == ["hmm"]
    assert last_usage == {"prompt_tokens": 5}
    assert len(responses) == 2  # two content-bearing messages (role-only skipped)


@pytest.mark.asyncio
async def test_consume_stream_surfaces_mid_stream_sse_error() -> None:
    # A 200-OK stream that fails mid-way with an SSE `event: error` frame must
    # surface as an error, not be reported as a truncated/empty reply.
    resp = _FakeResponse([b"event: error\n: upstream exploded\n\n"])
    with pytest.raises(SSEResponseError):
        await _consume_stream(resp)  # type: ignore[arg-type]


class _FakeStreamResponse:
    """Async-context-manager stand-in for the aiohttp response in `_run_turn`."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        url: str = "http://h/v1/chat/completions",
        status: int = 200,
        reason: str = "OK",
        body: bytes = b"",
    ) -> None:
        self.content = _FakeContent(chunks)
        self._url = url
        self.status = status
        self.reason = reason
        self._body = body

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise aiohttp.ClientResponseError(
                SimpleNamespace(real_url=self._url),
                (),
                status=self.status,
                message=self.reason,
            )

    async def text(self) -> str:
        return self._body.decode()

    async def __aenter__(self) -> _FakeStreamResponse:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in returning canned SSE chunks."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        status: int = 200,
        reason: str = "OK",
        body: bytes = b"",
    ) -> None:
        self._chunks = chunks
        self._status = status
        self._reason = reason
        self._body = body
        self.posted: list[dict] = []

    def post(self, url: str, *, data: bytes, headers: dict) -> _FakeStreamResponse:
        self.posted.append({"url": url, "data": data, "headers": headers})
        return _FakeStreamResponse(
            self._chunks,
            url=url,
            status=self._status,
            reason=self._reason,
            body=self._body,
        )


_CHAT_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":" there"}}]}\n\n',
    b'data: {"choices":[],"usage":{"prompt_tokens":4,'
    b'"prompt_tokens_details":{"cached_tokens":2}}}\n\n',
    b"data: [DONE]\n\n",
]


@pytest.mark.asyncio
async def test_run_turn_streams_reply_and_prints_stats(capsys) -> None:
    session = _FakeSession(_CHAT_CHUNKS)
    text = await _run_turn(
        session,  # type: ignore[arg-type]
        url="http://h/v1/chat/completions",
        headers={},
        model="m",
        conversation=[{"role": "user", "content": "hey"}],
        encode=lambda s: list(s),  # char-per-token
    )
    assert text == "Hi there"
    # Body is orjson-serialized bytes; include_usage is set so ISL/cache resolve.
    sent = orjson.loads(session.posted[0]["data"])
    assert sent["stream_options"] == {"include_usage": True}
    assert session.posted[0]["headers"]["Content-Type"] == "application/json"
    out = capsys.readouterr().out
    assert "Hi there" in out
    for label in ("TTFT:", "TPS:", "ITL:", "Cache:"):
        assert label in out


@pytest.mark.asyncio
async def test_run_turn_reasoning_only_reply_returns_reasoning_for_history() -> None:
    # A reply with only reasoning (no content) must return the reasoning text so
    # multi-turn history records a non-empty assistant turn, not "".
    chunks = [
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking hard"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    text = await _run_turn(
        _FakeSession(chunks),  # type: ignore[arg-type]
        url="http://h/v1/chat/completions",
        headers={},
        model="m",
        conversation=[{"role": "user", "content": "hey"}],
        encode=lambda s: list(s),
    )
    assert text == "thinking hard"


@pytest.mark.asyncio
async def test_run_turn_surfaces_http_error_body() -> None:
    # Servers put the real diagnostic in the response body; it must reach the
    # user, not be discarded by raise_for_status().
    session = _FakeSession(
        [], status=404, reason="Not Found", body=b'{"detail":"model x does not exist"}'
    )
    with pytest.raises(aiohttp.ClientResponseError) as excinfo:
        await _run_turn(
            session,  # type: ignore[arg-type]
            url="http://h/v1/chat/completions",
            headers={},
            model="m",
            conversation=[{"role": "user", "content": "hi"}],
            encode=lambda s: list(s),
        )
    assert "model x does not exist" in str(excinfo.value)


@pytest.mark.asyncio
async def test_read_user_message_returns_line(monkeypatch) -> None:
    monkeypatch.setattr("builtins.input", lambda prompt="": "hello")
    assert await _read_user_message("> ") == "hello"


@pytest.mark.asyncio
async def test_read_user_message_propagates_eof(monkeypatch) -> None:
    def _raise(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _raise)
    with pytest.raises(EOFError):
        await _read_user_message("> ")


def _fake_reader(lines: list[str]):
    """Return an async `_read_user_message` stand-in yielding `lines` then EOF."""
    it = iter(lines)

    async def _read(prompt: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise EOFError from None

    return _read


@pytest.mark.asyncio
async def test_chat_loop_quick_sends_one_turn(monkeypatch) -> None:
    seen = []

    async def _fake_run_turn(session, *, url, headers, model, conversation, encode):
        seen.append(conversation)
        return "reply"

    monkeypatch.setattr(chat_mod, "_run_turn", _fake_run_turn)
    await chat_mod._chat_loop(
        model="m",
        url="http://h/v1/chat/completions",
        headers={},
        system_prompt=None,
        quick="hi",
        no_history=False,
        encode=lambda s: [],
    )
    assert len(seen) == 1
    assert seen[0] == [{"role": "user", "content": "hi"}]


@pytest.mark.parametrize(
    "no_history,expected_turns",
    [
        # Default: turn 2 resends turn 1's user message (history retained).
        param(False, [["one"], ["one", "two"]], id="retains_history"),
        # --no-history: each turn carries only its own message (stateless).
        param(True, [["one"], ["two"]], id="stateless"),
    ],
)  # fmt: skip
@pytest.mark.asyncio
async def test_chat_loop_history_behavior(
    monkeypatch, no_history: bool, expected_turns: list[list[str]]
) -> None:
    turns = []

    async def _fake_run_turn(session, *, url, headers, model, conversation, encode):
        turns.append([m["content"] for m in conversation if m["role"] == "user"])
        return "answer"

    monkeypatch.setattr(chat_mod, "_read_user_message", _fake_reader(["one", "two"]))
    monkeypatch.setattr(chat_mod, "_run_turn", _fake_run_turn)
    await chat_mod._chat_loop(
        model="m",
        url="http://h/v1/chat/completions",
        headers={},
        system_prompt=None,
        quick=None,
        no_history=no_history,
        encode=lambda s: [],
    )
    assert turns == expected_turns


@pytest.mark.asyncio
async def test_chat_loop_reports_error_and_continues(monkeypatch, capsys) -> None:
    async def _boom(session, *, url, headers, model, conversation, encode):
        raise SSEResponseError("upstream exploded", error_code=502)

    monkeypatch.setattr(chat_mod, "_read_user_message", _fake_reader(["x"]))
    monkeypatch.setattr(chat_mod, "_run_turn", _boom)
    await chat_mod._chat_loop(
        model="m",
        url="http://h/v1/chat/completions",
        headers={},
        system_prompt=None,
        quick=None,
        no_history=False,
        encode=lambda s: [],
    )
    assert "request failed" in capsys.readouterr().err


def test_run_chat_builds_auth_header_and_resolves_url(monkeypatch) -> None:
    import aiperf.common.tokenizer as tokenizer_mod

    monkeypatch.setattr(
        tokenizer_mod.Tokenizer,
        "from_pretrained",
        classmethod(lambda cls, name: type("_T", (), {"encode": staticmethod(list)})()),
    )
    captured: dict = {}

    async def _fake_loop(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(chat_mod, "_chat_loop", _fake_loop)
    chat_mod._run_chat(
        model="m",
        url="http://h",
        system_prompt=None,
        quick="hi",
        no_history=False,
        api_key="secret",
        tokenizer=None,
    )
    assert captured["headers"]["Authorization"] == "Bearer secret"
    assert captured["url"] == "http://h/v1/chat/completions"


def test_chat_entry_inverts_history_flag(monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(chat_mod, "_run_chat", lambda **kwargs: captured.update(kwargs))
    chat_mod.chat(model="m", history=False)
    assert captured["no_history"] is True
    assert captured["model"] == "m"


@pytest.mark.asyncio
async def test_settle_ignores_already_done_future() -> None:
    # A Ctrl-C / cancellation can complete the future before input() returns;
    # settling it again must be a no-op, not an InvalidStateError.
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()
    future.set_result("first")
    _settle(loop, future, result="second")
    await asyncio.sleep(0)  # let the scheduled callback run
    assert future.result() == "first"
