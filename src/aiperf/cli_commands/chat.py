# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI command for the interactive ``chat`` subcommand.

A lightweight, interactive sanity-check companion to ``aiperf profile``: chat
with an OpenAI-compatible endpoint one message at a time and print per-turn
speed stats (TTFT, TPS, generated tokens, end-to-end latency) after each reply.

Deliberately minimal. For the full benchmarking surface (concurrency,
datasets, sampling knobs, artifacts) use ``aiperf profile``. The per-turn
numbers are produced by the same metric classes ``profile`` uses (see
``_chat_stats``), so they stay definitionally consistent -- including
reasoning-token accounting.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Annotated

import aiohttp
import orjson
from cyclopts import App, Parameter

from aiperf.cli_commands._chat_stats import (
    build_record,
    compute_record_metrics,
    count_tokens,
    format_stats,
    input_tokens_from_usage,
    make_response_data,
    split_delta,
)
from aiperf.common.environment import Environment
from aiperf.common.exceptions import SSEResponseError
from aiperf.common.models.record_models import ParsedResponse
from aiperf.config.loader.parsing import normalize_http_url
from aiperf.transports.sse_utils import AsyncSSEStreamReader

app = App(name="chat")

# Fail fast on an unreachable endpoint, but leave generous time for a slow first
# token / long generation -- no ``total`` cap so streaming is never truncated
# mid-reply; ``sock_read`` only fires if the server stalls between chunks. Both
# are tunable via AIPERF_CHAT_CONNECT_TIMEOUT / AIPERF_CHAT_READ_TIMEOUT.
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(
    connect=Environment.CHAT.CONNECT_TIMEOUT,
    sock_read=Environment.CHAT.READ_TIMEOUT,
)


def _chat_completions_url(url: str) -> str:
    """Resolve a user-supplied base URL to a chat-completions endpoint.

    Accepts a bare host (``http://host:8000``), a ``/v1`` base, or a full
    ``/chat/completions`` path. A scheme-less URL (``localhost:8000``) gets
    ``http://`` prepended, matching how ``aiperf profile`` normalizes ``--url``;
    and appending ``/v1/chat/completions`` to a bare base matches how ``profile``
    joins the endpoint's metadata path, so a server mounted under a sub-path
    (e.g. ``/openai``) resolves the same way in both commands.
    """
    base = normalize_http_url(url).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _build_turn_messages(
    system_messages: list[dict], history: list[dict], user_message: str
) -> list[dict]:
    """Assemble the messages to send for one turn: optional system prompt,
    accumulated history (empty under ``--no-history``), then the new user
    message."""
    return [*system_messages, *history, {"role": "user", "content": user_message}]


async def _consume_stream(
    resp: aiohttp.ClientResponse,
) -> tuple[list[ParsedResponse], list[str], list[str], dict | None]:
    """Read the SSE stream via aiperf's ``AsyncSSEStreamReader``: collect parsed
    responses (timestamped at arrival), print content/reasoning live, and
    capture the final usage chunk.

    Returns ``(responses, output_parts, reasoning_parts, last_usage)``.

    Raises:
        SSEResponseError: if the server signals a mid-stream ``event: error``
            after a 200 OK (surfaced instead of reported as a truncated reply).
    """
    responses: list[ParsedResponse] = []
    output_parts: list[str] = []
    reasoning_parts: list[str] = []
    last_usage: dict | None = None
    async for message in AsyncSSEStreamReader(resp.content):
        AsyncSSEStreamReader.inspect_message_for_error(message)
        chunk = message.get_json()
        if not chunk:  # keep-alive, ``[DONE]``, or unparsable frame
            continue
        if chunk.get("usage"):
            last_usage = chunk["usage"]
        choices = chunk.get("choices")
        if not choices:
            continue
        content, reasoning = split_delta(choices[0].get("delta") or {})
        data_obj = make_response_data(content, reasoning)
        if data_obj is None:
            continue
        responses.append(ParsedResponse(perf_ns=message.perf_ns, data=data_obj))
        if reasoning:
            reasoning_parts.append(reasoning)
            print(reasoning, end="", flush=True)
        if content:
            output_parts.append(content)
            print(content, end="", flush=True)
    return responses, output_parts, reasoning_parts, last_usage


async def _run_turn(
    session: aiohttp.ClientSession,
    *,
    url: str,
    headers: dict[str, str],
    model: str,
    conversation: list[dict],
    encode: Callable[[str], list],
) -> str:
    """Stream one chat completion, print the reply live + a stats block, and
    return the assistant's text content (for appending to the conversation)."""
    # include_usage gets the trailing usage chunk (prompt tokens + prompt-cache
    # reads) for the ISL and cache-hit lines; it does not change the reply.
    payload = {
        "messages": conversation,
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start_ns = time.perf_counter_ns()
    timestamp_ns = time.time_ns()
    # Serialize with orjson (repo-wide JSON rule) and send bytes; setting the
    # content type ourselves since we bypass aiohttp's ``json=`` encoder.
    body = orjson.dumps(payload)
    post_headers = {**headers, "Content-Type": "application/json"}
    async with session.post(url, data=body, headers=post_headers) as resp:
        try:
            resp.raise_for_status()
        except aiohttp.ClientResponseError as e:
            # OpenAI-compatible servers put the real diagnostic (unknown model,
            # context-length exceeded, ...) in the response body, which
            # raise_for_status() discards. Surface a bounded prefix of it -- for
            # a sanity-check tool that error message is the whole point.
            detail = (await resp.text())[:500].strip()
            if detail:
                e.message = f"{e.message}: {detail}"
            raise
        responses, output_parts, reasoning_parts, last_usage = await _consume_stream(
            resp
        )
    end_ns = time.perf_counter_ns()
    print()

    # A usage-only response (data=None) carries the server usage to
    # ``record.final_usage`` without affecting the content-timing metrics.
    if last_usage is not None:
        responses.append(
            ParsedResponse(perf_ns=time.perf_counter_ns(), usage=last_usage)
        )
    output_tokens = await asyncio.to_thread(count_tokens, encode, "".join(output_parts))
    reasoning_tokens = await asyncio.to_thread(
        count_tokens, encode, "".join(reasoning_parts)
    )
    record = build_record(
        model=model,
        start_ns=start_ns,
        end_ns=end_ns,
        timestamp_ns=timestamp_ns,
        responses=responses,
        input_tokens=input_tokens_from_usage(last_usage),
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
    )
    print(format_stats(compute_record_metrics(record), reasoning_tokens))
    # Fall back to the reasoning text only when the reply had no visible content,
    # so a reasoning-only turn isn't recorded as an empty assistant message.
    # Intentionally content-only when content exists: unlike profile's per-chunk
    # build_assistant_turn (which concatenates reasoning + content), we don't
    # resend reasoning, matching how chat APIs replay assistant turns.
    return "".join(output_parts) or "".join(reasoning_parts)


async def _read_user_message(prompt: str) -> str:
    """Read one input line off a daemon thread.

    Using a daemon thread (rather than ``asyncio.to_thread``'s default
    executor) means a Ctrl-C while we're blocked at the prompt exits on the
    first press: the abandoned read thread never blocks asyncio's executor
    shutdown or the interpreter's exit-time thread join. Raises ``EOFError`` on
    Ctrl-D / closed stdin, matching ``input()``.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def _read() -> None:
        try:
            line = input(prompt)
        # noqa: BLE001 - forward any read failure (EOFError, OSError, ...) to
        # the awaiting coroutine so it surfaces there instead of dying silently
        # in this thread. Signals never reach a non-main thread, so Exception
        # (not BaseException) is the correct breadth.
        except Exception as e:  # noqa: BLE001
            _settle(loop, future, exc=e)
        else:
            _settle(loop, future, result=line)

    threading.Thread(target=_read, daemon=True, name="aiperf-chat-input").start()
    return await future


def _settle(
    loop: asyncio.AbstractEventLoop,
    future: asyncio.Future[str],
    *,
    result: str | None = None,
    exc: BaseException | None = None,
) -> None:
    """Resolve ``future`` from the read thread.

    No-ops if the future is already done -- Ctrl-C / task cancellation can
    complete it before ``input()`` returns, and setting a result on a done
    future raises ``InvalidStateError`` -- or if the loop is already closed.
    """

    def _apply() -> None:
        if future.done():
            return
        if exc is not None:
            future.set_exception(exc)
        else:
            future.set_result(result)

    with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(_apply)


async def _chat_loop(
    *,
    model: str,
    url: str,
    headers: dict[str, str],
    system_prompt: str | None,
    quick: str | None,
    no_history: bool,
    encode: Callable[[str], list],
) -> None:
    """Drive the (single-shot or interactive) chat session."""
    system_messages: list[dict] = (
        [{"role": "system", "content": system_prompt}] if system_prompt else []
    )

    async with aiohttp.ClientSession(timeout=_REQUEST_TIMEOUT) as session:
        if quick is not None:
            await _run_turn(
                session,
                url=url,
                headers=headers,
                model=model,
                conversation=_build_turn_messages(system_messages, [], quick),
                encode=encode,
            )
            return

        # ``history`` stays empty under ``--no-history``, making each turn an
        # independent, stateless request (completion-style) instead of resending
        # the growing conversation.
        history: list[dict] = []
        print("Please enter a message for the chat model (Ctrl-D to exit):")
        while True:
            try:
                message = await _read_user_message("> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            user_message = {"role": "user", "content": message}
            try:
                assistant = await _run_turn(
                    session,
                    url=url,
                    headers=headers,
                    model=model,
                    conversation=_build_turn_messages(
                        system_messages, history, message
                    ),
                    encode=encode,
                )
            except (TimeoutError, aiohttp.ClientError, SSEResponseError) as e:
                # Keep the REPL alive on a transient failure (HTTP error, dropped
                # connection, stall, or a mid-stream SSE error) -- report it and
                # let the user retry instead of tearing down the whole session.
                print(f"request failed: {e}", file=sys.stderr)
                continue
            if not no_history:
                history.append(user_message)
                history.append({"role": "assistant", "content": assistant})


def _run_chat(
    *,
    model: str,
    url: str,
    system_prompt: str | None,
    quick: str | None,
    no_history: bool,
    api_key: str | None,
    tokenizer: str | None,
) -> None:
    """Load the tokenizer, build auth headers, and drive the chat session."""
    from aiperf.cli_utils import exit_on_error

    with exit_on_error(title="Error running aiperf chat", show_traceback=False):
        from aiperf.common.tokenizer import Tokenizer

        tok = Tokenizer.from_pretrained(tokenizer or model)
        headers: dict[str, str] = {}
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"

        # Ctrl-C during a request should exit quietly, not dump a traceback.
        with contextlib.suppress(KeyboardInterrupt):
            asyncio.run(
                _chat_loop(
                    model=model,
                    url=_chat_completions_url(url),
                    headers=headers,
                    system_prompt=system_prompt,
                    quick=quick,
                    no_history=no_history,
                    encode=tok.encode,
                )
            )


@app.default
def chat(
    *,
    model: Annotated[
        str,
        Parameter(name=["--model", "-m"], help="Model name served by the endpoint."),
    ],
    url: Annotated[
        str,
        Parameter(
            name=["--url", "-u"],
            help="Base URL of the OpenAI-compatible server "
            "(e.g. http://localhost:8000), matching `aiperf profile`.",
        ),
    ] = "http://localhost:8000",
    system_prompt: Annotated[
        str | None,
        Parameter(help="Optional system prompt prepended to the conversation."),
    ] = None,
    quick: Annotated[
        str | None,
        Parameter(
            name=["--quick", "-q"],
            help="Send a single MESSAGE and print the response + stats, then exit.",
        ),
    ] = None,
    history: Annotated[
        bool,
        Parameter(
            help="Retain and resend conversation history each turn (default). "
            "Pass --no-history for stateless, completion-style turns. "
            "Ignored with --quick.",
        ),
    ] = True,
    api_key: Annotated[
        str | None,
        Parameter(
            help="API key sent as a Bearer token. "
            "Defaults to the OPENAI_API_KEY environment variable."
        ),
    ] = None,
    tokenizer: Annotated[
        str | None,
        Parameter(
            help="Tokenizer for client-side token counts. Defaults to the model "
            "name. Pass `builtin` for a zero-network tokenizer."
        ),
    ] = None,
) -> None:
    """Chat interactively with an endpoint, printing per-turn speed stats.

    A lightweight sanity-check sibling of ``aiperf profile``: send one message
    at a time and see TTFT, TPS, generated tokens, and end-to-end latency for
    each reply. Multi-turn by default (history retained and resent each turn);
    pass ``--no-history`` for stateless turns, or ``--quick`` for a single
    request. For full benchmarking, use ``aiperf profile``.

    Examples:
        # Interactive multi-turn chat
        aiperf chat --model Qwen/Qwen3-0.6B --url http://localhost:8000

        # Stateless turns (no history retained between messages)
        aiperf chat --model Qwen/Qwen3-0.6B --no-history

        # Single-shot sanity check
        aiperf chat --model Qwen/Qwen3-0.6B --quick "hello, who are you?"
    """
    _run_chat(
        model=model,
        url=url,
        system_prompt=system_prompt,
        quick=quick,
        no_history=not history,
        api_key=api_key,
        tokenizer=tokenizer,
    )
