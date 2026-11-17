# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal OpenAI-compatible chat client for LLM-judge accuracy graders.

Shared by judge-backed graders (``darkbench_overseer``). Configured entirely
from ``Environment.ACCURACY.JUDGE_*``. One client lives per record processor;
its semaphore bounds in-flight judge requests from that processor.
"""

from __future__ import annotations

import asyncio
import random

import aiohttp
import orjson

from aiperf.common.environment import Environment

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class JudgeError(RuntimeError):
    """The judge returned an unusable HTTP response."""

    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


def backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter for retry ``attempt`` (0-based), capped at the judge timeout."""
    return min(Environment.ACCURACY.JUDGE_TIMEOUT, 2.0**attempt + random.random())


class LLMJudgeClient:
    """Async client for ``POST {JUDGE_URL}/chat/completions``.

    The aiohttp session is created lazily on first use so it binds to the
    record processor's running event loop, and closed by ``aclose``.
    """

    def __init__(self) -> None:
        settings = Environment.ACCURACY
        if not settings.JUDGE_URL or not settings.JUDGE_MODEL:
            raise ValueError(
                "LLM judge is not configured: set AIPERF_ACCURACY_JUDGE_URL and "
                "AIPERF_ACCURACY_JUDGE_MODEL"
            )
        self.model: str = settings.JUDGE_MODEL
        self._url = f"{settings.JUDGE_URL.rstrip('/')}/chat/completions"
        self._max_tokens = settings.JUDGE_MAX_TOKENS
        self._headers = {"Content-Type": "application/json"}
        if settings.JUDGE_API_KEY is not None:
            self._headers["Authorization"] = (
                f"Bearer {settings.JUDGE_API_KEY.get_secret_value()}"
            )
        self._timeout = aiohttp.ClientTimeout(total=settings.JUDGE_TIMEOUT)
        self._semaphore = asyncio.Semaphore(settings.JUDGE_CONCURRENCY)
        self._session: aiohttp.ClientSession | None = None

    def __repr__(self) -> str:
        return f"LLMJudgeClient(url={self._url!r}, model={self.model!r})"

    async def complete(self, system: str, user: str) -> str:
        """Return the judge's message content for one system+user exchange.

        Raises:
            JudgeError: non-200 status (``retryable`` for 408/409/425/429/5xx)
                or a body without ``choices[0].message``.
            aiohttp.ClientError, asyncio.TimeoutError: transport failures.
        """
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout, headers=self._headers
            )
        body = orjson.dumps(
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": self._max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }
        )
        async with (
            self._semaphore,
            self._session.post(self._url, data=body) as resp,
        ):
            raw = await resp.read()
            if resp.status != 200:
                raise JudgeError(
                    f"judge HTTP {resp.status}: {raw[:200].decode(errors='replace')}",
                    retryable=resp.status in _RETRYABLE_STATUS,
                )
        try:
            content = orjson.loads(raw)["choices"][0]["message"].get("content")
        except (
            orjson.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
        ) as e:
            raise JudgeError(
                f"judge returned a malformed completion: {e!r}", retryable=True
            ) from e
        return content or ""

    async def aclose(self) -> None:
        """Close the HTTP session if one was opened."""
        if self._session is not None:
            await self._session.close()
            self._session = None
