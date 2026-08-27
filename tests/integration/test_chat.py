# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for the ``aiperf chat`` command against the mock server.

Complements the unit tests (which cover the pure parsing/metric logic) by
exercising the full path end to end: real HTTP streaming, the metric pipeline,
and the printed stats block. Uses the in-repo mock server, which reports
``prompt_tokens_details.cached_tokens`` so the cache-hit line resolves.

Statefulness is asserted via the prompt-token count in each turn's ``Cache:``
line (the server counts the real prompt it received): with history it grows
turn over turn; with ``--no-history`` and an identical message it does not.
"""

import asyncio
import os
import re

import pytest

from tests.harness.utils import AIPerfMockServer, AIPerfRunnerFn
from tests.integration.conftest import get_venv_python

# Captures the prompt-token count (ISL) from a ``Cache: <hit>/<prompt> ...`` line.
_PROMPT_TOKENS = re.compile(r"\d+/(\d+) prompt tokens cached")


def _prompt_token_counts(stdout: str) -> list[int]:
    """Per-turn prompt-token counts parsed from the cache lines, in order."""
    return [int(m) for m in _PROMPT_TOKENS.findall(stdout)]


@pytest.mark.integration
@pytest.mark.asyncio
class TestChatCommand:
    """End-to-end checks that ``aiperf chat`` streams a reply and prints stats."""

    async def test_quick_prints_stats(
        self, aiperf_runner: AIPerfRunnerFn, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """``--quick`` streams one reply and prints the stats block, including
        the cache line (prefix caches are server-side, so even a single-shot
        request reports a hit rate when the server surfaces cached tokens)."""
        result = await aiperf_runner(
            [
                "chat",
                "--model",
                "mock-model",
                "--url",
                aiperf_mock_server.url,
                "--tokenizer",
                "builtin",
                "--quick",
                "hello there, who are you?",
            ]
        )
        assert result.exit_code == 0, result.stderr
        assert "TTFT:" in result.stdout
        assert "TPS:" in result.stdout
        assert "Cache:" in result.stdout

    async def test_multi_turn_resends_history(
        self, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """Default mode prints the ITL/decode + cache lines per turn, and the
        prompt grows turn over turn because history is resent."""
        stdout = await _run_chat_over_stdin(
            aiperf_mock_server.url, "tell me a short story\ncontinue the story\n"
        )
        # The full per-turn block (all four metrics) prints for each turn.
        assert stdout.count("TTFT:") == 2
        for label in ("TPS:", "ITL:", "Cache:"):
            assert label in stdout
        prompts = _prompt_token_counts(stdout)
        assert len(prompts) == 2
        # Turn 2 resends turn 1 (user + assistant), so its prompt is larger.
        assert prompts[1] > prompts[0]

    async def test_no_history_is_stateless(
        self, aiperf_mock_server: AIPerfMockServer
    ) -> None:
        """``--no-history`` sends each message independently: an identical
        message yields an identical prompt size across turns (no history)."""
        stdout = await _run_chat_over_stdin(
            aiperf_mock_server.url,
            "repeat this exactly\nrepeat this exactly\n",
            extra_args=["--no-history"],
        )
        assert stdout.count("TTFT:") == 2
        assert "ITL:" in stdout
        prompts = _prompt_token_counts(stdout)
        assert len(prompts) == 2
        # No history resent -> identical message -> identical prompt size.
        assert prompts[0] == prompts[1]


async def _run_chat_over_stdin(
    url: str,
    stdin_text: str,
    timeout: float = 60.0,
    extra_args: list[str] | None = None,
) -> str:
    """Run ``aiperf chat`` interactively, feeding ``stdin_text`` then EOF.

    Returns captured stdout; asserts a clean exit.
    """
    cmd = [
        get_venv_python(),
        "-m",
        "aiperf",
        "chat",
        "--model",
        "mock-model",
        "--url",
        url,
        "--tokenizer",
        "builtin",
        *(extra_args or []),
    ]
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(input=stdin_text.encode()), timeout=timeout
        )
    except TimeoutError:
        process.kill()
        raise
    assert process.returncode == 0, stderr_bytes.decode("utf-8", "replace")
    return stdout_bytes.decode("utf-8", "replace")
