# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``aiperf.cli_commands.chat`` helpers -- URL resolution and
per-turn message assembly. The streaming/REPL I/O is covered by the
integration tests; the stats/metric logic lives in ``test_chat_stats``.
"""

from __future__ import annotations

import pytest
from pytest import param

from aiperf.cli_commands.chat import _build_turn_messages, _chat_completions_url


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
