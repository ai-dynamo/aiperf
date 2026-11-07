# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A documented vLLM launch must cap its context, and the cap must fit the guides.

Qwen3-0.6B is a reasoning model. A request with no output-token cap generates
until EOS or the context limit, and a small reasoning model reaches the limit
often. Uncapped that is 40960 tokens, about 15 minutes at the ~47 tokens/s a
single request gets on the CI runner, which outlasts even the raised stall
timeout. Capping the context bounds that worst case.

The cap cuts both ways. Set it below what a guide asks for and vLLM rejects
every request with 400, which is how ``docs/tutorials/sessions.md`` broke when
the first cap landed at 20480 and the guide wanted 21000 for its context
prompts alone. So this test checks the cap from both sides, and it accounts
for multi-turn guides, where history accumulates across the session.

Only servers that enable a reasoning parser are covered. An instruct model can
in principle run to the context limit too, but it emits EOS instead, and none
of the vision, audio or embedding servers has ever stalled; capping a context
those models need for image or audio tokens on no evidence would trade a real
failure for a speculative one.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# What a single request gets on the CI runner, measured from vLLM's own
# "Avg generation throughput" during the 2026-11-06 stalls.
OBSERVED_TOKENS_PER_SECOND = 47

# Share of the stall timeout a worst-case generation may consume. The observed
# rate is a single sample from one runner; a slower one must not turn a green
# shard red, so the cap has to leave room rather than merely fit.
SAFETY_FACTOR = 0.75

# Prompt lengths that are paid once per session, whatever the turn count.
_ONCE_PER_SESSION = (
    "shared-system-prompt-length",
    "user-context-prompt-length",
    "prefix-prompt-length",
)
# Lengths paid again on every turn.
_PER_TURN_INPUT = ("synthetic-input-tokens-mean", "prompt-input-tokens-mean", "isl")
_PER_TURN_OUTPUT = ("output-tokens-mean", "prompt-output-tokens-mean", "osl")


def _flag(command: str, name: str) -> int:
    match = re.search(rf"--{re.escape(name)}\s+(\d+)", command)
    return int(match.group(1)) if match else 0


def _context_needed(command: str) -> int:
    """Tokens the longest turn of this command puts in front of the model."""
    turns = _flag(command, "session-turns-mean") or 1
    per_turn = max(_flag(command, n) for n in _PER_TURN_INPUT) + max(
        _flag(command, n) for n in _PER_TURN_OUTPUT
    )
    return sum(_flag(command, n) for n in _ONCE_PER_SESSION) + turns * per_turn


def _reasoning_servers() -> dict:
    from parser import MarkdownParser

    parser = MarkdownParser()
    parser.parse_directory(str(REPO_ROOT))
    return {
        name: server
        for name, server in parser.servers.items()
        if server.setup_command
        and "vllm" in server.setup_command.command
        and "--reasoning-parser" in server.setup_command.command
    }


def _cap(server) -> int:
    match = re.search(r"--max-model-len\s+(\d+)", server.setup_command.command)
    assert match, (
        "this vLLM launch has no --max-model-len, so one runaway reasoning "
        "generation can fill 40960 tokens and outlast the stall watchdog"
    )
    return int(match.group(1))


def test_a_runaway_generation_ends_before_the_stall_watchdog_fires() -> None:
    from constants import RECORD_PROGRESS_STALL_TIMEOUT

    servers = _reasoning_servers()
    assert servers, "no reasoning server is defined; this test has lost its subject"

    for name, server in servers.items():
        seconds = _cap(server) / OBSERVED_TOKENS_PER_SECOND
        budget = RECORD_PROGRESS_STALL_TIMEOUT * SAFETY_FACTOR
        assert seconds < budget, (
            f"{name} caps the context at {_cap(server)} tokens, which takes "
            f"{seconds:.0f}s to fill at {OBSERVED_TOKENS_PER_SECOND} tokens/s; "
            f"that leaves no room under the {RECORD_PROGRESS_STALL_TIMEOUT}s "
            f"stall timeout, which allows {budget:.0f}s"
        )


def test_the_cap_fits_every_guide_that_server_runs() -> None:
    for name, server in _reasoning_servers().items():
        cap = _cap(server)
        for command in server.aiperf_commands:
            needed = _context_needed(command.command)
            assert needed <= cap, (
                f"{command.file_path} asks {name} for {needed} tokens of "
                f"context but the cap is {cap}; vLLM answers 400 Bad Request "
                f"to every request and the guide stops being executable"
            )
