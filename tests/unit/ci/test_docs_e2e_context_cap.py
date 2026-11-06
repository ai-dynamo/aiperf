# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A documented vLLM launch must cap the context it will generate into.

Qwen3-0.6B is a reasoning model. A request that carries no output-token cap
generates until EOS or the context limit, and a small reasoning model reaches
the limit often. At the ~50 tokens/s a single request gets on the CI runner,
the stock 40960-token context takes about 13 minutes to exhaust, which is
longer than AIPerf's 600s record-progress stall timeout. The watchdog then
aborts a benchmark that is working: on 2026-11-06 three shards died this way
while vLLM logged ``Running: 1 reqs`` at a steady 54 tokens/s throughout, and
an earlier run recovered at 722s -- the time it takes to fill 40960 tokens.

Capping the context bounds that worst case. The cap has to clear the largest
input any tagged guide asks for, or the guide stops being executable at all,
so this test ties the two together rather than pinning a number.

Only servers that enable a reasoning parser are covered. An instruct model
can in principle run to the context limit too, but it emits EOS instead, and
none of the vision, audio or embedding servers has ever stalled; capping a
context those models need for image or audio tokens on no evidence would
trade a real failure for a speculative one.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Seconds the records manager waits before declaring a phase stalled, and the
# per-request generation rate a single request gets on the CI runner.
STALL_TIMEOUT_SECONDS = 600
OBSERVED_TOKENS_PER_SECOND = 47


def _vllm_servers() -> dict:
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


def test_every_vllm_server_caps_its_context() -> None:
    servers = _vllm_servers()
    assert servers, "no reasoning server is defined; this test has lost its subject"

    for name, server in servers.items():
        cap = re.search(r"--max-model-len\s+(\d+)", server.setup_command.command)
        assert cap, (
            f"{name} launches vLLM without --max-model-len, so one runaway "
            f"generation can outlast the {STALL_TIMEOUT_SECONDS}s stall timeout"
        )

        budget = STALL_TIMEOUT_SECONDS * OBSERVED_TOKENS_PER_SECOND
        assert int(cap.group(1)) < budget, (
            f"{name} caps the context at {cap.group(1)} tokens, which at "
            f"{OBSERVED_TOKENS_PER_SECOND} tokens/s takes longer to fill than "
            f"the {STALL_TIMEOUT_SECONDS}s stall timeout allows"
        )


def test_the_cap_still_fits_the_largest_documented_input() -> None:
    """The cap must not make a tagged guide unservable."""
    for name, server in _vllm_servers().items():
        cap = re.search(r"--max-model-len\s+(\d+)", server.setup_command.command)
        assert cap

        for command in server.aiperf_commands:
            requested = [
                int(value)
                for value in re.findall(
                    r"--(?:synthetic-input-tokens-mean|prefix-prompt-length"
                    r"|output-tokens-mean)\s+(\d+)",
                    command.command,
                )
            ]
            if not requested:
                continue
            assert sum(requested) < int(cap.group(1)), (
                f"{command.file_path} asks {name} for {sum(requested)} tokens "
                f"but the context is capped at {cap.group(1)}"
            )
