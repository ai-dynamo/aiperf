# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay the documented command corpus against another model family.

The docs end-to-end suite answers "is this guide correct?" by running each
documented command verbatim against Qwen3-0.6B. It cannot answer "does AIPerf
work on model family X", because every guide names one model.

A sweep reuses the same corpus for the second question: take a group's
commands, substitute the model, and run them against a server for that family.
A failure here is a **product** bug on that family, not a documentation bug --
the guide is known good, only the model changed.

Serving current families outright is not an option: the smallest member in
Dynamo's `recipes/` outside Qwen is ~31B against a ~20 GiB budget. Families do
ship small members that fit, and those carry the family's own tokenizer and
chat template, which is where the breakage actually lives.

Why the server command lives here rather than being patched out of the guide:
`docs/tutorial.md`'s vLLM setup passes `--reasoning-parser qwen3`, which is
Qwen-specific. Rewriting a documented command well enough to serve a different
family means guessing at per-family server flags, so each sweep target states
its own server command explicitly instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from data_types import Command, Server


@dataclass(frozen=True)
class SweepTarget:
    """One model family to replay the corpus against."""

    name: str
    """Shard name, e.g. ``sweep-nemotron``."""

    model: str
    """HuggingFace id substituted into every replayed command."""

    replays: str
    """Server group whose commands are borrowed."""

    server_args: list[str] = field(default_factory=list)
    """vLLM flags after the model. Stated per target because they are
    family-specific -- a reasoning or tool-call parser that fits one family
    rejects or silently mis-parses another."""

    notes: str = ""


# Chosen for fit and availability, verified against the HuggingFace API:
# weights must leave room on a ~20.27 GiB budget, the model must be ungated so
# CI needs no terms acceptance, and it must be a genuinely different family
# from Qwen rather than a distill carrying Qwen's tokenizer.
SWEEP_TARGETS: dict[str, SweepTarget] = {
    "sweep-nemotron": SweepTarget(
        name="sweep-nemotron",
        model="nvidia/Nemotron-Mini-4B-Instruct",
        replays="vllm-default-openai",
        server_args=["--enforce-eager"],
        notes="8.38 GB weights, ungated. No reasoning parser: that flag is Qwen-specific.",
    ),
}

_SETUP_TEMPLATE = """docker pull vllm/vllm-openai:latest
docker run --gpus all -p 8000:8000 -e HF_TOKEN vllm/vllm-openai:latest \\
  --model {model} \\
  {server_args} \\
  --host 0.0.0.0 --port 8000"""

_HEALTH_TEMPLATE = """for _ in $(seq 180); do
  curl -sf http://localhost:8000/v1/models >/dev/null && break
  sleep 5
done
curl -sf http://localhost:8000/v1/models >/dev/null"""

# Matches the flag and its value, whether separated by a space or '='.
_MODEL_FLAG = re.compile(r"(--model|--tokenizer|-m)(\s+|=)(\S+)")


def substitute_model(command: str, model: str) -> str:
    """Point every model/tokenizer flag in a command at ``model``.

    Only the flag values change. Everything else about the documented command
    -- concurrency, request counts, dataset flags, endpoint type -- is left
    exactly as written, because the point is to run the *same* workload against
    a different family.
    """
    return _MODEL_FLAG.sub(lambda m: f"{m.group(1)}{m.group(2)}{model}", command)


def build_sweep_server(target: SweepTarget, servers: dict[str, Server]) -> Server:
    """Build a synthetic server that replays another group's commands.

    Raises:
        KeyError: if the group named by ``replays`` was not discovered, which
            means the corpus moved and the sweep would silently test nothing.
    """
    if target.replays not in servers:
        raise KeyError(
            f"sweep target '{target.name}' replays '{target.replays}', which "
            f"was not discovered. Found: {sorted(servers)}"
        )
    base = servers[target.replays]

    def _rewrite(command: Command, suffix: str) -> Command:
        return Command(
            tag_name=f"{command.tag_name}{suffix}",
            command=substitute_model(command.command, target.model),
            file_path=command.file_path,
            start_line=command.start_line,
            end_line=command.end_line,
            weight=command.weight,
            timeout=command.timeout,
        )

    setup = _SETUP_TEMPLATE.format(
        model=target.model,
        server_args=" \\\n  ".join(target.server_args),
    )
    return Server(
        name=target.name,
        setup_command=Command(
            tag_name=f"setup-{target.name}-endpoint-server",
            command=setup,
            file_path=f"<sweep:{target.name}>",
            start_line=0,
            end_line=0,
        ),
        health_check_command=Command(
            tag_name=f"health-check-{target.name}-endpoint-server",
            command=_HEALTH_TEMPLATE,
            file_path=f"<sweep:{target.name}>",
            start_line=0,
            end_line=0,
        ),
        aiperf_commands=[_rewrite(c, f"[{target.name}]") for c in base.aiperf_commands],
        files=list(base.files),
    )
