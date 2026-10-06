# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replaying the corpus against another family must change only the model.

The sweep answers a different question from the rest of the suite: not "is this
guide correct" but "does AIPerf work on model family X". That only holds if the
*workload* is untouched -- concurrency, request counts, dataset flags and
endpoint types must survive verbatim, or a failure says nothing about the
family.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pytest import param

HARNESS = Path(__file__).resolve().parents[3] / "tests/ci/test_docs_end_to_end"
sys.path.insert(0, str(HARNESS))

from data_types import Command, Server  # noqa: E402
from model_sweep import (  # noqa: E402
    SWEEP_TARGETS,
    build_sweep_server,
    substitute_model,
)

MODEL = "nvidia/Nemotron-Mini-4B-Instruct"


@pytest.mark.parametrize(
    "original, expected",
    [
        param("aiperf profile --model Qwen/Qwen3-0.6B", f"aiperf profile --model {MODEL}", id="model"),
        param("aiperf profile --tokenizer Qwen/Qwen3-0.6B", f"aiperf profile --tokenizer {MODEL}", id="tokenizer"),
        param("aiperf profile -m Qwen/Qwen3-0.6B", f"aiperf profile -m {MODEL}", id="short-flag"),
        param("aiperf profile --model=Qwen/Qwen3-0.6B", f"aiperf profile --model={MODEL}", id="equals-form"),
    ],
)  # fmt: skip
def test_model_flags_are_repointed(original: str, expected: str) -> None:
    assert substitute_model(original, MODEL) == expected


def test_the_workload_is_untouched() -> None:
    """A sweep failure must implicate the family, not a changed workload."""
    original = (
        "aiperf profile --model Qwen/Qwen3-0.6B --tokenizer Qwen/Qwen3-0.6B "
        "--endpoint-type chat --streaming --concurrency 8 --request-count 200 "
        "--synthetic-input-tokens-mean 550 --extra-inputs '{\"temperature\": 0}'"
    )
    swept = substitute_model(original, MODEL)

    for fragment in [
        "--endpoint-type chat",
        "--streaming",
        "--concurrency 8",
        "--request-count 200",
        "--synthetic-input-tokens-mean 550",
        "--extra-inputs '{\"temperature\": 0}'",
    ]:
        assert fragment in swept, f"sweep altered the workload: lost {fragment}"
    assert "Qwen" not in swept


def test_a_model_named_inside_another_flag_is_not_rewritten() -> None:
    """Only the model flags move; a path or id elsewhere stays put."""
    original = "aiperf profile --model Qwen/Qwen3-0.6B --artifact-dir ./Qwen-run"
    assert "./Qwen-run" in substitute_model(original, MODEL)


def _server(commands: list[str]) -> dict[str, Server]:
    return {
        "vllm-default-openai": Server(
            name="vllm-default-openai",
            setup_command=Command("setup", "docker run ...", "f.md", 1, 2),
            health_check_command=Command("health", "curl ...", "f.md", 3, 4),
            aiperf_commands=[
                Command("run", c, "f.md", 5 + i, 6 + i) for i, c in enumerate(commands)
            ],
        )
    }


def test_the_sweep_borrows_every_command_from_its_base() -> None:
    servers = _server(["aiperf profile --model Qwen/Qwen3-0.6B --concurrency 4"] * 3)
    swept = build_sweep_server(SWEEP_TARGETS["sweep-nemotron"], servers)

    assert len(swept.aiperf_commands) == 3
    assert all(MODEL in c.command for c in swept.aiperf_commands)
    assert all("--concurrency 4" in c.command for c in swept.aiperf_commands)


def test_the_sweep_brings_its_own_server_command() -> None:
    """The documented setup passes --reasoning-parser qwen3, which is Qwen-only.

    Patching a guide's server command well enough to serve another family means
    guessing per-family flags, so each target states its own.
    """
    swept = build_sweep_server(SWEEP_TARGETS["sweep-nemotron"], _server([]))
    setup = swept.setup_command.command
    assert MODEL in setup
    assert "reasoning-parser" not in setup


def test_a_missing_base_group_fails_loudly() -> None:
    """Otherwise the sweep would quietly run zero commands and pass."""
    with pytest.raises(KeyError, match="was not discovered"):
        build_sweep_server(SWEEP_TARGETS["sweep-nemotron"], {})


def test_every_target_names_a_distinct_family() -> None:
    """A distill carrying Qwen's tokenizer would test Qwen again under a new name."""
    for name, target in SWEEP_TARGETS.items():
        assert "qwen" not in target.model.lower(), (
            f"sweep target '{name}' points at a Qwen-derived model, which would "
            "re-test the family the corpus already covers"
        )
