# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test for the agentic-replay warmup-failure abort.

Pins the bug that hung production runs: when warmup requests fail terminally,
the run used to (a) drain the whole warmup phase, then (b) raise at teardown,
after which the records manager never finalized and the process hung until an
external watchdog killed it. The fix aborts on the FIRST terminal warmup
failure (broadcast ProfileCancelCommand) and exits NON-ZERO -- a real process,
real HTTP mock server returning 100% errors, driven end-to-end.

Uses the real-process ``tests/integration`` tier on purpose: the
component-integration FakeTransport bypasses error injection, so only a real
mock server with ``error_rate=100`` can force terminal warmup failures.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.harness.utils import AIPerfCLI

_MODEL = "claude-opus-4-5-20251101"
_TOKENIZER = "openai/gpt-oss-120b"  # pre-cached + offline in integration conftest
_BLOCK_SIZE = 16


def _req(t: float, hash_ids: list[int], in_tokens: int) -> dict:
    return {
        "t": t,
        "type": "n",
        "model": _MODEL,
        "in": in_tokens,
        "out": 8,
        "hash_ids": hash_ids,
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
        "api_time": 0.05,
        "think_time": 0.0,
    }


def _write_linear_fixture(target_dir: Path, *, num_traces: int = 4) -> Path:
    """Minimal multi-turn weka traces (>=2 turns so agentic_replay can split
    into a warmup turn k_i and a profiling tail)."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for n in range(1, num_traces + 1):
        requests = []
        for k in range(max(2, n)):
            user_blocks = k + 1
            in_tokens = (1 + user_blocks) * _BLOCK_SIZE + 4
            hash_ids = list(range(1, 1 + 1 + user_blocks))
            requests.append(_req(k * 1.0, hash_ids, in_tokens))
        trace = {
            "id": f"lin_trace_{n:02d}",
            "models": [_MODEL],
            "block_size": _BLOCK_SIZE,
            "hash_id_scope": "local",
            "tool_tokens": 8,
            "system_tokens": 8,
            "requests": requests,
        }
        (target_dir / f"lin_trace_{n:02d}.json").write_text(json.dumps(trace))
    return target_dir


def _build_cmd(weka_dir: Path, url: str) -> str:
    return f"""
        aiperf profile \
            --model {_MODEL} \
            --url {url} \
            --endpoint-type chat \
            --streaming \
            --custom-dataset-type weka_trace \
            --input-file {weka_dir} \
            --no-fixed-schedule \
            --benchmark-duration 8 \
            --concurrency 3 \
            --random-seed 42 \
            --tokenizer {_TOKENIZER} \
            --extra-inputs ignore_eos:true \
            --workers-max 2 \
            --scenario inferencex-agentx-mvp \
            --unsafe-override \
            --export-level raw \
            --ui simple
    """


@pytest.mark.integration
@pytest.mark.asyncio
async def test_warmup_failure_aborts_nonzero_without_hanging(
    cli: AIPerfCLI,
    mock_server_factory,
    tmp_path: Path,
) -> None:
    """Every warmup request errors -> the run aborts on the first terminal
    failure and exits NON-ZERO, completing well within the timeout (the bug was
    a hang until an external watchdog killed it; a hang now fails by timeout)."""
    weka_dir = _write_linear_fixture(tmp_path / "lin_warmup_fail")

    async with mock_server_factory(fast=True, error_rate=100) as server:
        result = await cli.run(
            _build_cmd(weka_dir, server.url),
            timeout=120.0,
            assert_success=False,
        )

    # Aborted, not silently "successful": the warmup-failure abort exits non-zero.
    assert result.exit_code != 0, (
        f"expected non-zero exit on warmup failure, got {result.exit_code}\n"
        f"{(result.log or '')[-1500:]}"
    )

    # Prove it aborted via the LIVE warmup early-cancel path -- not an incidental
    # non-zero exit (e.g. a setup/validation failure that never reached warmup).
    log = result.log or ""
    assert "aborting run early" in log, (
        f"missing live early-abort marker; abort may not be the warmup path\n{log[-2000:]}"
    )
    assert "Run aborted (warmup_failure)" in log, (
        f"missing warmup-failure abort-reason marker (exit-code path)\n{log[-2000:]}"
    )
