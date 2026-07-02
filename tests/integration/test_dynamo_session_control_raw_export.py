# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end raw-export checks of the Dynamo nvext.session_control lifecycle.

Runs a real ``aiperf profile`` subprocess against the in-repo mock server with
``--use-dynamo-conv-aware-routing`` and ``--export-level raw``, then reads the
exported wire payloads and asserts the per-session session_control lifecycle:

- modern (default): ``bind`` on every non-final turn, ``close`` on the last.
- legacy (``--use-legacy-dynamo-session-control``): ``open`` on the first turn,
  ``session_id`` only on intermediate turns, ``close`` on the last -- and never
  ``bind`` (which released Dynamo v1.2.x rejects with HTTP 400).
"""

from __future__ import annotations

import json
from collections import defaultdict

import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

_TOKENIZER = "openai/gpt-oss-120b"  # pre-cached + offline in integration conftest

_NUM_SESSIONS = 2
_TURNS_PER_SESSION = 3
_TIMEOUT_SECONDS = 300


def _payload_dict(record) -> dict:
    if record.payload is not None:
        return record.payload
    if record.payload_bytes is not None:
        return json.loads(record.payload_bytes)
    return {}


def _build_cmd(url: str, *, legacy: bool) -> str:
    legacy_flag = "--use-legacy-dynamo-session-control" if legacy else ""
    return f"""
        aiperf profile \
            --model {_TOKENIZER} \
            --url {url} \
            --endpoint-type chat \
            --num-sessions {_NUM_SESSIONS} \
            --session-turns-mean {_TURNS_PER_SESSION} \
            --session-turns-stddev 0 \
            --random-seed 42 \
            --workers-max 1 \
            --use-dynamo-conv-aware-routing \
            {legacy_flag} \
            --dynamo-session-timeout-seconds {_TIMEOUT_SECONDS} \
            --export-level raw \
            --ui simple
    """


async def _session_controls_by_session(
    cli: AIPerfCLI, url: str, *, legacy: bool
) -> dict[str, list[dict]]:
    """Run a benchmark and return each session's session_control blocks, ordered
    by turn_index, keyed by X-Correlation-ID."""
    result = await cli.run(_build_cmd(url, legacy=legacy), timeout=300.0)

    records = list(result.raw_records or [])
    assert records, f"no raw records\n{(result.log or '')[-1500:]}"

    grouped: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for rec in records:
        sc = _payload_dict(rec).get("nvext", {}).get("session_control")
        assert sc is not None, "every request must carry nvext.session_control"
        grouped[rec.metadata.x_correlation_id].append((rec.metadata.turn_index, sc))

    assert len(grouped) == _NUM_SESSIONS, (
        f"expected {_NUM_SESSIONS} sessions, got {len(grouped)}"
    )
    out: dict[str, list[dict]] = {}
    for xcorr, turns in grouped.items():
        assert len(turns) == _TURNS_PER_SESSION, (
            f"session {xcorr}: expected {_TURNS_PER_SESSION} turns, got {len(turns)}"
        )
        turns.sort(key=lambda t: t[0])
        out[xcorr] = [sc for _, sc in turns]
    return out


@pytest.mark.integration
@pytest.mark.asyncio
async def test_modern_session_control_bind_close_lifecycle_on_wire(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
):
    """Default mode: every non-final turn re-binds (with timeout), the final
    turn closes, every turn shares one session_id, and 'open' never appears."""
    by_session = await _session_controls_by_session(
        cli, aiperf_mock_server.url, legacy=False
    )

    for xcorr, scs in by_session.items():
        actions = [sc.get("action") for sc in scs]
        assert all(a == "bind" for a in actions[:-1]), (
            f"session {xcorr}: non-final turns must bind; {actions}"
        )
        assert actions[-1] == "close", f"session {xcorr}: actions={actions}"
        assert "open" not in actions, f"session {xcorr}: emitted open; {actions}"

        # Every non-final 'bind' carries the timeout; 'close' does not.
        for sc in scs[:-1]:
            assert sc["timeout"] == _TIMEOUT_SECONDS
        assert "timeout" not in scs[-1]

        # One stable session_id == the X-Correlation-ID, on every turn.
        assert {sc["session_id"] for sc in scs} == {xcorr}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_legacy_session_control_open_close_lifecycle_on_wire(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
):
    """Legacy mode: open on the first request, session_id only on intermediate
    turns, close on the final turn -- never bind."""
    by_session = await _session_controls_by_session(
        cli, aiperf_mock_server.url, legacy=True
    )

    for xcorr, scs in by_session.items():
        actions = [sc.get("action") for sc in scs]
        assert actions[0] == "open", f"session {xcorr}: actions={actions}"
        assert actions[-1] == "close", f"session {xcorr}: actions={actions}"
        assert all(a is None for a in actions[1:-1]), (
            f"session {xcorr}: intermediate turns must carry no action; {actions}"
        )
        assert "bind" not in actions, f"session {xcorr}: emitted bind; {actions}"

        # One stable session_id == the X-Correlation-ID, on every turn.
        assert {sc["session_id"] for sc in scs} == {xcorr}

        # 'open' carries the timeout; 'close' / bare turns do not.
        assert scs[0]["timeout"] == _TIMEOUT_SECONDS
        assert "timeout" not in scs[-1]
