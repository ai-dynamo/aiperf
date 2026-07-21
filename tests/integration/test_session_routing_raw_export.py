# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end raw-export proof that every ``--session-routing`` mode reaches the wire.

Runs a real ``aiperf profile`` subprocess against the in-repo mock server with
``--export-level raw`` for each session-routing mode, then reads the exported
wire payloads / request headers and asserts the per-mode contract:

- ``dynamo_headers``: ``X-Dynamo-Session-ID`` on every request equals the
  session's ``x_correlation_id``; no parent header on root sessions; body is
  untouched (``"nvext" not in payload``).
- ``dynamo_nvext``: ``nvext.session_control`` carries ``bind`` (+ timeout) on
  every non-final turn and ``close`` (no timeout) on the final turn, with one
  stable ``session_id == x_correlation_id`` across the session.
- ``smg_routing_key``: ``X-SMG-Routing-Key`` equals ``x_correlation_id``.
- ``session_id_header`` (``--session-routing-opt header_name=X-Affinity``): the
  configured header equals ``x_correlation_id``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence

import pytest
from pytest import param

from aiperf.common.models import RawRecordInfo
from tests.harness.utils import AIPerfCLI, AIPerfMockServer

_TOKENIZER = "openai/gpt-oss-120b"  # pre-cached + offline in integration conftest

_NUM_SESSIONS = 2
_TURNS_PER_SESSION = 3
_TIMEOUT_SECONDS = 300


def _payload_dict(record: RawRecordInfo) -> dict:
    """Decode the exported wire payload regardless of which field carries it."""
    if record.payload is not None:
        return record.payload
    if record.payload_bytes is not None:
        return json.loads(record.payload_bytes)
    return {}


def _build_cmd(url: str, *, mode: str, opts: Sequence[str]) -> str:
    opt_flags = " ".join(f"--session-routing-opt {kv}" for kv in opts)
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
            --session-routing {mode} \
            {opt_flags} \
            --export-level raw \
            --ui simple
    """


async def _records_by_session(
    cli: AIPerfCLI, url: str, *, mode: str, opts: Sequence[str]
) -> dict[str, list[RawRecordInfo]]:
    """Run a benchmark and return each session's raw records ordered by
    turn_index, keyed by X-Correlation-ID."""
    result = await cli.run(_build_cmd(url, mode=mode, opts=opts), timeout=300.0)

    records = list(result.raw_records or [])
    assert records, f"no raw records\n{(result.log or '')[-1500:]}"

    grouped: dict[str, list[RawRecordInfo]] = defaultdict(list)
    for rec in records:
        grouped[rec.metadata.x_correlation_id].append(rec)

    assert len(grouped) == _NUM_SESSIONS, (
        f"expected {_NUM_SESSIONS} sessions, got {len(grouped)}"
    )
    out: dict[str, list[RawRecordInfo]] = {}
    for xcorr, recs in grouped.items():
        assert len(recs) == _TURNS_PER_SESSION, (
            f"session {xcorr}: expected {_TURNS_PER_SESSION} turns, got {len(recs)}"
        )
        recs.sort(key=lambda r: r.metadata.turn_index)
        out[xcorr] = recs
    return out


def _verify_dynamo_headers(by_session: dict[str, list[RawRecordInfo]]) -> None:
    for xcorr, recs in by_session.items():
        for rec in recs:
            headers = rec.request_headers or {}
            assert headers.get("X-Dynamo-Session-ID") == xcorr, (
                f"session {xcorr} turn {rec.metadata.turn_index}: headers={headers}"
            )
            # Root sessions have no parent, so the parent header must be absent.
            assert "X-Dynamo-Parent-Session-ID" not in headers, (
                f"session {xcorr}: root session emitted a parent header; {headers}"
            )
            assert "nvext" not in _payload_dict(rec), (
                f"session {xcorr}: header-mode must not mutate the body"
            )


def _verify_smg_routing_key(by_session: dict[str, list[RawRecordInfo]]) -> None:
    for xcorr, recs in by_session.items():
        for rec in recs:
            headers = rec.request_headers or {}
            assert headers.get("X-SMG-Routing-Key") == xcorr, (
                f"session {xcorr} turn {rec.metadata.turn_index}: headers={headers}"
            )
            assert "nvext" not in _payload_dict(rec), (
                f"session {xcorr}: header-mode must not mutate the body"
            )


def _verify_session_id_header(by_session: dict[str, list[RawRecordInfo]]) -> None:
    for xcorr, recs in by_session.items():
        for rec in recs:
            headers = rec.request_headers or {}
            assert headers.get("X-Affinity") == xcorr, (
                f"session {xcorr} turn {rec.metadata.turn_index}: headers={headers}"
            )
            assert "nvext" not in _payload_dict(rec), (
                f"session {xcorr}: header-mode must not mutate the body"
            )


def _verify_dynamo_nvext(by_session: dict[str, list[RawRecordInfo]]) -> None:
    for xcorr, recs in by_session.items():
        scs = []
        for rec in recs:
            sc = _payload_dict(rec).get("nvext", {}).get("session_control")
            assert sc is not None, (
                f"session {xcorr} turn {rec.metadata.turn_index}: "
                "every request must carry nvext.session_control"
            )
            scs.append(sc)

        actions = [sc.get("action") for sc in scs]
        assert all(a == "bind" for a in actions[:-1]), (
            f"session {xcorr}: non-final turns must bind; {actions}"
        )
        assert actions[-1] == "close", f"session {xcorr}: actions={actions}"
        assert "open" not in actions, f"session {xcorr}: emitted open; {actions}"

        # Every non-final 'bind' carries the timeout; 'close' does not.
        for sc in scs[:-1]:
            assert sc["timeout"] == _TIMEOUT_SECONDS, f"session {xcorr}: {sc}"
        assert "timeout" not in scs[-1], (
            f"session {xcorr}: close carried timeout; {scs[-1]}"
        )

        # One stable session_id == the X-Correlation-ID, on every turn.
        assert {sc["session_id"] for sc in scs} == {xcorr}, (
            f"session {xcorr}: session_id drift; {[sc.get('session_id') for sc in scs]}"
        )


_VERIFIERS: dict[str, Callable[[dict[str, list[RawRecordInfo]]], None]] = {
    "dynamo_headers": _verify_dynamo_headers,
    "dynamo_nvext": _verify_dynamo_nvext,
    "smg_routing_key": _verify_smg_routing_key,
    "session_id_header": _verify_session_id_header,
}


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, opts",
    [
        param("dynamo_headers", (), id="dynamo_headers"),
        param("dynamo_nvext", (f"timeout_seconds={_TIMEOUT_SECONDS}",), id="dynamo_nvext"),
        param("smg_routing_key", (), id="smg_routing_key"),
        param("session_id_header", ("header_name=X-Affinity",), id="session_id_header"),
    ],
)  # fmt: skip
async def test_session_routing_mode_reaches_wire(
    cli: AIPerfCLI,
    aiperf_mock_server: AIPerfMockServer,
    mode: str,
    opts: tuple[str, ...],
):
    """Each session-routing mode stamps its per-session identity on the wire and
    it survives to the raw export exactly as the plugin specifies."""
    by_session = await _records_by_session(
        cli, aiperf_mock_server.url, mode=mode, opts=opts
    )
    _VERIFIERS[mode](by_session)
