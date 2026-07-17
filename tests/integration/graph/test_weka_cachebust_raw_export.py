# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-mp assertion that --cache-bust first_turn_prefix stamps the marker.

Runs the full multiprocess ``aiperf`` stack over a 2-trace weka directory with
``--cache-bust first_turn_prefix --export-level raw`` and verifies on the EXPORTED
raw artifact that the AgentX-format marker ``[rid:<12hex>]\\n\\n`` is:

* present on the FIRST user message of every dispatched request,
* SHARED across all of one trace instance's dispatches (per-trace value), and
* DISTINCT across the two trace instances,

plus a ``--cache-bust none`` control proving the stamp is a no-op when off.
"""

from __future__ import annotations

import re
from pathlib import Path

import orjson
import pytest

from tests.harness.utils import AIPerfCLI, AIPerfMockServer

MULTIGRAPH_DIR = Path(__file__).parent / "fixtures" / "weka_multigraph_dir"
# AgentX marker: ``[rid:<12 hex>]`` followed by a blank line, at content start.
_RID_PREFIX = re.compile(r"^\[rid:([0-9a-f]{12})\]\n\n")


def _first_user_contents(raw_lines: list[str]) -> list[tuple[str, str]]:
    """Return ``(conversation_id, first_user_message_content)`` per raw record."""
    out: list[tuple[str, str]] = []
    for line in raw_lines:
        if not line.strip():
            continue
        rec = orjson.loads(line)
        conv = rec["metadata"]["conversation_id"]
        messages = rec["payload"]["messages"]
        user = next((m for m in messages if m.get("role") == "user"), None)
        if user is not None and isinstance(user.get("content"), str):
            out.append((conv, user["content"]))
    return out


async def _run(cli, mock, monkeypatch, cache_bust: str):
    result = await cli.run(
        f"""
        aiperf profile \
            --model claude-opus-4-5-20251101 \
            --url {mock.url} \
            --endpoint-type chat \
            --input-file {MULTIGRAPH_DIR} \
            --tokenizer gpt2 \
            --cache-bust {cache_bust} \
            --num-conversations 2 \
            --concurrency 2 \
            --workers-max 2 \
            --export-level raw \
            --ui simple
        """,
        timeout=300.0,
    )
    assert result.exit_code == 0, result.stderr[-2000:]
    raw = next(result.artifacts_dir.glob("**/*profile_export_raw.jsonl"), None)
    assert raw is not None, (
        "profile_export_raw.jsonl must exist with --export-level raw"
    )
    return _first_user_contents(raw.read_text(encoding="utf-8").splitlines())


@pytest.mark.integration
@pytest.mark.asyncio
class TestCacheBustRawExport:
    async def test_first_turn_prefix_stamped_per_trace(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contents = await _run(cli, aiperf_mock_server, monkeypatch, "first_turn_prefix")
        assert contents, "no dispatched user messages in the raw export"

        # Every dispatched first user message carries the marker at content start.
        # The cache-bust marker is stamped per TRACE INSTANCE; a subagent now
        # carries its own conversation_id (``{instance}::{subagent}``) but SHARES
        # its parent trace instance's marker -- so group rids by the trace
        # instance (the conversation_id minus any ``::subagent`` suffix).
        rid_by_instance: dict[str, set[str]] = {}
        for conv, content in contents:
            m = _RID_PREFIX.match(content)
            assert m is not None, (
                f"cache-bust marker missing/misplaced on {conv}: {content[:80]!r}"
            )
            instance = conv.split("::", 1)[0]
            rid_by_instance.setdefault(instance, set()).add(m.group(1))

        # Per trace instance: exactly one rid, shared across the root + every
        # subagent dispatch of that instance.
        for instance, rids in rid_by_instance.items():
            assert len(rids) == 1, (
                f"{instance} has multiple rids (not per-trace-instance): {rids}"
            )

        # Distinct across the two trace instances.
        all_rids = {next(iter(r)) for r in rid_by_instance.values()}
        assert len(rid_by_instance) >= 2 and len(all_rids) == len(rid_by_instance), (
            f"rids must be distinct per trace instance: {rid_by_instance}"
        )

    async def test_cache_bust_none_is_noop(
        self,
        cli: AIPerfCLI,
        aiperf_mock_server: AIPerfMockServer,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contents = await _run(cli, aiperf_mock_server, monkeypatch, "none")
        assert contents, "no dispatched user messages in the raw export"
        for conv, content in contents:
            assert _RID_PREFIX.match(content) is None, (
                f"cache-bust=none must NOT stamp a marker, but {conv} has one: "
                f"{content[:60]!r}"
            )
