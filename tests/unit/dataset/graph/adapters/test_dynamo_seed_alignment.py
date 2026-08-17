# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that ``from_dynamo_trace`` threads a concrete content root seed into the trie-lowering synthesizer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aiperf.common import random_generator as rng
from aiperf.dataset.graph.adapters.dynamo import trie_lowering
from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace
from tests.unit.dataset.graph.adapters.conftest import write_jsonl


def _dynamo_record(ts: int, sid: str, input_tokens: int, hashes: list[int]) -> dict:
    """Build a replay-bearing ``request_end`` record for session ``sid`` at ``ts``."""
    return {
        "schema": "dynamo.request.trace.v1",
        "event_type": "request_end",
        "event_time_unix_ms": ts,
        "event_source": "dynamo",
        "agent_context": {"session_id": sid},
        "request": {
            "request_id": f"r{ts}",
            "model": "m",
            "input_tokens": input_tokens,
            "output_tokens": 8,
            "cached_tokens": 0,
            "replay": {
                "trace_block_size": 16,
                "input_length": input_tokens,
                "input_sequence_hashes": hashes,
            },
        },
    }


@pytest.fixture
def trace_path(tmp_path: Path) -> Path:
    """A two-turn single-session dynamo trace with a growing shared prefix."""
    return write_jsonl(
        tmp_path / "dyn_seed.jsonl",
        [
            _dynamo_record(1000, "s1", 32, [111, 222]),
            _dynamo_record(2000, "s1", 64, [111, 222, 333, 444]),
        ],
    )


def _parse_capturing_seed(
    monkeypatch: pytest.MonkeyPatch, path: Path, seed: int | None
) -> int | None:
    """Parse ``path`` and return the root seed observed by ``dynamo_recon_callbacks``."""
    captured: list[int | None] = []
    real = trie_lowering.dynamo_recon_callbacks

    def spy(tokenizer: str, corpus: str, root_seed: int | None, **kwargs: Any):
        captured.append(root_seed)
        return real(tokenizer, corpus, root_seed, **kwargs)

    monkeypatch.setattr(trie_lowering, "dynamo_recon_callbacks", spy)
    from_dynamo_trace(path, content_root_seed=seed)
    assert len(captured) == 1
    return captured[0]


def test_from_dynamo_trace_explicit_seed_passes_through(
    monkeypatch: pytest.MonkeyPatch, trace_path: Path
) -> None:
    """An explicit ``content_root_seed`` reaches the synthesizer unchanged."""
    assert _parse_capturing_seed(monkeypatch, trace_path, 1234) == 1234


def test_from_dynamo_trace_none_seed_uses_ambient_root_seed(
    monkeypatch: pytest.MonkeyPatch, trace_path: Path
) -> None:
    """With no explicit seed, the ambient AIPerf root seed is used."""
    rng.reset()
    rng.init(777)
    assert _parse_capturing_seed(monkeypatch, trace_path, None) == 777


def test_from_dynamo_trace_unseeded_generates_per_run_seed(
    monkeypatch: pytest.MonkeyPatch, trace_path: Path
) -> None:
    """Unseeded parses each resolve fresh OS entropy into a concrete, distinct int seed."""
    rng.reset()
    first = _parse_capturing_seed(monkeypatch, trace_path, None)
    rng.reset()
    second = _parse_capturing_seed(monkeypatch, trace_path, None)
    assert isinstance(first, int)
    assert isinstance(second, int)
    assert first != second
