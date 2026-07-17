# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson

from aiperf.common import random_generator as rng
from aiperf.dataset.graph.adapters.dynamo import trie_lowering
from aiperf.dataset.graph.adapters.dynamo.trace import from_dynamo_trace


def _dynamo_record(ts: int, sid: str, input_tokens: int, hashes: list[int]) -> dict:
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


def _write_trace(tmp_path: Path) -> Path:
    p = tmp_path / "dyn_seed.jsonl"
    records = [
        _dynamo_record(1000, "s1", 32, [111, 222]),
        _dynamo_record(2000, "s1", 64, [111, 222, 333, 444]),
    ]
    p.write_bytes(b"\n".join(orjson.dumps(r) for r in records))
    return p


def _parse_capturing_seed(monkeypatch, path: Path, seed: int | None) -> int | None:
    captured: list[int | None] = []
    real = trie_lowering.dynamo_recon_callbacks

    def spy(tokenizer: str, corpus: str, root_seed: int | None, **kwargs: Any):
        captured.append(root_seed)
        return real(tokenizer, corpus, root_seed, **kwargs)

    monkeypatch.setattr(trie_lowering, "dynamo_recon_callbacks", spy)
    from_dynamo_trace(path, content_root_seed=seed)
    assert len(captured) == 1
    return captured[0]


def test_from_dynamo_trace_explicit_seed_passes_through(monkeypatch, tmp_path) -> None:
    assert _parse_capturing_seed(monkeypatch, _write_trace(tmp_path), 1234) == 1234


def test_from_dynamo_trace_none_seed_uses_ambient_root_seed(
    monkeypatch, tmp_path
) -> None:
    rng.reset()
    rng.init(777)
    assert _parse_capturing_seed(monkeypatch, _write_trace(tmp_path), None) == 777


def test_from_dynamo_trace_unseeded_generates_per_run_seed(
    monkeypatch, tmp_path
) -> None:
    # No ambient root seed: each parse resolves fresh OS entropy — concrete
    # int threaded to the synthesizer, distinct across parses.
    rng.reset()
    path = _write_trace(tmp_path)
    first = _parse_capturing_seed(monkeypatch, path, None)
    rng.reset()
    second = _parse_capturing_seed(monkeypatch, path, None)
    assert isinstance(first, int)
    assert isinstance(second, int)
    assert first != second
