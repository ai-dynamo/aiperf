# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic Mooncake v2 input preserves the request graph and hash sharing."""

import json
from pathlib import Path
from types import SimpleNamespace

import orjson
import pytest

from aiperf.common.enums import ReplayDependencyEvent
from aiperf.common.environment import Environment
from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset.content import PromptConfig
from aiperf.dataset.generator.coding_content import CodingContentGenerator
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.agentic_mooncake_v2 import AgenticMooncakeV2Loader

# Copied from ai-dynamo/aisim at 739c9d1585a73dc092b2afa95575644d47480adc.
FIXTURE = Path("tests/fixtures/agentic-mooncake-v2.jsonl")


class PromptGeneratorStub:
    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace(resolved_name="test-tokenizer")
        self._cache = {}
        self._hash_id_corpus_rng = SimpleNamespace(set_trace_id=lambda _: None)

    def _build_token_sequence(self, input_length, hash_ids, block_size):
        return [token for block in hash_ids for token in [block] * block_size][
            :input_length
        ]


def _loader(path, monkeypatch):
    monkeypatch.setattr(
        "aiperf.dataset.loader.hash_ids_synthesis.parallel_decode",
        lambda sequences, *args, **kwargs: [
            " ".join(map(str, seq)) for seq in sequences
        ],
    )
    return AgenticMooncakeV2Loader(
        filename=path, prompt_generator=PromptGeneratorStub()
    )


def test_aisim_fixture_preserves_explicit_edges(monkeypatch):
    loader = _loader(FIXTURE, monkeypatch)
    assert loader.can_load(orjson.loads(FIXTURE.read_bytes().splitlines()[0]))
    conversations = loader.convert_to_conversations(loader.load_dataset())
    by_id = {conversation.session_id: conversation for conversation in conversations}
    play = by_id["v2-play:agent-play"]
    assert play.is_orchestrator and play.turns[0].no_request
    assert set(play.branches[0].child_conversation_ids) == {
        "v2-request:agent-root",
        "v2-request:agent-child",
        "v2-request:agent-join",
    }
    child = by_id["v2-request:agent-child"].turns[0]
    assert child.timestamp == 0
    assert child.extra_headers == {
        "x-dynamo-session-id": "child",
        "x-dynamo-parent-session-id": "root",
    }
    assert [
        (r.conversation_id, r.event, r.delay_ns, r.relation)
        for r in child.replay_predecessors
    ] == [("v2-request:agent-root", ReplayDependencyEvent.DISPATCH, 5_000_000, "spawn")]
    join = by_id["v2-request:agent-join"].turns[0]
    assert {
        (r.conversation_id, r.event, r.delay_ns, r.relation)
        for r in join.replay_predecessors
    } == {
        (
            "v2-request:agent-root",
            ReplayDependencyEvent.COMPLETION,
            1_000_000,
            "sequence",
        ),
        ("v2-request:agent-child", ReplayDependencyEvent.COMPLETION, 1_000_000, "join"),
    }


def test_unknown_dependency_is_rejected(tmp_path, monkeypatch):
    rows = [orjson.loads(line) for line in FIXTURE.read_bytes().splitlines()]
    rows[2]["dependencies"][0]["request_id"] = "missing"
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    loader = _loader(path, monkeypatch)
    with pytest.raises(ValueError, match="missing"):
        loader.convert_to_conversations(loader.load_dataset())


def test_multiple_roots_preserve_shared_hash_prefix_and_independent_floors(
    tmp_path, monkeypatch
):
    rows = [orjson.loads(line) for line in FIXTURE.read_bytes().splitlines()]
    first = rows[1] | {"input_length": 8, "hash_ids": [301, 401], "not_before_ms": 100}
    second = rows[2] | {
        "input_length": 8,
        "hash_ids": [301, 402],
        "not_before_ms": 120,
        "dependencies": [],
    }
    child = rows[3] | {
        "not_before_ms": 0,
        "dependencies": [
            {
                "request_id": "agent-child",
                "trigger": "completion",
                "delay_ms": 1,
                "relation": "join",
            }
        ],
    }
    path = tmp_path / "multi-root.jsonl"
    path.write_bytes(
        b"\n".join(orjson.dumps(row) for row in [rows[0], first, second, child]) + b"\n"
    )
    loader = _loader(path, monkeypatch)
    conversations = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }
    root_a = conversations["v2-request:agent-root"].turns[0]
    root_b = conversations["v2-request:agent-child"].turns[0]
    join = conversations["v2-request:agent-join"].turns[0]
    assert (root_a.timestamp, root_b.timestamp, join.timestamp) == (0, 20, 0)
    tokens_a = root_a.texts[0].contents[0].split()
    tokens_b = root_b.texts[0].contents[0].split()
    assert tokens_a[:4] == tokens_b[:4] == ["301"] * 4
    assert tokens_a[4:] == ["401"] * 4
    assert tokens_b[4:] == ["402"] * 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("priority", 1),
        ("strict_priority", 1),
        ("policy_class", "gold"),
        ("output_token_ids", [1, 2]),
        ("output_length", 0),
    ],
)
def test_unmapped_request_features_are_not_silently_discarded(
    tmp_path, monkeypatch, field, value
):
    rows = [orjson.loads(line) for line in FIXTURE.read_bytes().splitlines()]
    rows[1][field] = value
    path = tmp_path / "unsupported.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(row) for row in rows) + b"\n")
    with pytest.raises(
        NotImplementedError,
        match="play 'agent-play' session 'root' request 'agent-root': .*not supported",
    ):
        _loader(path, monkeypatch).load_dataset()


@pytest.mark.parametrize("hash_id", [-1, 2**64])
def test_out_of_range_hash_id_is_rejected(tmp_path, monkeypatch, hash_id):
    rows = [orjson.loads(line) for line in FIXTURE.read_bytes().splitlines()]
    rows[1]["hash_ids"] = [hash_id]
    path = tmp_path / "invalid-hash.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash_ids"):
        _loader(path, monkeypatch).load_dataset()


def test_out_of_range_strict_priority_is_malformed(tmp_path, monkeypatch):
    rows = [orjson.loads(line) for line in FIXTURE.read_bytes().splitlines()]
    rows[1]["strict_priority"] = 2**32
    path = tmp_path / "invalid-priority.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(row) for row in rows) + b"\n")
    with pytest.raises(ValueError, match="strict_priority"):
        _loader(path, monkeypatch).load_dataset()


def test_same_hash_in_full_and_partial_blocks_uses_one_block(tmp_path, monkeypatch):
    header = orjson.loads(FIXTURE.read_bytes().splitlines()[0])
    base = {
        "play_id": "play",
        "session_id": "session",
        "model": "model",
        "output_length": 1,
        "not_before_ms": 0,
    }
    rows = [
        base | {"request_id": "a", "input_length": 2, "hash_ids": [301]},
        base | {"request_id": "b", "input_length": 4, "hash_ids": [301]},
        base | {"request_id": "c", "input_length": 3, "hash_ids": [301]},
        base | {"request_id": "d", "input_length": 6, "hash_ids": [301, 301]},
    ]
    path = tmp_path / "partial.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(row) for row in [header, *rows]) + b"\n")
    sequences = []

    def decode(token_sequences, *args, **kwargs):
        sequences.extend(token_sequences)
        return [" ".join(map(str, tokens)) for tokens in token_sequences]

    monkeypatch.setattr(
        "aiperf.dataset.loader.hash_ids_synthesis.parallel_decode", decode
    )
    tokenizer = Tokenizer.from_pretrained("builtin")
    generator = PromptGenerator(
        prompts=PromptConfig(block_size=4), prefix_prompts=None, tokenizer=tokenizer
    )
    loader = AgenticMooncakeV2Loader(filename=path, prompt_generator=generator)
    loader.convert_to_conversations(loader.load_dataset())

    partial_two, full, partial_three, repeated = sequences
    assert partial_two == full[:2]
    assert partial_three == full[:3]
    assert repeated == full + full[:2]


@pytest.mark.parametrize("corpus", ["sonnet", "coding"])
def test_one_token_blocks_preserve_shared_hashes(tmp_path, monkeypatch, corpus):
    header = orjson.loads(FIXTURE.read_bytes().splitlines()[0]) | {"block_size": 1}
    base = {
        "play_id": "play",
        "session_id": "session",
        "model": "model",
        "output_length": 1,
        "not_before_ms": 0,
    }
    rows = [
        base | {"request_id": "a", "input_length": 2, "hash_ids": [301, 302]},
        base | {"request_id": "b", "input_length": 1, "hash_ids": [301]},
    ]
    path = tmp_path / "single-token-blocks.jsonl"
    path.write_bytes(b"\n".join(orjson.dumps(row) for row in [header, *rows]) + b"\n")
    sequences = []

    def decode(token_sequences, *args, **kwargs):
        sequences.extend(token_sequences)
        return [" ".join(map(str, tokens)) for tokens in token_sequences]

    monkeypatch.setattr(
        "aiperf.dataset.loader.hash_ids_synthesis.parallel_decode", decode
    )
    tokenizer = Tokenizer.from_pretrained("builtin")
    config = PromptConfig(block_size=1)
    generator = (
        CodingContentGenerator(config, tokenizer)
        if corpus == "coding"
        else PromptGenerator(prompts=config, prefix_prompts=None, tokenizer=tokenizer)
    )
    loader = AgenticMooncakeV2Loader(filename=path, prompt_generator=generator)
    loader.convert_to_conversations(loader.load_dataset())
    assert len(sequences[0]) == 2
    assert sequences[0][:1] == sequences[1]


def test_derived_dynamo_session_headers_cannot_replace_source_ids(monkeypatch):
    monkeypatch.setattr(
        Environment.HTTP, "X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID", True
    )
    with pytest.raises(ValueError, match="X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID"):
        _loader(FIXTURE, monkeypatch)


def test_nested_and_ambiguous_source_session_lineage(tmp_path, monkeypatch):
    rows = [orjson.loads(line) for line in FIXTURE.read_bytes().splitlines()]
    root, child = rows[1:3]
    grandchild = child | {
        "request_id": "grandchild",
        "session_id": "grandchild",
        "dependencies": [
            {
                "request_id": "agent-child",
                "trigger": "dispatch",
                "delay_ms": 0,
                "relation": "spawn",
            }
        ],
    }
    ambiguous = child | {
        "request_id": "ambiguous",
        "session_id": "ambiguous",
        "dependencies": [
            {
                "request_id": "agent-root",
                "trigger": "dispatch",
                "delay_ms": 0,
                "relation": "spawn",
            },
            {
                "request_id": "agent-child",
                "trigger": "dispatch",
                "delay_ms": 0,
                "relation": "spawn",
            },
        ],
    }
    path = tmp_path / "nested.jsonl"
    path.write_bytes(
        b"\n".join(
            orjson.dumps(row) for row in [rows[0], root, child, grandchild, ambiguous]
        )
        + b"\n"
    )
    loader = _loader(path, monkeypatch)
    conversations = {
        c.session_id: c for c in loader.convert_to_conversations(loader.load_dataset())
    }
    assert conversations["v2-request:grandchild"].turns[0].extra_headers == {
        "x-dynamo-session-id": "grandchild",
        "x-dynamo-parent-session-id": "child",
    }
    assert conversations["v2-request:ambiguous"].turns[0].extra_headers == {
        "x-dynamo-session-id": "ambiguous"
    }
