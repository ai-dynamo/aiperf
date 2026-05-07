# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.dataset.loader.weka_trace import WekaTraceLoader

FIXTURES = Path(__file__).parents[3] / "fixtures" / "weka_traces"


def _mk_user_config():
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.loadgen.inter_turn_delay_cap_seconds = None
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = None
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "test-tok"
    uc.endpoint.model_names = ["claude-opus-4-5-20251101", "claude-haiku-4-5-20251001"]
    return uc


def _stub_prompt_generator_for_reconstructor(loader) -> None:
    """Wire a MagicMock prompt_generator with the attrs the reconstructor needs.

    Reconstructor calls `_decode_blocks(hash_ids)` -> `_cache` lookup +
    `_sample_tokens` fallback + `tokenizer.decode`. ``sample_partial_tail`` (the
    mixin method) needs `_tokenized_corpus` and `_corpus_size`. ``_decode_block_tokens``
    consumes ``_hash_id_corpus_rng`` so its reseed/randrange surface is stubbed
    via ``stub_hash_id_corpus_rng``.
    """
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader.prompt_generator._sample_tokens.side_effect = lambda n: [0] * n
    loader.prompt_generator._tokenized_corpus = list(range(10000, 11000))
    loader.prompt_generator._corpus_size = 1000
    stub_hash_id_corpus_rng(loader.prompt_generator)
    loader.prompt_generator.tokenizer.decode.side_effect = (
        lambda toks: f"<dec:{len(toks)}>"
    )


def test_can_load_single_weka_file():
    assert WekaTraceLoader.can_load(filename=FIXTURES / "simple.json") is True


def test_can_load_detects_directory():
    assert WekaTraceLoader.can_load(filename=FIXTURES) is True


def test_can_load_rejects_non_weka_json(tmp_path: Path):
    p = tmp_path / "x.json"
    p.write_text('{"not": "weka"}')
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_rejects_non_json_file(tmp_path: Path):
    p = tmp_path / "x.txt"
    p.write_text("not json")
    assert WekaTraceLoader.can_load(filename=p) is False


def test_can_load_rejects_empty_directory(tmp_path: Path):
    assert WekaTraceLoader.can_load(filename=tmp_path) is False


def test_load_dataset_single_file_yields_one_trace():
    loader = WekaTraceLoader(
        filename=str(FIXTURES / "simple.json"), user_config=_mk_user_config()
    )
    data = loader.load_dataset()
    assert set(data.keys()) == {"trace_simple"}
    assert len(data["trace_simple"]) == 1  # one WekaTrace object


def test_load_dataset_directory_yields_one_per_file():
    loader = WekaTraceLoader(filename=str(FIXTURES), user_config=_mk_user_config())
    data = loader.load_dataset()
    # simple.json, one_subagent.json, terminal_subagent.json, multi_model.json
    assert "trace_simple" in data
    assert "trace_sa" in data
    assert "trace_term" in data


def test_load_dataset_rejects_extra_fields_with_filename(tmp_path):
    import shutil

    good = FIXTURES / "simple.json"
    bad = FIXTURES.parent / "weka_traces_invalid" / "bad_extra_field.json"
    d = tmp_path / "traces"
    d.mkdir()
    shutil.copy(good, d)
    shutil.copy(bad, d)
    loader = WekaTraceLoader(filename=str(d), user_config=_mk_user_config())
    with pytest.raises(ValueError, match="bad_extra_field.json"):
        loader.load_dataset()


def test_convert_to_conversations_builds_one_conversation_per_normal_request(
    monkeypatch,
):
    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES / "simple.json"), user_config=uc)

    # Required attributes set by __init__ (we bypass the real PromptGenerator wiring).
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    data = loader.load_dataset()
    convs = loader.convert_to_conversations(data)
    assert len(convs) == 1
    conv = convs[0]
    assert conv.session_id == "trace_simple"
    assert len(conv.turns) == 2
    assert conv.turns[0].model == "claude-opus-4-5-20251101"
    assert conv.turns[0].max_tokens == 30
    # Trace `t` is in seconds; Turn.timestamp/delay contract is milliseconds.
    assert conv.turns[0].timestamp == 0.0
    assert conv.turns[1].timestamp == 5000.0
    assert conv.turns[1].delay == pytest.approx(5000.0)
    # weka loader populates only ``Turn.raw_messages`` (the multi-message chat
    # form consumed by ChatEndpoint.build_messages). ``Turn.texts`` is left
    # at its default empty list — a separate full-prompt decode previously
    # populated it but no consumer reads it for chat-shape traces, so the
    # decode was removed.
    assert conv.turns[0].texts == []
    # Weka now emits delta-encoded turns. Turn 0 carries the full initial
    # state (system + user). Turn 1 may either be a strict append (just
    # asst + user_k) or a full re-emit (reset_context=True) if the LCP
    # truncate disturbed an emitted segment — both forms are valid; we
    # assert on the accumulated wire shape instead.
    turn_0_roles = [m["role"] for m in conv.turns[0].raw_messages]
    assert "user" in turn_0_roles
    assert "assistant" not in turn_0_roles
    assert conv.turns[0].reset_context is False
    turn_1_roles = [m["role"] for m in conv.turns[1].raw_messages]
    assert "assistant" in turn_1_roles
    assert "user" in turn_1_roles
    # If turn 1 was a strict append, system stays in turn 0 only; if it
    # was a reset, turn 1 carries the full state including system. Either
    # is correct under DELTAS_WITH_RESPONSES semantics.
    if conv.turns[1].reset_context:
        assert "system" in turn_1_roles
    else:
        assert "system" not in turn_1_roles
    # Accumulated state across both turns (mimicking what
    # BaseEndpoint.build_messages produces at request time) must contain
    # the full message-array prefix.
    accumulated: list[dict] = []
    for t in conv.turns:
        if t.reset_context:
            accumulated = list(t.raw_messages)
        else:
            accumulated.extend(t.raw_messages)
    accumulated_roles = [m["role"] for m in accumulated]
    assert "system" in accumulated_roles
    assert "assistant" in accumulated_roles
    assert "user" in accumulated_roles


def test_convert_to_conversations_emits_alternating_roles(monkeypatch):
    """Turn 1+ should have an assistant segment between the prefix-user content
    and the new user_k content (symmetric attribution rule, spec section 4.4.1)."""
    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES / "simple.json"), user_config=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    conv = convs[0]

    # Turn 0: just system / user (no asst).
    turn_0_roles = [m["role"] for m in conv.turns[0].raw_messages]
    assert "assistant" not in turn_0_roles

    # Turn 1: asst should appear before the new user_k segment.
    turn_1_roles = [m["role"] for m in conv.turns[1].raw_messages]
    assert "assistant" in turn_1_roles
    asst_idx = turn_1_roles.index("assistant")
    user_indices = [i for i, r in enumerate(turn_1_roles) if r == "user"]
    assert max(user_indices) > asst_idx, (
        f"asst should precede the new user_k segment; got roles={turn_1_roles}"
    )


def test_subagent_produces_child_conversation_and_branch_plus_prereq(monkeypatch):
    from aiperf.common.enums import ConversationBranchMode, PrerequisiteKind

    uc = _mk_user_config()
    loader = WekaTraceLoader(
        filename=str(FIXTURES / "one_subagent.json"), user_config=uc
    )

    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    # Parent + one subagent = 2 conversations.
    assert {c.session_id for c in convs} == {"trace_sa", "trace_sa::sa:agent_001"}
    parent = next(c for c in convs if c.session_id == "trace_sa")
    child = next(c for c in convs if c.session_id == "trace_sa::sa:agent_001")

    # Parent root turn declares one SPAWN branch.
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.mode == ConversationBranchMode.SPAWN
    assert branch.child_conversation_ids == ["trace_sa::sa:agent_001"]
    assert parent.turns[0].branch_ids == [branch.branch_id]

    # Parent's next turn carries a SPAWN_JOIN prereq referencing the branch.
    assert len(parent.turns[1].prerequisites) == 1
    p = parent.turns[1].prerequisites[0]
    assert p.kind == PrerequisiteKind.SPAWN_JOIN
    assert p.branch_id == branch.branch_id

    # Child conversation has one inner turn.
    assert len(child.turns) == 1
    assert child.turns[0].model == "claude-haiku-4-5-20251001"


def test_terminal_subagent_becomes_background_branch_no_prereq(monkeypatch):
    from aiperf.common.enums import ConversationBranchMode

    uc = _mk_user_config()
    loader = WekaTraceLoader(
        filename=str(FIXTURES / "terminal_subagent.json"), user_config=uc
    )
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "trace_term")
    assert len(parent.branches) == 1
    branch = parent.branches[0]
    assert branch.is_background is True
    assert branch.mode == ConversationBranchMode.SPAWN
    # Only one parent turn exists -> no prereq anywhere.
    assert all(not t.prerequisites for t in parent.turns)


def test_filters_requests_exceeding_max_isl(monkeypatch):
    uc = _mk_user_config()
    uc.input.synthesis.max_isl = 210  # simple.json has in=200 and in=250
    loader = WekaTraceLoader(filename=str(FIXTURES / "simple.json"), user_config=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    convs = loader.convert_to_conversations(loader.load_dataset())
    conv = convs[0]
    assert len(conv.turns) == 1
    assert conv.turns[0].timestamp == 0.0


def test_caps_max_osl(monkeypatch):
    uc = _mk_user_config()
    uc.input.synthesis.max_osl = 25
    loader = WekaTraceLoader(filename=str(FIXTURES / "simple.json"), user_config=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    convs = loader.convert_to_conversations(loader.load_dataset())
    for t in convs[0].turns:
        assert t.max_tokens <= 25


def test_trace_model_rewritten_to_configured_model_zero(monkeypatch):
    """Trace's per-request model is unconditionally rewritten to model_names[0]."""
    uc = _mk_user_config()
    uc.endpoint.model_names = ["override-model"]
    loader = WekaTraceLoader(filename=str(FIXTURES / "simple.json"), user_config=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    convs = loader.convert_to_conversations(loader.load_dataset())
    for c in convs:
        for t in c.turns:
            assert t.model == "override-model"


def test_orphaned_subagent_is_dropped_when_preceding_turn_filtered(monkeypatch):
    # Raise the bar so BOTH parent turns in one_subagent.json get filtered (in=200, in=400).
    uc = _mk_user_config()
    uc.input.synthesis.max_isl = 50
    loader = WekaTraceLoader(
        filename=str(FIXTURES / "one_subagent.json"), user_config=uc
    )
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    convs = loader.convert_to_conversations(loader.load_dataset())
    parent = next(c for c in convs if c.session_id == "trace_sa")
    # No parent turns remain -> subagent branch also dropped.
    assert parent.branches == []


# --- Hash content scoped per (trace_id, hash_id) ---


def _real_pg():
    """Build a PromptGenerator-shape mock with a real HashIdRandomGenerator.

    We bypass full PromptGenerator init (it loads a tokenizer corpus) and only
    populate the surface ``_decode_block_tokens`` actually touches: the int-keyed
    cache, the hash-id rng, and a tiny synthetic tokenized corpus.
    """
    from aiperf.common.hash_id_random_generator import HashIdRandomGenerator
    from aiperf.common.random_generator import RandomGenerator

    pg = MagicMock()
    base_rng = RandomGenerator(0, _internal=True)
    pg._hash_id_corpus_rng = HashIdRandomGenerator.from_base_rng(base_rng)
    pg._cache = {}
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    return pg


def _real_loader_with_pg(pg):
    uc = _mk_user_config()
    loader = WekaTraceLoader(filename=str(FIXTURES / "two_turns.json"), user_config=uc)
    loader.prompt_generator = pg
    loader._block_size = 64
    return loader


def test_decode_block_tokens_distinct_across_scopes():
    """Same hash_id under different trace scopes must produce different tokens.

    The kv-cache-tester corpus declares ``hash_id_scope: "local"``; identical
    ``hash_id`` values in different traces must map to distinct content so
    the model under test sees the cache MISSES the recording cluster saw,
    not artificial cross-trace HITS.
    """
    pg = _real_pg()
    loader = _real_loader_with_pg(pg)

    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id("trace_alpha")
    a = loader._decode_block_tokens([1])

    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id("trace_beta")
    b = loader._decode_block_tokens([1])

    assert a != b
    assert len(a) == 64 and len(b) == 64


def test_decode_block_tokens_deterministic_within_scope():
    """Same (scope, hash_id) called twice (after cache clear and reseed) is
    byte-identical — required for cross-process reproducibility."""
    pg = _real_pg()
    loader = _real_loader_with_pg(pg)

    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id("trace_alpha")
    a1 = loader._decode_block_tokens([7])

    pg._cache.clear()
    pg._hash_id_corpus_rng.set_trace_id("trace_alpha")
    a2 = loader._decode_block_tokens([7])

    assert a1 == a2


def test_decode_block_tokens_deterministic_across_loaders():
    """Two freshly built loaders with the same seed produce identical bytes for
    the same (scope, hash_id) — stand-in for cross-process reproducibility."""
    pg1 = _real_pg()
    loader1 = _real_loader_with_pg(pg1)
    pg1._hash_id_corpus_rng.set_trace_id("trace_x")
    a = loader1._decode_block_tokens([3, 5, 11])

    pg2 = _real_pg()
    loader2 = _real_loader_with_pg(pg2)
    pg2._hash_id_corpus_rng.set_trace_id("trace_x")
    b = loader2._decode_block_tokens([3, 5, 11])

    assert a == b


def test_ignore_trace_delays_nulls_timestamp_and_delay(monkeypatch):
    """When ``ignore_trace_delays=True``, parent and child turns must have
    ``timestamp`` and ``delay`` set to None so concurrency / request-rate
    timing modes dispatch back-to-back instead of replaying recorded gaps."""
    uc = _mk_user_config()
    uc.input.ignore_trace_delays = True
    loader = WekaTraceLoader(
        filename=str(FIXTURES / "one_subagent.json"), user_config=uc
    )
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    assert len(convs) >= 2  # parent + at least one subagent child
    for conv in convs:
        for turn in conv.turns:
            assert turn.timestamp is None
            assert turn.delay is None


def test_use_think_time_only_emits_recorded_think_time_as_delay(monkeypatch, tmp_path):
    """When ``use_think_time_only=True``, ``Turn.delay`` should equal each
    request's recorded ``think_time * 1000`` (ms), not the full
    ``(t_curr - t_prev) * 1000`` inter-request delta. The first turn always has
    delay=None. Falls back to the full delta if a request's ``think_time`` is
    None."""
    import orjson

    trace = {
        "id": "trace_tt",
        "models": ["claude-opus-4-5-20251101"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 100,
                "out": 10,
                "hash_ids": [1, 2],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 5.5,
                "think_time": 0.0,
            },
            {
                "t": 12.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 200,
                "out": 20,
                "hash_ids": [3, 4],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 4.0,
                "think_time": 7.0,
            },
            {
                "t": 25.0,
                "type": "n",
                "model": "claude-opus-4-5-20251101",
                "in": 300,
                "out": 30,
                "hash_ids": [5, 6],
                "input_types": ["text"],
                "output_types": ["text"],
                "stop": "end_turn",
                "api_time": 3.0,
                "think_time": None,  # forces fallback to full delta
            },
        ],
    }
    f = tmp_path / "trace_tt.json"
    f.write_bytes(orjson.dumps(trace))

    uc = _mk_user_config()
    uc.input.use_think_time_only = True
    loader = WekaTraceLoader(filename=str(f), user_config=uc)
    _stub_prompt_generator_for_reconstructor(loader)
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64

    convs = loader.convert_to_conversations(loader.load_dataset())
    turns = convs[0].turns
    assert len(turns) == 3
    assert turns[0].delay is None  # first turn always
    assert (
        turns[1].delay == 7000.0
    )  # think_time=7.0s -> 7000ms (NOT 12000ms full delta)
    assert turns[2].delay == 13000.0  # think_time=None -> falls back to (25-12)*1000
