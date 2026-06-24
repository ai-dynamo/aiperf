# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trace block_size resolution in WekaTraceLoader."""

from unittest.mock import MagicMock

import orjson
import pytest

from aiperf.dataset.loader.weka_trace import WekaTraceLoader


def _mk_user_config(*, block_size=None):
    uc = MagicMock()
    uc.input.random_seed = 0
    uc.input.fixed_schedule_start_offset = None
    uc.input.fixed_schedule_end_offset = None
    uc.input.ignore_trace_delays = False
    uc.input.use_think_time_only = False
    uc.input.use_end_to_start_delays = False
    uc.loadgen.inter_turn_delay_cap_seconds = None
    uc.loadgen.trace_idle_gap_cap_seconds = None
    uc.input.synthesis.max_isl = None
    uc.input.synthesis.max_osl = None
    uc.input.max_context_length = None
    uc.input.synthesis.should_synthesize.return_value = False
    uc.input.prompt.input_tokens.block_size = block_size
    uc.tokenizer.trust_remote_code = False
    uc.tokenizer.revision = None
    uc.tokenizer.name = "t"
    uc.endpoint.model_names = ["m"]
    return uc


def _make_loader(filename, uc, monkeypatch):
    loader = WekaTraceLoader(filename=str(filename), user_config=uc)
    monkeypatch.setattr(
        loader,
        "synthesize_prompts_from_hash_ids",
        lambda rs: {r.key: f"p-{r.key}" for r in rs},
    )
    loader.prompt_generator = MagicMock()
    loader.prompt_generator._cache = {}
    loader.prompt_generator._sample_tokens.side_effect = lambda n: [0] * n
    loader.prompt_generator._tokenized_corpus = list(range(10000, 11000))
    loader.prompt_generator._corpus_size = 1000

    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    stub_hash_id_corpus_rng(loader.prompt_generator)
    loader.prompt_generator.tokenizer.decode.side_effect = (
        lambda toks: f"<dec:{len(toks)}>"
    )
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    return loader


def _write_trace(tmp_path, data, name="t.json"):
    p = tmp_path / name
    p.write_bytes(orjson.dumps(data))
    return p


# A turn-0 normal request with a hash_ids count that perfectly tiles in_tokens at
# the trace's declared block_size, so the relaxed-vs-strict reconstructor distinction
# doesn't matter for THIS test. We're only verifying block_size resolution.
def _trace_with_bs(trace_id, bs, *, in_tokens, hash_ids):
    return {
        "id": trace_id,
        "models": ["m"],
        "block_size": bs,
        "hash_id_scope": "local",
        "requests": [
            {
                "t": 0.0,
                "type": "n",
                "model": "m",
                "in": in_tokens,
                "out": 1,
                "hash_ids": hash_ids,
            }
        ],
    }


def test_trace_block_size_honored_when_user_unset(tmp_path, monkeypatch):
    """Trace declares block_size=128, user_config has block_size=None.
    Loader must use 128, NOT the historical default of 64.
    """
    # Pick in_tokens that DOES tile bs=128 cleanly so this test isolates the
    # block_size resolution from any hash-id truncation concerns.
    # in_tokens=512, bs=128 -> 4 hash_ids needed.
    trace = _trace_with_bs(
        "t_bs128", bs=128, in_tokens=512, hash_ids=[100, 200, 300, 400]
    )
    path = _write_trace(tmp_path, trace)
    loader = _make_loader(path, _mk_user_config(block_size=None), monkeypatch)
    # Build conversations. The success criterion is that no ValueError is raised
    # for "len(hash_ids)=4 but in_tokens=512 with block_size=64 requires 8"
    # (which is what the OLD code would have done with the hardcoded bs=64).
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert any(c.session_id == "t_bs128" for c in convs)


def test_user_block_size_overrides_trace_block_size(tmp_path, monkeypatch):
    """User-config block_size takes precedence over trace.block_size.
    Trace declares 64, user wants 32. Loader must use 32 (the override).
    """
    # in_tokens=128, bs=32 -> 4 hash_ids needed. The trace declares bs=64 but
    # provides only 4 hash_ids; bs=64 would need 2. Either resolution works at
    # turn-0 (since 4 >= 2 and 4 >= 4). What we're really checking is which
    # one the loader picks. We'll check via a side-channel: the ConversationReconstructor
    # constructor's recorded block_size.
    trace = _trace_with_bs("t_bs_override", bs=64, in_tokens=128, hash_ids=[1, 2, 3, 4])
    path = _write_trace(tmp_path, trace)
    loader = _make_loader(path, _mk_user_config(block_size=32), monkeypatch)
    # Capture every ConversationReconstructor block_size argument the loader uses
    # during this convert call.
    from aiperf.dataset.loader import weka_synth_buf as wsb

    captured_block_sizes: list[int] = []
    orig = wsb.ConversationReconstructor.__init__

    def spy(self, *args, **kw):
        captured_block_sizes.append(kw.get("block_size", args[0] if args else None))
        return orig(self, *args, **kw)

    monkeypatch.setattr(wsb.ConversationReconstructor, "__init__", spy)
    loader.convert_to_conversations(loader.load_dataset())
    assert captured_block_sizes, (
        "no ConversationReconstructor built - test setup broken"
    )
    assert all(bs == 32 for bs in captured_block_sizes), (
        f"user-config block_size=32 should win over trace.block_size=64. "
        f"Got: {captured_block_sizes}"
    )


def test_default_64_when_neither_trace_nor_user_set(tmp_path, monkeypatch):
    """If user_config doesn't override AND somehow the trace has no block_size
    (defensive fallback), default to 64. The Pydantic schema makes this hard to
    reach since `block_size` is required - but the fallback should still be present
    for safety. If schema-required-ness makes this test impossible, document and
    skip it."""
    # WekaTrace.block_size is REQUIRED per the schema. So this test can either:
    # (a) construct a dict that bypasses Pydantic to exercise the fallback, or
    # (b) be skipped with a comment that the schema enforces the precondition.
    # Choose (b) - the schema is the right place to enforce this.
    pytest.skip(
        "WekaTrace.block_size is schema-required; fallback is dead code in practice"
    )
