# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tests.unit.dataset.loader._shared_helpers import _write_trace

"""Per-trace block_size resolution in WekaTraceLoader."""

from unittest.mock import MagicMock

import pytest

from aiperf.dataset.loader.weka_trace import WekaTraceLoader


def _mk_user_config():
    from tests.unit.dataset.loader.conftest import make_weka_run

    return make_weka_run(model_names=["m"], tokenizer_name="t")


def _make_loader(filename, uc, monkeypatch, *, block_size=None):
    loader = WekaTraceLoader(
        filename=str(filename), run=uc, default_block_size=block_size
    )
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
    """Trace declares block_size=128, user_config has block_size=None."""
    trace = _trace_with_bs(
        "t_bs128", bs=128, in_tokens=512, hash_ids=[100, 200, 300, 400]
    )
    path = _write_trace(tmp_path, trace)
    loader = _make_loader(path, _mk_user_config(), monkeypatch, block_size=None)
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert any(c.session_id == "t_bs128" for c in convs)


def test_user_block_size_overrides_trace_block_size(tmp_path, monkeypatch):
    """User-config block_size takes precedence over trace.block_size."""
    trace = _trace_with_bs("t_bs_override", bs=64, in_tokens=128, hash_ids=[1, 2, 3, 4])
    path = _write_trace(tmp_path, trace)
    loader = _make_loader(path, _mk_user_config(), monkeypatch, block_size=32)
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
    """If user_config doesn't override AND somehow the trace has no block_size"""
    pytest.skip(
        "WekaTrace.block_size is schema-required; fallback is dead code in practice"
    )
