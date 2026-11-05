# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-trace block_size resolution in WekaTraceLoader."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pytest import param

from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.plugin import plugins
from aiperf.plugin.enums import CustomDatasetType
from tests.unit.dataset.loader._shared_helpers import _write_trace


def _mk_user_config():
    from tests.unit.dataset.loader.conftest import make_weka_run

    return make_weka_run(model_names=["m"], tokenizer_name="t")


def _make_loader(filename, uc, monkeypatch, *, block_size=None):
    # ``block_size`` is passed as ``default_block_size``, as CustomDatasetComposer
    # does with the plugin metadata value; it must not override the trace's own.
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
    """Trace block_size=128 with user override unset: loader must use 128, not the historical default of 64."""
    # Pick in_tokens that DOES tile bs=128 cleanly so this test isolates the
    # block_size resolution from any hash-id truncation concerns.
    # in_tokens=512, bs=128 -> 4 hash_ids needed.
    trace = _trace_with_bs(
        "t_bs128", bs=128, in_tokens=512, hash_ids=[100, 200, 300, 400]
    )
    path = _write_trace(tmp_path, trace)
    loader = _make_loader(path, _mk_user_config(), monkeypatch, block_size=None)
    # Build conversations. The success criterion is that no ValueError is raised
    # for "len(hash_ids)=4 but in_tokens=512 with block_size=64 requires 8"
    # (which is what the OLD code would have done with the hardcoded bs=64).
    convs = loader.convert_to_conversations(loader.load_dataset())
    assert any(c.session_id == "t_bs128" for c in convs)


def _capture_reconstructor_block_sizes(monkeypatch) -> list[int]:
    """Record the block_size of every ConversationReconstructor the loader builds."""
    from aiperf.dataset.loader import weka_synth_buf as wsb

    captured: list[int] = []
    orig = wsb.ConversationReconstructor.__init__

    def spy(self, *args, **kw):
        captured.append(kw.get("block_size", args[0] if args else None))
        return orig(self, *args, **kw)

    monkeypatch.setattr(wsb.ConversationReconstructor, "__init__", spy)
    return captured


@pytest.mark.parametrize(
    "trace_bs",
    [
        param(16, id="smaller-than-plugin-default"),
        param(128, id="larger-than-plugin-default"),
    ],
)  # fmt: skip
def test_trace_block_size_wins_over_plugin_default_block_size(
    tmp_path, monkeypatch, trace_bs
):
    """A trace's declared block_size is replayed as-is although the composer always passes the plugin's default_block_size (64).

    Regression: the plugin default was treated as a user override, so a 128-token
    trace was rebuilt from 64-token blocks -- its hash_ids covered only half of each
    prompt and the rest was synthesized fresh every turn, halving prefix-cache reuse.
    """
    plugin_default = plugins.get_dataset_loader_metadata(
        CustomDatasetType.WEKA_TRACE
    ).default_block_size
    assert plugin_default == 64, "precondition: plugin default differs from the trace"
    trace = _trace_with_bs(
        f"t_bs{trace_bs}_plugin_default",
        bs=trace_bs,
        in_tokens=4 * trace_bs,
        hash_ids=[100, 200, 300, 400],
    )
    path = _write_trace(tmp_path, trace)
    # Build the loader the way CustomDatasetComposer does for weka_trace.
    loader = _make_loader(
        path, _mk_user_config(), monkeypatch, block_size=plugin_default
    )
    captured = _capture_reconstructor_block_sizes(monkeypatch)
    loader.convert_to_conversations(loader.load_dataset())
    assert captured, "no ConversationReconstructor built - test setup broken"
    assert all(bs == trace_bs for bs in captured), (
        f"trace-declared block_size={trace_bs} must win over the plugin default "
        f"{plugin_default}. Got: {captured}"
    )


class _DatasetWithPromptsBlockSize:
    """The run's real default dataset plus ``prompts.block_size`` -- the loader's user override."""

    def __init__(self, dataset, block_size):
        self._dataset = dataset
        self.prompts = SimpleNamespace(block_size=block_size)

    def __getattr__(self, name):
        return getattr(self._dataset, name)


def test_user_block_size_overrides_trace_block_size(tmp_path, monkeypatch):
    """A configured prompts.block_size overrides trace.block_size: trace declares 64, user sets 32, loader must use 32."""
    # in_tokens=128, bs=32 -> 4 hash_ids needed. The trace declares bs=64 but
    # provides only 4 hash_ids; bs=64 would need 2. Either resolution works at
    # turn-0 (since 4 >= 2 and 4 >= 4). What we're really checking is which
    # one the loader picks, via the ConversationReconstructor's block_size.
    trace = _trace_with_bs("t_bs_override", bs=64, in_tokens=128, hash_ids=[1, 2, 3, 4])
    path = _write_trace(tmp_path, trace)
    uc = _mk_user_config()
    dataset = uc.cfg.get_default_dataset()
    monkeypatch.setattr(
        type(uc.cfg),
        "get_default_dataset",
        lambda self: _DatasetWithPromptsBlockSize(dataset, 32),
    )
    plugin_default = plugins.get_dataset_loader_metadata(
        CustomDatasetType.WEKA_TRACE
    ).default_block_size
    loader = _make_loader(path, uc, monkeypatch, block_size=plugin_default)
    captured = _capture_reconstructor_block_sizes(monkeypatch)
    loader.convert_to_conversations(loader.load_dataset())
    assert captured, "no ConversationReconstructor built - test setup broken"
    assert all(bs == 32 for bs in captured), (
        f"user-config block_size=32 should win over trace.block_size=64. Got: {captured}"
    )


def test_default_64_when_neither_trace_nor_user_set(tmp_path, monkeypatch):
    """With neither trace nor user block_size set the fallback default is 64; skipped since the schema makes block_size required."""
    # WekaTrace.block_size is REQUIRED per the schema. So this test can either:
    # (a) construct a dict that bypasses Pydantic to exercise the fallback, or
    # (b) be skipped with a comment that the schema enforces the precondition.
    # Choose (b) - the schema is the right place to enforce this.
    pytest.skip(
        "WekaTrace.block_size is schema-required; fallback is dead code in practice"
    )
