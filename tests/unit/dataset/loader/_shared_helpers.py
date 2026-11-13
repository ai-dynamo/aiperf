# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock

import orjson

from aiperf.dataset.loader.weka_trace import WekaTraceLoader
from aiperf.dataset.loader.weka_trace_models import WekaNormalRequest

_MODEL = "claude-opus-4-5-20251101"
_HAIKU = "claude-haiku-4-5-20251001"


def _mk_user_config(**overrides):
    from tests.unit.dataset.loader.conftest import make_weka_run

    overrides.setdefault("model_names", [_MODEL, _HAIKU])
    return make_weka_run(**overrides)


def _make_loader(filename, uc, monkeypatch):
    loader = WekaTraceLoader(filename=str(filename), run=uc)
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
    loader.prompt_generator.tokenizer.decode.side_effect = lambda toks: (
        f"<dec:{len(toks)}>"
    )
    loader._tokenizer_name = "t"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    return loader


def _stub_loader(loader: WekaTraceLoader) -> None:
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    pg = MagicMock()
    pg._cache = {}
    pg._sample_tokens.side_effect = lambda n: [0] * n
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    pg._bpe_stable_terminator_tokens = []
    stub_hash_id_corpus_rng(pg)
    pg.tokenizer.decode.side_effect = lambda toks: f"<dec:{len(toks)}>"
    pg._hash_id_corpus_rng.seed = 12345
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64


def _write_trace(tmp_path, data, name="t.json"):
    p = tmp_path / name
    p.write_bytes(orjson.dumps(data))
    return p


def _req(
    t: float, hash_ids: list[int], api_time: float = 1.0, model: str = "m"
) -> WekaNormalRequest:
    return WekaNormalRequest(
        type="n",
        t=t,
        model=model,
        input_length=len(hash_ids) * 64,
        output_length=10,
        hash_ids=hash_ids,
        api_time=api_time,
    )


def _normals(*reqs: WekaNormalRequest) -> list[tuple[int, WekaNormalRequest]]:
    return list(enumerate(reqs))


def _chain_outer_indices(result, chain_index: int) -> list[int]:
    return [oi for oi, _ in result.chains[chain_index].requests]
