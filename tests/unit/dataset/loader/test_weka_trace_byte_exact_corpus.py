# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Byte-exact replay structural smoke tests (``slow``) that process each kv-cache-tester trace one at a time to bound RSS while checking structural invariants."""

from __future__ import annotations

import gc
import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiperf.dataset.loader.weka_trace import WekaTraceLoader

CORPUS = Path(__file__).parents[4] / "artifacts" / "kv-cache-tester" / "traces"

pytestmark = pytest.mark.slow


def _collect_corpus_models() -> set[str]:
    models: set[str] = set()
    for path in sorted(CORPUS.glob("trace_*.json")):
        blob = json.loads(path.read_text())
        _walk_models(blob.get("requests", []), models)
    return models


def _walk_models(reqs: list, models: set[str]) -> None:
    for r in reqs:
        if r.get("type") in ("n", "s"):
            models.add(r["model"])
        elif r.get("type") == "subagent":
            _walk_models(r.get("requests", []), models)


def _recorded_per_turn(blob: dict) -> tuple[list[int], list[int]]:
    ins: list[int] = []
    outs: list[int] = []
    for r in blob.get("requests", []):
        if r.get("type") in ("n", "s"):
            ins.append(r["in"])
            outs.append(r["out"])
    return ins, outs


def _make_user_config(model_names: set[str]):
    from tests.unit.dataset.loader.conftest import make_weka_run

    return make_weka_run(model_names=sorted(model_names))


def _make_stubbed_loader(traces_dir: Path, models: set[str]) -> WekaTraceLoader:
    """Build a loader pointed at a single-file directory with a stubbed pg."""
    loader = WekaTraceLoader(filename=str(traces_dir), run=_make_user_config(models))
    pg = MagicMock()
    pg._cache = {}
    pg._sample_tokens.side_effect = lambda n: [0] * n
    pg._tokenized_corpus = list(range(10000, 11000))
    pg._corpus_size = 1000
    from tests.unit.dataset.loader.conftest import stub_hash_id_corpus_rng

    stub_hash_id_corpus_rng(pg)
    pg.tokenizer.decode.side_effect = lambda toks: "x" * len(toks)
    loader.prompt_generator = pg
    loader._tokenizer_name = "test-tok"
    loader._trust_remote_code = False
    loader._tokenizer_revision = None
    loader._block_size = 64
    loader.synthesize_prompts_from_hash_ids = lambda reqs: {r.key: "x" for r in reqs}
    return loader


def _iter_corpus_traces():
    """Yield (trace_path, blob) for every trace in the corpus, single-file at a time."""
    if not CORPUS.exists() or not any(CORPUS.glob("trace_*.json")):
        pytest.skip(f"Corpus not present at {CORPUS}; submodule not initialized")
    for path in sorted(CORPUS.glob("trace_*.json")):
        yield path, json.loads(path.read_text())


def _convert_one(trace_path: Path, models: set[str]):
    """Load + convert a single trace; return (convs, blob). Caller drops both."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        shutil.copy(trace_path, td_path / trace_path.name)
        loader = _make_stubbed_loader(td_path, models)
        return loader.convert_to_conversations(loader.load_dataset())


def test_corpus_loads_without_error():
    """Sanity: every trace in the corpus parses end-to-end without exception."""
    models = _collect_corpus_models()
    traces_seen = 0
    for trace_path, _ in _iter_corpus_traces():
        convs = _convert_one(trace_path, models)
        assert len(convs) > 0, f"{trace_path.name}: zero conversations"
        traces_seen += 1
        del convs
        gc.collect()
        gc.collect()
    assert traces_seen > 0


def test_corpus_every_turn_has_at_least_one_segment():
    """Every non-filtered turn must carry at least one role segment."""
    models = _collect_corpus_models()
    failures: list[str] = []
    for trace_path, _ in _iter_corpus_traces():
        convs = _convert_one(trace_path, models)
        for conv in convs:
            for k, turn in enumerate(conv.turns):
                if not turn.raw_messages:
                    failures.append(f"{conv.session_id} turn {k}: empty raw_messages")
        del convs
        gc.collect()
        gc.collect()
    assert not failures, (
        "raw_messages structural failures (showing first 20):\n  "
        + "\n  ".join(failures[:20])
    )


def test_corpus_per_turn_role_structure():
    """k>=1 turns whose prior turn produced output_tokens must include the assistant role (symmetric attribution, spec §4.4.1)."""
    models = _collect_corpus_models()
    failures: list[str] = []
    for trace_path, blob in _iter_corpus_traces():
        convs = _convert_one(trace_path, models)
        _ins, outs = _recorded_per_turn(blob)
        for conv in convs:
            if "::sa:" in conv.session_id:
                continue
            for k in range(1, len(conv.turns)):
                if k >= len(outs) or outs[k - 1] == 0:
                    continue
                roles = [m["role"] for m in (conv.turns[k].raw_messages or [])]
                if "assistant" not in roles:
                    failures.append(
                        f"{conv.session_id} turn {k}: missing assistant role "
                        f"(prev_out={outs[k - 1]}, roles={roles})"
                    )
        del convs
        gc.collect()
        gc.collect()
    assert not failures, (
        "role-structure failures (showing first 20):\n  " + "\n  ".join(failures[:20])
    )
