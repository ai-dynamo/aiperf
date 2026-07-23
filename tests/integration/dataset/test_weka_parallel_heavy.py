# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Heavy parity tests for WekaTraceLoader's parallel reconstruction path."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[2] / "fixtures" / "weka_traces"
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"


def _tokenizer_in_cache() -> bool:
    try:
        from aiperf.common.tokenizer import _is_hf_cached

        return _is_hf_cached(TOKENIZER_NAME)
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _tokenizer_in_cache(),
        reason=f"Tokenizer {TOKENIZER_NAME} not in local HF cache",
    ),
]


@pytest.fixture(autouse=True)
def _rng_init():
    """Each test starts with a deterministic global RNG seed so the"""
    import contextlib

    from aiperf.common import random_generator as rng_mod

    with contextlib.suppress(Exception):
        rng_mod.reset()
    rng_mod.init(0)
    yield


def _make_corpus_dir(tmp_path: Path, n_copies: int, fixture_name: str) -> str:
    """Copy ``fixture_name`` ``n_copies`` times with unique trace IDs."""
    src_text = (FIXTURES / fixture_name).read_text()
    src_id = json.loads(src_text)["id"]
    for i in range(n_copies):
        new_id = f"{src_id}__copy_{i:04d}"
        new_text = src_text.replace(f'"{src_id}"', f'"{new_id}"')
        (tmp_path / f"trace_{i:04d}.json").write_text(new_text)
    return str(tmp_path)


def _convs_signature(convs) -> str:
    """Stable hash over a list of Conversation model dumps."""
    h = hashlib.sha256()
    for c in convs:
        h.update(json.dumps(c.model_dump(), sort_keys=True, default=str).encode())
    return h.hexdigest()


def _build_loader(
    filename: str,
    *,
    force_parallel: bool,
    workers: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Real WekaTraceLoader with real PromptGenerator + tokenizer."""
    from aiperf.common import environment as env_mod
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.config.dataset.content import PrefixPromptConfig, PromptConfig
    from aiperf.dataset.generator.prompt import PromptGenerator
    from aiperf.dataset.loader.weka_trace import WekaTraceLoader
    from tests.unit.dataset.loader.conftest import make_weka_run

    if force_parallel:
        monkeypatch.setenv("AIPERF_DATASET_WEKA_PARALLEL_THRESHOLD", "1")
        monkeypatch.setenv(
            "AIPERF_DATASET_WEKA_PARALLEL_WORKERS",
            str(workers) if workers else "0",
        )
    else:
        monkeypatch.setenv("AIPERF_DATASET_WEKA_PARALLEL_THRESHOLD", "100000")
        monkeypatch.setenv("AIPERF_DATASET_WEKA_PARALLEL_WORKERS", "1")

    env_mod.Environment.DATASET = type(env_mod.Environment.DATASET)()

    run = make_weka_run(
        model_names=[
            "claude-opus-4-5-20251101",
            "claude-haiku-4-5-20251001",
        ],
        tokenizer_name=TOKENIZER_NAME,
    )
    tok = Tokenizer.from_pretrained(TOKENIZER_NAME)
    pg = PromptGenerator(
        prompts=PromptConfig(),
        prefix_prompts=PrefixPromptConfig(),
        tokenizer=tok,
    )
    return WekaTraceLoader(
        filename=filename,
        run=run,
        prompt_generator=pg,
        default_block_size=64,
    )


def _convert(
    filename: str,
    *,
    force_parallel: bool,
    workers: int = 0,
    monkeypatch: pytest.MonkeyPatch,
):
    loader = _build_loader(
        filename,
        force_parallel=force_parallel,
        workers=workers,
        monkeypatch=monkeypatch,
    )
    data = loader.load_dataset()
    return loader.convert_to_conversations(data)


@pytest.mark.parametrize(
    "fixture_name",
    ["simple.json", "one_subagent.json", "terminal_subagent.json", "multi_model.json"],
    ids=lambda s: s.removesuffix(".json"),
)
def test_serial_parallel_byte_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixture_name: str
):
    """Serial path and parallel path must produce byte-identical model dumps"""
    corpus = _make_corpus_dir(tmp_path, 16, fixture_name)
    serial = _convert(corpus, force_parallel=False, monkeypatch=monkeypatch)
    parallel = _convert(corpus, force_parallel=True, workers=4, monkeypatch=monkeypatch)
    assert _convs_signature(serial) == _convs_signature(parallel)


def test_terminal_subagent_emits_background_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``terminal_subagent.json`` has a subagent with no following parent"""
    corpus = _make_corpus_dir(tmp_path, 4, "terminal_subagent.json")
    convs = _convert(corpus, force_parallel=True, workers=2, monkeypatch=monkeypatch)
    bg_count = sum(
        1
        for c in convs
        for b in getattr(c, "branches", [])
        if getattr(b, "is_background", False)
    )
    assert bg_count > 0


def test_parallel_run_twice_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Running the parallel path twice in the same process produces identical"""
    corpus = _make_corpus_dir(tmp_path, 16, "simple.json")
    a = _convert(corpus, force_parallel=True, workers=4, monkeypatch=monkeypatch)
    b = _convert(corpus, force_parallel=True, workers=4, monkeypatch=monkeypatch)
    assert _convs_signature(a) == _convs_signature(b)


@pytest.mark.parametrize("workers", [2, 4, 8, 16])
def test_worker_count_invariance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int
):
    """Output bytes do not depend on worker count: the byte signature for"""
    corpus = _make_corpus_dir(tmp_path, 20, "simple.json")
    target = _convs_signature(
        _convert(corpus, force_parallel=True, workers=workers, monkeypatch=monkeypatch)
    )
    baseline = _convs_signature(
        _convert(corpus, force_parallel=True, workers=4, monkeypatch=monkeypatch)
    )
    assert target == baseline


def test_cross_process_signature_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Running the same conversion in a fresh subprocess produces the same"""
    corpus = _make_corpus_dir(tmp_path, 8, "simple.json")
    here_sig = _convs_signature(
        _convert(corpus, force_parallel=True, workers=4, monkeypatch=monkeypatch)
    )

    repo_root = Path(__file__).resolve().parents[3]
    script = (
        "import os, sys, json, hashlib;"
        "os.environ['AIPERF_DATASET_WEKA_PARALLEL_THRESHOLD']='1';"
        "os.environ['AIPERF_DATASET_WEKA_PARALLEL_WORKERS']='4';"
        "os.environ.setdefault('HF_HUB_OFFLINE','1');"
        "os.environ.setdefault('TRANSFORMERS_OFFLINE','1');"
        "os.environ.setdefault('TOKENIZERS_PARALLELISM','false');"
        "sys.path.insert(0, 'src');"
        "sys.path.insert(0, '.');"
        "from aiperf.common import random_generator as rng_mod\n"
        "try: rng_mod.reset()\n"
        "except Exception: pass\n"
        "rng_mod.init(0)\n"
        "from aiperf.common.tokenizer import Tokenizer;"
        "from aiperf.config.dataset.content import PrefixPromptConfig, PromptConfig;"
        "from aiperf.dataset.generator.prompt import PromptGenerator;"
        "from aiperf.dataset.loader.weka_trace import WekaTraceLoader;"
        "from tests.unit.dataset.loader.conftest import make_weka_run;"
        "run=make_weka_run(model_names=['claude-opus-4-5-20251101','claude-haiku-4-5-20251001'], "
        f"tokenizer_name={TOKENIZER_NAME!r});"
        f"tok=Tokenizer.from_pretrained({TOKENIZER_NAME!r});"
        "pg=PromptGenerator(prompts=PromptConfig(), prefix_prompts=PrefixPromptConfig(), tokenizer=tok);"
        f"loader=WekaTraceLoader(filename={corpus!r}, run=run, prompt_generator=pg, default_block_size=64);"
        "convs=loader.convert_to_conversations(loader.load_dataset());"
        "h=hashlib.sha256()\n"
        "for c in convs:\n"
        "    h.update(json.dumps(c.model_dump(), sort_keys=True, default=str).encode())\n"
        "print('SIG=' + h.hexdigest())\n"
    )

    res = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(repo_root),
    )
    assert res.returncode == 0, (
        f"subprocess failed: rc={res.returncode}\nstderr tail:\n{res.stderr[-800:]}"
    )
    sig_lines = [ln for ln in res.stdout.splitlines() if ln.startswith("SIG=")]
    assert sig_lines, f"no SIG= line in subprocess output: {res.stdout!r}"
    sub_sig = sig_lines[-1].removeprefix("SIG=")
    assert sub_sig == here_sig


def test_scope_isolation_same_hash_id_different_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The ``hash_id_scope:'local'`` invariant: the same hash_id appearing in"""
    corpus = _make_corpus_dir(tmp_path, 2, "simple.json")
    convs = _convert(corpus, force_parallel=True, workers=2, monkeypatch=monkeypatch)
    a = next(
        m["content"] for m in convs[0].turns[0].raw_messages if m["role"] == "user"
    )
    b = next(
        m["content"] for m in convs[1].turns[0].raw_messages if m["role"] == "user"
    )
    assert a != b


def test_stress_500_simple_traces(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """500 simple-fixture traces x 16 workers -- exercises full pickle/spawn/"""
    corpus = _make_corpus_dir(tmp_path, 500, "simple.json")
    convs = _convert(corpus, force_parallel=True, workers=16, monkeypatch=monkeypatch)
    assert len(convs) == 500
    for c in convs:
        assert len(c.turns) == 2
        for turn in c.turns:
            assert turn.raw_messages
            assert all(m.get("content") for m in turn.raw_messages)


def test_stress_mixed_fixtures_1000(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """1000 traces drawn from all four fixture layouts in interleaved order;"""
    fixtures = [
        "simple.json",
        "one_subagent.json",
        "terminal_subagent.json",
        "multi_model.json",
    ]
    written = 0
    for batch in range(250):
        for j, fname in enumerate(fixtures):
            src_text = (FIXTURES / fname).read_text()
            src_id = json.loads(src_text)["id"]
            new_id = f"{src_id}__b{batch}_v{j}"
            new_text = src_text.replace(f'"{src_id}"', f'"{new_id}"')
            (tmp_path / f"trace_{written:05d}.json").write_text(new_text)
            written += 1
    convs = _convert(
        str(tmp_path), force_parallel=True, workers=16, monkeypatch=monkeypatch
    )
    assert len(convs) == 1750


def test_oversubscribed_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Worker count exceeding cpu_count works without resource contention or"""
    corpus = _make_corpus_dir(tmp_path, 32, "simple.json")
    convs = _convert(corpus, force_parallel=True, workers=32, monkeypatch=monkeypatch)
    assert len(convs) == 32


def test_helper_reused_across_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Three sequential parallel convert calls reuse the forkserver helper --"""
    corpus = _make_corpus_dir(tmp_path, 8, "simple.json")
    times: list[float] = []
    for _ in range(3):
        t0 = time.time()
        _convert(corpus, force_parallel=True, workers=4, monkeypatch=monkeypatch)
        times.append(time.time() - t0)
    assert max(times[1:]) < times[0] + 2.0, (
        f"helper reuse appears broken: timings={times}"
    )
