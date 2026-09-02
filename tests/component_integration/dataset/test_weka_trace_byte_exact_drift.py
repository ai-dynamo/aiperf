# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI-enforced byte-exact ISL drift contract: re-tokenize each Weka ``raw_messages`` turn with real Qwen3-0.6B and verify per-turn drift vs recorded ``in[k]`` stays within a per-message bound (tier 1 fixtures every PR, tier 2 corpus subset opt-in via ``slow``)."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from aiperf.common.tokenizer import Tokenizer
from aiperf.config.dataset.content import PrefixPromptConfig, PromptConfig
from aiperf.dataset.generator.prompt import PromptGenerator
from aiperf.dataset.loader.weka_trace import WekaTraceLoader

pytestmark = pytest.mark.component_integration

TOKENIZER_NAME = "Qwen/Qwen3-0.6B"
"""Matches the mock-server replay and ``tools/weka_byte_exact_verify.py``."""

MAX_TOKENIZER_DIVERGENCE_PER_MSG = 3
"""Tier-2 per-message ISL drift tolerance; must equal the constant in ``tools/weka_byte_exact_verify.py`` (empirical corpus per-msg max 0.96)."""

FIXTURE_TIER_PER_MSG_BOUND = 25
"""Looser tier-1 per-message drift tolerance absorbing synthetic-fixture block-alignment noise; the real correctness bound is enforced by tier 2."""

CORPUS_SUBSET = (
    "trace_0012",
    "trace_0058",
    "trace_0095",
    "trace_0103",
    "trace_0128",
    "trace_0184",
    "trace_0187",
    "trace_0546",
)
"""Preserved so the bound can be re-justified against the same population."""

CORPUS_MODELS = (
    "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-20250514",
)


@pytest.fixture(scope="module")
def real_qwen_tokenizer() -> Tokenizer:
    """Load the real Qwen3-0.6B tokenizer directly from HuggingFace, bypassing the package-scoped mock autouse fixture, skipping when it is not in the local HF cache."""
    from transformers import AutoTokenizer

    try:
        auto = AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)
    except Exception as e:
        pytest.skip(
            f"Real Qwen tokenizer ({TOKENIZER_NAME}) not in local HF cache: {e}. "
            'Run `python -c "from transformers import AutoTokenizer; '
            f"AutoTokenizer.from_pretrained('{TOKENIZER_NAME}')\"` to populate."
        )
    tokenizer = Tokenizer()
    tokenizer._tokenizer = auto
    tokenizer._resolved_name = TOKENIZER_NAME
    tokenizer._apply_kwarg_overrides()
    return tokenizer


@pytest.fixture(scope="module")
def real_prompt_generator(real_qwen_tokenizer: Tokenizer) -> PromptGenerator:
    """Build a real ``PromptGenerator`` so ``raw_messages`` content decodes via the same Qwen tokenizer the drift test counts against."""
    # Module-scoped fixture runs before the function-scoped reset_random_generator,
    # so seed here to make it self-contained; per-test reset re-seeds before each test.
    from aiperf.common import random_generator as rng

    rng.reset()
    rng.init(42)
    # The weka reconstructor sizes prompts from the trace's recorded in/hash_id
    # blocks (block_size=64), not from isl, so only block_size matters here.
    prompts = PromptConfig(block_size=64)
    prefix_prompts = PrefixPromptConfig(pool_size=None, length=None)
    return PromptGenerator(
        prompts=prompts,
        prefix_prompts=prefix_prompts,
        tokenizer=real_qwen_tokenizer,
    )


def _make_real_loader(
    filename: Path,
    model_names: tuple[str, ...],
    prompt_generator: PromptGenerator,
) -> WekaTraceLoader:
    """Build a real ``WekaTraceLoader`` off a real ``BenchmarkRun``, threading the real Qwen ``prompt_generator`` through."""
    from tests.unit.dataset.loader.conftest import make_weka_run

    run = make_weka_run(
        model_names=sorted(model_names),
        tokenizer_name=TOKENIZER_NAME,
    )
    loader = WekaTraceLoader(
        filename=str(filename),
        run=run,
        prompt_generator=prompt_generator,
        default_block_size=64,
    )
    # Match the trace files' default; no auto-detection in the loader.
    loader._block_size = 64
    return loader


def _tokenize_messages(tokenizer: Tokenizer, messages: list[dict]) -> int:
    """Sum content-only tokens across all messages joined with a single space, mirroring aiperf's client-side ISL formula in ``inference_result_parser._compute_token_count``."""
    if not messages:
        return 0
    joined = " ".join(m["content"] for m in messages)
    return len(tokenizer.encode(joined))


def _verify_drift_bound(
    loader: WekaTraceLoader,
    tokenizer: Tokenizer,
    recorded_per_trace: dict[str, list[int]],
    per_msg_bound: int = MAX_TOKENIZER_DIVERGENCE_PER_MSG,
) -> tuple[list[str], list[int], list[float]]:
    """Run ``convert_to_conversations``, accumulate delta-encoded turns (skipping subagents), and return ``(failures, abs_drifts, per_msg_drifts)`` for the per-turn drift bound."""
    convs = loader.convert_to_conversations(loader.load_dataset())

    failures: list[str] = []
    drifts: list[int] = []
    per_msg_drifts: list[float] = []
    for conv in convs:
        if "::sa:" in conv.session_id:
            continue
        ins = recorded_per_trace.get(conv.session_id)
        if ins is None:
            continue
        accumulated: list[dict] = []
        for k, turn in enumerate(conv.turns):
            if turn.raw_messages is not None:
                if getattr(turn, "reset_context", False):
                    accumulated = list(turn.raw_messages)
                else:
                    accumulated = accumulated + list(turn.raw_messages)
            if k >= len(ins):
                break
            tokenized = _tokenize_messages(tokenizer, accumulated)
            recorded = ins[k]
            n_msgs = len(accumulated)
            bound = per_msg_bound * max(n_msgs, 1)
            drift = abs(tokenized - recorded)
            drifts.append(drift)
            per_msg_drifts.append(drift / max(n_msgs, 1))
            if drift > bound:
                failures.append(
                    f"{conv.session_id} turn {k}: drift={drift} > bound={bound} "
                    f"(n_msgs={n_msgs}, recorded={recorded}, tokenized={tokenized})"
                )

    return failures, drifts, per_msg_drifts


def _restore_real_corpus_open():
    """Undo the package-scoped ``mock_corpus_file`` patch on ``builtins.open`` so the real Shakespeare corpus is read (currently unused; kept for future bound tightening)."""
    import builtins

    return patch("builtins.open", builtins.__dict__["open"])


def test_byte_exact_isl_drift_simple_fixture(
    real_qwen_tokenizer: Tokenizer,
    real_prompt_generator: PromptGenerator,
) -> None:
    """Tier 1: small fixture exercising a 2-turn normal-only trace."""
    fixture = Path(__file__).parents[2] / "fixtures" / "weka_traces" / "simple.json"
    loader = _make_real_loader(
        fixture,
        model_names=("claude-opus-4-5-20251101",),
        prompt_generator=real_prompt_generator,
    )
    # in[0]=200, in[1]=250 from simple.json.
    recorded = {"trace_simple": [200, 250]}
    failures, drifts, _per_msg = _verify_drift_bound(
        loader, real_qwen_tokenizer, recorded, per_msg_bound=FIXTURE_TIER_PER_MSG_BOUND
    )
    assert not failures, "byte-exact drift bound violated:\n  " + "\n  ".join(failures)
    assert len(drifts) >= 2, (
        f"expected at least 2 turn drifts measured; got {len(drifts)}"
    )


def test_byte_exact_isl_drift_multi_model_fixture(
    real_qwen_tokenizer: Tokenizer,
    real_prompt_generator: PromptGenerator,
) -> None:
    """Tier 1: subagent fixture; only the parent's normal turns are checked."""
    fixture = (
        Path(__file__).parents[2] / "fixtures" / "weka_traces" / "multi_model.json"
    )
    loader = _make_real_loader(
        fixture,
        model_names=(
            "claude-opus-4-5-20251101",
            "claude-haiku-4-5-20251001",
        ),
        prompt_generator=real_prompt_generator,
    )
    # Parent normal requests: in[0]=200, in[1]=400 (subagent at index 1 is
    # filtered by ``_verify_drift_bound``).
    recorded = {"trace_multi": [200, 400]}
    failures, drifts, _per_msg = _verify_drift_bound(
        loader, real_qwen_tokenizer, recorded, per_msg_bound=FIXTURE_TIER_PER_MSG_BOUND
    )
    assert not failures, "byte-exact drift bound violated:\n  " + "\n  ".join(failures)
    assert len(drifts) >= 2, (
        f"expected at least 2 turn drifts measured; got {len(drifts)}"
    )


def _sequential_decode_patch(real_tokenizer: Tokenizer):
    """Replace ``parallel_decode`` with in-process sequential decode to avoid the fork-from-multithreaded-parent xdist flake, reusing the real tokenizer."""

    def _seq_decode(token_sequences, tokenizer_name, **_kwargs):
        return [real_tokenizer.decode(tokens) for tokens in token_sequences]

    return patch(
        "aiperf.dataset.loader.hash_ids_synthesis.parallel_decode",
        _seq_decode,
    )


@pytest.mark.slow
def test_byte_exact_isl_drift_corpus_subset(
    real_qwen_tokenizer: Tokenizer,
    real_prompt_generator: PromptGenerator,
    tmp_path: Path,
) -> None:
    """Tier 2: assert the drift bound holds across the 41 turns of the 8-trace kv-cache-tester subset that backed the empirical baseline."""
    corpus = Path(__file__).parents[3] / "artifacts" / "kv-cache-tester" / "traces"
    if not corpus.exists():
        pytest.skip(f"Corpus not present at {corpus}")

    # Stage the 8-trace subset into a fresh directory the loader can scan.
    subset_dir = tmp_path / "subset"
    subset_dir.mkdir()
    recorded: dict[str, list[int]] = {}
    for tid in CORPUS_SUBSET:
        src = corpus / f"{tid}.json"
        if not src.exists():
            pytest.skip(f"Required trace missing from corpus: {src}")
        dst = subset_dir / f"{tid}.json"
        dst.write_bytes(src.read_bytes())
        blob = json.loads(src.read_text())
        recorded[blob["id"]] = [
            r["in"] for r in blob["requests"] if r.get("type") in ("n", "s")
        ]

    loader = _make_real_loader(
        subset_dir,
        model_names=CORPUS_MODELS,
        prompt_generator=real_prompt_generator,
    )

    t0 = time.perf_counter()
    with _sequential_decode_patch(real_qwen_tokenizer):
        failures, drifts, per_msg = _verify_drift_bound(
            loader, real_qwen_tokenizer, recorded
        )
    elapsed = time.perf_counter() - t0

    assert not failures, "byte-exact drift bound violated:\n  " + "\n  ".join(failures)
    # 41 comparable turns measured across this subset.
    assert len(drifts) >= 30, (
        f"expected ~41 turn drifts; got {len(drifts)} (corpus may have changed)"
    )
    # Informational summary; useful when the bound is re-tuned.
    print(
        f"\ncorpus subset drift: n={len(drifts)} median={statistics.median(drifts)} "
        f"mean={statistics.mean(drifts):.1f} max={max(drifts)} "
        f"per_msg_max={max(per_msg):.2f} per_msg_median={statistics.median(per_msg):.2f} "
        f"elapsed={elapsed:.2f}s"
    )
