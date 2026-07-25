# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-corpus golden for the main-conversation reconstruction of a fixture.

Drives the REAL CodingContentGenerator corpus + the REAL loader token closures
(`_decode_block_tokens` / `sample_partial_tail_tokens` semantics) + the REAL
ConversationReconstructor over a no-subagent Weka fixture, with the actual
Qwen3-0.6B tokenizer. Dumps per-turn raw_messages so the Rust convert_trace
(build_coding_corpus + CorpusTokenSynth) can diff byte-for-byte.

Skips (exit 0, no file) when Qwen3-0.6B is not in the local HF cache.

Run: ``python tools/agentx_realcorpus_golden.py``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RUN_SEED = 42
FIXTURE = "tests/fixtures/weka_traces/simple.json"


def main():
    import aiperf.common.random_generator as rng

    try:
        from transformers import AutoTokenizer
    except Exception:
        print("skip: transformers unavailable", file=sys.stderr)
        return
    try:
        auto = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as e:
        print(f"skip: Qwen not cached: {e}", file=sys.stderr)
        return

    rng.reset()
    rng.init(RUN_SEED)
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.config.dataset.content import PromptConfig
    from aiperf.dataset.generator.coding_content import CodingContentGenerator
    from aiperf.dataset.loader.weka_synth_buf import (
        ConversationReconstructor,
        compute_asst_block_caps,
    )
    from aiperf.dataset.loader.weka_metric_prepass import (
        MetricRecord,
        compute_shared_prefix_cache_metrics,
    )
    from aiperf.dataset.loader.weka_trace import _classify_turn_input, _end_to_start_delay_ms
    from aiperf.dataset.loader.weka_trace_models import WekaTrace

    tok = Tokenizer()
    tok._tokenizer = auto
    tok._resolved_name = "Qwen/Qwen3-0.6B"
    tok._apply_kwarg_overrides()

    root = Path(__file__).resolve().parents[1]
    trace = WekaTrace.model_validate_json((root / FIXTURE).read_text())
    trace_id = trace.id
    bs = trace.block_size

    g = CodingContentGenerator(config=PromptConfig(block_size=bs), tokenizer=tok)
    corpus = g._tokenized_corpus
    corpus_size = g._corpus_size
    hidr = g._hash_id_corpus_rng
    hidr.set_trace_id(trace_id)
    hash_base_seed = hidr.seed
    cache: dict[int, list[int]] = {}

    def decode_block_tokens(hash_ids):
        out = []
        for h in hash_ids:
            c = cache.get(h)
            if c is None:
                hidr.reseed_for_hash_id(h)
                start = hidr.randrange(corpus_size)
                end = start + bs
                c = corpus[start:end]
                if end > corpus_size:
                    c = c + corpus[: end - corpus_size]
                cache[h] = c
            out.extend(c)
        return out

    def sample_partial_tail_tokens(n, seed):
        if n <= 0:
            return []
        digest = hashlib.sha256(seed.encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % max(corpus_size - n, 1)
        return list(corpus[offset : offset + n])

    def decode_tokens_to_text(tokens):
        return tok.decode(tokens)

    normals = [
        (i, r)
        for i, r in enumerate(trace.requests)
        if getattr(r, "type", "n") in ("n", "s")
    ]
    caps = compute_asst_block_caps([(r.hash_ids, r.input_length) for _, r in normals], bs)
    records = [
        MetricRecord(sort_key=(r.t, oi, 0, 0), session_id=trace_id, k=k, hash_ids=list(r.hash_ids))
        for k, (oi, r) in enumerate(normals)
    ]
    metrics = compute_shared_prefix_cache_metrics(records)

    recon = ConversationReconstructor(
        block_size=bs,
        decode_block_tokens=decode_block_tokens,
        sample_partial_tail_tokens=sample_partial_tail_tokens,
        decode_tokens_to_text=decode_tokens_to_text,
    )
    turns = []
    for k, (oi, r) in enumerate(normals):
        seed = f"{trace_id}:turn_{k}:partial_tail"
        prev = normals[k - 1][1] if k else None
        ik = _classify_turn_input(r, prev)
        is_tool = ik is not None and ik.value == "tool_result"
        if k == 0:
            recon.init_turn_0(r.hash_ids, r.input_length, trace.tool_tokens, trace.system_tokens, seed, is_tool)
        else:
            recon.advance_turn(prev.hash_ids, prev.input_length, prev.output_length, r.hash_ids, r.input_length, seed, is_tool, max_asst_blocks=caps[k])
        t_ms = r.t * 1000.0
        delay = None if k == 0 else _end_to_start_delay_ms(t_ms - prev.t * 1000.0, prev.api_time)
        if delay is not None:
            delay = max(delay, 0.0)
        delta = recon.turn_delta()
        hit, total = metrics[(trace_id, k)]
        turns.append({
            "timestamp_ms": t_ms,
            "delay_ms": delay,
            "source_outer_idx": oi,
            "max_tokens": r.output_length if r.output_length >= 1 else 1,
            "raw_messages": delta.delta_messages,
            "reset_context": delta.reset_context,
            "hit": hit, "total": total,
            "input_kind": None if ik is None else ik.value,
        })

    out = {
        "fixture": FIXTURE,
        "trace_id": trace_id,
        "block_size": bs,
        "tool_tokens": trace.tool_tokens,
        "system_tokens": trace.system_tokens,
        "run_seed": RUN_SEED,
        "hash_base_seed": hash_base_seed,
        "turns": turns,
    }
    dest = root / "tests/fixtures/agentx/realcorpus_golden.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest} (hash_base_seed={hash_base_seed}, {len(turns)} turns)")


if __name__ == "__main__":
    main()
