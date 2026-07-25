# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-corpus golden for full trace reconstruction (main + subagent children).

Drives the REAL CodingContentGenerator corpus + REAL loader token semantics +
REAL ConversationReconstructor + REAL subagent expansion over Weka fixtures with
the actual Qwen3-0.6B tokenizer, replicating
`WekaTraceLoader.convert_to_conversations` (no flat-chain split / no idle-warp).
Dumps every conversation's per-turn raw_messages + timing so the Rust
`convert_trace_to_conversations` can diff byte-for-byte.

Skips (exit 0) when Qwen3-0.6B is absent from the local HF cache.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

RUN_SEED = 42
FIXTURES = ["tests/fixtures/weka_traces/simple.json", "tests/fixtures/weka_traces/one_subagent.json"]


def build_env():
    import aiperf.common.random_generator as rng

    try:
        from transformers import AutoTokenizer
        auto = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", local_files_only=True)
    except Exception as e:
        print(f"skip: Qwen not available: {e}", file=sys.stderr)
        return None
    rng.reset()
    rng.init(RUN_SEED)
    from aiperf.common.tokenizer import Tokenizer
    from aiperf.config.dataset.content import PromptConfig
    from aiperf.dataset.generator.coding_content import CodingContentGenerator

    tok = Tokenizer()
    tok._tokenizer = auto
    tok._resolved_name = "Qwen/Qwen3-0.6B"
    tok._apply_kwarg_overrides()
    return tok, PromptConfig, CodingContentGenerator, rng


def main():
    env = build_env()
    if env is None:
        return
    tok, PromptConfig, CodingContentGenerator, rng = env

    from aiperf.dataset.loader.weka_synth_buf import ConversationReconstructor, compute_asst_block_caps
    from aiperf.dataset.loader.weka_metric_prepass import MetricRecord, compute_shared_prefix_cache_metrics
    from aiperf.dataset.loader.weka_trace import (
        _classify_turn_input, _end_to_start_delay_ms, _expand_subagent_to_child_plans,
        _dropped_subagent_indices, _ParentPlan,
    )
    from aiperf.dataset.loader.weka_trace_models import (
        WekaNormalRequest, WekaStreamingRequest, WekaSubagentEntry, WekaTrace,
    )

    root = Path(__file__).resolve().parents[1]
    out_all = []

    for fixture in FIXTURES:
        trace = WekaTrace.model_validate_json((root / fixture).read_text())
        trace_id, bs = trace.id, trace.block_size
        g = CodingContentGenerator(config=PromptConfig(block_size=bs), tokenizer=tok)
        corpus, corpus_size = g._tokenized_corpus, g._corpus_size
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

        def sample_tail(n, seed):
            if n <= 0:
                return []
            off = int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "big") % max(corpus_size - n, 1)
            return list(corpus[off : off + n])

        def dtt(tokens):
            return tok.decode(tokens)

        # Split + expand subagents.
        normals = [(i, r) for i, r in enumerate(trace.requests) if isinstance(r, (WekaNormalRequest, WekaStreamingRequest))]
        subagents = [(i, r) for i, r in enumerate(trace.requests) if isinstance(r, WekaSubagentEntry)]
        child_plans = []
        for sa_index, (outer, entry) in enumerate(subagents):
            child_plans.extend(_expand_subagent_to_child_plans(trace_id, sa_index, outer, entry, bs))

        parent = _ParentPlan(trace_id, normals, subagents, block_size=bs)
        dropped = _dropped_subagent_indices(parent)
        sa_outer_by_index = {i: outer for i, (outer, _) in enumerate(subagents)}

        # Shared metric values (parent normals + active children).
        records = [MetricRecord(sort_key=(r.t, oi, 0, 0), session_id=trace_id, k=k, hash_ids=list(r.hash_ids))
                   for k, (oi, r) in enumerate(normals)]
        for cp in child_plans:
            if cp.subagent_index in dropped:
                continue
            for k, cr in enumerate(cp.requests):
                records.append(MetricRecord(sort_key=(cr.t, sa_outer_by_index[cp.subagent_index], cp.chain_index, k),
                                            session_id=cp.session_id, k=k, hash_ids=list(cr.hash_ids)))
        metrics = compute_shared_prefix_cache_metrics(records)

        def reconstruct(session_id, init_tool, init_system, reqs, source_kind, source_outer_fn):
            recon = ConversationReconstructor(block_size=bs, decode_block_tokens=decode_block_tokens,
                                              sample_partial_tail_tokens=sample_tail, decode_tokens_to_text=dtt)
            caps = compute_asst_block_caps([(r.hash_ids, r.input_length) for r in reqs], bs)
            turns = []
            for k, r in enumerate(reqs):
                seed = f"{session_id}:turn_{k}:partial_tail"
                prev = reqs[k - 1] if k else None
                ik = _classify_turn_input(r, prev)
                is_tool = ik is not None and ik.value == "tool_result"
                if k == 0:
                    recon.init_turn_0(r.hash_ids, r.input_length, init_tool, init_system, seed, is_tool)
                else:
                    recon.advance_turn(prev.hash_ids, prev.input_length, prev.output_length, r.hash_ids, r.input_length, seed, is_tool, max_asst_blocks=caps[k])
                t_ms = r.t * 1000.0
                delay = None if k == 0 else _end_to_start_delay_ms(t_ms - prev.t * 1000.0, prev.api_time)
                if delay is not None:
                    delay = max(delay, 0.0)
                delta = recon.turn_delta()
                hit, total = metrics[(session_id, k)]
                turns.append({"timestamp_ms": t_ms, "delay_ms": delay, "source_outer_idx": source_outer_fn(k, r),
                              "source_kind": source_kind, "max_tokens": r.output_length if r.output_length >= 1 else 1,
                              "raw_messages": delta.delta_messages, "reset_context": delta.reset_context,
                              "hit": hit, "total": total, "input_kind": None if ik is None else ik.value})
            return turns

        conversations = [{
            "session_id": trace_id, "parent_conversation_id": None, "source_kind": "weka_main",
            "turns": reconstruct(trace_id, trace.tool_tokens, trace.system_tokens,
                                 [r for _, r in normals], "weka_main", lambda k, r, nn=normals: nn[k][0]),
        }]
        for cp in child_plans:
            if cp.subagent_index in dropped:
                continue
            conversations.append({
                "session_id": cp.session_id, "parent_conversation_id": trace_id, "source_kind": "weka_subagent",
                "turns": reconstruct(cp.session_id, cp.init_tool_tokens, cp.init_system_tokens,
                                     list(cp.requests), "weka_subagent", lambda k, r, so=cp.source_outer_idx: so),
            })

        out_all.append({"fixture": fixture, "trace_id": trace_id, "block_size": bs,
                        "tool_tokens": trace.tool_tokens, "system_tokens": trace.system_tokens,
                        "run_seed": RUN_SEED, "hash_base_seed": hash_base_seed, "conversations": conversations})

    dest = root / "tests/fixtures/agentx/realcorpus_golden.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out_all, indent=1))
    print(f"wrote {dest} ({len(out_all)} fixtures)")


if __name__ == "__main__":
    main()
