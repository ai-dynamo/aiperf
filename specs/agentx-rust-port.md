<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AgentX Rust port (legacy-parity, switchable)

## Purpose

Port the Python **AgentX v1.0** feature (WEKA trace replay + agentic-replay timing
+ scenario locks; PR #1165) to Rust as a **standalone, self-contained subsystem**
that reproduces the Python behavior **byte-for-byte**. It is deliberately parallel
to — and shares no logic with — the next-gen graph-ir recorded path
(`graph/recorded/`). The two are **switchable** at runtime (`agentx` semantics vs
`graph-ir` semantics) so a run can be executed either way and the outputs compared.

This subsystem is **transitional**: once graph-ir supersedes AgentX it is deleted
wholesale. It therefore lives behind a Cargo feature and never entangles the rest
of the runtime.

Parity is *proven*, not asserted: the `raw` export level emits per-request
byte-exact timing and content, and an A/B harness diffs the Rust output against the
Python AgentX output over identical deterministic (SimClock) virtual time.

## Built

The `agentx` feature and module tree exist under `rust/runtime/src/agentx/`, off by
default. Slice 1 is in progress; these units are byte-exact against their Python
counterparts with unit tests (golden values captured from CPython where relevant):

- `rng.rs` — `HashIdRandomGenerator`, reusing the runtime CPython-MT compat; golden
  parity against `random.Random(sha256-derived-seed).randrange`.
- `trace.rs` — the `WekaTrace` serde schema; parses every in-repo
  `tests/fixtures/weka_traces/` file and rejects unknown fields.
- `selection.rs` — `filter_then_cap` selection with stats.
- `prepass.rs` — the shared per-trace theoretical prefix-cache prepass.
- `prompt.rs` — `compose_weka_prompt_tokens` (three ISL layouts, callback-injected
  token generation).
- `synth.rs` — the LCP-driven `ConversationReconstructor` (init_turn_0, advance_turn,
  turn_delta, block geometry, `compute_asst_block_caps`, `truncate_synth_buf_at_block`),
  token generation injected via the `TokenSynth` trait. **Proven byte-exact against
  the real Python reconstructor** by `tools/agentx_synth_golden.py` +
  `rust/runtime/tests/agentx_synth_parity.rs` (5 scenarios, full segment + TurnDelta
  diff).
- `tool_shape.rs` — OpenAI tool-call wire shaping (`demote_unpaired_tool_marks`,
  `tool_shape_segment_messages`), wired into `turn_delta`.
- `chains.rs` — flat-agent chain detection (`detect_agent_chains` two-phase greedy
  build + seam splicing, `is_aux_chain`/`is_reduction_chain`). **Proven byte-exact
  against Python** by `tools/agentx_chains_golden.py` +
  `rust/runtime/tests/agentx_chains_parity.rs` (6 scenarios, full partition diff).
  Grouping helpers (`worker_group_assignment`, `compute_chain_prefix_blocks`,
  `overlap_components`, `observed_group_prefix`) now built and **byte-exact**
  (chains golden + parity extended to diff worker-group coordinates and per-chain
  setup-prefix blocks). `chains.rs` is complete.

- `corpus.rs` — `CorpusTokenSynth`, the token-generation bridge (rng → corpus →
  synth): per-hash-id `randrange(corpus_size)` block window (no separator token,
  unlike the graph-ir `content.rs` scheme) + sha256-offset partial tail. **Proven
  byte-exact against the real Python `HashIdRandomGenerator`** (embedded golden).

Note (separation confirmed): AgentX legacy block tokens are `corpus[start..start+bs]`
with NO block-separation token; the graph-ir recorded path
(`graph/recorded/content.rs`) uses `[sep] + corpus[start..]`. The two schemes
differ, so `agentx` must not reuse `content.rs` — it reuses only the tokenized
corpus itself (a leaf util).

- `loader.rs` — the reconstruction hub. **Main-conversation path built and proven
  byte-exact against the real Python `_reconstruct_serial`** by
  `tools/agentx_loader_golden.py` + `rust/runtime/tests/agentx_loader_parity.rs`
  (3 scenarios; every turn field diffed: timing, `raw_messages`, `reset_context`,
  `input_kind`, prefix-cache tallies, `max_tokens`). Includes the loop helpers
  (`classify_turn_input`, `end_to_start_delay_ms`, `api_time_ms`, `clamp_delay_ms`,
  `cap_output`) and `ReconstructedConversation`/`ReconstructedTurn`.

- `config.rs` / `subagent.rs` — the `WEKA_*` config defaults (`WekaConfig`) and
  subagent expansion (`expand_subagent_to_child_plans` + `split_off_preamble`,
  `chain_init_tokens`, `worker_suffix`). **Proven byte-exact against Python**
  (`tools/agentx_subagent_golden.py` + `agentx_subagent_parity.rs`, 4 scenarios:
  single-chain / spawn-worker / cross-model-aux / relative-timestamp).

- `plan.rs` / `loader.rs` — trace orchestration: `build_shared_metric_values`
  (trace-wide prefix-cache over parent + children + flat chains),
  `detect_and_split_flat_chains` (`FlatChainPlan`), `reconstruct_conversation`
  (shared main/child/flat loop), `convert_trace_to_conversations` (root
  `weka_main` + `weka_subagent` children + `weka_flat` chains), and
  `build_model_map`. Composed from byte-exact-proven pieces; unit-tested.

**Corpus-generator finding (blocks the real-fixture e2e):** Python's
`PromptGenerator` corpus for weka (`coding`) comes from the AgentX-added
`dataset/generator/coding_content.py`; the runtime's `dataset::coding::build_coding_corpus`
is the *graph-ir* procedural corpus. Whether they are byte-identical is unverified —
the real-fixture golden diff must first establish that `build_coding_corpus`
reproduces `PromptGenerator._tokenized_corpus` for `coding`, or port
`coding_content.py` separately. Token-level sampling (`CorpusTokenSynth`) is already
proven byte-exact given an identical corpus.

Remaining Slice 1 (see below): idle-gap time-warp timing; SPAWN/JOIN branch
metadata; and the real-corpus e2e (pending the corpus-generator match above). The
child-conversation reconstruction and flat-chain paths are now composed; flat-chain splitting; the idle-gap
time-warp timing; model mapping; and — the gating
dependency for a real fixture-level diff — wiring the real Qwen-tokenized corpus
(reused from the runtime as a leaf util) into `CorpusTokenSynth` so it byte-matches
Python's `PromptGenerator._tokenized_corpus`, then the full-loader golden-diff over
`tests/fixtures/weka_traces/`.

## Future requirements

Delivered in dependency order as separate implementation slices. Each slice is
byte-exact vs its Python counterpart and lands with tests before the next begins.

### Slice 1 — WEKA reconstruction core (loader foundation)

Deterministic, clock-free reconstruction of a `WekaTrace` JSON into
`ReconstructedConversation`s (root + one child per `subagent` entry), byte-identical
to Python `WekaTraceLoader` (serial path).

Modules (`rust/runtime/src/agentx/`):

- `trace.rs` — serde schema for `WekaTrace` / `WekaNormalRequest` /
  `WekaStreamingRequest` / `WekaSubagentEntry` (`deny_unknown_fields`, `in`/`out`
  aliases, `type` discriminator, `hash_id_scope: "local"` only).
- `rng.rs` — `HashIdRandomGenerator`: `sha256("{seed}:{trace_id}:{hash_id}")`,
  first 8 bytes big-endian → seed the ported CPython Mersenne-Twister
  (`rng::compat::python_random`). The #1 byte-exact hazard.
- `prompt.rs` — `compose_weka_prompt_tokens` (hash-id blocks + sha256-keyed partial
  tail, three ISL layouts) and the hash-ids→prompt synthesis pipeline.
- `synth.rs` — `ConversationReconstructor`: the LCP-driven per-turn state machine
  (`RoleSegment` of exact token IDs, `TurnDelta { delta_messages, reset_context }`).
  The correctness heart.
- `chains.rs` — flat-agent chain detection over hash-id LCP evidence
  (join-seam vs spawn classification, same-model rule).
- `prepass.rs` — theoretical prefix-cache prepass:
  `{(session_id, k): (hit_blocks, total_blocks)}` over one shared per-trace seen-set
  in global time order.
- `tool_shape.rs` — optional OpenAI tool-call wire shaping.
- `selection.rs` — filter-by-`max_context_length`-then-cap-N.
- `loader.rs` — the `WekaTraceLoader` hub producing `ReconstructedConversation`s.

Byte-exact substrate: reuse `rng::compat::python_random` (CPython MT), the Qwen3-0.6B
tokenizer via the `tokenizers` crate, `serde_json`, `sha2`.

Validation: a Python dumper emits canonical JSON (conversations → ordered turns →
messages + per-turn token IDs + prefix-cache tallies) over the in-repo fixtures
(`tests/fixtures/weka_traces*/`); the Rust loader emits identical canonical JSON;
parity = byte-identical diff. Secondary: the existing per-turn ISL-drift bound.

### Slice 2 — parallel reconstruction

Byte-identical multithreaded (`rayon`) reconstruction: one trace-scoped
`HashIdRandomGenerator` per trace, shared token corpus via `Arc`. Output must equal
slice 1 exactly.

### Slice 3 — agentic-replay runtime (timing)

`trajectory_source` (t* sampling, alive-at-t* reconstruction, recycle),
`session_tree` (one slot per live tree), `replay_dependencies` (recorded interval
gating), `cache_bust` (per-tree deterministic marker), and the `agentic_replay`
strategy (WARMUP prime-to-t* then PROFILING resume). Clock-driven via `Clock`
(SimClock in A/B, RealClock online).

### Slice 4 — scenario locks + config + metrics glue

`ScenarioSpec`/validator, `is_context_overflow_response` + `ContextOverflowCount`
metric, `TheoreticalPrefixCache` accumulator, and the AgentX config/enum surface
(`--scenario`, `--cache-bust`, trajectory ratios, agentic warmup, etc.).

### Slice 5 — HF-backed loader

`semianalysis_cc_traces_weka` HuggingFace download wrapper (`hf-hub`), delegating
reconstruction to slice 1.

### Switchable semantics + raw-export parity gate

A run selects `agentx` (this port) or `graph-ir` semantics. The `raw` export level
emits per-request byte-exact timing + content. The A/B harness runs both the Python
AgentX and the Rust port over identical SimClock virtual time and asserts a
byte-identical raw export — the definition of done for parity.

## Source anchors

- `rust/runtime/src/agentx/` — the subsystem (feature `agentx`).
- `rust/runtime/src/rng/compat/python_random.rs` — reused CPython MT.
- Python reference: `src/aiperf/dataset/loader/weka_*.py`,
  `src/aiperf/timing/strategies/agentic_replay.py`,
  `src/aiperf/timing/{trajectory_source,session_tree,replay_dependencies}.py`,
  `src/aiperf/common/scenario/*`.
- Fixtures: `tests/fixtures/weka_traces*/`; drift contract:
  `tests/component_integration/dataset/test_weka_trace_byte_exact_drift.py`.
