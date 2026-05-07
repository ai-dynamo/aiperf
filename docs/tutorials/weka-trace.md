<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Replaying Agentic Coding Sessions with Weka Traces

Benchmark your LLM inference server with real-world agentic coding sessions captured via the [Weka KV-Cache-Tester](https://github.com/callanjfox/kv-cache-tester) research project. These traces preserve per-request timing, cache-block hash IDs (for KV-cache-aware replay), and nested subagent topology.

> **Looking for the SemiAnalysis InferenceX AgentX-MVP submission flow?** That benchmark is built on this corpus with extra rules locked in. See [InferenceX AgentX MVP](agentx-mvp.md) — the scenario preset (`--scenario inferencex-agentx-mvp`) bundles the AgentX rules into a single flag on top of the loader documented here.

---

## What Is a Weka Trace?

Each trace file is a single JSON object describing one coding conversation:

- `requests` is an ordered list of normal API calls (`type: "n"`), streaming API calls (`type: "s"`), and subagent markers (`type: "subagent"`).
- Each normal/streaming request carries `hash_ids` (KV-cache block identifiers) used to simulate cache reuse during replay.
- Subagent markers point at nested sub-conversations — AIPerf replays them as separate concurrent child sessions that the parent waits on before resuming.

AIPerf maps the format directly onto its DAG datastructure:

- One root `Conversation` per trace file.
- One child `Conversation` per `type: "subagent"` entry (session id `<trace_id>::sa:<agent_id>`).
- A `SPAWN` branch on the parent's preceding turn; a `SPAWN_JOIN` prerequisite on the parent's following turn. Three nuances: (a) subagents with no preceding parent turn are dropped (logged at load time); (b) subagents with no following parent turn become `is_background=True` branches with no `SPAWN_JOIN` prerequisite (the parent doesn't wait); (c) adjacent subagents sharing the same `(preceding, following)` anchors collapse into one multi-child branch.

---

## Quick Start

```bash
aiperf profile \
    --url localhost:8000 \
    --model claude-opus-4-5-20251101 \
    --model claude-haiku-4-5-20251001 \
    --endpoint-type chat \
    --streaming \
    --input-file artifacts/kv-cache-tester/traces/ \
    --fixed-schedule
```

Whatever you pass to `--model` becomes the model the server actually sees. Trace requests are rewritten to use your configured model(s) — the trace's recorded model names don't have to match what you're serving. See [Per-Trace Model Rewriting](#per-trace-model-rewriting) below for how multi-model traces map onto multiple `--model` values.

The `--fixed-schedule` flag replays requests at their recorded timestamps; subagents run in parallel and the parent's next turn waits until they complete.

### Directory vs Single File

Both work:

```bash
# Directory (739 traces in the shipped corpus)
aiperf profile ... --input-file artifacts/kv-cache-tester/traces/

# Single trace
aiperf profile ... --input-file artifacts/kv-cache-tester/traces/trace_0001.json
```

### Filtering

Standard trace filters apply:

- `--synthesis-max-isl <N>`: drop any request whose input length exceeds N tokens. Subagents whose preceding parent turn is filtered out are dropped; subagents whose only-following parent turn is filtered out fall back to background branches (no anchor turn to wait on).
- `--synthesis-max-osl <N>`: cap any request's `max_tokens` to N.
- `--fixed-schedule-start-offset` / `--fixed-schedule-end-offset`: time window on the outer `t` field.

---

## Loading From HuggingFace (No Download Required)

If you don't already have the trace corpus on disk, the same dataset is published on HuggingFace as [`semianalysisai/cc-traces-weka-042026`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-042026) and can be pulled directly by AIPerf with a single flag:

```bash
aiperf profile \
    --url localhost:8000 \
    --model claude-opus-4-5-20251101 \
    --model claude-haiku-4-5-20251001 \
    --endpoint-type chat \
    --streaming \
    --public-dataset semianalysis_cc_traces_weka \
    --fixed-schedule
```

On first run, the full corpus downloads upfront (657 MB compressed, 739 traces) and is cached locally by the HuggingFace `datasets` library; subsequent runs reuse the cache. The dataset is public — no HuggingFace authentication or token is required.

> **`--num-dataset-entries` caps the loaded subset.** The HF loader reads at most `--num-dataset-entries` rows out of the cached download (default 100). To load the full 739-trace corpus, pass `--num-dataset-entries 739`. The loader logs `Loading <n>/739 traces` at INFO so you can see the actual count. (The file-based `--input-file <dir>` path loads every JSON file it finds; there is no per-trace cap on that path. Use a smaller directory or the HF loader with `--num-dataset-entries N` if you want a controlled subset.)

The HuggingFace path and the file-based `--input-file` path produce **byte-identical conversations** because the public-dataset loader is a thin wrapper that delegates 100% of trace reconstruction (hash_id replay, per-trace model mapping, branch + spawn-join topology, delay capping, parallel reconstruction) to the same `WekaTraceLoader.convert_to_conversations()` used by `--input-file`. There is one source of truth for trace reconstruction.

### File-Based vs HuggingFace: Which to Use

| Path | When to use |
|---|---|
| `--input-file <dir-or-file>` (file-based) | You already have a local trace directory, you need offline runs (no outbound network), or you're developing/debugging the loader against a specific subset of traces. |
| `--public-dataset semianalysis_cc_traces_weka` (HuggingFace) | You want zero-setup against the canonical 739-trace corpus and don't mind a one-time ~657 MB download (cached afterward). |

All existing tunables work identically in both paths: `--synthesis-max-isl`, `--synthesis-max-osl`, `--inter-turn-delay-cap-seconds`, `--ignore-trace-delays`, `--use-think-time-only`, `--scenario inferencex-agentx-mvp`, `--cache-bust`, the per-trace model rewriting rules below — same flags, same behavior, same output bytes on the wire.

A tokenizer is required in both paths (the prompt is reconstructed from `hash_ids`); pass `--tokenizer <name-or-path>` if your `--model` doesn't resolve a default tokenizer.

---

## Replay Timing Controls

By default, AIPerf auto-enables `--fixed-schedule` for trace datasets — turns are sent at their recorded timestamps, subagents run in parallel, and the parent waits on `SPAWN_JOIN`. The Quick Start above is what you want for most cases.

If you need different replay pacing, several flags are available (recent additions, all weka-trace-aware):

| Flag | What it does |
|---|---|
| `--no-fixed-schedule` | Opt out of the auto-enabled fixed-schedule. Turns dispatch at whatever pace your other timing flags imply (concurrency, request rate, `agentic_replay`, etc.) instead of the recorded `t` timestamps. |
| `--ignore-trace-delays` | Strip per-turn timestamps and inter-turn delays at load time — every turn becomes back-to-back. Mutually exclusive with `--use-think-time-only`. |
| `--use-think-time-only` | Inter-turn delay uses only the trace's recorded `think_time` (client-side wait before each request), not `t_curr - t_prev` (which would include the original server's response time). Useful when your server is faster or slower than the recording — you don't want it punished or rewarded for the *previous* server's latency. Mutually exclusive with `--ignore-trace-delays`. |
| `--inter-turn-delay-cap-seconds <S>` | Clamp any single inter-turn delay to at most `S` seconds. Defaults to `None` (no clamp); pass `60` to cap "coffee-break" gaps in real coding traces. |

`--fixed-schedule` and `--no-fixed-schedule` are mutually exclusive — passing both errors at startup.

### `agentic_replay` Timing Mode

For multi-turn steady-state benchmarking with FIFO trace recycle and trajectory-based warmup (the agent-load-generation pattern AgentX MVP requires), AIPerf has a dedicated timing mode: `agentic_replay`. It is **scenario-locked** — there is no direct CLI flag to select it. Pass `--scenario inferencex-agentx-mvp` (the only built-in scenario that pins this mode today) and AIPerf's scenario validator sets `timing_mode=agentic_replay` for you:

```bash
aiperf profile \
    --scenario inferencex-agentx-mvp \
    --input-file artifacts/kv-cache-tester/traces/ \
    --concurrency 50 \
    --benchmark-duration 900 \
    ...
```

For the full mechanics (trajectory selection, recycle queue, warmup barrier) and the locked submission rules on top, see [InferenceX AgentX MVP](agentx-mvp.md).

### Cache-Bust Markers

AIPerf can prepend a unique per-conversation marker to every prompt, so that recycled plays of the same trace produce different prompt bytes and don't progressively warm the server's KV-cache prefix as the run goes on. Pass `--cache-bust system_prefix` (or `system_suffix` / `first_turn_prefix` / `first_turn_suffix`) to enable it. The default is `none` (no marker injected).

The marker looks like `[rid:8a3f2c1b9e7d]` and is derived deterministically within a run from the auto-generated benchmark ID, the trace's recycle count, the trajectory index, and the trace ID — same trace, same recycle pass, same marker for every turn in that play. Markers differ across runs (the benchmark ID is a fresh UUID each time).

This is locked on for the AgentX MVP scenario — auto-injected as `system_prefix` when you don't pass `--cache-bust` yourself, and any explicit conflicting value is rejected at startup. Outside that scenario it's optional and defaults to `none`.

A few details worth knowing if you're using `--cache-bust` outside the scenario:

- **Compatibility is checked at startup.** `--cache-bust` requires the `agentic_replay` timing mode (set by `--scenario inferencex-agentx-mvp`) and a chat-shaped endpoint (`--endpoint-type chat` or `responses`). Other combinations error before the run starts with a message naming the offending flag, not silently mid-run.
- **Multimodal turns are supported.** When a turn carries images or audio alongside text, the marker is added as a new `{type: "text", text: "<marker>"}` content part at the start (prefix) or end (suffix) of the parts list; existing text/image/audio parts pass through untouched.
- **`system_*` falls back to the first user turn when there's no system message.** If a trace has no system role anywhere (neither a conversation-level system message nor a `raw_messages[0].role=='system'`), `--cache-bust system_prefix` and `system_suffix` route the marker to the first user turn (turn index 0) with the same orientation (prefix stays prefix, suffix stays suffix). Because the fallback only fires on turn 0, later turns of that session can't re-inject — the worker logs this once per worker process at WARN level so you can spot it in mixed corpora.
- **Incompatible with `payload_bytes` workloads.** AIPerf's pre-encoded mmap fast path bypasses the per-request rendering that injection needs. If your dataset would otherwise pick the `PAYLOAD_BYTES` format, AIPerf refuses the run with a clear error rather than silently dropping markers. Either drop `--cache-bust` or use a workload that goes through the normal compose path.

If you're tracking how the marker contributes to the **wire-token total** the model actually sees, see [Input Sequence Length (ISL) Tokenization](../reference/isl-tokenization.md). With `--apply-chat-template`, AIPerf compensates the synthetic prompt budget for the marker's token cost so `--isl N` lands on `N` tokens at the wire after the chat template wraps it.

---

## Per-Trace Model Rewriting

The WEKA corpus was captured against specific models (typically Claude Opus for the agent and Claude Haiku for subagents). You don't have to serve those exact models to replay it. AIPerf rewrites every request's `model` field at load time to whatever you pass via `--model`.

The mapping is built **per trace**, in this order:

1. The trace's **main model** — the `model` of the first parent (non-subagent) request, falling back to the first request of the first subagent for parent-less traces — maps to your **first** `--model`.
2. Other distinct models in the trace map to your **second**, **third**, … `--model` in **first-appearance order**.
3. If a trace has more distinct models than you passed `--model` values, the mapping wraps with modulo (so every request still resolves to one of your configured models).

Practical implications:

- **One `--model`**: every request — parent, subagents, all of it — gets routed to that one model.
- **Two `--model` values**: a typical Opus-parent + Haiku-subagent trace replays with parent → first model, subagent → second model. Same shape as the recording, just relabeled.
- **Multi-model traces against fewer configured models**: extras reuse the configured list from the start. This is intentionally lossy (you asked for fewer routes) but the run still completes.
- **Trace's own `models` list is ignored** — the mapping is built from per-request `model` fields, not the trace-level metadata field.

The mapping is rebuilt for every trace independently, so a corpus with mixed-model and single-model traces all work side-by-side under one `--model` set.

---

## What Gets Replayed

Per turn:

- **Prompt** is synthesized deterministically from the recorded `hash_ids` via the shared `hash_ids -> token sequence -> decoded prompt` pipeline, so cache structure is preserved across runs.
- **Model** is rewritten via a per-trace mapping (see [Per-Trace Model Rewriting](#per-trace-model-rewriting)) — the trace's per-request `model` field is used to *pick which* configured model gets sent for that request, not as the routing model itself.
- **Max tokens** comes from the `out` field (after `--synthesis-max-osl` capping).
- **Timing** preserves the recorded `t` field for `--fixed-schedule`. By default, inter-turn `delay` is computed as `t_n - t_{n-1}`. With `--use-think-time-only`, `delay` instead uses the recorded per-request `think_time`. With `--ignore-trace-delays`, both `timestamp` and `delay` are stripped at load time. See [Replay Timing Controls](#replay-timing-controls) above.

The trace's recorded `type: "s"` (streaming) vs `type: "n"` (non-streaming) is independent of how AIPerf sends the request — the transport is controlled by `--streaming`. Both types are replayed identically.

---

## Related Tutorials

- [InferenceX AgentX MVP](agentx-mvp.md) — the SemiAnalysis multi-turn agentic-coding benchmark scenario built on this corpus.
- [DAG Benchmarking (Sub-Agents)](../benchmark-modes/dag.md) — the gating mechanism subagent support relies on.
- [Fixed Schedule](fixed-schedule.md) — precise timestamp-based execution.
- [Trace Benchmarking](../benchmark-modes/trace-replay.md) — general deterministic workload replay.
- [Input Sequence Length (ISL) Tokenization](../reference/isl-tokenization.md) — how `--isl` is reconciled across bare-text, chat-template wrapping, and cache-bust marker overhead.
- [ISL Budget Compensation Derivation](../reference/isl-budget-compensation.md) — the math behind chat-template overhead compensation in the synthetic composer.
