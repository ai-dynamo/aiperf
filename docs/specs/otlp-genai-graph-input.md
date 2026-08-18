<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# OTLP GenAI graph input

## Purpose

`dataset.format: otlp_genai` converts OTLP/HTTP JSON exports containing
OpenInference or OpenTelemetry GenAI semantic-convention spans into native,
per-trace flat Graph-IR programs. It is a direct graph-input adapter, not an
OTLP exporter or a generic OTel workload.

## Built

The adapter in `rust/runtime/src/engine/graph_input/otlp_genai.rs` accepts
`.json`, `.jsonl`, `.json.gz`, and `.jsonl.gz` files, or one inline OTLP JSON
record. It decodes every OTel `AnyValue` representation used by attributes:
string, bool, integer, double, bytes, array, and key/value-list. The input
envelope is strict (`type: file`, `format: otlp_genai`, one of `path` or
`records`, sequential selection), while each OTLP record is decoded with
contextual errors instead of silently manufacturing executable requests.

Spans are grouped by `traceId` in input order. For every trace the compiler
keeps spans carrying `openinference.span.kind` or a `gen_ai.*` attribute and
every known ancestor. Classification has fixed precedence:

1. `openinference.span.kind`: `LLM` and `AGENT` are executable LLM spans;
   every other value is recorded non-LLM work.
2. `gen_ai.operation.name`: chat/text completion/content generation/agent
   invocation/message creation are executable; every other operation is
   recorded non-LLM work.
3. OTLP `kind == CLIENT` is executable; all other kinds are recorded non-LLM
   work.

LLM spans retain their input messages (`input.value` first, then
`gen_ai.input.messages`), model override, recorded request/output token cap,
`gen_ai.choice` streaming signal, and span/server provenance metadata. The adapter does not route requests to
an endpoint inferred from telemetry: endpoint selection remains the authored
native run configuration, which is immutable across a run.

The Graph-IR has no generic `ReplayNode`, and the adapter intentionally does
not add one. A non-LLM span's `output.value` or `gen_ai.output.messages` is
seeded under a declared trace-state channel. The span is removed from the
executable topology; the union of its measured OTLP time intervals is added to the rerouted
successor's completion delay, or to the successor's start floor for a leading
root chain. This is the same lowering boundary used by `conditional_graph`:
the executor receives only `LlmNode` and `StaticEdge`, preserving its static,
worker-local hot path. A missing recorded output still preserves topology and
timing but contributes no seed.

OTLP parentage establishes execution ordering; it does not express a
channel-reference position inside a child request. Consequently, as in the
source adapter, recorded span output is retained as trace state but is not
invented into the child LLM's prompt.

## Source anchors

- `rust/runtime/src/engine/graph_input/otlp_genai.rs` — source decoding,
  classification, and lowering.
- `rust/runtime/src/engine/graph_input.rs` — built-in adapter registration and
  adapter-level fidelity tests.
- `rust/runtime/src/graph/conditional/fold.rs` — corresponding replay folding
  contract for authored conditional graphs.
- `rust/runtime/src/graph/executor.rs` — static edge timing anchors.
