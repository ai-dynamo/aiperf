<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Graph IR schema reference

This page documents the native AIPerf graph intermediate representation (IR): the authorable JSON/YAML/JSONL shape, the typed in-memory model it decodes to, and the boundary between stable workload fields and adapter/runtime internals.

The canonical typed models live in `src/aiperf/dataset/graph/models.py`. Native files are parsed by `parser.py`, loose/legacy shapes are normalized by `decode.py`, and auto-derived defaults are applied by `auto_derive.py` (which also normalizes runtime channels), followed by interned-segment lowering in `native_lowering.py`. There is no native serializer: the loader reads native files but does not emit them.

## Native file formats

A native graph workload can be written as either:

- **JSONL records**: one JSON object per non-empty line. Each object is a record with a `kind` field.
- **JSON/YAML document**: a single mapping with top-level `graph` and `traces` sections. YAML multi-document files may also use one `kind` record per document.

Benchmark workload detection selects native parsing only when the caller chooses the native graph format, for example via `--graph-format native`; auto-detection excludes `native` so arbitrary JSONL/YAML chat datasets are not hijacked into graph mode. The lower-level `parse_graph(...)` helper can still auto-detect the native adapter from a `.yaml`, `.yml`, or `.jsonl` suffix when called directly by tooling.

### JSONL record shape

```jsonl
{"kind":"graph","version":"2.0","nodes":{"llm":{"node_type":"llm","prompt":["@messages"],"output":"messages"}}}
{"kind":"trace","id":"trace-1","initial_state":{"messages":[{"role":"user","content":"hi"}]}}
```

Supported record kinds:

| `kind` | Body decodes as | Notes |
| --- | --- | --- |
| `graph` | `GraphRecord` | At most one. Must appear before any `trace` record. |
| `trace` | `TraceRecord` | If `kind` is omitted in JSONL or multi-document YAML, the parser defaults the record to `trace`. |

Earlier revisions of this schema carried a rich node-kind zoo (`replay`, `tool_call`/`tool_result`, `subgraph`, `spawn`/`await`, `delay`, `barrier`, `compact`, `bootstrap`, `loop`), conditional edges, `mix` records, top-level subgraphs, and per-graph endpoint pools. All of these were retired with their runtime: every live producer (the weka and dynamo trace adapters, and the native lowering) emits flat `llm`-node topologies wired with static edges. Authored files that still carry one of these constructs fail at parse like any other invalid input — as an unknown node kind, unknown record kind, or unknown field (`branches`, `endpoints`).

Parser ordering and cardinality rules:

- A `graph` record must precede all `trace` records.
- More than one `graph` record is rejected.
- Unknown record kinds (anything other than `graph` and `trace`, including the former `mix` and `subgraph`) are rejected.
- A `graph` record carrying an `endpoints` block is rejected as an unknown field; use the global `--url`/`--model` run configuration.

### Single-document JSON/YAML shape

The same workload is usually easier to author as one JSON/YAML mapping:

```yaml
graph:
  version: "2.0"
  provenance:
    source: hand-authored
    tool: manual
  state:
    messages:
      type: messages
      reducer: add_messages
  nodes:
    llm:
      node_type: llm
      prompt:
        - "@messages"
      output: messages
  edges:
    - source: START
      target: llm
    - source: llm
      target: END

traces:
  - id: trace-1
    tags: [chat]
    initial_state:
      messages:
        - role: user
          content: hi
```

Single-document expansion maps:

- `graph:` to a `kind: graph` record.
- Each item in `traces:` to a `kind: trace` record.

A `mix:` or `subgraphs:` section is rejected at the top-level-key gate like any other unknown key (the record kinds they used to expand to were retired with their runtime).

When authoring by hand, the single-document layout (a `graph:` block followed by `traces:`) is the recommended form; the equivalent JSONL form is an ordered `kind: graph` record followed by `kind: trace` records.

## Top-level in-memory shape

Parsing native files produces a `ParsedGraph`:

| Field | Type | Authoring status |
| --- | --- | --- |
| `graph` | `GraphRecord` | Main/top-level graph. Native files author this under `graph:` or `kind: graph`. Defaults to an empty graph if absent. |
| `graphs` | map of graph key to `GraphRecord` | Per-trace top-level graphs for multi-graph workloads (Weka heterogeneous directories, per-trace native lowering), keyed by the value each `TraceRecord.graph_ref` names. Resolved via `resolve_trace_graph`. Not authored. |
| `traces` | list of `TraceRecord` | Native files author this under `traces:` or `kind: trace` records. |
| `segment_pool` | `SegmentPool` or null | Content-addressed segment pool that backs every `LlmNode`'s interned prompt; set by every live producer and drained into the unified store by the build plane. Do not author. |

## Graph records

A `GraphRecord` declares topology and channel state:

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `version` | string | `"2.0"` | Known schema version. |
| `provenance` | `ProvenanceSpec` | `{source: hand-authored, tool: manual}` | Origin metadata. Unknown provenance keys round-trip through `extra`. |
| `system` | string or null | null | Linear-chat shorthand system prompt. Used only when no explicit nodes are authored. |
| `state` | map of channel name to `ChannelSpec` | `{}` | Explicit channel declarations. Missing safe runtime channels may be auto-derived. |
| `nodes` | map of node id to node spec | `{}` | Node ids must not be `START`, `END`, or start with `_aiperf`. |
| `edges` | list of `StaticEdge` | `[]` | `START` and `END` are sentinel ids. |

If a main graph has nodes but omits explicit sentinel edges, the native parser auto-injects `START -> <root>` for nodes with no incoming edge and `<leaf> -> END` for nodes with no outgoing edge. Existing explicit `START` or `END` edges are preserved. This auto-injection applies to the main graph parse path.

If a graph has no nodes, `auto_derive.py` synthesizes a linear chat graph:

- node id `_llm`
- prompt containing optional `graph.system` plus `@messages`
- `messages` channel with `type: messages`, `reducer: add_messages`
- `START -> _llm -> END`

Trace-level `messages` are lifted into `initial_state.messages` for every workload (with or without explicit nodes), before linear-chat synthesis runs.

## Channels and reducers

`graph.state` maps channel names to `ChannelSpec`:

| Field | Values | Default | Meaning |
| --- | --- | --- | --- |
| `type` | `text`, `messages` | `text` | Value modality. |
| `reducer` | `overwrite`, `add_messages` | `overwrite` | How writes merge. |

Reducer semantics:

- `overwrite`: one writer value wins; static validation catches multi-writer conflicts, and runtime catches duplicate overwrite writes from any writer that reaches the channel.
- `add_messages`: appends message lists and replaces prior messages with the same message `id`. Pair with `type: messages`.

Auto-derived channels:

- Node output channels, trace `initial_state` channels, and trace `replay_outputs` channels are declared if missing.
- A missing channel named `messages` defaults to `type: messages`, `reducer: add_messages`.
- Other missing runtime channels default to `type: text`, `reducer: overwrite`.
- LLM prompt references in the main graph at top-level prompt-array position, such as `"@messages"`, auto-declare missing channels as `messages/add_messages`.
- LLM prompt references in the main graph inside a message content block, such as `content: ["@context"]`, auto-declare missing channels as `text/overwrite`.

Prompt channel references use `@channel`; `@@literal` escapes a leading at-sign.

## Channel requirements and `count` semantics

Every node inherits a common `inputs` field:

```yaml
inputs:
  - channel: draft_ready
    count: 1
  - channel: fanout_done
    count: all
```

`inputs` is an AND-fan-in gate: a node with non-empty `inputs` fires only after every requirement is satisfied. An empty `inputs` list keeps the legacy successor-walk behavior.

`ChannelRequirement.count` is not a token count, byte count, or list length. It counts node-write arrivals on the named channel:

- `count: N` waits until the Nth node write to that channel has arrived. `N` must be at least 1.
- `count: all` resolves to the static number of declared producers for that channel and waits for that many arrivals.
- `trace.initial_state` seeds a channel value but does not increment the arrival count.
- The read snapshot captures the first `N` node writes plus any initial seed as reducer input.
- If producer completion/cancellation makes the requested count unreachable, runtime raises an orphaned-channel error instead of waiting forever.

Use `count: 1` for “fire when any one producer writes this gate channel” and `count: all` for “wait for every statically declared producer of this gate channel.”

## Node common fields

Every node inherits these optional fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `metadata` | mapping | Free-form span metadata, emitted under `aiperf.meta.*`. |
| `min_start_delay_us` | non-negative float or null | Minimum delay after all predecessors/inputs are satisfied before this node may start. |
| `arrival_offset_us` | non-negative integer or null | Recorded offset from trace arrival, used by adapters/snapshot replay. |
| `inputs` | list of `ChannelRequirement` | Explicit AND-fan-in channel gate. |

`decode.py` accepts both `node_type` and legacy `kind` as the node discriminator; the only valid value is `llm`. Any other value fails as unknown, and a node with no discriminator decodes as `llm`. `messages` is accepted as an alias for `prompt`; a node that sets both is rejected. A node with neither fails to decode (`prompt` is required) — in particular, a prompt-less node carrying only `expected.input_tokens` is rejected with an explicit error: synth-token prompt fabrication is not supported on the native lowering path. Unknown node fields are rejected (`forbid_unknown_fields`), so a typo'd field name fails loudly instead of silently changing what is benchmarked; the same applies to `graph` and `trace` records, with `provenance` remaining the vendor-key catch-all (unknown provenance keys fold into `extra`).

## Node kinds

`llm` is the only node kind.

### `llm`

Dispatches an LLM request.

Canonical fields:

- `prompt`: list of prompt items (`messages` is accepted as an alias; a node without either fails to decode).
- `output`: output channel name.

Important optional fields:

- `streaming`: defaults to `true`.
- `expected`: expected token accounting (`input_tokens`, `output_tokens`, cache fields).
- `extra_body`: request-body overrides such as temperature or provider-specific keys (Turn naming; model / streaming / max_tokens / raw_tools / extra_headers are first-class node fields).

Writes: `[output]`.

## Edges

Edges are `StaticEdge` records, tagged `edge_type: static` in canonical output; `decode.py` also accepts loose edge objects without the tag. An edge carrying a `branches` mapping (the former conditional-edge shape) is rejected at decode as an unknown field.

```yaml
- source: START
  target: llm
  delay_after_predecessor_us: 1000
  min_start_delay_us: 500
```

Fields:

- `source`: source node id or `START` (a source that is neither is rejected by validator rule 56).
- `target`: target node id or `END` (a target that is neither is rejected by validator rule 56).
- `delay_after_predecessor_us`: optional non-negative idle/scheduling delay after predecessor completion.
- `delay_after_predecessor_start_us`: optional non-negative idle delay measured from the moment the predecessor DISPATCHES (the successor does not await the predecessor's completion or output). Mutually exclusive with `delay_after_predecessor_us`; an edge that sets both is rejected by validator rule 54.
- `delay_after_predecessor_first_token_us`: optional non-negative idle delay measured from the predecessor's OBSERVED FIRST TOKEN; the runtime gates the successor at `first_token_wall + this delay`, falling back to dispatch + `delay_after_predecessor_start_us` when the predecessor terminates without a first token. Only valid alongside `delay_after_predecessor_start_us` (the dispatch fallback), must not combine with `delay_after_predecessor_us`, and cannot be `START`-sourced — validator rule 55. The native lowering also enforces the missing-fallback case at parse time.
- `min_start_delay_us`: optional non-negative minimum wait on the successor after predecessors are satisfied.

All delay values must be finite: `inf`/NaN are rejected at decode time (`GraphDecodeError`) and by validator rule 57 for graphs constructed directly from typed structs — a non-finite gate never clears and would hang the trace.

## Trace records

A `TraceRecord` supplies per-session data:

| Field | Type | Notes |
| --- | --- | --- |
| `id` | string | Required stable trace id. |
| `tags` | list of strings | Opaque provenance labels, round-tripped through the codec/sidecar; no runtime consumer reads them. |
| `graph_ref` | string or null | Multi-graph selector naming a key in `ParsedGraph.graphs`; resolution goes through `resolve_trace_graph`. Set by adapters and the native lowering; native single-graph workloads leave this null. |
| `messages` | list of messages or null | Linear-chat shorthand, equivalent to `initial_state.messages`; lifted into `initial_state.messages` for every trace (explicit-node graphs included) unless `initial_state.messages` is already set, which wins. |
| `initial_state` | mapping | Initial channel values at trace start. Seeds reducers but does not satisfy `ChannelRequirement.count`. |
| `replay_outputs` | mapping of node id to channel values | Authorable native-file surface read by `auto_derive` for channel inference; adapters never populate it, and the structural sidecar strip (`graph_meta_sidecar`) clears it. |

Channel values spliced by `@channel` prompt references must be plain data: a messages splice requires a list of `{role, content}` message dicts and a text splice requires a string. Typed/multimodal directive blocks and non-string content blocks are rejected by the unified-store lowering with an actionable error.

## Advanced and internal fields boundary

The graph IR intentionally preserves more information than the runtime needs to execute a hand-authored workload. Treat fields in three groups:

### Stable authoring surface

These are normal native workload fields:

- `graph.version`, `provenance`, `system`, `state`, `nodes`, `edges`
- `traces`
- Node common fields, `llm` node fields, `inputs`, edge timing anchors

### Advanced preserved metadata

These may be authored, but are primarily produced by adapters or used for fidelity/analysis rather than core scheduling:

- `expected`
- `extra_body`
- `arrival_offset_us`
- vendor-specific keys inside the `ProvenanceSpec` catch-all

Unknown vendor keys round-trip only for models that explicitly preserve extras (`provenance.extra`). Native single-document parsing rejects unknown root keys with the offending key named and the nearest valid key suggested (only `graph` and `traces` are recognized).

### Runtime/adapter internals

Do not hand-author these in native graph files unless you are extending the loader/runtime:

- `ParsedGraph.graphs`
- `ParsedGraph.segment_pool`

## Validation highlights

The validator enforces cross-field rules. Common authoring errors include:

- cycles in static topology
- unreachable nodes
- multiple writers to an `overwrite` channel, with broader duplicate-write protection enforced at runtime
- node ids colliding with `START`/`END` or the `_aiperf` reserved prefix
- unknown `graph.version`, or a non-hand-authored graph missing `provenance.tool` (error) or still carrying the default `"manual"` (warning)
- an edge setting both `delay_after_predecessor_us` and `delay_after_predecessor_start_us` (mutually exclusive; validator rule 54)
- a `START`-sourced edge that is start-anchored (`delay_after_predecessor_start_us`); the `START` pseudo-node never dispatches, so its start-anchored successors would never fire — use `min_start_delay_us` for absolute offsets (validator rule 54)
- a first-token-anchored edge (`delay_after_predecessor_first_token_us`) that sets no `delay_after_predecessor_start_us` fallback, combines with `delay_after_predecessor_us`, or is `START`-sourced; the START pseudo-node never dispatches or streams a first token (validator rule 55)
- an edge whose `source`/`target` is neither a declared node nor the matching `START`/`END` sentinel (validator rule 56)
- a non-finite delay value on an edge or a node `min_start_delay_us` (validator rule 57; also rejected at decode time)

`validate()` runs every rule over the main graph and over every per-trace graph in `ParsedGraph.graphs`; issues in a `graphs` entry are located under `graphs[<name>].…`.

Only parse-time checks run automatically before a benchmark: schema/type decoding (surfaced as `GraphParseError`) plus the record-ordering and singleton-graph rules (rule-19, rule-20) enforced in `parser.py`. The full semantic rule set (`validate()` in `validator.py`) is not invoked by the profile runtime — it is an adapter-output contract exercised by unit tests and available to tooling that lints a `ParsedGraph`. See [Graph IR Validation](./graph-ir-validation.md).
