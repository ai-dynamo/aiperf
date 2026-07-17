<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Graph IR Validation Reference

This reference describes the static validation layer for AIPerf Graph IR workloads. The rule set is deliberately small: it is the contract for what the live trace adapters (weka, dynamo) emit — `LlmNode` topologies wired with `StaticEdge` timing anchors — plus the shared graph-record invariants (version, provenance, reserved names). Rules for constructs no live producer emits (conditional edges, mix records, subgraphs, loops, barriers, spawn/await, replay coverage, endpoint pools, channel directives, streaming reducers, placement hints, retry policies) were retired; see git history for the removal rationale.

Primary implementation files:

- `src/aiperf/dataset/graph/validator.py`
- `src/aiperf/dataset/graph/parser.py`

## Validation lifecycle

Graph IR validation happens after a workload has been parsed into a `ParsedGraph`.

1. `parse_graph(...)` reads native Graph IR files or delegates to a graph adapter.
2. Native parsing enforces record-level shape and ordering, including parse-time rule-19 and rule-20 failures.
3. The parser decodes typed Graph IR models and wraps schema decode failures as `GraphParseError` messages.
4. `validate(parsed)` runs the validator rule set and returns all findings.

The profile runtime does not invoke `validate(...)` automatically; the rule set is exercised as an adapter-output contract by unit tests and is available to tooling that wants to lint a `ParsedGraph`.

The validator does not stop after the first problem. It accumulates every implemented `ValidationIssue` so related wiring mistakes can be fixed in one pass.

## Finding format and severity

Each validator finding is a `ValidationIssue` with these fields:

| Field | Meaning |
|---|---|
| `rule_id` | Stable rule label such as `rule-1`, when the rule is implemented in source. |
| `severity` | `error` or `warning`. Errors block execution. Warnings are informational unless the caller elects to treat them as blocking. |
| `location` | Human-readable pointer to the offending graph, node, edge, or trace field. |
| `message` | Plain-English explanation of the problem. |
| `suggested_fix` | Optional remediation hint. |

Use the `location` first when debugging. It is intentionally close to author-facing Graph IR paths, for example `graph.nodes.<node_id>`, `graph.edges[<source>-><target>]`, `graph.provenance.tool`, or `graph.version`.

## Rule categories

The current validator dispatches the following verified rule IDs. Rule IDs not listed here should not be documented as implemented without checking source first.

| Area | Verified rule IDs | What is checked |
|---|---:|---|
| Structural topology | `rule-1`, `rule-21` | Cycles (iterative DFS — safe on 100k+-node chains); reachability from `START`. |
| Writer conflicts | `rule-9` | Two or more nodes writing the same overwrite-reducer channel. |
| Versioning and provenance | `rule-11`, `rule-12`, `rule-13` | Reserved node names, known graph version, provenance tool (missing tool on a non-hand-authored graph is an error; the default `"manual"` on a non-hand-authored graph is a warning — it means the generating adapter never stamped itself). |
| Engine expectations | `rule-15` | Warning when `expected.cache_read_tokens` / `expected.cache_creation_tokens` are set — not all engines report cache fields. |
| Timing edge anchors | `rule-54`, `rule-55` | Edge delay-anchor exclusivity and first-token-anchor shape (dispatch fallback present, no completion anchor, no `START` source). |
| Edge endpoints | `rule-56` | Every edge `source` is a declared node or `START`; every edge `target` is a declared node or `END`. Dangling endpoints (typos, `END` as source, `START` as target) are errors. |
| Delay finiteness | `rule-57` | Every edge delay field (`delay_after_predecessor_us`, `min_start_delay_us`, `delay_after_predecessor_start_us`, `delay_after_predecessor_first_token_us`) and node `min_start_delay_us` must be finite — a non-finite gate never clears and hangs the trace. Catches typed-struct producers; the loose decoder rejects non-finite values at decode time. |
| Parse-time record ordering | `rule-19`, `rule-20` | Native record order and the singleton graph record, enforced by `parser.py` before `validate(...)` returns findings. |

`validate(parsed)` runs every rule over the main `parsed.graph` **and** over every per-trace graph in `parsed.graphs` (multi-graph workloads: weka heterogeneous directories, per-trace native lowering). Issues found in a `parsed.graphs` entry are re-located under `graphs[<name>].…` so the offending graph is identifiable; the aliased first entry (identical object to `parsed.graph`) is not double-reported.

## Common failures and remediations

### Graph topology errors

Symptoms:

- Cycle findings on `graph.edges` (rule-1).
- Nodes reported unreachable from `START` (rule-21).

Fixes:

- Pre-unroll loops into acyclic trace topology.
- Add missing `START -> node`, `node -> node`, or `node -> END` edges, or remove dead nodes. For adapter-emitted IR, fix the adapter's edge lowering rather than the generated file.

### Writer conflicts

Symptoms:

- Multiple nodes write the same overwrite channel (rule-9).

Fixes:

- Rename outputs so overwrite channels have one writer. For adapter-emitted IR this indicates a node-id / output-channel collision in the lowering.

### Versioning, provenance, and reserved names

Symptoms:

- A node id collides with `START`/`END` or begins with `_aiperf` (rule-11).
- `graph.version` is not a known major version (rule-12).
- A non-hand-authored graph is missing `provenance.tool` (error), or still carries the field default `"manual"` (warning — the generating adapter never stamped itself) (rule-13).

Fixes:

- Rename the node; set `version` to a known value; have the generating adapter stamp `provenance.tool` as `<tool-name>/<version>`.

### Timing anchor errors

Symptoms:

- An edge sets both `delay_after_predecessor_us` and `delay_after_predecessor_start_us`, or a `START`-sourced edge is start-anchored (rule-54).
- A first-token-anchored edge lacks its `delay_after_predecessor_start_us` dispatch fallback, combines with the completion anchor, or sources at `START` (rule-55).

Fixes:

- Keep exactly one anchor per edge; give first-token anchors their start-anchored fallback; use `min_start_delay_us` for absolute offsets from trace start.

## Parse-time failures versus validation findings

Not every invalid workload reaches `validate(...)`.

Native Graph IR parsing raises `GraphParseError` for file and record-shape problems such as malformed JSON/YAML, non-object JSONL records, non-mapping YAML documents, malformed top-level `graph`/`traces` blocks, retired record kinds (`kind: mix`, `kind: subgraph`, or a `graph` record carrying an `endpoints` block), and unknown record kinds. A single-document workload with an unknown top-level key (anything other than `graph`/`traces`, including the retired `mix`/`subgraphs` sections) is rejected with the offending key named and the nearest valid key suggested; a document containing none of those sections is rejected rather than silently discarded.

The loose decoder (`decode.py`, surfaced as `GraphDecodeError` / `GraphParseError`) additionally rejects:

- unknown fields on `graph`, node, edge, and trace records (`forbid_unknown_fields` — a typo'd field name fails instead of silently changing what is benchmarked; vendor keys are still preserved under `provenance.extra`),
- a node setting both `prompt` and `messages` (aliases; keep exactly one),
- a prompt-less node carrying `expected.input_tokens` (synth-token prompt fabrication is not supported on the native lowering path),
- non-finite (`inf`/NaN) edge delay values and node `min_start_delay_us` (mirrored by rule-57 for typed-struct producers),
- a non-dict `provenance.extra`.

The native lowering additionally rejects a first-token-anchored edge with no `delay_after_predecessor_start_us` fallback (the rule-55 shape) at parse time, because the profile pipeline lowers without invoking `validate(...)` and would otherwise treat the edge as a completion edge.

The parser also enforces these verified rule IDs directly:

- `rule-19`: a `kind: graph` record must precede trace records.
- `rule-20`: only one `kind: graph` record is allowed.

Typed model decode failures are wrapped as `GraphParseError` with an `IR error` message that preserves the decoder's field path.

## Unsupported-feature `NotImplementedError` location convention

Validator findings are preferred for invalid author input that can be reported as a `ValidationIssue`. Use `NotImplementedError` only for unsupported constructs or guardrails that are outside the static rule set.

When adding an unsupported-feature gate, follow the repository validator-gate convention:

```text
<loc>: <reason>
```

The `<loc>` prefix must identify the exact author-owned construct. For Graph IR, use the same location style as `ValidationIssue.location`, for example:

- `graph.nodes.<node_id>: <reason>`
- `graph.nodes.<node_id>.<field>: <reason>`
- `traces[<trace_id>].<field>: <reason>`

This keeps unsupported-feature failures actionable without requiring users to grep the workload. Do not raise a bare `NotImplementedError` such as `unsupported graph shape` when the offending node, trace, or field is known.

## Developer checklist for new rules

When adding a Graph IR validator rule:

- Add the rule implementation to `validator.py` as a `_rule_NN_<name>` function.
- Return `list[ValidationIssue]`; do not raise for ordinary authoring errors.
- Include `rule_id`, `severity`, `location`, `message`, and a `suggested_fix` where practical.
- Add the new rule to `validate(...)` in `validator.py`.
- Keep `location` paths human-readable and specific enough to locate the offending workload field.
- Add or update tests that pin both positive and negative behavior against IR the live adapters can actually emit.
- Do not document a new exact rule ID until it exists in source.
- Only add a rule when a live producer can emit the construct it validates (or the rule guards an adapter-lowering contract); rules for hypothetical authoring surface get retired.
