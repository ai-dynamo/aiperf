<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native Graph-IR inspection tools

## Purpose

This record describes the built Rust-native `aiperf graph` surface: `validate`,
`explain`, and `visualize`. It is available in the native binary without
Python, an inference endpoint, or benchmark traffic.

The central decision is that these commands inspect the same lowered
`GraphInputBundle` that benchmark execution consumes. They do not port the
Python `ParsedGraph` authoring model into Rust and do not construct a second
graph representation from the source.

## Built

The native runtime owns the graph input adapters, Graph-IR data model,
structural validator, scheduler, and filesystem-free inspection seam at
`rust/runtime/src/graph/inspect.rs`. `aiperf graph` is routed natively before
the Python fallback. The stock command uses the built-in resolver and its shared
seven-format inventory; unrelated top-level commands retain their existing
delegation behavior.

## Problem

The historical Python DAG-v3 branch provides useful offline commands:

- `aiperf graph validate <path>` parses and validates a graph workload
  (`ajc/dag-v3:src/aiperf/cli_commands/graph_validate.py:21-80`).
- `aiperf graph explain <path>` prints topology, nodes, channels, traces, and an
  illustrative schedule
  (`ajc/dag-v3:src/aiperf/cli_commands/graph_explain.py:32-99`).
- `aiperf graph visualize <path>` emits docs-ready Markdown or an HTML page
  (`ajc/dag-v3:src/aiperf/cli_commands/graph_visualize.py:41-149`).

Historically, `graph` fell through to Python delegation
(`rust/cli/src/dispatch.rs:11-35`). Native inspection closes that offline gap
while retaining the runtime-owned graph input adapters, Graph-IR data model,
structural validator, and scheduler as the authority.

A direct command-for-command translation would be incorrect. Python explains a
rich pre-lowering `ParsedGraph` that includes conditional edges, replay-style
nodes, subgraphs, placement, waits, expected-token metadata, and authored branch
selection (`ajc/dag-v3:src/aiperf/cli_commands/graph_explain.py:102-252` and
`:255-393`). Native `GraphRecord` deliberately contains only runtime-relevant
channels, executable `Llm`/`Tool` nodes, and static edges
(`rust/runtime/src/graph/model.rs:206-311` and `:319-347`). In particular,
`conditional_graph` resolves branches, prunes untaken paths, folds recorded
content, and emits one flat graph per trace before execution
(`rust/runtime/src/graph/conditional/mod.rs:55-66` and `:89-120`).

The native tools therefore have an explicit, truthful contract: they inspect
the lowered, retained execution-shaped Graph-IR for each trace. They are not an authored
source editor or a complete replacement for every DAG-v3 graph command.

## Goals

1. Make `validate`, `explain`, and `visualize` work in the native `aiperf`
   binary without Python, a server, an inference endpoint, or benchmark traffic.
2. Use exactly one existing graph input adapter and retain its
   `GraphInputBundle` as the sole semantic input.
3. Report every runtime-relevant structural problem, adapter warning, and
   optional arrival-pacing problem with stable machine-readable identifiers.
4. Explain and render the per-trace resolved topology, including the distinction
   between profiling and warmup plans.
5. Produce deterministic text, JSON, Mermaid, and Markdown suitable for tests,
   scripts, and documentation.
6. Keep semantic analysis in `aiperf-runtime` and CLI parsing, presentation, and
   filesystem output in `aiperf-cli`.

## Non-goals

- Reintroducing Python `ParsedGraph`, Pydantic models, directives, auto-derive,
  or Python plugin discovery in the Rust binary.
- Matching Python validator `rule-N` identifiers. The Python validator runs 53
  rules over a richer pre-lowering representation
  (`ajc/dag-v3:src/aiperf/dataset/loader/graph/validator.py:92-177`); reusing its
  identifiers for different native checks would be a false compatibility claim.
- Showing the full authored conditional topology. The native input bundle holds
  the selected and pruned topology for each trace.
- Executing a trace, issuing model/tool requests, simulating latency, or writing
  benchmark metric artifacts.
- Porting `graph view`, `replay-audit`, `scaffold`, `fix`, `export`, `convert`,
  `normalize`, or `merge`, all of which are separate commands in the historical
  Python namespace (`ajc/dag-v3:src/aiperf/cli_commands/graph.py:29-39`).
- Browser rendering, HTML, Cytoscape, Dagre, ELK, Mermaid JavaScript, or any CDN
  dependency.

## CLI contract

### Command family

The public commands are:

```text
aiperf graph validate <PATH> --graph-format <FORMAT>
    [--tokenizer <NAME_OR_LOCAL_PATH>] [--endpoint-type <TYPE>]
    [--source-format <FORMAT>] [--seed <U64>]
    [--pace arrival] [--output-format text|json]

aiperf graph explain <PATH> --graph-format <FORMAT>
    [--tokenizer <NAME_OR_LOCAL_PATH>] [--endpoint-type <TYPE>]
    [--source-format <FORMAT>] [--seed <U64>]
    [--output-format text|json]

aiperf graph visualize <PATH> --graph-format <FORMAT>
    [--tokenizer <NAME_OR_LOCAL_PATH>] [--endpoint-type <TYPE>]
    [--source-format <FORMAT>] [--seed <U64>]
    [--trace <TRACE_ID>] [--output <PATH>]
    [--output-format markdown|mermaid] [--no-validate]
```

`aiperf graph` without a subcommand prints namespace help and exits 2. This
matches the useful part of the Python namespace behavior
(`ajc/dag-v3:src/aiperf/cli_commands/graph.py:22-26`). Clap usage errors also
exit 2, following the existing native config command's `try_parse_from` pattern
(`rust/cli/src/config/mod.rs:64-80`).

Common defaults are:

- `--tokenizer builtin`;
- `--endpoint-type chat`, passed only to the selected adapter to validate its
  request-profile compatibility; it never resolves or contacts an endpoint;
- no `--source-format`; when supplied, it selects the imported-session source
  discriminator for `agent_recording` input and has no role in other formats;
- `--seed 0`;
- `--output-format text` for `validate` and `explain`;
- `--output-format markdown` for `visualize`;
- stdout when `--output` is absent;
- the first program in authored bundle order when `visualize --trace` is absent.

`--pace` accepts only `arrival`. When present, it requires every profiling plan
to have an `arrival_offset_ns`; a retained plan without one receives an
`arrival-offset-missing` validation issue. This is the native equivalent of Python's opt-in
arrival-time rule (`ajc/dag-v3:src/aiperf/cli_commands/graph_validate.py:33-39`
and `ajc/dag-v3:src/aiperf/dataset/loader/graph/validator.py:158-177`).

### Input shorthand

The first release accepts a local file or directory path plus an explicit graph
format. The CLI constructs the existing adapter-owned dataset object:

```json
{
  "type": "file",
  "format": "<FORMAT>",
  "path": "<PATH>",
  "sampling": "sequential"
}
```

This is not a new public graph-file schema. It is a shorthand for the same raw
dataset object used by native graph preparation. File datasets already carry
`format`, `path`, `sampling`, `options`, `records`, graph replay settings, and
other adapter-owned fields (`rust/runtime/src/config/model/dataset.rs:398-441`).
The shorthand deliberately authors only the four fields above.

`--graph-format` is required. The Python parser can use extension and content
sniffing (`ajc/dag-v3:src/aiperf/dataset/loader/graph/parser.py:33-67`), but the
native resolver intentionally decodes only the explicit `format` identity and
then lets the selected adapter own full strict decoding
(`rust/runtime/src/engine/graph_input.rs:193-221` and `:264-301`). Native JSON
and JSONL sources overlap across formats, so content auto-detection would add a
second source read or a second parser and violate the lower-once boundary.

The initial command is local/offline. It rejects public, URL, and Hugging Face
sources. A local tokenizer path or built-in tiktoken encoding is allowed. A
tokenizer name that would require network acquisition is rejected with an
actionable error rather than silently downloading it.

### Format inventory

The command obtains supported values from the shared native inventory rather
than a second handwritten list. The stock built-in resolver composes:

- `dag_jsonl`;
- `conditional_graph`;
- `weka_trace`;
- `dynamo_trace`;
- `aiperf_trace`;
- `agent_recording`;
- `otlp_genai`.

The shared `config::model::workload_kind::GRAPH_FORMATS` inventory includes
`aiperf_trace` and drives workload classification, resolver composition tests,
and graph-command help/completion.

### Exit codes

| Code | Meaning |
|---:|---|
| 0 | Input loaded and the requested operation completed. Warnings do not fail a command. |
| 1 | The adapter lowered the input successfully, but validation found one or more error-severity issues. |
| 2 | CLI usage, missing/unreadable source, tokenizer preparation, adapter decode/lowering, trace selection, or output I/O failure. |

`explain` is best-effort after successful lowering: it reports validation issues
and omits analyses that require a valid graph, but returns 0. `visualize`
validates the entire retained bundle and returns 1 without rendering if any
bundle, profiling, or warmup error exists, unless
`--no-validate` is present. This preserves Python visualize's useful validation
gate (`ajc/dag-v3:src/aiperf/cli_commands/graph_visualize.py:138-149` and
`:176-191`) while keeping `validate` the command whose exit status represents
graph validity.

## Lower-once architecture

The complete command flow is:

1. Clap parses flags. The CLI verifies that `<PATH>` exists and is local, but
   does not inspect or sniff its contents.
2. The CLI serializes the four-field dataset shorthand to one `RawValue`.
3. One local tokenizer is built from `--tokenizer` by the shared preparation
   seam. It handles directories, tokenizer files, and built-in encodings.
4. A current-thread Tokio runtime and `LocalSet` call exactly one
   `GraphInputAdapterResolver::load` with `run_random_seed: Some(seed)`. This
   mirrors production's direct Graph-IR preparation at
   `rust/runtime/src/engine/online_execution.rs:1214-1239`.
5. The adapter performs source acquisition, strict decode, normalization,
   content interning, and lowering. It returns one `GraphInputBundle` containing
   ordered `GraphTraceProgram`s, one immutable segment store, and input metadata
   (`rust/runtime/src/graph/input.rs:34-74`).
6. The selected command borrows that retained bundle. Validation, explanation,
   trace selection, and rendering never reread the source, call the adapter
   again, or reconstruct `ParsedGraph`.

The seed is always present so conditional weighted-branch lowering and any
recorded content synthesis are deterministic. The conditional adapter already
threads `run_random_seed` into its compiler
(`rust/runtime/src/engine/graph_input.rs:387-402`).

This flow preserves the execution boundary: `GraphTraceProgram` is the complete
placement-owned trace command, with profiling, optional warmup, environment,
replay context, and driver identity (`rust/runtime/src/graph/model.rs:378-415`).
Inspection retains projected plan facts, arrival timing, driver kind, and
environment/replay presence; it intentionally excludes environment recipes,
replay payload/context, and other execution payloads from its reports.

## Module boundaries

### `aiperf-runtime`

`rust/runtime/src/graph/inspect.rs`, exported from `graph/mod.rs`, owns pure,
filesystem-free analysis:

- detailed structural validation DTOs and codes;
- bundle/program/plan summaries;
- deterministic topology ordering;
- illustrative static readiness waves;
- the data DTOs consumed by text, JSON, and Mermaid renderers.

It does not print, colorize, serialize a complete CLI envelope, write files, or
know Clap arguments.

`rust/runtime/src/engine/graph_input.rs` exposes a small public preparation
entry point and the supported-format inventory. These wrap the existing
resolver; they do not create a second registry.

Local tokenizer construction is re-exported from the preparation boundary, so
graph tooling and production share tokenizer semantics without depending on
private execution internals.

### `aiperf-cli`

The CLI contains:

```text
rust/cli/src/graph/
  mod.rs          Clap types, common loading, command dispatch, exit policy
  validate.rs     validation text/JSON envelopes
  explain.rs      explanation text/JSON envelopes
  visualize.rs    Markdown/Mermaid rendering and output handling
  report.rs       versioned serde CLI output DTOs
```

`rust/cli/src/lib.rs` exports `graph`, and `rust/cli/src/dispatch.rs` routes
`Some("graph")` before Python fallback. No other delegated command changes.

Presentation belongs in the CLI:

- human table layout;
- JSON envelope serialization;
- Mermaid escaping and style declarations;
- Markdown assembly;
- stdout/stderr selection;
- atomic `--output` writes.

No new crate is needed. `aiperf-cli` already depends on Clap, Serde,
`serde_json`, Tokio, and `tempfile` (`rust/cli/Cargo.toml:82-105`).

## Validation contract

### Validation layers

Validation visits every profiling plan and every present warmup plan.

1. **Preparation failures.** Source acquisition, strict adapter decoding,
   normalization, interning, or lowering failures are fatal command errors and
   exit 2. They are not graph issues because no trustworthy bundle exists.
2. **Adapter warnings.** Inspection retains every
   `GraphInputMetadata.warning_facts` entry as a bundle-level warning-severity
   issue. Its stable code is exposed as `adapter-warning.<adapter-code>` and
   its deterministic context map is retained in the report
   (`rust/runtime/src/graph/input.rs:34-63`,
   `rust/runtime/src/graph/inspect.rs:315-328`). Warnings remain observable in
   `validate` and `explain` without making validation fail.
3. **Structural validation.** Inspection uses the existing checks for unknown
   edge endpoints, undeclared read/write channels, unreachable nodes, and
   unsatisfiable firing gates (`rust/runtime/src/graph/validate.rs:27-149`).
4. **Scheduler construction.** Inspection constructs `Scheduler::new` to expose
   mixed or multiple start-anchored fan-in, a separate executor invariant
   (`rust/runtime/src/graph/scheduler.rs:31-75` and `:123-163`).
5. **Bundle invariants.** Check a nonempty program list, unique profiling trace
   IDs, and agreement between bundle metadata and the retained programs. Validate
   that a warmup plan, when present, has a nonempty trace ID.
6. **Arrival policy.** With `--pace arrival`, inspection requires
   `program.profiling.arrival_offset_ns` for every program. Arrival offset is the
   native plan field (`rust/runtime/src/graph/model.rs:367-376`).

### Detailed issue API

Keep the existing `validate(&GraphRecord) -> Vec<ValidationError>` compatibility
API. `graph::inspect::validate_detailed` exposes the detailed typed API, while
the compatibility API and detailed API both derive from the shared validation
findings. The detailed issue shape is:

```rust
pub struct GraphInspectionIssue {
    pub code: String,
    pub severity: GraphInspectionSeverity,
    pub trace_id: Option<String>,
    pub phase: Option<GraphPlanPhase>,
    pub location: Option<String>,
    pub message: String,
    pub context: BTreeMap<String, String>,
}
```

Initial stable codes are:

- `edge-source-unknown`;
- `edge-target-unknown`;
- `channel-write-undeclared`;
- `channel-read-undeclared`;
- `node-unreachable`;
- `node-never-fireable`;
- `mixed-anchor-fan-in`;
- `multi-start-anchor-fan-in`;
- `bundle-empty`;
- `trace-id-empty`;
- `trace-id-duplicate`;
- `metadata-root-count-mismatch`;
- `metadata-node-count-mismatch`;
- `arrival-offset-missing`;
- `adapter-warning.<adapter-code>`.

Codes are a compatibility contract. Messages can improve without requiring
machine consumers to parse prose.

### Human output

Human validation writes issues and the summary to stdout:

```text
ERROR [node-unreachable] trace=t-1 phase=profiling graph.nodes.foo: node is unreachable from START
WARNING [adapter-warning.missing-model] trace=t-1: model was not recorded
FAIL: 1 error(s), 1 warning(s).
```

Clean or warnings-only input ends with:

```text
OK: 0 errors, 1 warning(s).
```

Fatal human diagnostics go to stderr and do not print a success/failure summary.

### JSON output

Successful lowering produces `aiperf.graph.validate.v1`:

```json
{
  "schema_version": "aiperf.graph.validate.v1",
  "source": "/absolute/input.dag.jsonl",
  "format": "dag_jsonl",
  "root_count": 2,
  "node_count": 7,
  "issues": [
    {
      "code": "node-unreachable",
      "severity": "error",
      "trace_id": "t-1",
      "phase": "profiling",
      "location": "graph.nodes.foo",
      "message": "node is unreachable from START",
      "context": {}
    }
  ],
  "summary": {"errors": 1, "warnings": 0}
}
```

Fatal errors in JSON mode still produce parseable stdout, with exit 2:

```json
{
  "schema_version": "aiperf.graph.error.v1",
  "operation": "validate",
  "code": "input-lowering-failed",
  "message": "...",
  "source": "/absolute/input.dag.jsonl"
}
```

Logging remains on stderr. The command handler catches expected failures and
serializes this envelope rather than letting the process-level `aiperf: {error}`
handler write an unstructured error (`rust/cli/src/main.rs:47-53`).

## Explain contract

### Human report

`explain` prints resolved execution facts in deterministic program order:

1. **Input:** canonical source path, lowered format, root count, aggregate node
   count, segment count, adapter warnings, and bundle findings.
2. **Traces:** trace ID, driver kind, arrival offset, warmup presence,
   environment/replay presence, and profiling counts for all nodes, LLM nodes,
   tool nodes, edges, and channels.
3. **Plans:** one profiling block per trace and a warmup block when present.
   Each block contains:
   - nodes: ID, `llm`/`tool`, output channel, input gates, prompt splice channels,
     streaming, model override, and maximum tokens where the Graph-IR carries
     them;
   - channels: name, type, reducer;
   - edges: source, target, anchor, and nonzero timing fields;
   - validation issues scoped to that plan;
   - illustrative readiness waves when available.

The report does not invent Python-only fields such as `terminal_for_user`,
placement, replay-node expected counts, wait distributions, or subgraph trees.
The native node accessors expose actual read/write channels and static request
counts at `rust/runtime/src/graph/model.rs:245-289`.

### Readiness waves

The Python renderer calls its rows an executor schedule but also states that the
runtime has no global barrier between them
(`ajc/dag-v3:src/aiperf/cli_commands/graph_visualize.py:231-244`). The native
report uses the narrower term **illustrative readiness waves**.

For a structurally valid static graph, the analyzer:

1. starts with `Scheduler::entry_nodes`;
2. tracks completed producer counts per channel;
3. admits a successor only after its incoming-edge and channel-count gates can
   be satisfied;
4. treats `delay_after_predecessor_start_us` successors as dispatch-anchored,
   not completion-anchored;
5. preserves deterministic edge order and deduplicates node IDs;
6. stops after every node has appeared once, or returns a typed analysis issue.

The scheduler distinguishes entry, completion successors, and dispatch-anchored
successors at `rust/runtime/src/graph/scheduler.rs:78-120`. Waves are explanatory
dependency levels, not timestamps, barriers, or a prediction of concurrent
completion order. For a non-static/custom driver, or an invalid graph, the
report says why waves are unavailable.

### JSON report

`--output-format json` emits `aiperf.graph.explain.v1`. It contains typed input
metadata and a `programs` array. Every program contains `trace_id`, driver,
arrival offset, environment/replay flags, profiling summary/topology/issues,
optional warmup summary/topology/issues, and optional readiness waves. It does
not embed ANSI strings or preformatted tables.

The JSON topology is an inspection summary, not a serialization of
`GraphTraceProgram`. It excludes payload/content values such as prompt bodies,
initial-state values, replay outputs, segment bytes, tool-request bodies, and
environment values. It intentionally exposes topology and request-shape metadata:
trace/node/channel identifiers, node kinds, channel gates and reducers, static
edges and timing, streaming, model override, maximum tokens, and environment or
replay presence flags.

## Visualization contract

### Trace selection

Visualization renders exactly one profiling plan. Without `--trace`, it selects
`bundle.programs[0]`. With `--trace`, it matches
`program.profiling.trace.id` exactly. A missing trace exits 2 and lists available
IDs in authored order, following the useful Python behavior
(`ajc/dag-v3:src/aiperf/cli_commands/graph_visualize.py:194-213`). An empty
bundle is a validation/preparation failure and cannot reach rendering.

### Mermaid

The Mermaid graph is deterministic:

- direction is `flowchart LR`;
- each authored node receives a synthetic Mermaid identifier `n0`, `n1`, and so
  on, while the escaped authored node ID is the visible label;
- IDs are ordered from reachable START traversal, then remaining nodes in
  `BTreeMap` order;
- START and END use dedicated terminal nodes;
- LLM and tool nodes use separate static class definitions;
- edges are ordered by source rank, source ID, target rank, and target ID;
- edge labels show dispatch/completion/first-token anchor and nonzero delay;
- terminal nodes with no explicit END edge receive a dashed explanatory edge to
  END only in the rendering, without mutating `GraphRecord`.

Synthetic Mermaid IDs are mandatory because Graph-IR node IDs are arbitrary
strings. Rendering them directly can create invalid Mermaid or identifier
collisions.

### Markdown

Markdown output contains:

~~~~text
## Graph topology

```mermaid
...
```

## Resolved plan

- Source: ...
- Format: ...
- Trace: ...
- Driver: ...

## Illustrative readiness waves

| Wave | Nodes ready | Trigger |
|---:|---|---|
...
~~~~

The document says that the topology is the selected trace's **resolved Graph-IR**.
It does not copy Python's “full possible topology” claim
(`ajc/dag-v3:src/aiperf/cli_commands/graph_visualize.py:239-243`).

`--output-format mermaid` emits only the Mermaid source plus one trailing
newline. `--output-format markdown` emits the complete document plus one trailing
newline.

### Validation and output I/O

By default, any bundle, profiling-plan, or warmup-plan validation error suppresses rendering and returns 1.
`--no-validate` permits best-effort topology rendering after successful
lowering, but cannot bypass adapter/lowering failures.

Without `--output`, rendered bytes go to stdout. With `--output`, the CLI writes
to a temporary file in the destination directory, flushes it, and leaves stdout
empty. Unix atomically replaces a nonexistent or existing regular file. Other
platforms write a nonexistent destination but reject a preexisting destination
rather than promise a non-atomic overwrite. It refuses a directory target and
reports output errors on stderr with exit 2.

### Why HTML is excluded

Python HTML mode imports Mermaid from a CDN and its interactive mode loads
Cytoscape, Dagre, ELK, and expansion plugins
(`ajc/dag-v3:src/aiperf/cli_commands/graph_visualize.py:557-604`). Those
dependencies do not improve the native inspection boundary. Markdown already
renders on GitHub and docs systems, while raw Mermaid remains consumable by
external renderers. The minimal native command therefore adds no JavaScript,
browser, server, or network surface.

## Error taxonomy

Fatal error codes used by `aiperf.graph.error.v1` are stable kebab-case strings:

- `invalid-arguments`;
- `source-not-found`;
- `source-not-local`;
- `tokenizer-unsupported`;
- `tokenizer-load-failed`;
- `input-decode-failed`;
- `input-lowering-failed`;
- `trace-not-found`;
- `output-invalid`;
- `output-write-failed`.

Errors retain their `anyhow` source chain internally, but the JSON `message` is
one bounded human-readable string. It does not include payload/content values or
arbitrary debug output. Human mode may include the source chain, subject to the
same redaction rule.

## Determinism and security

- The default seed is fixed at zero, and an authored seed is explicit.
- The adapter is invoked exactly once.
- Program order is adapter-authored order. `BTreeMap` node/channel order is
  stable; edge order is explicitly normalized for reports.
- Canonical source paths appear in metadata, but input contents do not.
- Segment store contents and prompt items are not dumped. `GraphInputBundle`
  retains an immutable `Arc<dyn SegmentStore>` specifically because Graph-IR
  handles refer to frozen content (`rust/runtime/src/graph/input.rs:66-74`).
- Local source checks reject URLs and Hugging Face dataset descriptors before
  adapter invocation.
- Tool environments are described only by presence and backend/driver identity;
  the inspection command never provisions or runs them.

## Verification evidence

### Runtime unit tests

Runtime unit coverage beside `graph::inspect` covers:

- a valid chain;
- parallel fan-out and gated fan-in;
- dangling source and target;
- undeclared read and write channels;
- unreachable nodes;
- self-deadlock, producer-count shortage, and mutual gate cycle;
- mixed-anchor and multi-start-anchor fan-in;
- empty bundle and duplicate trace IDs;
- metadata count mismatch;
- missing arrival offsets under `pace=arrival`;
- deterministic readiness waves and normalized topology order;
- non-static drivers omitting readiness waves without losing summary facts.

The structural validator also has focused behavioral tests for a valid chain and
exotic deadlocks (`rust/runtime/src/graph/validate.rs:614-675`).

### Adapter/load tests

`rust/runtime/tests/graph_inspection_load.rs` exercises the real built-in
resolver and local tokenizer once for each supported format using these fixtures:

- `tests/fixtures/dag/small.dag.jsonl`;
- `tests/fixtures/weka_traces/simple.json`;
- `tests/fixtures/graph_inspection/dynamo-trace.jsonl`;
- `tests/fixtures/graph_inspection/aiperf-trace.json`;
- `rust/e2e-tests/tests/fixtures/conditional/conditional_shopping.yaml`, already
  used by the real conditional Graph-IR E2E
  (`rust/e2e-tests/tests/test_conditional_graph.rs:15-18`);
- existing `rust/runtime/tests/fixtures/recorded_agent_replay/` inputs;
- existing OTLP GenAI adapter fixtures.

The integration test asserts that the seven-format inventory, resolver
selection, and `bundle.metadata.format` agree, including `aiperf_trace`.

### CLI process tests

`rust/cli/tests/graph_tools.rs` invokes `env!("CARGO_BIN_EXE_aiperf")` in a
Python-free environment and verifies:

- namespace help and exit 2;
- valid, invalid, and fatal exit codes;
- no Python delegation in a Python-free environment;
- clean JSON output for success, validation failure, and fatal failure;
- text summary and issue formatting;
- first-trace default and exact `--trace` selection;
- missing trace with available IDs;
- deterministic Mermaid and Markdown goldens;
- validation blocking and `--no-validate` rendering;
- output-file success, overwrite, directory rejection, and stdout silence.

The CLI suite uses native graph-tool fixtures and deterministic Mermaid/Markdown
goldens. It does not import Python renderer goldens because the native output
represents post-lowering Graph-IR and intentionally has a narrower node
vocabulary.

### Verification commands

Run the following focused runtime and CLI suites plus formatting/static checks
when validating this implementation:

```bash
source .venv/bin/activate
cd rust
cargo test -p aiperf-runtime --features engine graph::inspect
cargo test -p aiperf-runtime --features engine graph::validate
cargo test -p aiperf-runtime --features engine graph::scheduler
cargo test -p aiperf-runtime --features engine --test graph_inspection_load
cargo test -p aiperf-cli --test graph_tools
cargo fmt --check
cargo clippy -p aiperf-runtime -p aiperf-cli --all-targets
```

The implementation also updates the synchronized agent files, `llms.txt`, and
the spec index with the native CLI behavior. Repository documentation checks
cover those synchronized artifacts.

## Delivery

The shared inventory/preparation seam, native route, detailed inspection API,
versioned validate/explain reports, deterministic Mermaid/Markdown renderer,
atomic output behavior, process goldens, and product documentation are all
delivered. The graph namespace owns only these three subcommands; the remaining
Python graph namespace continues to delegate unchanged.

## Explicit exclusions from the first release

- `--graph-format auto` or source-content sniffing.
- stdin (`<PATH> = -`).
- Config-v2 `--config` extraction or a new graph-tool config file.
- Public dataset names, URLs, Hugging Face sources, or remote tokenizer download.
- Passing arbitrary adapter `options`, `synthesis`, replay-root, or tool-execution
  settings through the CLI shorthand.
- Authored conditional topology, branch alternatives, and subgraph expansion.
- Python placement, wait-distribution, expected-token, terminal-for-user,
  directive, endpoint, and authored-provenance diagnostics.
- Full prompt, initial-state value, replay output, segment, tool command, or
  environment-content output.
- HTML or interactive visualization.
- Live watch mode or opening a browser.
- Executing Graph-IR, model calls, tool calls, Docker, simulation, or dry-run
  metrics.
- A stable serialization of internal `GraphTraceProgram`; only the versioned
  inspection report schemas are public.

## Resolved product decisions

1. **`aiperf_trace` classification.** It is included in the shared seven-format
   inventory used by workload classification, resolver composition, and graph help.
2. **Resolver scope.** The first public command uses the stock built-in resolver.
   Extension-provided adapters are not part of this command's advertised surface.
3. **Advanced configuration.** The public input is the narrowly typed local-path
   shorthand; it does not accept Config-v2 extraction or arbitrary adapter options.
4. **Explain volume.** Explain reports every retained program; no filtering
   capability changes that all-program behavior.
5. **Output portability.** Unix replaces a regular output file atomically. Other
   platforms reject an existing output instead of deleting it first.
6. **Schema publication.** The report schemas are documented in this record;
   generated JSON Schema files are not part of the public contract.

## Source anchors

- `rust/cli/src/dispatch.rs:11-35` — native top-level routing and Python fallback.
- `rust/cli/src/main.rs:37-54` — process-level dispatch, unstructured error
  fallback, and exit.
- `rust/cli/src/config/mod.rs:12-80` — established native Clap subcommand pattern.
- `rust/runtime/src/engine/application.rs:51-95` — frozen application composition
  and injected graph-input resolver.
- `rust/runtime/src/engine/graph_input.rs:193-303` — adapter and resolver
  contracts, explicit discriminator selection, and one-call loading.
- `rust/runtime/src/engine/graph_input.rs:244-260` — built-in adapter composition.
- `rust/runtime/src/engine/online_execution.rs:1214-1257` — production tokenizer
  resolution and one-pass direct Graph-IR preparation.
- `rust/runtime/src/graph/input.rs:26-74` — graph input configuration, metadata,
  warning facts, and `GraphInputBundle`.
- `rust/runtime/src/graph/model.rs:206-415` — executable node, graph, trace plan,
  and trace program contracts.
- `rust/runtime/src/graph/validate.rs:27-169` — built structural validation.
- `rust/runtime/src/graph/inspect.rs` — lower-once, filesystem-free inspection,
  typed findings, normalized topology, and readiness analysis.
- `rust/runtime/src/graph/scheduler.rs:24-163` — adjacency, entry/successor seams,
  and start-anchor fan-in refusal.
- `rust/runtime/src/graph/conditional/mod.rs:55-120` — per-trace conditional
  resolution and flat Graph-IR lowering.
- `ajc/dag-v3:src/aiperf/cli_commands/graph_validate.py:21-80` — Python validate
  UX and exit behavior reference.
- `ajc/dag-v3:src/aiperf/cli_commands/graph_explain.py:32-99` and `:395-505` —
  Python explain output and illustrative schedule reference.
- `ajc/dag-v3:src/aiperf/cli_commands/graph_visualize.py:41-213` and `:216-247` —
  Python visualize CLI, validation, trace selection, and Markdown contract
  reference.
