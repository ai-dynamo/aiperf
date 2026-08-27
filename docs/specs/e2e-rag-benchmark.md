<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native end-to-end RAG benchmark

## Purpose

Define the native Rust workloads, seams, and artifacts that let AIPerf measure a
complete Retrieval-Augmented Generation deployment end to end, in the shape
MLCommons published for MLPerf Inference `e2e-rag-db` / `e2e-rag-qna`
(2026-08-26): one **ingestion** pipeline that turns a document corpus into a
vector database, and one **QnA** pipeline that answers multi-hop queries by
looping retrieve-and-reason steps across several models of different roles and
sizes until the accumulated evidence is sufficient.

The scored quantity is not a token rate. Ingestion reports **documents per
second**; QnA reports **tasks per second**, where one task is one query answered
end to end across a variable number of hops and a dozen or more model calls
spread over at least four distinct served endpoints.

This record fixes AIPerf's position in that system. AIPerf is a load generator
and measurement front end: it owns orchestration, HTML parsing, chunking, the
vector index, the hop control loop, measurement, and scoring. It never performs
model inference in-process. Every embedding, rerank, grade, sufficiency,
rewrite, answer, and judge call leaves the process as a request to a served
endpoint over the existing HTTP/gRPC transports, so the system under test is the
serving stack — exactly the surface a submitter optimizes through model
placement, co-residency, cross-stage scheduling, precision selection, and prefix
caching.

## Built

This section is scoped to what the code does today, verified against source. It
separates substrate that composes as-is from substrate that is present but does
not reach the RAG use case — the latter is prerequisite work, not free reuse.

### Per-node endpoint profile routing — works at dispatch

Protocol-v2 carries an id'd list of endpoint profiles with a `"default"` entry,
decoded strictly and validated once at bootstrap (`EndpointProfileConfigV2` /
`ValidatedEndpointProfileV2`, `rust/runtime/src/engine/registry.rs:1329`). Each
profile independently selects endpoint kind, URLs, path, streaming, timeouts,
headers, API key, TLS, UDS, proxy, and connection limit.

Graph execution binds a per-node endpoint profile end to end. An `LlmNode`
carries `metadata["endpoint"] = "<profile id>"`
(`rust/runtime/src/graph/lowering.rs:612`), references are validated before any
dispatch (`validate_graph_endpoint_profile_references`,
`rust/runtime/src/engine/online_execution.rs:1868`, called from `:1358`), the
selector is read at dispatch (`rust/runtime/src/engine/graph_execution.rs:2224`)
and resolved against the prepared profile map with a hard error on an unknown id
(`:1030`-`:1043`), and each profile gets its own dispatcher carrying its own URLs
and HTTP client (`:500`-`:509`). A per-node `model` override reaches the wire body
(`:2154`-`:2156` → `:1054`-`:1065`). Cellular carries `endpoint_profiles` verbatim
into each cell envelope (`rust/runtime/src/engine/cellular_controller.rs:1999`),
and each cell re-enters the same validated lowering.

**Consequence: routing distinct nodes of one measured trace to distinct served
endpoints is a real, exercised capability.** It is the load-bearing seam this
design stands on, and it works.

### The same seam's four gaps

Each of these is a prerequisite, sized in `## Design`.

1. **No authoring surface.** `rust/runtime/src/config/resolve.rs:1682` hardcodes
   `endpoint_profiles: Map::new()`, and `rust/cli/src/yaml.rs` has no
   corresponding section — with `deny_unknown_fields` on every section, authoring
   one in YAML is a hard parse error, not a silent ignore. The only reachable
   surfaces are the protocol-v2 stdio envelope and the eval NativeGraph
   `model-bindings.toml`. `docs/dev/python-rust-parity-gaps.md:514` already
   records this as a missing feature.
2. **One tokenizer for every profile.** On the benchmark path
   `PreparedRunnerGraphEndpointRuntimeFactory::new` clones a single
   `input_token_counter` into every profile entry
   (`rust/runtime/src/engine/graph_execution.rs:353`-`:379`); only the eval path's
   `new_with_profile_bindings` (`:383`-`:429`) binds them independently. Role-level
   ISL across models with different tokenizers is therefore not trustworthy today.
3. **One dialect for every node.** Graph lowering passes a single `endpoint_id`
   — the default profile's — into the input adapter
   (`rust/runtime/src/engine/online_execution.rs:1320`-`:1332`, consumed at
   `rust/runtime/src/engine/graph_input.rs:944`). A node bound to a
   different-dialect profile is still lowered against the default's dialect.
4. **gRPC forbids divergence outright**:
   `rust/runtime/src/engine/grpc_execution.rs:130` asserts every profile shares the
   default's URL list. Multi-endpoint RAG is HTTP-only.

No test in the tree stands up two servers and asserts that node A hit server 1
while node B hit server 2. `rust/cli/tests/online_v2_stdio.rs:544`-`:663`
configures a profile literally named `judge`, but both profiles point at the same
URL and no node selects it.

### Embedding and rerank endpoints exist — but not on the graph path

The endpoint kinds are registered members of the open interner: `embeddings`,
`chat_embeddings`, `nim_embeddings`, `cohere_rankings`, `hf_tei_rankings`,
`nim_rankings` (`rust/runtime/src/endpoints/type_id.rs:16`-`:36`), with KServe v2
variants (`rust/runtime/src/endpoints/kserve.rs:649`). `aiperf-mock-server` serves
all of them deterministically (`rust/mock-server/src/handlers.rs:2445`, `:2559`,
`:2622`, `:2644`), with `generate_embedding` (`:2500`) and `compute_mock_score`
(`:2509`) as pure BLAKE2s functions of the request text.

**They do not currently work as graph nodes, in either direction.** This was the
central error in an earlier draft of this record and it is corrected here.

*Request side.* `rust/runtime/src/engine/graph_execution.rs:2204`-`:2221` builds
the `Turn` from `raw_messages`/`lowered` plus `..Turn::default()`, so `Turn.texts`
is always empty for a graph node. Every non-chat endpoint reads only `Turn.texts`:
`EmbeddingsEndpoint::format_prepared_payload`
(`rust/runtime/src/endpoints/implementation.rs:1090`-`:1094`) emits
`{"input": []}` — a silently empty request — and `format_rankings`, shared by all
three rankings kinds, hard-errors before any HTTP request
(`rust/runtime/src/endpoints/tier2.rs:367`-`:370`).
`rust/runtime/src/graph/materialize.rs:134`-`:161` can only produce
`{"role", "content"}` message objects; no `PromptItem` variant and no
`MaterializedGraphRequest` field can carry named texts. The sole exception is
`chat_embeddings`, which delegates to `ChatEndpoint::format_prepared_payload` and
declares `splices_lowered_wires() = true`
(`rust/runtime/src/endpoints/implementation.rs:1114`-`:1127`).

*Reply side.* `rust/runtime/src/transport/reduce.rs:206`-`:211` explicitly no-ops
for `ResponseData::Embeddings` and `ResponseData::Rankings`;
`rust/runtime/src/endpoints/models.rs:246`-`:277` returns `""` from `get_text()`
for both. `graph_execution.rs:2409` therefore calls `GraphReply::from_text("")`,
which yields `message: None` (`rust/runtime/src/graph/sink.rs:96`-`:103`) and
writes `ChanVal::encoded_messages(vec![])` or `Value::Null` into the channel
(`rust/runtime/src/graph/executor.rs:478`-`:510`). The vector *is* parsed
(`implementation.rs:2137`-`:2200`) and then discarded. `ChannelType` is
`Text | Messages` only (`rust/runtime/src/graph/model.rs:22`-`:28`), and splicing a
JSON value is a silent no-op (`materialize.rs:129`).

Nothing detects the mismatch: `splices_lowered_wires()` is consulted only on the
scheduled path (`rust/runtime/src/dataset/request.rs:1610`), never on the graph
path, and the only endpoint-kind gate in graph execution is
`!descriptor.requires_raw_token_ids` (`graph_execution.rs:395`), which
embeddings and rankings both pass. There are zero tests driving a graph node
against a non-chat endpoint.

**Consequence: a graph node bound to an embedding or rerank profile starts,
dispatches, and benchmarks an empty request against a discarded reply.** Making
embed and rerank first-class graph steps is prerequisite work, not composition.

### Bounded live-output staging — the mechanism works, its host does not

The flat Graph-IR core forbids branch-on-live-output
(`docs/specs/conditional-graph-lowering.md`): pinned, recorded, and weighted
branches resolve eagerly at lowering.

The escape hatch is the staged trace-program driver
(`rust/runtime/src/graph/driver.rs:511`). A `TraceProgramDriver` alternates
`next_stage()` → `TraceStageDirective::Execute(GraphTracePlan)` with
`observe_stage(TraceStageResult)`. Three properties are confirmed against the
code:

- `TraceStageResult.channels` is the **full post-reduction snapshot of every
  declared channel**, carrying real content rather than opaque handles
  (`rust/runtime/src/engine/graph_execution.rs:1871`-`:1910` ←
  `rust/runtime/src/graph/executor.rs:204`-`:209` ←
  `rust/runtime/src/graph/channel_store.rs:338`-`:350`; the reply becomes a value at
  `executor.rs:474`-`:497`).
- `terminal_output_channels()` is honored by placement
  (`graph_execution.rs:1876`-`:1885` → `freeze_terminal_outputs` `:1748`-`:1797`).
  It governs handle freezing only; `channels` is handed over in full regardless.
  The returned slice must be sorted — the existing consumer uses `binary_search`
  (`rust/runtime/src/eval/native_graph/live_driver.rs:487`).
- `stage_bound()` is enforced independently by placement
  (`graph_execution.rs:1820`-`:1822` reads it and hard-errors if a staged driver
  omits it; `:1861`-`:1867` refuses `Execute` past the bound). A lying driver
  cannot spin.

Two sharp edges the driver must handle. Channel values arrive as **chat message
envelopes**, so a model replying `{"sufficient": true}` is seen as
`{"role":"assistant","content":"{\"sufficient\": true}"}` — or an array of those
for a messages-typed channel. And never-written channels arrive as
`{"$unset": true}` (`rust/runtime/src/graph/reducers.rs:47`-`:58`), which the
existing driver propagates unfiltered into the next stage's `initial_state`
(`live_driver.rs:479`, `:983`-`:988` → `channel_store.rs:114`-`:125`), where it is
seeded as a real write at seq 0.

**The blocking constraint is the host, not the mechanism.** The benchmark phase
runtime cannot run *any* driver-bearing program:
`rust/runtime/src/engine/graph_phase_runtime.rs:2247`-`:2318` accepts `all_static`
or `all_recorded_replay` and otherwise bails. The one live driver in the tree runs
only through the one-shot eval path `execute_native_graph_trace`
(`graph_execution.rs:903`-`:991`), which asserts `completed_traces == 1`. A
tasks-per-second RAG benchmark needs a third `GraphTraceProgram` family in the
phase runtime — with its own t\*, warmup-handoff, and cache-pressure decisions,
all of which exist today only for static plans. This is the single largest piece
of work in this design.

The driver-kind registry is likewise **open as a data structure, closed as a
product surface**: `RegisteredTraceProgramDrivers::stock()`
(`rust/runtime/src/engine/execution_factories.rs:44`-`:70`) is a hardcoded
three-entry registry with a private field and no `register` method, and
`with_trace_driver_factory` (`:172`-`:179`) has zero callers anywhere in the repo.
No config or CLI can select a driver kind; `TraceDriverSpec::kind` is set only by
lowering, and the authority that makes a live driver work
(`with_source_provenance`, `driver.rs:163`-`:176`) is `pub(crate)` and
serde-skipped, so a wire-supplied kind cannot smuggle one in.

Finally, there is **no partial-failure path**. Placement hardcodes
`terminal_status: Completed` (`graph_execution.rs:1906`), and any node failure
aborts the whole trace before the driver observes anything. A driver cannot drop
one failed grader and continue.

### The existing loop construct

`NativeGraphControlContract` already carries a declarative bounded loop steered by
live model output: `selector_node`, `selector_channel`, `continue_match`,
`retry_match`, `max_iterations`, `members`, `entry`/`backedge`/`exit` (applied by
`apply_loop_decisions`, `live_driver.rs:1145`-`:1223`; branches by
`apply_model_decisions`, `:1014`-`:1063`). Its selector reads the channel value's
`.content` string and compares it to declared literals
(`selector_value`, `:1278`-`:1295`); an undeclared string is a hard error, not a
fallback. There is no JSON-pointer or predicate decoder anywhere in the tree.

Critically, the existing driver does **not** author fresh graphs. Each stage is a
*projection* of the whole lowered source graph: ready nodes are selected
(`:895`-`:923`), a plan is rebuilt over just those nodes with synthetic
`START → node → END` edges (`:931`-`:985`), and the projection is re-validated
byte-identically against the source (`:987`-`:1010`), then re-validated again
independently by placement (`graph_execution.rs:1868`).

### The rest of the substrate

Composes unchanged: content-addressed BLAKE3 segment identity
(`rust/runtime/src/graph/segment.rs`), trace-level cellular ownership
(`docs/specs/cellular.md`), phase orchestration
(`rust/runtime/src/phase_runtime.rs`), the record/aggregate/derived metric planes,
the exporter plane, and `--steady-state`.

Directly reusable for this design, and not to be rebuilt:

- **Batched requests.** `--batch-size` already ships N independent texts in one
  request (`rust/runtime/src/endpoints/implementation.rs:1080`-`:1094`;
  `rust/cli/src/flags.rs:724`). One batched request is **one record**, with
  `input_sequence_length` the batch sum — proven by
  `rust/dry-run-tests/tests/random_pool_batches.rs:11`-`:66`.
- **Corpus chunking.** `rust/runtime/src/dataset/corpus.rs` `build_corpus_chunks`
  is character-bounded specifically so token boundaries stay machine-stable — the
  same reproducibility contract the 768-char/32-overlap MLPerf chunking needs.
- **Cache-aware corpus fetch.** `rust/runtime/src/dataset/fetch.rs` (BLAKE3-keyed
  on-disk cache over AIPerf's own transport) plus `hf_hub.rs` and the
  `public_loader!` registration in `rust/runtime/src/dataset/loader/public.rs`.
- **Pre-dispatch worker CPU.** `--dispatch global-push` already runs meaningful
  per-request materialization inline on the worker before dispatch
  (`rust/runtime/src/engine/turn_execution.rs:2035`-`:2055`), and the latency origin
  is `on_admit` inside the HTTP sink
  (`rust/runtime/src/transport/http/sink/endpoint_dispatch.rs:289`), which fires
  *after* it. Client-side work is already excluded from request latency for free.
- **`ParsedResponse.sources`.** An existing, plumbed, untyped carrier for
  retrieved-document identities (`rust/runtime/src/endpoints/tier2.rs:1128`),
  introduced by `solido_rag` — which is itself a chat-shaped dialect with
  server-side opaque retrieval, not a RAG capability. Nothing consumes `sources`
  for metrics today.
- **Large binary artifacts.** A registered `Exporter` writing into the run's
  `artifact_dir` (`rust/runtime/src/export/mod.rs:310`, `:397`), with
  `ParquetExporter` as the existing multi-hundred-megabyte precedent and no size
  or quota limit. `ResourceRequirementsV2 { artifacts }`
  (`rust/runtime/src/engine/registry.rs:107`-`:122`) already lets a workload declare
  an artifact `Required`.

Dependency inventory, verified against `rust/Cargo.toml` and `rust/Cargo.lock`:
there is **no HTML parser** (`html5ever`, `scraper`, `lol_html`, `tl`, `select`
all absent), no `memmap2`, no ANN crate, and no linear-algebra crate — `ndarray`
appears only transitively behind the optional Dynamo dependency and is not
dependable. `rayon`, `regex`, `blake3`, `parquet`/`arrow`, `hf-hub`, and
`tokenizers` are available.

## Design

### Prerequisites

Four pieces of work gate everything below. They are stated first because
sequencing the RAG workloads ahead of them produces code that cannot be exercised
through the product surface.

**P1 — structured request and reply for embed and rerank on the graph path.**
The narrowest correct fix is a structured channel path: extend `ChannelType` and
`ChanVal` to carry a typed non-text payload, populate `Turn.texts` from a new
`PromptItem`/`MaterializedGraphRequest` field so named texts (`query`,
`passages`, `input`) can be authored, and stop discarding
`ResponseData::{Embeddings, Rankings}` in `reduce.rs` so the parsed vector or
ranking reaches the channel. Until then, embed and rerank nodes are not
expressible.

*Immediate partial mitigation, worth landing first on its own:* add a
`splices_lowered_wires()` / service-kind refusal beside the
`requires_raw_token_ids` check at `graph_execution.rs:393`-`:400`, so a graph node
bound to an incompatible endpoint kind fails at startup instead of silently
benchmarking empty requests. This is a bug fix independent of RAG.

**P2 — a staged `GraphTraceProgram` family in the benchmark phase runtime.**
`prepare_graph_phase` must accept driver-bearing programs alongside `all_static`
and `all_recorded_replay`, with defined t\*, warmup-handoff, and cache-pressure
behavior for staged traces, and without the eval path's `completed_traces == 1`
assertion. Nothing multi-hop can be benchmarked before this exists.

**P3 — an `endpointProfiles:` authoring surface.** A Config-v2 YAML section
projecting into `BenchmarkConfig.endpoint_profiles`, replacing the hardcoded empty
map at `resolve.rs:1682`, with one e2e test standing up two mock servers on
different ports and asserting the per-node request split. Without it the whole
design is reachable only through the internal stdio envelope.

**P4 — per-profile input token counters.** Bind counters independently on the
benchmark path as the eval path already does
(`graph_execution.rs:383`-`:429`), so role-level ISL is trustworthy across models
with different tokenizers.

### Two workloads, two very different costs

| Workload | Shape | Unit of work | Scored metric |
|---|---|---|---|
| `rag_ingest` | `scheduled` run over a new dataset format `rag_corpus` | one source document | `rag_documents_per_second` |
| `rag_qna` | new graph format `rag_qna` + staged driver | one query task | `rag_tasks_per_second` |

`rag_ingest` is deliberately **not** a new workload kind. There are two seams
called "workload" in this runtime: the registry-level `WorkloadFactory`
(`rust/runtime/src/engine/registry.rs:342`) and the inner per-phase `Workload`
trait (`rust/runtime/src/scheduled.rs:1434`, selected by a `match` on `PhaseSpec`
at `rust/runtime/src/engine/execute/dataset_build.rs:184`). Registering a new
inner workload means extending seven exhaustive `match phase` sites; registering
a new factory means a descriptor, a requirements matrix, a `WorkloadKind` variant,
and its own lowering — and the one in-tree attempt at that path,
`static_accuracy`, was written and then never wired into the stock composition
(`register_http_static_accuracy_workload`,
`rust/runtime/src/engine/online_execution.rs:277`, has no non-test call site).

Ingestion needs none of it. It is an ordinary request-bounded scheduled run whose
dataset loader emits passage batches, dispatched through the existing
`--batch-size` path. That reduces it to a `DatasetLoader` + `Composer` pair
registered in `register_builtin_formats`
(`rust/runtime/src/dataset/loader/mod.rs:379`), which the `hf` format established
in four commits and roughly 460 lines (`205a8212d4`, `77bacfdef2`, `8cacaa2ae9`,
`d8fc7e9d7c`).

`rag_qna` genuinely is a graph format, and adding one is not free: it obliges an
inventory entry (`BUILTIN_GRAPH_FORMATS`,
`rust/runtime/src/config/model/workload_kind.rs:28` — eight entries today, so
`rag_qna` is the **ninth**), a resolver adapter whose set is assert-checked
against that inventory (`rust/runtime/src/engine/graph_input.rs:296`-`:306`,
`:345`-`:352`, hard `assert_eq!` at `:2177`), the `[Arc<dyn GraphInputAdapter>; 8]`
array length at `:286`, and the `aiperf graph validate`/`explain`/`visualize`
projections. Inspection itself is inventory-driven and needs no per-format code
(`rust/cli/src/graph/mod.rs:176`, `:294`, `:380`), but four enumeration tests
break on any addition. Two pre-existing inventory bugs should be fixed in the same
change: `rust/runtime/src/engine/cellular_controller.rs:2452` omits `otlp_genai`
and `rust/runtime/src/engine/cellular_kind.rs:21` omits `aiperf_trace`; both hand-
written lists should be driven off `is_builtin_graph_format()`.

### One corpus identity

The two runs are independent. `rag_ingest` writes a **vector database artifact**;
`rag_qna` declares it `Required` and refuses to start without it. The artifact
carries a `corpus_digest` — BLAKE3 over the canonical normalized corpus projection
(sorted document identities, each binding its bytes; then the frozen chunker
parameters and the embedding profile's model identity) — and the QnA run records
that digest in its report, so two submissions are only comparable when they
answered from the same corpus under the same chunking. The digest and manifest
shape follows `rust/runtime/src/eval/native_graph/artifacts.rs:62`-`:83`
(`FrozenArtifact { digest, length }`, `FrozenArtifactManifest`, digest-mismatch
refusal) without inheriting that module's episode-scoped quota model.

### Ingestion: parse → chunk → embed → index

The client-side stages are pure functions run on the worker before dispatch,
following the `CreditMaterializer` precedent:

- **Parse.** Extract the article body from HTML, dropping navigation, reference,
  and metadata subtrees; flatten tables and lists to text row by row rather than
  dropping them. This is the one new third-party dependency in this design, and
  the inventory above confirms nothing suitable is present in either language.
  Parsing is deterministic and content-addressed, so a parsed-corpus cache makes
  re-runs skip it.
- **Chunk.** Slice to `chunk_chars` (default 768) with `chunk_overlap_chars`
  (default 32) on UTF-8-respecting character boundaries, tagging each passage with
  its source document identity and byte offsets. This extends
  `dataset/corpus.rs`'s existing character-bounded chunker rather than replacing
  it. Defaults match the MLPerf reference; both are authorable and both
  participate in `corpus_digest`.
- **Embed.** Dispatch passage batches to the `embed` profile through the existing
  batch path. Batch size is authorable because macro-batching the embedder is one
  of the optimization levers the benchmark exposes.
- **Index.** Append returned vectors to the builder, then seal through a
  registered `Exporter` into the run's `artifact_dir`.

**`rag_documents_per_second` is not derivable from the record plane as it
stands.** One batched request is one record, a document spans several batches, and
a batch spans several documents. It requires a document-completion notion carried
on the record — a new catalog tag plus record-plane plumbing, not a derived
aggregate over existing columns. The spec's earlier framing of this as "a derived
aggregate" was wrong.

Parse and chunk time are reported as named client-side stage timings. The half of
that claim that says they are *not folded into request latency* is already true
for free (`on_admit` fires after materialization). The half that says they are
*measured* is new machinery: a timer around the worker-side stages, new catalog
tags, record-plane fields, and exporter projections. `CreditToStartLatency`
(`rust/runtime/src/metrics_core/store.rs:1675`) contains materialization time but
also queueing, so it cannot attribute a slow parser.

### The vector index seam

A new registry category, `VectorIndex`, with a two-trait split so the read path is
`!Send` worker-local and the write path is owned by the ingestion sealer:

```rust
pub trait VectorIndexBuilder {
    fn add(&mut self, passages: &[PassageVector]) -> Result<(), VectorIndexError>;
    fn seal(self: Box<Self>, sink: &mut dyn ArtifactWrite) -> Result<VectorIndexManifest, VectorIndexError>;
}

pub trait VectorIndex {
    fn dim(&self) -> usize;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<PassageHit>, VectorIndexError>;
}
```

Three registered implementations:

- `flat` (**default**): exact inner-product/cosine scan over an `f32` matrix. For
  the MLPerf corpus shape (~107k passages × 768 dims ≈ 330 MB) an exact scan is
  ~82 MFLOP per query — single-digit milliseconds with a chunked, auto-vectorized
  kernel — and it is *exact*, which removes approximate-search recall from the
  list of things that can silently differ between two submissions. The kernel is
  hand-written `f32` over slices; no linear-algebra dependency is available or
  needed. `rayon` is available if the scan needs parallelism. Memory-mapping
  requires adding `memmap2`; a plain read is the fallback if that dependency is
  refused.
- `hnsw`: approximate graph index for corpora where the exact scan stops being
  negligible against the model calls. Parameters are recorded in the manifest and
  the report; a run using it is flagged approximate. This needs a new dependency
  or a hand-rolled implementation and is explicitly last in the delivery order.
- `http`: retrieval delegated to an external vector database over the existing
  HTTP transport. This is the path where the vector DB *is* part of the system
  under test, and its latency lands on the measured request timeline like any
  other endpoint.

The sealed artifact is `header || vectors || passage table || manifest`, where the
header pins `{dim, count, metric, chunker params, embed model identity,
corpus_digest}`. Under `--cells N` each cell process reads the same file on a
same-host run; a cross-host cell lands the artifact privately before execution
through the existing pinned-TLS artifact channel and refuses to execute on a
digest mismatch.

### Retrieval as a graph node

Retrieval is not a model call, but it is a scheduled step with real latency that
can run in parallel across sub-queries within a hop, and under the `http` index it
is a genuine wire request. Expressing it as driver-side work between stages would
hide it from the scheduler, from per-node records, and from concurrency
accounting. `ExecutableGraphNode` therefore gains a third variant, `Retrieval`,
reading a query-vector channel and writing a passage-set channel.

Modelling retrieval as a `Tool` node instead is not viable: tool nodes force the
driver to supply a `ToolDispatcher` — pulling in trace-local dispatcher lifecycle
this workload otherwise does not need — and are explicitly rejected by the
non-driver static path (`rust/runtime/src/graph/execution.rs:245`-`:255`) and by
`static_graph_plans` (`graph_phase_runtime.rs:2189`-`:2197`).

**The variant is additive at the wire boundary but expensive in the tree, and
this record does not soften that.** `Serialize` is `#[serde(tag = "kind")]` with a
hand-written `Deserialize` carrying an explicit unknown-kind rejection
(`rust/runtime/src/graph/model.rs:249`-`:286`), so every existing document decodes
byte-identically and new-writer/old-reader fails closed. `GraphRecord` is not on
the cellular wire — only `GraphCellSupplement` crosses — so there is no cellular
DTO change. And `is_flat_graph` fails closed by construction: it matches a single
`ExecutableGraphNode::Llm` positionally and returns `false` for anything else
(`rust/runtime/src/graph/flat.rs:48`-`:66`).

But the `ToolNode` precedent — a **non-executing** node kind, rejected at every
execution boundary — cost 37 files and roughly +1400/−480 lines (`33780ead7f`)
plus eight follow-ups, two of them cancellation-semantics bugs found after the
fact. A node that actually dispatches costs strictly more. Around ten exhaustive
matches break the build, which is the good case. Thirteen sites would silently
mishandle a new variant, of which these must be decided deliberately:

- `rust/runtime/src/graph/snapshot.rs:36`-`:40` — `has_tool_node` is the *only*
  guard on `chop_trie_at_tstar`/`rewrite_for_warmup`, and `:390`-`:421` silently
  **deletes** non-`Llm` nodes during warmup rewrite. Highest severity, and it
  forces an unmade semantic choice: what does a t\* snapshot of a retrieval node
  mean?
- `rust/runtime/src/graph/model.rs:380`-`:386` — `llm_node_count` drives credits
  and budgets across twelve consumers, and `rust/runtime/src/graph/inspect.rs:440`
  computes `tool_node_count` as *total minus llm*, so a retrieval node would be
  **reported as a tool**.
- `rust/runtime/src/engine/graph_execution.rs:1556` (worker node index) and
  `:2579` (`terminal_graph_nodes`) would **disagree** with each other.
- `rust/runtime/src/graph/validate.rs:478`-`:490` gates the splice check behind
  `as_llm`, turning a missing declaration into a deadlock instead of a finding.
- `rust/runtime/src/graph/execution.rs:245` is written as a denylist
  (`matches!(Tool(_))`) rather than an allowlist.

Three pieces of de-risking pre-work make the variant tractable and are worth
landing independently: flip the `execution.rs` reject gate to allowlist form,
replace `has_tool_node` with `!matches!(Llm(_))`, and make the llm/tool counts
explicit rather than derived by subtraction. `inspect.rs`'s
`GraphNodeInspection` also carries only LLM-shaped optional fields, so retrieval
attributes (index binding, top-k) require widening that CLI DTO.

### QnA: `rag_qna`

One trace is one query task; one stage is one hop. The per-hop node set:

| Node | Kind | Endpoint profile | Reference model |
|---|---|---|---|
| `rewrite` | `Llm` | `rewriter` | large reasoning model |
| `embed_subquery[i]` | `Llm` | `embed` | embedding model |
| `retrieve[i]` | `Retrieval` | index-bound | — |
| `rerank` | `Llm` | `rerank` | late-interaction reranker |
| `grade[j]` | `Llm` | `grader` | small model, high volume |
| `sufficient` | `Llm` | `sufficiency` | large reasoning model |

with `answer` (profile `answer`) emitted in a final stage when the driver observes
a sufficient verdict or reaches its hop bound. The `rewriter`, `sufficiency`, and
`answer` profiles may name the same served model — that they are *separate
profiles* is what lets a submitter place them differently. Fan-out counts are
authored, bounded, and validated before execution; `stage_bound()` is
`max_hops + 1` (default 5 + 1).

**Prefer the existing loop construct over a new driver kind.** If the sufficiency
verdict is expressible as a bounded literal match on the reply content, the
existing `loops` machinery in `NativeGraphControlContract` already implements a
bounded live-output-steered loop, and `rag_qna` needs only a non-eval lowering
path plus P2 — not a fourth driver kind. A new kind is justified only if the
verdict genuinely requires structured decoding. Either way the driver-kind
registry must be edited inside `aiperf-runtime`
(`execution_factories.rs:44`-`:70`); it is not externally extensible.

Whichever path is taken, the stage plan must be a **projection of an authored
source graph**, never synthesized from nothing: placement re-runs
`validate_native_graph_trace_plan`, which enforces every-node-reachable-from-START,
every-node-reaches-END, acyclicity, and full read/write channel declaration. The
driver must also filter `{"$unset": true}` out of carried channels, and emit a
`plan_identity` matching placement's `"{trace_id}::stage-{n}"` format
(`graph_execution.rs:1905`) or observation is rejected.

Verdict extraction is a declared, authored **verdict decoder** — a strict
JSON-pointer-plus-predicate rule, or a bounded literal match — never an ad-hoc
substring sniff. No such decoder exists in the tree; the only precedent stops at
exact string equality on `.content`. Because channel values are chat envelopes,
the decoder must unwrap array → last element → `.content` → parse → pointer →
predicate as declared steps, each with a typed error attributed to the sufficiency
node.

Two execution modes:

- **`live`** — verdicts and retrieved passages come from the run itself. This is
  the accuracy mode; hop count varies per task and per run.
- **`pinned`** — the performance mode. This record's earlier description of it was
  internally inconsistent and is corrected: pinned mode does **not** discard model
  outputs while still feeding them downstream. It replays a recorded plan in which
  *every node's request inputs are the recorded ones* — the recorded sub-query
  texts drive `embed_subquery`, the recorded passage sets drive `rerank` and
  `grade` — so each stage's input comes from the plan rather than from the
  previous stage's live output. Every model call still executes against the served
  endpoint and its latency and token counts are still measured; no live output
  steers control or content. This is the same semantics `recorded_replay` already
  has, extended to a staged program. A pinned plan is produced by a `live` run
  (`--rag-record-plan <path>`) and is bound to a `corpus_digest`.

  Pinning fixes hop count, retrieved sets, and request bytes; it does **not** fix
  generated output length, which remains stochastic. Reproducibility of
  tasks-per-second therefore rests on fixed work, not fixed outputs, and residual
  output-length variance is exactly what the compliance check below bounds. Where
  the endpoint supports it, the recorded `max_tokens`/`ignore_eos` are replayed to
  narrow that variance further.

### Live branching: the driver seam is a stepping stone

The staged driver lets the multi-hop loop ship without reactive machinery in the
flat core. It is not the end state.

The cost of staging is real and measured. A stage is a full barrier: within one
hop every grade must complete before the driver observes the verdict, even though
a production RAG system can begin the sufficiency check as soon as enough passages
are graded and can abandon in-flight grading once the answer is decided. Staging
reports a hop latency that is the maximum over its fan-out rather than the real
critical path, and cannot express early termination at all. Cross-*task*
concurrency is preserved — each trace is its own `spawn_local`
(`rust/runtime/src/graph/placement.rs:396`-`:401`) — so the barrier stalls only its
own task. Per-stage there is also a fresh executor, context, and channel store,
and `EngineGraphSink::configure_stage` clears the prepared-metadata cache every
stage (`graph_execution.rs:2448`-`:2459`), so per-node metadata is re-parsed on
every hop: six times the setup a static trace pays once.

For a benchmark whose whole subject is overlapping many concurrent tasks across
heterogeneous accelerators, understating the intra-task critical path understates
exactly the optimization the benchmark exists to reward.

The destination is **branch-on-live-output inside the graph**: a conditional edge
resolved at dispatch time from a channel value, with in-flight cancellation of
untaken arms. That is the machinery `docs/specs/conditional-graph-lowering.md`
deliberately omits, and its ban is correct for what the flat core is today. Lifting
it is a deliberate future change to that record, not a loophole in it.

The RAG QnA loop is the first product workload that genuinely wants it, so the
migration contract is stated now, while the shape is cheap to preserve:

- **The authored graph is the stable surface.** A `rag_qna` graph authors its hop
  body once, with the sufficiency verdict as a declared channel and a declared
  decoder. Today the driver reads that channel between stages; under live
  branching a conditional edge reads the same channel mid-stage. The authored
  document, the endpoint profiles, the node set, and the artifacts do not change.
- **The verdict decoder is the seam.** Because extraction is a declared strict
  rule rather than driver-internal logic, it moves to the edge resolver unchanged.
  This is the main reason it is specified declaratively.
- **The hop bound survives.** `stage_bound()` becomes a bounded-loop iteration cap
  on the edge. Boundedness is a property of the workload, not of the staging
  mechanism, and must remain enforced independently of whoever resolves the branch.
- **Determinism is preserved by `pinned`, not by staging.** `pinned` replays a
  recorded plan and is untouched by live branching, so the reproducible
  performance number keeps working across the migration. Only `live` mode's
  execution shape changes.
- **Cellular is the hard part, deferred honestly.** Eager resolution is what lets a
  compiled `GraphRecord` mean the same thing on every cell. A dispatch-time branch
  means a cell resolves control locally, so the fold has to carry which arms ran.
  Nothing in this record depends on solving that; the migration must.

The concrete test that the seam was kept honest: when live branching lands, a
`rag_qna` run should switch by changing the execution mode, not by rewriting the
graph, the decoder, the profiles, or the metrics.

### Metrics

New catalog tags, reported per-run and per-phase-window: `rag_task_latency`
(record: query admitted → final answer terminal), `rag_tasks_per_second`,
`rag_hops_per_task` (distribution, not just mean), `rag_task_llm_calls`;
`rag_documents_per_second`, `rag_passages_per_second`, `rag_parse_latency`,
`rag_chunk_latency`, `rag_index_latency`; `rag_retrieval_latency`,
`rag_retrieval_candidates`.

The catalog is closed and positional. `MetricTagId`
(`rust/runtime/src/metrics_core/tag_id.rs:12`) is generated by
`define_builtin_metric_tags!`, whose header states the governing contract: the
ordinal is embedded in MessagePack payloads crossing the cell/controller boundary,
and new built-ins append at the end only. `MetricTagRegistry::register` exists but
is a name interner — `catalog.rs:2034` indexes a `[MetricSpec; COUNT]` array
unchecked, so a registered non-builtin tag panics on first use. Each addition
touches `tag_id.rs` and `catalog.rs` at the same ordinal (out of sync, every
downstream tag silently re-points at a neighbour's unit and flags with no error),
plus the length and `catalog_fingerprint` constants, one insta snapshot, and eight
console goldens. A new `Record` metric also adds a column to `profile_export.csv`
and `.parquet` for **every** run, RAG or not.

**The role dimension is a label, not a tag.** An earlier draft conflated the two.
Roles are run configuration, not compile-time constants, and cannot be enumerated
in a positional `[MetricSpec; COUNT]` static. The correct seam already exists:
`MetricSeries.labels` (`rust/runtime/src/metrics_core/report.rs:140`-`:150`), fed
per dimension by `compute_inference_series`
(`rust/runtime/src/metrics_core/accumulator.rs:750`-`:828`) via
`InferenceDimensions { endpoint_url, model }`
(`rust/runtime/src/metrics_core/ingest.rs:21`-`:27`), stored as interned dimension
codes. Adding a third field and stamping it at ingest is around two days of engine
work.

Delivery is staged, because the exporter half is where the cost actually lives:

1. Thread `endpoint_profile_id` into `RecordIngest` and out to the per-record row.
   It is absent today — `RecordMetadata` (`rust/runtime/src/engine/records.rs:113`)
   carries no model, endpoint, profile, or node identity — even though the value
   exists at dispatch (`rust/runtime/src/graph/driver.rs:115`) and at run level
   (`report.rs:264`). Nullable, omitted for single-profile runs. This alone
   satisfies this record's own e2e checks by reading the JSONL.
2. Add the dimension and surface it as `labels["role"]` in the **native report JSON
   only**, which already has a labeled-series schema and is genuinely additive
   there.
3. Only then decide on console, genai_perf, and timeslice per-role columns.

Step 3 is gated on a **latent bug that should be filed independently of RAG**:
`report_inference_series` *replaces* the aggregate rather than adding to it
(`report.rs:1284`-`:1288`), and `summary_series` (`rust/runtime/src/export/mod.rs:80`-`:96`)
returns `NoAggregate` for multiple non-unique series — which means the metric is
**dropped** from genai_perf JSON/CSV (`export/genai_perf.rs:207`), the console
table (`console_txt.rs:568`), timeslices (`timeslice.rs:201`), MLflow
(`mlflow.rs:182`), and W&B (`wandb/mod.rs:381`). Any run with more than one model
or endpoint silently loses TTFT, ITL, latency, ISL, and OSL from five exporters
today. RAG would make that the common case.

Two honest limitations. "Existing artifacts byte-identical" holds only if the role
label is **omitted, not defaulted**, when there is one profile — which makes the
feature conditional rather than additive. And sketch mode drops the dimension
outright: `export_results_sketch` hardcodes empty inference series, and
`accumulator.rs:622`-`:643` already documents why. Per-role sketch statistics need a
new key dimension in `SketchColumns`, a MessagePack wire change, and one t-digest
per tag per role; it is out of scope here and documented as a limitation beside the
existing ones.

Offline scenario is the existing concurrency workload sized to the full task set.
Server scenario is out of scope for this record, exactly as for the first MLPerf
instantiation.

### Accuracy and compliance

Both are post-run passes over recorded artifacts, native, and off the timed path.

- **`aiperf rag score`** drives the `judge` endpoint profile over the recorded
  answers artifact and reports answer accuracy plus precision/recall/F1. The judge
  is deliberately a separate profile so it can name a model from a different family
  than the models it grades. It runs on the endpoint-profile wire, **not** through
  the accuracy evaluator plane: `EvaluatorGradeItem`
  (`rust/runtime/src/accuracy_core/protocol.rs:196`) carries only
  `{problem_id, response}` with no retrieved context, the evaluator worker is a
  stdio subprocess with no transport handle (so an in-worker judge would open a
  second, unmeasured HTTP path), grading is single-threaded at the join, and the
  static-accuracy workload is not registered in `Application::stock`
  (`docs/specs/accuracy.md:26`-`:34`). `EvaluatorGrade`'s
  `{correct, confidence, reasoning, extracted_answer}` remains the right *shape*
  for a verdict.
- **Retrieval integrity** compares each task's retrieved document identities
  against the ground-truth identities shipped with the query set and reports
  recall/precision/F1. It is a database-integrity check on the built index, not a
  scored metric, and is reported as such. `ParsedResponse.sources` is the existing
  carrier for those identities.
- **Output-length compliance** (`aiperf rag compliance`) re-reads a pinned run's
  records and asserts the answer-role mean OSL falls inside an authored band around
  an authored reference (the MLPerf analogue is 273.81 tokens ±10%). The check is a
  single aggregate over one role and the report states that scope explicitly rather
  than implying whole-pipeline coverage. It depends on step 1 of the role work,
  since attributing OSL to the answer role requires the per-record profile id.

Mock-server fixtures reuse the existing accuracy-fixture mechanism
(`rust/mock-server/src/accuracy.rs`, flags at `rust/mock-server/src/config.rs:753`-`:817`)
by adding a RAG format variant, rather than building a parallel canned-answer system.
Deterministic embeddings and rerank scores already exist as pure hash functions of the
request text, but the dimension is hardcoded at 768 and scores can only be *predicted*
by recomputing the hash, never *dictated*; a test that needs a specific ranking order
should add one score-by-index knob to `compute_mock_score` rather than fork it.

### Refusals

Typed refusals before any dispatch: a QnA run whose index `corpus_digest` disagrees
with its pinned plan or query set; a `retrieval` node in a non-RAG workload; a
`rag_qna` graph naming an unregistered endpoint profile; an authored hop, fan-out, or
`k` bound that is zero or unbounded; a pinned plan whose hop count exceeds
`stage_bound()`; an index artifact whose header dimension disagrees with the embed
profile's returned vector width; a sufficiency verdict decoder that is absent or
ambiguous; a graph node bound to an endpoint kind whose request cannot be
constructed from graph materialization (P1's startup gate); and a multi-endpoint
`rag_qna` run over gRPC, which the transport forbids.

## Future requirements

Everything in `## Design` is unbuilt. Delivery order, gates, and per-step
verification live in
`~/.aiperf/docs/superpowers/plans/2026-08-27-native-e2e-rag-benchmark.md`.

Branch-on-live-output inside the graph is explicitly **planned, not rejected**,
scoped as its own change against `docs/specs/conditional-graph-lowering.md`:
dispatch-time conditional-edge resolution, in-flight cancellation of untaken arms,
a bounded-loop iteration cap on the edge, and a cellular fold that carries resolved
control. This record's obligation to it is the migration contract above.

Explicitly deferred, and not designed here: Server (rate-paced) scenario scoring; a
knowledge-graph or otherwise structured index; table-aware parsing that keeps
figures intact through chunking; tool use in the answer step; per-component or
distribution-level compliance beyond the single answer-role aggregate; per-role
statistics under sketch mode; per-hop partial-failure resilience (placement has no
substrate for it); and cross-host sharding of a single index across cells.

## Source anchors

- Endpoint profiles and strict validation:
  `rust/runtime/src/engine/registry.rs:1329`; authoring gap
  `rust/runtime/src/config/resolve.rs:1682`.
- Per-node profile binding, resolution, and per-profile dispatchers:
  `rust/runtime/src/graph/lowering.rs:612`,
  `rust/runtime/src/engine/online_execution.rs:1868`,
  `rust/runtime/src/engine/graph_execution.rs:1030`-`:1043`, `:500`-`:509`, `:2224`.
- Shared token counter and default-profile dialect:
  `rust/runtime/src/engine/graph_execution.rs:353`-`:379`,
  `rust/runtime/src/engine/online_execution.rs:1320`-`:1332`.
- gRPC profile-divergence refusal: `rust/runtime/src/engine/grpc_execution.rs:130`.
- Embedding/rankings request construction and its `Turn.texts` dependency:
  `rust/runtime/src/engine/graph_execution.rs:2204`-`:2221`,
  `rust/runtime/src/endpoints/implementation.rs:1090`-`:1094`,
  `rust/runtime/src/endpoints/tier2.rs:357`-`:370`,
  `rust/runtime/src/graph/materialize.rs:134`-`:161`.
- Embedding/rankings reply discard:
  `rust/runtime/src/transport/reduce.rs:206`-`:211`,
  `rust/runtime/src/endpoints/models.rs:246`-`:277`,
  `rust/runtime/src/graph/executor.rs:478`-`:510`.
- Staged driver seam and its enforcement:
  `rust/runtime/src/graph/driver.rs:511`;
  `rust/runtime/src/engine/graph_execution.rs:1820`-`:1822`, `:1861`-`:1885`,
  `:1871`-`:1910`.
- Phase-runtime gate that blocks staged benchmarking:
  `rust/runtime/src/engine/graph_phase_runtime.rs:2247`-`:2318`;
  one-shot eval path `rust/runtime/src/engine/graph_execution.rs:903`-`:991`.
- Closed driver-kind registry:
  `rust/runtime/src/engine/execution_factories.rs:44`-`:70`, `:172`-`:179`.
- Live staged driver precedent, its loop construct, and stage projection:
  `rust/runtime/src/eval/native_graph/live_driver.rs:895`-`:1010`, `:1145`-`:1223`,
  `:1278`-`:1295`.
- `{"$unset": true}` propagation: `rust/runtime/src/graph/reducers.rs:47`-`:58`,
  `rust/runtime/src/eval/native_graph/live_driver.rs:479`, `:983`-`:988`.
- Graph node enum and its tagged decode:
  `rust/runtime/src/graph/model.rs:249`-`:286`; node-count derivation `:380`-`:386`;
  `rust/runtime/src/graph/inspect.rs:440`.
- Sites that silently mishandle a new node variant:
  `rust/runtime/src/graph/snapshot.rs:36`-`:40`, `:390`-`:421`;
  `rust/runtime/src/graph/validate.rs:478`-`:490`;
  `rust/runtime/src/graph/execution.rs:245`-`:255`;
  `rust/runtime/src/engine/graph_execution.rs:1556`, `:2579`.
- Flat fast-path eligibility (fails closed on a non-`Llm` node):
  `rust/runtime/src/graph/flat.rs:48`-`:66`.
- Graph-format inventory and its assert-enforced adapter agreement:
  `rust/runtime/src/config/model/workload_kind.rs:19`, `:28`;
  `rust/runtime/src/engine/graph_input.rs:286`, `:296`-`:306`, `:345`-`:352`.
  Inventory bugs to fix in the same change:
  `rust/runtime/src/engine/cellular_controller.rs:2452`,
  `rust/runtime/src/engine/cellular_kind.rs:21`.
- Graph inspection surface a new format must serve:
  `docs/specs/2026-08-19-native-graph-inspection-tools.md`,
  `rust/cli/src/graph/mod.rs:176`, `:294`, `:380`.
- Workload seams and their costs: `rust/runtime/src/engine/registry.rs:342`;
  `rust/runtime/src/scheduled.rs:1434`;
  `rust/runtime/src/engine/execute/dataset_build.rs:184`;
  unwired `static_accuracy` `rust/runtime/src/engine/online_execution.rs:277`.
- Dataset loader/composer registration:
  `rust/runtime/src/dataset/loader/mod.rs:313`, `:345`, `:379`;
  `rust/runtime/src/dataset/compose.rs:246`.
- Batching semantics (one batched request is one record):
  `rust/runtime/src/endpoints/implementation.rs:1080`-`:1094`;
  `rust/dry-run-tests/tests/random_pool_batches.rs:11`-`:66`.
- Existing character-bounded chunker: `rust/runtime/src/dataset/corpus.rs`.
- Pre-dispatch worker materialization and the latency origin:
  `rust/runtime/src/engine/turn_execution.rs:2035`-`:2055`;
  `rust/runtime/src/transport/http/sink/endpoint_dispatch.rs:289`.
- Artifact seams: `rust/runtime/src/export/mod.rs:310`, `:397`;
  `rust/core/src/artifact.rs:120`-`:131`;
  digest/manifest template `rust/runtime/src/eval/native_graph/artifacts.rs:62`-`:83`;
  `ResourceRequirementsV2` `rust/runtime/src/engine/registry.rs:107`-`:122`.
- Metric catalog closure and its positional contract:
  `rust/runtime/src/metrics_core/tag_id.rs:12`, `:41`-`:49`;
  `rust/runtime/src/metrics_core/catalog.rs:2034`.
- Labeled-series machinery the role dimension extends:
  `rust/runtime/src/metrics_core/report.rs:140`-`:150`;
  `rust/runtime/src/metrics_core/accumulator.rs:750`-`:828`;
  `rust/runtime/src/metrics_core/ingest.rs:21`-`:27`.
- Multi-series metric-drop bug: `rust/runtime/src/metrics_core/report.rs:1284`-`:1288`;
  `rust/runtime/src/export/mod.rs:80`-`:96`.
- Per-record row lacking profile identity:
  `rust/runtime/src/engine/records.rs:113`-`:143`.
- Accuracy evaluator seam and why the judge does not ride it:
  `rust/runtime/src/accuracy_core/protocol.rs:196`-`:222`; `docs/specs/accuracy.md`.
- Mock-server determinism and fixtures:
  `rust/mock-server/src/handlers.rs:2500`, `:2509`;
  `rust/mock-server/src/accuracy.rs`.
- Eager-branch doctrine this record does not violate:
  `docs/specs/conditional-graph-lowering.md`.
- Graph plane determinism live branching must not disturb:
  `docs/specs/graph-runtime.md`.
