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

### What this design optimizes for

AIPerf is a benchmarking tool. The bar this record holds itself to is **the
correctness of the pipeline and of the numbers it reports** — every other
consideration is subordinate to it, and where they conflict, measurement
correctness wins.

That ordering has a concrete consequence for how the artifact machinery below
should be read. `corpus_digest`, the index header, and the manifest digests are
**comparability devices, not integrity devices**. They exist because a
tasks-per-second number is meaningless unless the two runs being compared
answered from the same corpus, chunked the same way, embedded by the same model.
They are checks against silent drift — a stale index, a re-chunked corpus, a
swapped embedding model — not against a motivated attacker. They should be
implemented to that standard: cheap, always-on, and refusing loudly on mismatch,
without a threat model, key management, or signature scheme attached.

This design introduces **no new trust boundary**. It reuses existing transports,
the existing artifact channel, and the existing cellular authentication as they
are. Where a run crosses hosts, the pinned-TLS artifact path and controller
registration already in place carry it; nothing here extends, weakens, or needs
to reason about them.

Effort belongs instead in the places a benchmark can be quietly, plausibly wrong:
that request bodies are what the design says they are (the failure O1/O2 caught,
where empty requests benchmarked nothing and reported a clean number), that a
task's latency has a defined origin and terminal, that concurrency accounting
distinguishes in-flight *tasks* from in-flight *requests*, and that no exporter
silently drops the metric the run exists to produce. Those obligations are stated
in `### Measurement correctness` and carry the earliest positions in the delivery
order.

## Invariants

These are the normative core of this record. Everything in `## Design` exists to
establish them and everything in `## Built` is assessed against them. Each is
stated as a property that either holds or does not — not as a goal — and carries
the point that enforces it, its truth value in the tree today, and the wrong
number it prevents.

Status values: **HOLDS** (true today, must not regress), **VIOLATED** (false
today, with the anchor proving it), **NEW** (nothing enforces it yet).

### Corpus and index identity

**I1. A QnA run answers from the index the run's own configuration names, or it
does not run.** The index artifact's `corpus_digest` equals the digest recorded
in the pinned plan and in the query set, checked before dispatch.
*Enforces:* run bootstrap, artifact load. *Status:* NEW.
*Without it:* a stale index silently answers a newer query set and the accuracy
number describes a corpus nobody built.

**I2. `corpus_digest` is a pure function of exactly three inputs:** the sorted
document identities each binding its bytes, the frozen chunker parameters, and the
embedding profile's model identity. Nothing else enters it — not wall-clock, not
run id, not file order, not host.
*Enforces:* the digest constructor. *Status:* NEW.
*Without it:* two byte-identical corpora produce different digests, I1 fires
spuriously, and the check gets disabled.

**I3. The index header's vector dimension equals the width of the vectors the
embed profile actually returns.** Checked on first embed response, not assumed
from configuration.
*Enforces:* ingestion sealer. *Status:* NEW.
*Without it:* a mismatched model produces an index that searches successfully and
retrieves garbage.

**I4. Chunking is byte-stable across runs and hosts.** The same document under the
same parameters yields the same passages, with the same boundaries, in the same
order.
*Enforces:* the character-bounded chunker (`rust/runtime/src/dataset/corpus.rs`),
whose existing reproducibility contract is the reason it is reused rather than
replaced. *Status:* HOLDS for the existing chunker; NEW for the RAG parameters.

### Request construction

**I5. Every request AIPerf dispatches is well-formed for its endpoint kind and
carries the content the graph says it carries.** No dispatched body is empty
because a field the endpoint reads was never populated.
*Enforces:* a startup refusal beside `graph_execution.rs:393`-`:400` for kinds whose
request cannot be constructed from graph materialization; then P1's structured
path removing the need for the refusal for embed and rerank.
*Status:* **VIOLATED.** `graph_execution.rs:2204`-`:2221` builds the Turn with
`..Turn::default()`, so `Turn.texts` is always empty; `embeddings` therefore emits
`{"input": []}` (`implementation.rs:1090`-`:1094`) and rankings hard-error
(`tier2.rs:367`). The only kind gate on this path is `!requires_raw_token_ids`
(`:395`), which both pass.
*Without it:* the run completes, reports clean latency and throughput, and
benchmarked nothing. This is the defect the contest caught, and it is the reason
this list exists.

**I6. A node's endpoint profile determines its URL, its dialect, and its
tokenizer.** All three, or the profile is not a routing decision.
*Enforces:* per-node profile resolution at dispatch; lowering; the token-counter
binding.
*Status:* **VIOLATED in two of three.** URL holds
(`graph_execution.rs:1030`-`:1043`, `:500`-`:509`). Dialect does not — lowering
passes the *default* profile's `endpoint_id` to every node
(`online_execution.rs:1320`-`:1332` → `graph_input.rs:944`). Tokenizer does not —
the benchmark path clones one shared counter into every profile
(`graph_execution.rs:353`-`:379`), while only the eval path binds per profile
(`:383`-`:429`).
*Without it:* role-level ISL is not a measurement of the role.

### Reply handling

**I7. Every reply payload for which the graph declares a channel reaches that
channel.** No reply is silently reduced to empty, and no parsed value is discarded
after decoding.
*Enforces:* reduction; the channel write.
*Status:* **VIOLATED.** `reduce.rs:206`-`:211` no-ops for `Embeddings` and
`Rankings`; `models.rs:246`-`:277` returns `""`; the channel receives
`encoded_messages(vec![])` or `Value::Null` (`executor.rs:478`-`:510`). The vector
*is* correctly parsed (`implementation.rs:2137`-`:2200`) and then dropped.
*Without it:* `embed_subquery[i]` cannot write the query vector `Retrieval` reads,
and the pipeline's data dependencies are fiction.

**I8. A channel that was never written is never observed as a write.** The
`{"$unset": true}` sentinel does not enter a downstream stage's `initial_state`.
*Enforces:* the driver's channel carry-over.
*Status:* **VIOLATED** in the existing driver — `reducers.rs:47`-`:58` produces the
sentinel and `live_driver.rs:479`, `:983`-`:988` propagate it unfiltered into
`channel_store.rs:114`-`:125`, where it is seeded as a real write at seq 0.
*Without it:* a node reads a literal `{"$unset": true}` as its input and the
request is well-formed nonsense.

### Control flow

**I9. Hop count is bounded, and the bound is enforced by something other than the
driver that requests the hops.** A driver cannot exceed its own declared bound.
*Enforces:* placement, independently — `graph_execution.rs:1820`-`:1822` hard-errors
if a staged driver omits `stage_bound()`, `:1861`-`:1867` refuses `Execute` past it.
*Status:* **HOLDS.** Must not regress when the staged program family lands, and
must survive the move to live branching as a loop-iteration cap on the edge.

**I10. Every executed stage plan is a projection of the authored source graph.**
Never synthesized, never a superset, and re-validated against the same rules the
source passed.
*Enforces:* the driver's projection step — **and nothing else**. An earlier draft
of this record credited `graph_execution.rs:1868` with "placement's independent
re-validation"; that call is structural self-consistency only and never sees the
source graph. See I22.
*Status:* **HOLDS** for the existing driver (`live_driver.rs:895`-`:1010`),
because that driver constrains itself. It is not enforced on any driver that
declines to.
*Without it:* the benchmark measures a graph the author never wrote.

**I11. Verdict decoding is total and declared.** The authored decoder yields a
verdict or a typed error attributed to its node. There is no fallback, no
substring sniff, and no silent default.
*Enforces:* the verdict decoder. *Status:* NEW — no JSON-pointer or predicate
decoder exists in the tree; the only precedent is exact string equality on
`.content` (`live_driver.rs:1278`-`:1295`), which is total in the same sense and is
the model to follow.
*Without it:* an unparsed verdict reads as "insufficient", every task runs to the
hop bound, and tasks-per-second measures the timeout path.

### Measurement

**I12. Request latency excludes client-side work; task latency includes it, and
every stage the reference pipeline names is inside some measured window.**
Both scopes exist, they measure different things, and the report says which is
which. Parse and chunk are ingestion *stages*, not setup: they occur after the
run origin, on the worker, inside the document task.
*Enforces:* `on_admit` as the request origin
(`transport/http/sink/endpoint_dispatch.rs:289`), which already fires after
materialization; a separate task-level origin and terminal; and the ingestion
driver's own `next_stage` for parse and chunk. That seam runs on the shared
`LocalSet`, so it carries the additional obligation that measuring these stages
must not perturb co-resident traces — see the staging-cost discussion under
live branching.
*Status:* request half **HOLDS** for free. Task half is NEW. The parse/chunk
clause is NEW **and was contradicted by an earlier draft of this record**, which
scoped both stages to a `DatasetLoader`/`Composer` pair. That pair runs entirely
in preparation — `LoaderRegistry::build_dataset` awaits `load` and `compose` and
freezes the `Dataset` (`dataset/loader/mod.rs:543`, `:560`, `:568`, `:578`),
reached from `prepare_with_context` (`engine/online_execution.rs:362`, `:1276`
-`:1288`) before the run spec, before any phase plan, and before
`set_run_origin` (`engine/execute/sharding.rs:476`). Nothing in a loader or
composer can be timed by this benchmark.
*Without it:* `rag_documents_per_second` names the reference's four-stage
pipeline and measures two of them, which is a wrong number wearing the right
label.

**I13. In-flight tasks and in-flight requests are separate curves, and any window
derived from concurrency names which one it used.** `--steady-state` reads the
task curve for `rag_qna` or refuses.
*Enforces:* the sweep-line consumer. *Status:* NEW — every existing consumer counts
requests.
*Without it:* the steady-state window is shaped by hop fan-out rather than task
admission and excludes the wrong interval.

**I14. Every reported aggregate is the correct function of the records beneath
it, and no metric is dropped for a configuration this design makes normal.**
*Enforces:* the exporter plane.
*Status:* **VIOLATED.** `summary_series` returns `NoAggregate` for multiple
non-unique series (`export/mod.rs:80`-`:96`), removing TTFT, ITL, latency, ISL, and
OSL from genai_perf JSON/CSV, console, timeslice, MLflow, and W&B whenever a run
has more than one model or endpoint. RAG makes that the common case.
*Without it:* a reader sees no request-latency row and concludes the run did not
measure it. A dropped metric is a wrong number, not a missing one.

**I15. One batched request is one record, with `input_sequence_length` the batch
sum.** Every ingestion rate states its unit against this.
*Enforces:* the existing batch path.
**A batch never spans documents.** A document spans several batches; the converse
is refused. This is forced by the corrected execution design — one document is one
staged task and its batches are that task's stages, so a cross-document batch would
belong to two tasks at once and complete neither.
*Status:* the batch-record half **HOLDS** (`random_pool_batches.rs:11`-`:66`); the
containment half is NEW and refused-on-violation. An earlier draft of this
invariant said a batch spans several documents, which contradicted the execution
design and is corrected here.
*Without it:* document completion is undefined, because a record would attribute
to two documents.

**I16. No per-role aggregate is reported before per-record role attribution
exists.** A role-labelled number is backed by records that carry the role.
*Enforces:* `RecordIngest` and the per-record row.
*Status:* **VIOLATED as a precondition** — `RecordMetadata`
(`engine/records.rs:113`-`:143`) carries no model, endpoint, profile, or node
identity, so there is currently nothing to attribute to.
*Without it:* the answer-role OSL compliance check is computed from records that
cannot be attributed to the answer role.

**I17. A scored number is exact, never an estimate, and sketch mode is refused
for the whole scored surface rather than for one command.** Sketch mode's
percentiles and standard deviation are streaming estimates, and — the part an
earlier draft of this record missed — `export_results_sketch` drops the inference
series entirely, so the per-role dimension I20 requires does not survive it at all.
A sketch-mode run therefore refuses `aiperf rag compliance` *and* refuses to be a
scored `rag_qna` run: the two refusals are the same refusal, and scoping it to the
command alone would have let a pinned run report a rate with I20's per-role OSL
columns silently empty.
*Enforces:* the compliance command's precondition and the pinned run's report.
*Status:* NEW.

**I18. Every new metric is verified against raw per-record output.** An e2e test
against a deterministic `aiperf-mock-server` configuration reads the per-record
artifact and asserts the aggregate is the correct function of those records.
Summary-only assertions do not satisfy this.
*Enforces:* the repo's standing verification requirement.
*Status:* NEW for every metric in this design.
*Without it:* I5 and I7 fail undetected, which is exactly what happened.

### Replay

**I19. A pinned run issues byte-identical request bodies across runs, with fixed
hop count and fixed retrieved sets, and refuses every authored option that
deliberately perturbs request bytes.** Every node's request inputs come from the
recorded plan, not from the previous stage's live output. The second clause is not
decorative: `cache_bust_target: first_turn_prefix` exists precisely to make every
request differ, deriving its marker from the trace instance
(`agentic_replay.rs:83`-`:95`), so an authored cache bust and a pinned run are
directly contradictory. A pinned `rag_qna` run requires `cache_bust_target` to be
`None` and **refuses** any other value rather than silently issuing perturbed
bytes under a byte-identical claim. An earlier draft of this record stated the
guarantee without naming its one existing counterexample in the tree.
*Enforces:* the pinned plan loader. *Status:* NEW.

**I20. Pinning fixes work, not generation, and the unfixed part is reported
rather than assumed away.** A pinned run holds hop count, retrieved sets, and
request bytes constant; it does not constrain any role's generated output length.
Because service time is dominated by generation, `rag_tasks_per_second` is
reproducible only to the extent that the endpoint's per-role output-length
distributions are stable — a property this design does not establish and cannot.
The enforceable half is therefore reporting: a pinned run emits mean and p90 OSL
**per role** beside every scored rate, so run-to-run rate drift is attributable to
the role that generated it instead of being invisible.
*Enforces:* the pinned run's report. *Status:* NEW; depends on I16, and on
I17's sketch refusal, without which the per-role columns are empty by construction.
*Without it:* a rate difference between two pinned runs of the same plan is
unexplainable, and the reader attributes it to the system under test.

An earlier draft of this record said the compliance check bounds that residual
variance. It does not, and the two must not be collapsed: I17's check is a single
aggregate over the **answer** role only. Every rewriter, grader, and sufficiency
output is equally unbounded and equally moves service time, and none of them is
covered. Per-role bands are named as future work, not claimed here.


**I21. A stage that contained a failed request is never reported to the driver
as completed, and a loop that exits on an unrecognized verdict leaves a
receipt.** The signal a hop-controlling driver reads must distinguish "the model
said nothing" from "the request failed."
*Enforces:* the stage-result terminal status; the staged driver's failure policy;
the loop's non-match path.
*Status:* **VIOLATED, and this is the most dangerous defect in the substrate this
design builds on.** Three lines conspire. `graph_execution.rs:1905` hardcodes
`terminal_status: GraphReplyStatus::Completed` on every `TraceStageResult`, so a
driver can never observe a failed stage — the existing driver's
`ensure_completed_stage` (`live_driver.rs:885`-`:894`) is dead code. The default
`OnFailure::Continue` (`failure.rs:29`-`:32`) resolves through
`ResilientNodeFailurePolicy` to `NodeFailureDisposition::ContinueWithEmpty`
(`graph_execution.rs:1600`-`:1604`, `policy.rs:271`-`:275`,
`executor.rs:450`-`:456`), so a 500 writes an empty channel value and execution
proceeds. And the loop's unrecognized-verdict path is
`else { progress.awaits_backedge = false; continue; }`
(`live_driver.rs:1179`-`:1182`) — a silent exit, where the *branch* construct
hard-errors on the same condition (`:1036`-`:1046`).
*Without it:* a `rag_qna` task whose retrieval hop 500s reads an empty passage
set, generates a plausible answer from nothing, terminates early, and is scored
and counted as a successful task. Every reported number is well-formed and the
run is wrong. This is the pipeline-correctness failure mode the whole record is
ordered around, and it is not hypothetical: it is the current default
configuration of the substrate.

All three parts are fixable in our own code, and the fix is a prerequisite rather
than a hardening pass. Thread the real terminal classification into
`TraceStageResult` in place of the constant at `:1905` — the field and the
driver-side check already exist, so this is plumbing through `TraceResult`.
Select `AbortTraceNodeFailurePolicy` for staged-driver execution, or carry a
per-stage "a node failed" flag. And make the loop's non-match case explicit: a
closed declared verdict set, as branches already require, or a receipt when a
loop exits on a value it did not recognize.

**I22. A driver-authored stage plan is validated against the authored source
graph, not only against itself.**
*Enforces:* the driver's projection step.
*Status:* **NEW, and weaker in the substrate than an earlier draft of this record
claimed under I10.** Placement runs exactly one check —
`validate_native_graph_trace_plan` (`graph_execution.rs:1868`,
`lowering.rs:584`-`:662`) — and it is purely structural self-consistency: node
ids, channels declared within the plan itself, edge endpoints, reachability from
`START` to `END`, acyclicity. It never sees the source graph. The source-binding
validation exists but lives *inside* the driver
(`validate_native_graph_stage`, `lowering.rs:664`-`:688`;
`validate_dynamic_stage_projection`, `live_driver.rs:988`-`:1013`) and is called
only from `live_driver.rs:347` and `:384`. So any structurally valid DAG — new
node ids, new prompts, new models — is accepted by placement and dispatched.
I10 holds today only because the one existing driver constrains itself.
*Without it:* the benchmark measures a graph the author never wrote, and nothing
outside the driver would notice.

### Pipeline identity and comparability

I1–I4 protect one axis of comparability — *did these two runs answer from the
same corpus, chunked the same way, embedded by the same model?* — and nothing
protects the other. Every parameter that changes the scored number without
touching a single corpus byte is currently unguarded: hop bound, sub-query
fan-out, top-`k`, index kind and its build/search parameters, the role→profile
map, pinned-versus-live, exact-versus-sketch. Two runs can agree on
`corpus_digest` to the bit and still be measuring different benchmarks. The
asymmetry is the defect; these two invariants make the second axis as explicit as
the first.

**I23. Every parameter that changes the scored number is recorded in the run's
own output, and their canonical digest is `pipeline_digest`.** The digest covers,
at minimum: the ordered role→profile map with each profile's endpoint kind and
model identity; the hop bound, sub-query fan-out bound, and top-`k`; the index
kind and every parameter that changes which passages come back (for `hnsw`, `M`
and both `ef` values); the retrieval scoring metric; whether the run was pinned
or live; and the metrics mode. It is a pure function of the resolved
configuration in exactly the sense I2 requires of `corpus_digest` — no
wall-clock, no run id, no host, no worker or cell count. `corpus_digest` and
`pipeline_digest` together name the benchmark; either one alone names half of it.
*Enforces:* the digest constructor, at run bootstrap, from the resolved plan.
*Status:* NEW. The carrier already exists — `input_config` is written verbatim
into the run's JSON (`rust/runtime/src/export/genai_perf.rs:88`-`:89`, `:565`-`:566`,
and again in `export/timeslice.rs:44`, `:306`-`:307`), so the raw material is
present and only the canonicalization and the digest are missing.
*Without it:* a reader compares two numbers that agree on corpus and differ on
`k`, and the difference is attributed to the system under test.

**I24. A run declares its comparability class.** Either it matches an authored
reference profile in every parameter I23 covers, and says so, or the report names
each parameter that diverges and its authored value. There is no third state and
no silent divergence. The reference profile is a shipped artifact, not a
constant, so a run can declare itself comparable to the MLPerf reference
configuration, to a prior AIPerf run, or to nothing.
*Enforces:* the report writer, from I23's inputs.
*Status:* NEW.
*Without it:* every number leaves the tool with the same authority, whether it
was produced under the reference configuration or under a locally-tuned one, and
the burden of noticing lands on whoever reads the JSON.

**I25. A run records the served identity it can observe, and states plainly what
it cannot.** *Status:* NEW, and **partly unachievable today** — which is the
finding worth recording rather than the invariant worth asserting. Every identity
field in an AIPerf artifact is client-authored and never server-observed. The
per-record `model` is the string the client sent: `model: Some(self.model.clone())`
(`rust/runtime/src/transport/http/sink.rs:604`), with `turn.effective_model`
falling back to it (`:619`-`:628`); `InferenceDimensions` documents it as "model
carried by the dispatched request" (`metrics_core/ingest.rs:25`-`:26`);
`ExactRecordV1.model` as "model named on the request" (`rust/core/src/capture.rs:110`);
`run_info` as "requested model name" (`metrics_core/report.rs:171`-`:179`); and
`input_config` as "the authored `BenchmarkConfig` dump"
(`export/genai_perf.rs:88`-`:89`, echoed by `rust/cli/src/profile.rs:1794`,
"a reporting-only mirror of the authored config"). The chat response parser
extracts `data` and `usage` and drops the body's own `"model"` key
(`endpoints/implementation.rs:319`-`:333`); `reduce.rs` never touches a model
string and `measure.rs` contains no occurrence of "model" at all.

The trap is that provenance *looks* like it exists, because a discovery call
genuinely happens on the wire. `/v1/models` is queried
(`endpoints/implementation.rs:294`-`:297`), the server answers with the full model
object, and the classifier discards all of it for a boolean —
`data[].id == self.model` (`engine/readiness.rs:238`-`:261`) — and is skipped
entirely when `/v1/models` 404s and a base-URL 2xx is accepted (`readiness.rs:661`).
No quantization, precision, dtype, engine version, or model revision is captured
anywhere on the served-model path. Two runs at different served precisions under
the same authored name produce byte-identical identity fields. For a benchmark
where precision selection is the submitter's primary optimization lever, that is
a comparability hole and not a cosmetic one.

Three things are available and this design uses all three. Retain the discovery
response rather than reducing it to a bool, which is a change local to the
readiness classifier and costs one retained JSON value per profile. Capture
Prometheus `*_build_info` and `cache_config_info` series when
`--server-metrics-url` is configured: labels are retained losslessly with no
allowlist (`server_metrics/prom_text.rs:59`-`:86`,
`server_metrics/accumulator.rs:721`-`:722`, into `SidecarSeries.labels`,
`metrics_core/sidecar.rs:62`-`:63`), though derived atlas metrics drop labels
(`accumulator.rs:303`) and this path is opt-in. And where neither is available,
the report says so in as many words: `pipeline_digest` covers what the client
authored, and the served precision behind an authored model name is outside it.
*Without it:* an archived artifact cannot distinguish an fp8 result from a bf16
one, and I24's comparability claim silently overstates itself.

This reframes the **index default**, which the rest of this record decides on
cost grounds alone. `flat` is exact, which removes ANN recall as a confound and
makes it the right default for AIPerf-internal A/B work, where the question is
whether *this* change moved the number. `hnsw` matches the MLPerf reference, and
is therefore the only setting under which a number is comparable across
harnesses. Both stay; the decision is which question the run is asking, and I24
is what makes the answer legible — a `flat` run declares itself divergent from
the reference profile in the index-kind parameter and remains a perfectly valid
AIPerf measurement.

### What is deliberately not an invariant

This design introduces **no new trust boundary**, so there is no
confidentiality, authenticity, or tamper-resistance invariant here. The digests
in I1–I3 are comparability devices: they catch a stale index, a re-chunked
corpus, or a swapped embedding model, and they are budgeted as cheap always-on
drift checks with no threat model, key management, or signature scheme attached.
Cross-host runs reuse the existing pinned-TLS artifact channel and controller
registration unchanged.

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

**And the seam is unproven by test, which is a different claim from unbuilt.**
The routing above is read out of the code, not out of a passing assertion. The
closest existing test authors a second profile beside the default and asserts both
are *prepared* (`rust/cli/tests/online_v2_stdio.rs:589`-`:597`, `:659`-`:663`) —
but its graph is a single `dag_jsonl` turn with no `endpoint` field, both profiles
are `chat`, and both point at the same URL. It proves preparation and never
selection. No fixture anywhere populates a node-level `endpoint` selector; the
only writers of that metadata key are `graph/dag_source.rs:264` and
`graph/lowering.rs:615`. **There is no test in the tree in which a two-profile
graph run splits traffic between two endpoints.** Since this seam is the one this
whole design stands on, closing that gap is the cheapest risk reduction available
and belongs before any RAG-specific work, not after it.

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

**Two precisions on the size of that work, because "prerequisite" has been read
here as "large".**

*The channel plane needs no change at all.* `ChannelType` is two-valued, but that
enum selects only the write shape; the stored value is `ChanVal::Val(Value)`
(`rust/runtime/src/graph/reducers.rs:16`-`:29`), which holds arbitrary JSON, so a
768-float array round-trips through the store, the snapshots, and the reducers
untouched. The gap is *consumption*, and it is one line:
`materialize.rs:121`'s `Some(ChanVal::Val(_)) => Ok(Vec::new())`, where a
non-message value silently vanishes. One additive `PromptItem::Field { from, name }`
variant, materialized into `Turn.texts` as a named `Media`, covers both the
unnamed `input` of embeddings and the named `query`/`passages` groups of
rankings. `PromptItem` is `#[serde(untagged)]`, so the variant is invisible to
existing documents — add it last and confirm no existing shape newly matches.

*The reply-side defect is the small half and the shared half.* The irreducible
fix is a carrier for the parsed payload — one `Option<Value>` field on
`ModelResponseMetadata` (`rust/runtime/src/scheduled/observe.rs:37`-`:56`),
populated in the currently-empty match arm at `reduce.rs:203`-`:207`, threaded
through `TurnDispatchOutcome` (`observe.rs:60`-`:77`), and read at
`graph_execution.rs:2407` to build the reply from the value instead of the empty
text. It must **not** be fixed by making `get_text()` return the payload: that
string is what output-length metrics count, and doing so would corrupt OSL and
ITL for every retrieval and rerank record. `has_token_output()` already returns
false for these kinds, so such a record is legitimately latency-only.

This one defect is the shared prerequisite for embedding, rerank, **and**
retrieval, since all three are `Llm` nodes bound to non-chat profiles. That is
why it is sequenced first rather than alongside the RAG-specific work: three
features are blocked on approximately ten lines in two files, and no amount of
downstream design removes that block.

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
- **`ParsedResponse.sources` — not usable, and this record previously said
  otherwise.** It is defined at `rust/runtime/src/endpoints/models.rs:391`-`:402`
  as an untyped `Option<Value>`, written by exactly one endpoint
  (`SolidoRagEndpoint::parse_response`, `rust/runtime/src/endpoints/tier2.rs:1129`
  -`:1132`) and, as a full-workspace search establishes, **read by nothing**:
  `reduce_parsed_response` (`rust/runtime/src/transport/reduce.rs:61`-`:101`) reads
  only `data`, `usage`, and `perf_ns`, and the value dies with its local at
  `rust/runtime/src/transport/http/sink/endpoint_dispatch.rs:683`-`:697`. Every
  other construction site writes `None`. Calling it "plumbed" was wrong — it is
  write-only.

  The disqualifier that matters most is not the missing plumbing: `ParsedResponse`
  does not appear anywhere under `rust/runtime/src/graph/` or `.../eval/`, so it is
  not a carrier the graph plane can read even where it is written. Once retrieval
  is an `Llm` node bound to a retrieval profile, its evidence must ride the same
  route every other reply payload takes — the `Option<Value>` carrier added for
  embeddings and rankings — rather than a second, endpoint-private channel that
  only one dialect populates. (An earlier draft rested this bullet on retrieval
  being a *non-dispatching* node kind that produces no inference record. That
  premise is retired with the node-kind decision; the conclusion is unchanged and
  now rests on the carrier being unreadable from the graph plane rather than on
  the record being absent.)
- **Large binary artifacts.** A registered `Exporter` writing into the run's
  `artifact_dir` (`rust/runtime/src/export/mod.rs:310`, `:397`), with
  `ParquetExporter` as the existing multi-hundred-megabyte precedent and no size
  or quota limit *in the runtime*. The ceiling is downstream and hard: the
  Kubernetes publication path caps one artifact at 512 MiB and fails the entire
  results manifest on breach (`rust/cli/src/k8s.rs:208`-`:216`,
  `rust/cli/src/results_sidecar.rs:51`). `ResourceRequirementsV2 { artifacts }`
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
Establishes I5 and I7.
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
Must preserve I9 and I10, which hold today only on the eval path.
`prepare_graph_phase` must accept driver-bearing programs alongside `all_static`
and `all_recorded_replay`, with defined t\*, warmup-handoff, and cache-pressure
behavior for staged traces, and without the eval path's `completed_traces == 1`
assertion. Nothing multi-hop can be benchmarked before this exists.

**P3 — an `endpointProfiles:` authoring surface.** Makes I6 reachable at all. A Config-v2 YAML section
projecting into `BenchmarkConfig.endpoint_profiles`, replacing the hardcoded empty
map at `resolve.rs:1682`, with one e2e test standing up two mock servers on
different ports and asserting the per-node request split. Without it the whole
design is reachable only through the internal stdio envelope.

**P4 — per-profile input token counters.** Establishes I6's tokenizer clause. Bind counters independently on the
benchmark path as the eval path already does
(`graph_execution.rs:383`-`:429`), so role-level ISL is trustworthy across models
with different tokenizers.

### Two workloads, two very different costs

| Workload | Shape | Unit of work | Scored metric |
|---|---|---|---|
| `rag_ingest` | graph run over a new graph format `rag_corpus` + staged driver | one source document | `rag_documents_per_second` |
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

Ingestion needs none of it — but it is **not** a linear dataset format either, and
this record said twice that it was. The first draft put parse and chunk in a
`DatasetLoader`/`Composer` pair, which measures neither. The correction moved them
to the worker and made one document a task with a data-dependent stage count, and
then left the workload on the linear path anyway, which does not execute staged
programs.

`NativeDatasetPlan` is a closed enum — `PreparedLinear | StaticAccuracy | Graph`
(`rust/runtime/src/engine/execute/plan.rs:360`-`:368`). `lower_linear` produces
`PreparedLinear` (`rust/runtime/src/engine/online_execution.rs:1272`-`:1295`);
scheduled execution rejects a graph plan
(`rust/runtime/src/engine/execute/compose_sidecars.rs:73`-`:82`) and graph
execution rejects `PreparedLinear`
(`rust/runtime/src/engine/execute/entrypoints.rs:403`-`:421`). There is no bridge
and `CreditMaterializer` is not one: it is a scheduled-turn seam and cannot turn a
linear dataset into staged programs.

The cheap alternative does not exist either. A scheduled run fixes its credit
count before dispatch, and a document's passage count is unknown until parse
completes, so the linear path cannot express the unit of work at all — not as a
matter of plumbing, but because the count that would size the credits is the
output of the stage being measured.

**`rag_corpus` is therefore a graph input format, acquired and lowered through a
`GraphInputAdapter`, and `rag_ingest` is a graph run.** Its adapter fetches and
interns raw article bytes and emits one staged `GraphTraceProgram` per document,
whose stages are that document's embed batches. This is the price of measuring
parse and chunk, and it is charged honestly: ingestion now carries the same
graph-format obligations as `rag_qna` — a `BUILTIN_GRAPH_FORMATS` entry, a
resolver adapter, and the fixed-length adapter array — so the inventory grows by
**two**, to ten, not one, and ingestion depends on the staged-program prerequisite
(P2) exactly as QnA does. Nothing about ingestion is cheaper than QnA except the
absence of a control loop.

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

The digest is computed **per element first, then folded over the sorted element
digests with a `\x00` delimiter**, over raw bytes with no normalization. This is
the MLPerf reference's `_corpus_set_sha256` construction
(`db_manifest.py:106`-`:127`) and it is adopted deliberately: order-independence
falls out by construction rather than by sorting a large intermediate, and the
sealer can maintain it incrementally as passages land.

The manifest additionally carries a **seeded retrieval probe set** — 50 queries
sampled deterministically from the query set, each with its top-`k` result
identities — and the index-integrity check asserts mean overlap against it. The
digest and the probe set answer different questions: the digest proves the
passages are the same, the probe set proves the index *retrieves* the same, which
a content digest cannot establish for an insertion-order-dependent structure like
HNSW. The reference ships both and gates only on the probe set; we gate on both.

### Ingestion: parse → chunk → embed → index

The client-side stages are pure functions run on the worker inside the measured
window, following the deferred-materialization precedent. **This corrects an
earlier draft of this record**, which scoped parse and chunk to a
`DatasetLoader`/`Composer` pair while also claiming they were measured. Those two
statements cannot both hold: the loader/composer pipeline is fully awaited and
frozen during preparation (`dataset/loader/mod.rs:543`, `:560`, `:568`, `:578`,
reached from `engine/online_execution.rs:362`, `:1276`-`:1288`), before the run
origin exists (`engine/execute/sharding.rs:476`). A loader-based ingestion would
have reported `rag_documents_per_second` for embed and index while calling it the
four-stage pipeline.

**A second dataset plane now exists, and the honest statement names both.** Saying
only "the loader is eager" implies the loader is the whole dataset surface, and it
is not: `runtime/src/streaming/` is a complete streaming input contract —
`StreamingDatasetSourceFactory` → `PreparedStreamingDatasetSource`
(`streaming/source.rs:174`-`:200`), an incremental
`StreamingPartitionDecoder::next_batch(DecodeBatchBudget)` returning
`DecodeStep::{Batch, End}` under `FormatStateRetention::BoundedMemory`
(`streaming/format.rs:163`-`:186`, `:68`-`:75`), `StreamingSessionProgram` emitting
`ExecutableDatasetAction` (`streaming/unit.rs:615`), and `StreamingActionSink`
binding those actions to a transport and endpoint (`streaming/action.rs:227`-`:233`),
with working implementations for paged HF rows, Parquet, and synthetic prompts. It
ships in every default build (`rust/cli/Cargo.toml:34`, `:49`), holds five
registry slots (`extensions/mod.rs:416`-`:431`) and five `register_stream_*`
methods (`engine/registry.rs:726`, `:738`, `:750`, `:762`, `:774`), and reaches the
discovery `Catalog` (`engine/protocol.rs:153`-`:174`).

It has **no execution driver**. No code outside `streaming/` names
`PreparedStreamingDatasetSource`, `StreamingDatasetFormat`, `StreamingSessionProgram`,
`StreamingActionSink`, or `StreamingCheckpointBackend`; every `register_stream_*`
call site is a test using fakes, and `extensions/mod.rs:413`-`:415` states the slots
are "populated only by a real built-in or external extension" — today, none.
`dataset/loader/mod.rs` and `engine/dataset_input.rs` contain zero streaming
references and `build_dataset` still fully materializes.

So the current-truth statement is: **ingestion is not a `DatasetLoader` because the
loader is eager, and not a `StreamingDatasetFormat` because that plane has no
execution driver. The graph path is the only executable option.** That second
clause is a condition, not a permanent fact, and it has a checkable trigger:
`builtin_source_factories()` (`streaming/sources.rs:16`) acquiring a caller. If
that happens, ingestion-as-a-streaming-format becomes a real alternative to the
graph path and this section should be re-decided rather than inherited.

**The seam is the staged driver's `next_stage`, not deferred materialization.**
An earlier draft of this record named `materialize_credit`
(`engine/turn_execution.rs:2045`, `:2320`) as the worker-side seam, and that was
wrong in a way the O11 correction made worse rather than better: `materialize_credit`
belongs to the scheduled path, and once ingestion moved to
`NativeDatasetPlan::Graph` nothing calls it. The record kept the mechanism after
removing the execution path that reaches it.

The seam that actually runs client-side work on the worker inside a trace's
lifetime is `TraceProgramDriver::next_stage` (`graph/driver.rs:552`-`:559`),
awaited between stages by `execute_staged_driver`
(`engine/graph_execution.rs:1841`), with the prior stage's channels and frozen
terminal outputs handed back through `observe_stage` (`:1903`). A driver may do
arbitrary local compute there and return the next `GraphTracePlan` built from its
result, which is exactly the data-dependent stage construction ingestion needs and
is not available anywhere else on the graph path. So:

- The **loader** fetches and interns raw article bytes. It does not parse. Its
  only job is to make each document a frozen, content-addressed handle.
- The **worker** parses, chunks, and issues that document's passage batches.

The unit that makes this work is the document. Chunk count is not known until
parse completes, so batch membership cannot be computed ahead of dispatch — and in
particular the `rag_corpus` `GraphInputAdapter` **cannot pre-emit the batch
stages**, because it would have to parse to know how many there are, which is the
stage being measured. Batches are therefore formed **within** a document by the
driver, at execution time, and one document is one trace whose stages are its
passage batches.

Concretely: the `rag_corpus` adapter lowers one trace per document carrying the
interned raw bytes and nothing else. A **new driver kind**, `rag_ingest`, parses
those bytes on its first `next_stage`, chunks, and then returns one embed stage
per batch, `Complete` when the batches are exhausted. `stage_bound()` is the
authored cap on batches per document, enforced independently by placement
(`graph_execution.rs:1819`-`:1822`); a document whose chunk count exceeds it is
refused rather than truncated, because a truncated document is a silently
under-counted corpus. This makes P2's staged family a hard prerequisite for
ingestion (it already was, per O11) *and* adds a driver-kind registration to
ingestion's cost that the earlier draft did not carry. `--batch-size` retains its
meaning as the passages-per-request cap.

The stages:

- **Parse.** Extract the article body from HTML, dropping navigation, reference,
  and metadata subtrees; flatten tables and lists to text row by row rather than
  dropping them. This is the one new third-party dependency in this design, and
  the inventory above confirms nothing suitable is present in either language.
  Parsing is deterministic and content-addressed. A parsed-corpus cache is
  therefore possible and is **refused for a scored ingestion run**: parse is inside
  the measured window by I12, so a cache hit would silently delete a reference
  stage from `rag_documents_per_second`. Caching is available only for
  non-scored exploratory runs, which say so in their report.
- **Chunk.** Slice to `chunk_chars` (default 768) with `chunk_overlap_chars`
  (default 32) on UTF-8-respecting character boundaries, tagging each passage with
  its source document identity and byte offsets. This extends
  `dataset/corpus.rs`'s existing character-bounded chunker rather than replacing
  it. Defaults match the MLPerf reference; both are authorable and both
  participate in `corpus_digest`.
- **Embed.** Dispatch the document's passage batches to the `embed` profile
  through the existing batch path, one stage per batch. Batch size is authorable
  because macro-batching the embedder is one of the optimization levers the
  benchmark exposes; it caps passages per request and does not span documents.
- **Index.** Append returned vectors to a **worker-local shard builder**, and
  seal by merging shards at the run boundary. An earlier draft said "append to
  the builder, then seal through a registered `Exporter`", which named no owner and
  did not survive contact with the execution model. `VectorIndexBuilder` is `!Send`
  and there is one per worker thread, so there is no single builder for many
  document traces to append to; and the staged driver's `output_handles` are opaque
  handles resolved against a **trace-local** terminal-output store
  (`graph_execution.rs:1874`-`:1886`) that is not serialized, so graph output
  handles cannot carry vectors out of a trace, let alone across a cell boundary.

  The ownership chain is therefore explicit, and its shape already exists in the
  tree: `RecordArtifactLane` is a worker-local `Rc<RefCell<_>>` writer held open
  for the whole run (`rust/runtime/src/engine/record_lane.rs:1`-`:38`), each shard
  writes to `<artifact_dir>/.shard-<id>/<name>`
  (`rust/runtime/src/engine/execute/sharding.rs:391`-`:418`),
  `concatenate_shard_artifacts` merges at finalize
  (`rust/runtime/src/engine/execute/compose_sidecars.rs:794`-`:804`), and the
  controller runs the same merge over `cell-{id}` directories with cross-host
  uploads landing at the same paths
  (`rust/runtime/src/engine/cellular_controller.rs:1662`-`:1682`). Each worker
  owns one shard builder, appends the vectors its own traces returned with no
  cross-worker synchronization on the per-request path, and seals its shard to a
  content-addressed part file at end of phase; the cell merges its workers' parts;
  under `--cells N` each cell uploads its merged part and the **controller**
  performs the final merge and writes the sealed index.

  **Three pieces of that chain are new, and this record does not let the existing
  shape imply otherwise.**

  - *The merge must fail on a missing part.* Every file-level merge in the tree
    silently skips one — `shard_artifacts.rs:291`, `:308`, `:349`, and
    `per_record_parquet.rs:512` all `continue` or filter on `exists()`. That is
    defensible for a row union compared as a sorted set, which is what the module
    documents itself as (`shard_artifacts.rs:5`-`:13`, "never byte-for-byte"), and
    it is exactly the semantics an index seal must not inherit. Fail-closed merge
    is not a new *idea* — `cellular/shard.rs:150`-`:170` has typed
    `MissingOrdinal`/`DuplicateOrdinal`/`OrdinalOutOfRange` errors, and
    `eval/native_graph/artifacts.rs:311`-`:325` has declared-length staging — but
    it is new *at this layer*.
  - *Content addressing gives identity, not order.* An earlier draft said the
    merge is "order-independent because passage identity is content-addressed",
    which conflates the two: a set of content-addressed passages is topology-
    invariant, but the bytes of a merged file are not until a canonical order is
    chosen. The seal stamps each passage a global corpus ordinal at issue,
    validates the union across parts is exactly `0..N` as a permutation, and
    writes rows in ordinal order — the discipline `merge_records_in_global_order`
    (`cellular/shard.rs:139`-`:180`) uses to reach byte-identity against a
    single-cell run, which is the only topology-invariant merge in the tree.
  - *The digest must exclude topology fields.* The one canonical digest
    precedent, `streaming/results.rs:521`-`:546` with the field-wise form at
    `streaming/checkpoint.rs:1326`-`:1367`, is worth copying in shape —
    domain-separated, length-prefixed BLAKE3 — but its descriptor deliberately
    includes `cell_id` and `worker_id` (`results.rs:124`-`:126`), making that root
    topology-*dependent*. Nothing in the tree combines an invariant merge with a
    canonical digest; that combination is what I1 asks for and it is new code.

  **On closer reading the streaming result plane is a counter-example rather than
  a precedent, and this record no longer treats it as the closest structural fit.**
  Beyond the topology dependence above, its root is *order*-dependent:
  `CheckedResultStagePlan::from_partitions`
  (`streaming/checkpoints/local.rs:1743`-`:1762`) performs no sort, no dedup, and no
  canonicalization, folding the caller's `Vec` order straight into the hashed
  object, so staging the same segments in a different order yields a different
  root. And there is no cross-process merge in that plane at all: `ResultPartition`
  is not `Serialize` (only its descriptor is, `results.rs:112`-`:114`),
  `runtime/src/cellular/` contains zero streaming references, and neither shipped
  backend claims `CheckpointBackendPlacement::SharedAcrossCells`. Its `cell_id` and
  `worker_id` fields describe a reduction that does not exist.

  Three things from that plane are still worth taking, and they are the parts this
  record cites going forward. Its **verification ladder** is the strongest
  fail-closed discipline in the tree — reachability (`checkpoints/local.rs:2290`-`:2292`),
  length (`:2305`-`:2307`), digest, head re-check (`:2173`-`:2180`), missing object
  refused at `:1107`-`:1111` — and the seal should copy its shape. Its
  **length-prefixed domain-separated hashing** at `streaming/identity.rs:12`-`:24`
  (`domain_hash`/`update_field`) is the better citation than `results.rs:521`-`:546`
  for constructing a root, because it is the primitive that makes field
  concatenation unambiguous. And **`StableOrderKey`** (`streaming/identity.rs:88`-`:91`)
  is already this codebase's name for a "stable topology-independent tie-break key"
  — exactly the concept the global corpus ordinal needs, and exactly what the
  result index conspicuously does not use.

  Two further ceilings, both discovered in that plane and both reasons not to route
  the seal through it. Its byte budget tracks **one tokio semaphore permit per
  byte** and packs item and byte counters into a single `u64` as two `u32` halves
  (`streaming/budget.rs:195`, `:17`-`:18`, `:567`-`:572`, `:582`-`:584`), so a
  category tops out near 4.29 GB simultaneously charged and a W-way merge of 330 MB
  shards reaches `AccountingOverflow` before topology becomes the limit. And **no
  chunked write or ranged read exists anywhere**: the store's only write is
  `write_new(path, bytes: &[u8])` (`checkpoints/local.rs:140`), which copies the
  whole buffer before handing it to the blocking executor (`:404`-`:410`), so a
  330 MB publish peaks near 660 MB resident.

  Two ceilings bound the design at target scale and neither is negotiable from
  inside AIPerf. The Kubernetes publication path hard-caps a single artifact at
  512 MiB and fails the **entire** results manifest on breach
  (`rust/cli/src/k8s.rs:208`-`:216`, `rust/cli/src/results_sidecar.rs:51`,
  `:554`-`:558`), so a ~330 MB index fits with about 55% headroom and a 2×
  scale-up does not. And `AIPERF_CELL_ARTIFACT_UPLOAD_TIMEOUT` defaults to 300
  seconds for *all* cells combined
  (`rust/runtime/src/engine/cellular_controller.rs:1963`-`:1968`); every artifact
  shipped over that hop today is line-oriented JSONL that zstd crushes, and
  incompressible f32 at N × 330 MB has no precedent under that deadline. The
  cell→controller hop also carries **no digest and no declared length** —
  `received_bytes` is log-only (`artifact_shipping.rs:1682`, `:1721`-`:1735`) —
  so the per-part `{length, blake3}` contract is ours to add. Finally,
  `ArtifactSpec` is a flat list of named `Option<PathBuf>` fields
  (`rust/runtime/src/engine/protocol.rs:257`-`:302`) and `shippable_relatives`
  hardcodes them (`artifact_shipping.rs:2198`-`:2221`): a new artifact needs an
  entry in both or every upload is rejected by the allowlist, and a multi-block
  artifact family has no representation there at all.

**`rag_documents_per_second` is not derivable from the record plane as it
stands.** One batched request is one record and a document spans several batches,
so the rate needs a document identity and a final-batch marker on the record — a
new catalog tag plus record-plane plumbing, not a derived aggregate over existing
columns. The spec's earlier framing of this as "a derived aggregate" was wrong.

What batch-within-document containment (I15) buys is that this is the *whole*
requirement: a document completes when its final batch reaches terminal, and no
general many-to-many completion notion is needed. An earlier draft asserted the
opposite containment and would have needed one.

Parse and chunk time are reported as named client-side stage timings, scoped to
the document task rather than to any one request. The half of that claim that says
they are *not folded into request latency* is already true for free (`on_admit`
fires after materialization, and the transport origin is later still —
`transport/http/client/http_client.rs:434`-`:437`). The half that says they are
*measured* is new machinery: a timer around the worker-side stages, new catalog
tags, record-plane fields, and exporter projections. `CreditToStartLatency`
(`rust/runtime/src/metrics_core/store.rs:1675`) contains materialization time but
also queueing, so it cannot attribute a slow parser.

### The vector index seam

A new registry category, `VectorIndex`, with a two-trait split so the read path is
`!Send` worker-local and the write path is owned by the ingestion sealer.

**It is a runtime-owned `AIPerfRegistry` category in `aiperf-runtime` — the same
`TransactionalRegistry` shape as `RegisteredTraceProgramDrivers::stock()` — and
deliberately not a plugin category.** `PluginCategory` is sealed at three
(`Endpoint`, `Transport`, `Exporter`) with the seal stated in the code
(`rust/plugin-api/src/validation.rs:59`-`:79`) and normatively in the design record
(`docs/specs/2026-08-26-native-rust-runtime-plugins-design.md:1437`-`:1440`:
"adding a fourth dynamic category requires a new reviewed API generation"), whose
`:1434` puts everything else — including native-graph factories — in the
"host-owned or statically composed" bucket this belongs to. The point is not
deference to a rule: nothing could ship out-of-tree today regardless, because
`rust/plugin-host/` has no loader, `plugin-sdk-macros` has no entry macro, and
`PluginRegistrar` (`rust/plugin-api/src/extension.rs:97`-`:165`) has no
registration methods at all. The two-trait split below is nonetheless shaped so it
*could* become a plugin category at a later reviewed API generation; that is a
property of the design, not a claim about its present status.

**`PassageVector` and `PassageHit` stay out of `aiperf-core` and
`aiperf-plugin-api`.** `cargo xtask abi-gate` fails on any new `(name, file)` pair
reachable from a seed (`rust/xtask/src/abi_closure.rs:100`-`:114`) and
`abi-impl-budget` caps `MAX_ABI_TYPES = 177` / `MAX_ABI_FILES = 56`
(`rust/xtask/src/abi_impl_budget.rs:12`-`:17`) against a baseline already at
177/56 — zero headroom. The same ceiling governs the record plane: RAG record
fields are **scalar fields added to existing types**, never new nominal types hung
off `RecordIngest`, `Request`, or `PreparedTurn`, because a field addition is
invisible to the gate and a new type is not re-baselineable without breaking the
budget.

**`ArtifactWrite`, named in the `seal` signature below, does not exist in the
tree.** Zero hits across `core/src` and `runtime/src`. The boundary-side
`ArtifactAccess` (`rust/core/src/artifact.rs:120`-`:133`) offers only whole-buffer
`read`/`create`/`append`, and routing a 330 MB seal through it materializes the
whole index in one `Vec<u8>`. The chunked write seam is new work, and it belongs on
the in-tree `RecordArtifactLane` path this record already anchors the seal against,
not on `ArtifactAccess`.

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

- `flat` (**default**): exact inner-product/cosine scan over an `f32` matrix. The
  arithmetic in an earlier draft of this record was wrong and its conclusion did
  not follow. At the MLPerf corpus shape (~107k passages × 768 dims) a scan is
  82.2 million multiply-accumulate **pairs**, which is ~164 MFLOP under the
  standard convention, not 82. More importantly the FLOP count is the wrong figure
  of merit: the kernel has arithmetic intensity of about 0.5 FLOP per byte, so it
  is memory-bound, and every query streams the entire 329 MB matrix. Per-query
  latency is bandwidth ÷ 329 MB, and — the part the earlier framing missed
  entirely — **concurrent queries share that bandwidth**, so aggregate retrieval
  throughput has a hard ceiling of roughly (total memory bandwidth) ÷ 329 MB
  queries per second no matter how many cores the harness has. At a few hundred
  GB/s that is order hundreds of queries per second, and a multi-hop task issues
  several. No measurement backs any of these numbers, and the earlier
  "single-digit milliseconds, negligible" framing asserted a conclusion in a regime
  it had not entered.

  `flat` stays the default, and the reason is measurement correctness rather than
  cost: exact retrieval removes approximate-search recall as a confound from every
  scored number, so a QnA result difference is attributable to the system under
  test rather than to the harness's index. The cost is bounded by a **decision gate
  with a measurable trigger**: T7 measures single-query latency and the aggregate
  bandwidth ceiling on the target host before the default is locked. If the ceiling
  binds below the run's target task rate, `hnsw` becomes the default for scored QnA
  runs and `flat` is retained as the exactness reference the index-integrity check
  runs against. That decision is made from the measurement, not from this record.
  It is *exact*, which removes approximate-search recall from the
  list of things that can silently differ between two submissions. The gate
  settles the AIPerf-internal default only: the MLPerf reference has no flat path
  at all, so a run asserting reference comparability resolves to `hnsw` with the
  reference parameters or declares the divergence under I24. See
  `## Compared to the MLPerf reference implementation`. The kernel is
  hand-written `f32` over slices; no linear-algebra dependency is available or
  needed. `rayon` is available if the scan needs parallelism. Memory-mapping
  requires adding `memmap2`; a plain read is the fallback if that dependency is
  refused.
- `hnsw`: approximate graph index for corpora where the exact scan stops being
  negligible against the model calls, and the only cross-harness-comparable
  configuration. Parameters are recorded in the manifest and the report; a run
  using it is flagged approximate. The reference profile is `M = 32`,
  `efConstruction = 200`, `efSearch = 100` over an L2 metric on normalized
  vectors (`retrieve/vectordb.py:263`-`:266`, `:201`), which is order-equivalent
  to cosine. This needs a new dependency
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
accounting. It must be a node.

**It is an `Llm` node bound to a retrieval endpoint profile, not a third
`ExecutableGraphNode` variant.** An earlier draft of this record specified the
third variant and priced its landing surface honestly at roughly twice the
`ToolNode` precedent. The price was right and the choice was still wrong, because
the capability it was buying already exists.

The per-node endpoint selector is live on the benchmark path. `dispatch` reads
`metadata["endpoint"]` from the node
(`rust/runtime/src/engine/graph_execution.rs:2224`) and `materialize` resolves it
to a distinct `ValidatedEndpointProfileV2` — its own dialect, URL set, streaming
flag, and `input_token_counter` — falling back to the default profile only when
the node names none (`:1030`-`:1045`). Node selectors are validated up front
against the authored profile table
(`rust/runtime/src/engine/online_execution.rs:1868`-`:1888`), and the selector is
authorable end to end: `dag_jsonl` turns carry a per-turn `endpoint` field that
lowers into node metadata (`rust/runtime/src/graph/dag_source.rs:59`-`:60`,
`:264`; `rust/runtime/src/graph/lowering.rs:613`-`:619`). Retrieval-shaped
dialects are already registered — `nim_rankings`, `cohere_rankings`,
`hf_tei_rankings`, `image_retrieval`, `solido_rag`, `embeddings`
(`rust/runtime/src/endpoints/registry.rs:749`-`:760`).

A node authored that way is a first-class scheduled step **today**: input gating,
`min_start_delay_us`, splice successors, a `CapturedRecord`, a
`ReplayCallMeasurement`, correct `llm_node_count` budget accounting, cellular
fold, and dry-run support — all for zero graph-plane edits. `has_token_output()`
is correctly `false` for these response kinds, so a retrieval record is a
legitimate latency-only record and does not pollute OSL or ITL. Records separate
in the output plane by `InferenceDimensions.endpoint_url`
(`rust/runtime/src/metrics_core/ingest.rs:22`-`:27`) and by
`correlation_id = "{trace_id}:{node_id}"`.

What the third variant would have cost, and what it would have broken:

- **It buys nothing the `Llm` path does not already have.** Dispatch, records,
  scheduling anchors, and budget accounting all come free on the existing path.
- **`rust/runtime/src/graph/snapshot.rs:34`-`:40` would silently delete it.**
  `has_tool_node` is `matches!(node, Tool(_))`, so a third variant does not trip
  it; the three guards at `:61`, `:233`, and `:448` then take the LLM-only path,
  and `rewrite_for_warmup` (`:457`-`:462`) rebuilds the node set as `Llm` only.
  Retrieval nodes vanish from the warmup graph and the t\* chop drops them from
  the profiling graph. No error, no receipt.
- **Roughly fourteen further sites read through `as_llm()` or `matches!`** and
  would return zero, mislabel, or refuse with the wrong kind: `llm_node_count`
  under-counts the request budget (`rust/runtime/src/graph/workload.rs:181`),
  `inspect.rs:440` reports it as a *tool* node in `aiperf graph explain` JSON,
  `validate.rs:157`-`:165` never range-validates its timing gate,
  `executor.rs:658`-`:665` ignores its firing anchor, `scheduler.rs:275`-`:282`
  skips it in leading-offset collapse, and `execution.rs:245`-`:255` — written as
  a `Tool`-only denylist — fails *open*.
- **The `ToolNode` precedent was 37 files, +1387/−481 (`33780ead7f`)**, for a node
  that does not dispatch, has no inputs, no anchor, no record, and no metrics
  identity. A dispatching variant is 30–40 files, plus a `GraphNodeKindReport`
  change to a serialized public artifact contract, a second variant on
  `conditional_graph`'s separate `AuthoredNode` enum, and a new bucket in
  `TraceTerminalSupplement`'s two-vector fold vocabulary.

Modelling retrieval as a `Tool` node is not merely expensive but structurally
impossible: `ToolNode` (`rust/runtime/src/graph/model.rs:239`-`:248`) has only
`output`, `commands`, and `timeout_ns`; its `read_channels()` returns
`Vec::new()` and `input_requirements()` returns `&[]` (`:298`-`:317`), so **it
cannot consume the upstream query**, which is the entire step. It also gets no
native request record (`sink.rs:45`-`:51`), so retrieval latency would land only
as a `ToolCallMeasurement` in the replay artifact and never in the metrics plane.

Where the `Llm`-node approach genuinely breaks is the reply payload, and that is
one defect shared with embed and rerank rather than a retrieval-specific one — see
I7. The identity of that defect across all three steps is the reason it is a
Phase 1 prerequisite in the plan rather than a Phase 3 detail.

Two secondary consequences follow from the choice and are not free:

- **Distinguishing a retrieval step in inspection and reporting** wants a `kind`
  discriminator field on `LlmNode` rather than a new enum variant. Adding it
  there keeps every match arm, node count, budget, snapshot transform, flat-graph
  gate, and scheduler anchor correct by construction, while giving `aiperf graph
  explain` a real place to name the step.
- **The in-process worker-local index is a transport, not a node kind.** The
  non-eval constructor maps every profile to the same
  `Arc<dyn NativeTransportExecution>` (`graph_execution.rs:366`-`:374`); the
  heterogeneous per-profile `transports` map exists at `:387` but is supplied only
  by the eval caller. Serving a worker-local HNSW index means writing a new
  transport under either design, so it does not argue for the variant. Threading
  the existing per-profile map through the profile path is the smaller half of
  that work.

Revisit the third variant only if retrieval comes to need a channel payload the
`Llm` path cannot express. Note the shape of that constraint: `GraphReply<M>` is
generic over `WireMessage` and the engine sink is `GraphSink<OpenAiChatMessage>`,
so passages ride the channel encoded as assistant content.

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
  generated output length for any role, which remains stochastic. Reproducibility
  of tasks-per-second therefore rests on fixed work, not fixed outputs. The
  compliance check below does **not** close that gap — it covers one aggregate over
  the answer role, while the rewriter, grader, and sufficiency outputs are equally
  unbounded and equally move service time. What closes the reporting side is I20:
  a pinned run emits per-role mean and p90 OSL beside every scored rate, so
  run-to-run rate drift is attributable rather than mysterious. Where the endpoint
  supports it, the recorded `max_tokens`/`ignore_eos` are replayed to narrow the
  generation variance itself, which is a mitigation, not a fix.

### Live branching: the driver seam is a stepping stone

The staged driver lets the multi-hop loop ship without reactive machinery in the
flat core. It is not the end state.

The cost of staging is real, measured, and understated by an earlier draft of
this record in both directions. A stage is a full barrier: within one hop every
grade must complete before the driver observes the verdict, even though a
production RAG system can begin the sufficiency check as soon as enough passages
are graded and can abandon in-flight grading once the answer is decided.

The barrier is confirmed at the level that matters — `execute_static_trace_result`
ends at `handle.wait_idle().await` (`graph/execution.rs:226`), which waits on a
per-trace inflight counter (`graph/runtime.rs:112`-`:117`, decremented in
`InflightTask::drop`, `:67`-`:75`) including stragglers nobody consumed — so a
hop costs max-over-stage, not critical path. At five hops and fan-out three that
is roughly 1300 ms staged against 1067 ms pipelined and 900 ms with speculative
overlap; at fixed concurrency it deflates `rag_tasks_per_second` by something like
18–31%.

Four corrections to that statement, two mitigating and two aggravating:

- **The barrier is per-trace, not per-worker.** `placement.rs:369`-`:409`'s
  `worker_loop` spawns each `Execute` into a `JoinSet` at `:398` and returns
  immediately to `commands.recv()` at `:394`; the single-reactor path spawns at
  `graph_phase_runtime.rs:791`. **Stage barriers cost per-task latency, not
  aggregate worker throughput** — offered load is preserved as long as resident
  traces cover the gaps, and the deflation above bites at fixed concurrency, not
  open loop. An earlier draft implied the cost was unqualified.
- **"Max over fan-out" is a property of the stage boundary, not of the executor.**
  Nodes within a stage already overlap (`executor.rs:215`-`:236` `schedule()`
  spawning at `:233`), and `Count::N(k)` joins (`graph/model.rs:54`-`:57`,
  `channel_store.rs:207`-`:233`) already let a node fire on the first *k*
  arrivals — so intra-stage fan-in can be *min*-over-fan-out today. Modelling a
  hop as a stage forfeits an optimization the executor already supports. If the
  hop body can be expressed as one graph with a k-of-N join rather than N stages,
  most of the deflation disappears with no new runtime mechanism.
- **Per-stage marshalling is O(hops²), and was not costed at all.**
  `freeze_terminal_outputs` (`graph_execution.rs:1753`-`:1797`) calls
  `SegmentPool::thaw` at `:1774`, a full deep copy of every segment in the store
  (`dataset/segment.rs:317`-`:331`), and each stage's frozen store becomes the
  next stage's base (`:1874`-`:1883`). Retrieved passages are precisely what
  inflates that store, so the quadratic lands on the one workload that grows it.
  Each barrier also serializes every channel to JSON (`:1888`-`:1899`).
  Both are synchronous on the reactor. One escape hatch exists:
  `freeze_terminal_outputs` early-returns before `thaw` when no declared terminal
  channel holds a concrete value (`:1766`-`:1771`), so a driver that declares no
  terminal outputs mid-run avoids the copy entirely.
- **"No early termination" is structural, not a backlog item.** `Abortable` at
  `graph_execution.rs:1837` and `:1899` wraps only the driver's own futures, the
  `TraceProgramDriver` trait (`driver.rs:512`-`:576`) has no in-stage callback,
  and `set_abort` (`context.rs:105`-`:115`) is whole-trace. Grader-driven
  cancellation of sibling nodes needs a new trait method *and* per-node abort
  granularity — it cannot be added later as a driver-side optimization.

Per-stage there is also a fresh executor, context, and channel store, and
`EngineGraphSink::configure_stage` clears the prepared-metadata cache every stage
(`graph_execution.rs:2448`-`:2459`), so per-node metadata is re-parsed on every
hop: six times the setup a static trace pays once.

**`next_stage` and `observe_stage` run on the shared `LocalSet`.** The worker loop
spawns each trace and never awaits one, so synchronous CPU work inside a driver
body stalls every co-resident trace on that core. For `rag_qna` the driver bodies
are cheap. For `rag_ingest` they are not — HTML parsing and chunking are exactly
the kind of work that would inflate the measured latency of unrelated traces
sharing the worker, which is a measurement-correctness defect and not a
throughput note. The ingestion driver must therefore yield the reactor across its
parse and chunk stages rather than running them straight through, and I12's claim
that parse and chunk are *measured* stages carries the additional obligation that
measuring them does not perturb what else is being measured.

The mechanism for that already exists and should not be hand-rolled:
`StreamingBlockingExecutor` and `BudgetedBlockingOutput`
(`runtime/src/streaming/blocking.rs`) are a budgeted seam for moving blocking work
off the reactor, built for exactly this hazard on the streaming decode path. Route
parse and chunk through it. The corresponding assertion is behavioral rather than
structural: a corpus of deliberately slow-to-parse documents must move
`rag_documents_per_second` (I12's positive half) while leaving the *request*
latency of co-resident embed nodes unchanged (this obligation).

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

**The reasoning-type label.** The reference benchmark's query set partitions
into reasoning types (numerical, tabular, multi-constraint, post-processing,
and so on), and the interesting reading of a RAG result is per-type, not
aggregate: a system can hold aggregate accuracy while collapsing on one type.
This is the one gap in this record that gets structurally *harder* the longer it
waits, because the per-record schemas are fixed-column and the ABI budget is at
its cap, so it is specified here rather than deferred.

There is **no key-value metadata carrier anywhere in the dataset to record to
export path**. `tags`, `labels`, `custom_fields`, `annotations`, `attributes`,
and `user_data` do not exist under any spelling. `RequestMetricMetadata`
(`rust/runtime/src/metrics.rs:33`-`:63`) is the sole issue-time-to-record bridge and
is a fixed eleven-field struct; `RecordIngest`
(`rust/runtime/src/metrics_core/ingest.rs:136`-`:221`) is likewise closed;
`metric_overrides` (`ingest.rs:213`) is `f64`-only and cannot carry a string.
`Turn.extra_body`/`extra_headers`/`request_parameters` go on the wire, not into
the record. So this is an addition, and there are exactly three shapes it can
take, in ascending cost.

*The free breakdown.* `AccuracyAssociation { correlation_id, task }`
(`rust/runtime/src/dataset/model.rs:207`-`:213`) already flows
`Conversation.accuracy` to `MaterializedTurn.accuracy`
(`dataset/request.rs:70`) to `multiturn.rs:1112`-`:1115` to `ProblemAssociation.task`
(`accuracy.rs:178`) to `CapturedResponse.task` (`:501`) to
`AccuracyRecord.task` (`metrics_core/accuracy.rs:122`), and per-task rollups
already exist (`AccuracySummary.per_task`, `accuracy.rs:181`) and are already
exported (`export/accuracy_csv.rs:65`-`:69`). Authoring `task` = the reasoning
type yields accuracy-by-reasoning-type with **zero schema change**. That is the
right first move, and for the accuracy question it may be the only move needed.
Three constraints ride with it: `task` is single-valued; the evaluator must echo
it byte-identically or grading fails hard (`accuracy.rs:687`-`:691`); and
`AccuracyEvaluation.records` -- the per-request grades -- is serialized by no
exporter in the tree, so there is no per-record accuracy artifact today.

*The zero-file hack, named so it is not mistaken for the design.* `conversation_id`
is the only authored string that survives into all four per-record formats
(`RecordMetadata.conversation_id`, `rust/runtime/src/engine/records.rs:117`;
`RawRecordMetadata`, `:311`; `CSV_METADATA_COLUMNS`, `:517`;
`PerRecordRow`, `export/per_record_parquet.rs:91`). Folding the label into the
authored `session_id` therefore works with no code change at all. It also
pollutes the identity column and forces every consumer to string-split. It is
recorded as a fallback for an exploratory run, not as this design's answer.

*The column.* One `Option<Arc<str>>` on `RequestMetricMetadata`, materialized to
`Option<String>` at the `into_record` boundary exactly as `worker_id` already is
(`metrics.rs:44`-`:51` states the per-request-clone rationale), then
`Option<String>` on `RecordIngest`. This is roughly eleven production files --
`dataset/model.rs`, the authoring loader, `dataset/request.rs`,
`multiturn/model.rs:161` with both construction sites (`multiturn.rs:911`, `:1117`),
`engine/execute/capture.rs:350`-`:358`, `metrics.rs` (field, `Default` at `:65`,
`into_record` at `:528`), `metrics_core/ingest.rs` (field and `minimal()` at `:223`),
`engine/records.rs` in seven places including the paired
`CSV_METADATA_COLUMNS:513` const and the positional `record_csv_row:598` push,
`export/per_record_parquet.rs:88`, and on the graph path one
`metadata_string(&node, ...)` read at `graph_execution.rs:2261` -- plus roughly six
test fixtures, because `RecordIngest` derives no `Default` and every struct
literal must be updated to compile.

**It must be a scalar field, not a new type.** `MAX_ABI_TYPES = 177` /
`MAX_ABI_FILES = 56` (`rust/xtask/src/abi_impl_budget.rs:14`-`:16`) with the
baseline at exactly 177/56, and `ensure_no_growth`
(`rust/xtask/src/abi_closure.rs:100`-`:120`) rejects any new `(name, file)` pair
even at constant count. A `ReasoningType` enum or a `QueryLabels` struct
reachable from `RecordIngest` or `Request` fails the gate and cannot be
re-baselined. `Option<String>` introduces no nominal type and the carrier structs
are already in the closure, so the field addition is invisible to it; adding
fields raises `type_lines` and therefore *improves* `MAX_BOUNDARY_IMPL_RATIO`.
For the same reason a `HashMap<String, String>` bag is refused independently of
the gate: it puts an allocating map on the per-request clone path, and the CSV
and Parquet writers have fixed column lists a map cannot populate without a
schema-discovery pass.

**Multi-valued labels are canonicalized to one string, not modeled as a list.**
A query carrying two reasoning types is recorded as a sorted, `;`-joined single
value, keeping the Parquet column `Utf8` rather than introducing a
`List<Utf8>` builder and a `SCHEMA_VERSION` break
(`per_record_parquet.rs:58`). The separator is `;` or `|` and never `,`:
`csv_escape` (`engine/records.rs:552`-`:558`) would quote a comma-joined value
correctly, but naive splitters mangle the result. And the joined value must not
be fanned out into multiple accuracy buckets: `compute_results_for_context`
(`metrics_core/accuracy.rs:400`-`:431`) pushes each record into `overall` once and
into `per_task[task]` once, so a two-bucket record makes
`sum(per_task.n) != overall.n` and reads as a bug in `accuracy_results.csv`. A
composite bucket (`numerical+tabular`) is a legitimate additional bucket and
preserves the identity.

The graph path preserves almost nothing authored per-node into the record, which
is why the label rides the conversation rather than the node. `dag_jsonl` has no
metadata field at all -- `DagJsonlTurn` (`graph/dag_source.rs:54`-`:80`) is closed,
so an authored label dies at parse rather than at lowering.
`LlmNode.metadata` (`graph/model.rs:168`) *is* a free-form map that survives
lowering, but only six keys are ever read out (`"model"`, `"endpoint"`,
`"input_tokens"`, the two recorded-timing keys, `"turn_index"`), and nothing
generic is projected into `RequestMetricMetadata`. What the graph path does stamp
is `conversation_id = trace_id` and `correlation_id = "{trace_id}:{node_id}"`
(`graph_execution.rs:2270`-`:2278`).

**The Offline mapping, stated with what it is not.** Server scenario is out of
scope for this record, exactly as for the first MLPerf instantiation. Offline is
approximated by the existing concurrency workload with
`type: concurrency`, `concurrency == requests == N`, one turn, and no ramp.

That mapping is exact for `rag_ingest` and is *not* what the reference does for
QnA: the reference's scored QnA workload admits 10 concurrent queries against 824,
and declares it Offline anyway (`user.conf`, `reference_mlperf_perf.sh:31`-`:32`).
So `rag_qna` pins the admission bound explicitly, records it in
`pipeline_digest`, and reports a bounded-concurrency completion rate rather than
an unqualified Offline throughput. The four imperfections below apply to both, and
the full comparison is in
`## Compared to the MLPerf reference implementation`. The
approximation is good in the one place that matters most and imperfect in four
places that must be named, because each is a way for a run to look Offline and
not be.

It is good on admission: `PhaseKind::Concurrency` lowers to
`ArrivalPattern::ConcurrencyBurst` (`rust/runtime/src/engine/protocol.rs:880`-`:883`)
whose `next_interval_ns()` returns literal zero
(`rust/runtime/src/timing/intervals.rs:170`-`:172`), so the issuer never sleeps and
the only gate is a non-blocking `SlotPool::try_acquire` sized to `concurrency`
(`rust/runtime/src/phase_runtime.rs:299`-`:308`), which cannot bind at N-of-N. All
N are admissible from t=0 in policy. The workload named "concurrency" does no
rate metering at all, which is the opposite of the intuitive reading.

It is not equivalent in these ways. **Issuance is serial**, one request per
issuer-loop iteration (`rust/runtime/src/request_rate.rs:654`-`:663`), not a single
handoff of the whole set. **The draw is lazy**: one sample is cached before start
and the next is drawn only after a successful issue (`:519`-`:525`), so the system
under test never sees the query set as a set and cannot reorder across it — which
is precisely the freedom LoadGen's Offline scenario is designed to grant.
**Multi-turn corrupts it**: continuations take FIFO priority over new sessions
(`:640`-`:648`) and can block on prefill capacity (`:437`-`:441`), so the property
holds only for the single-turn shape. **`concurrency_ramp`**
(`config/model/phase.rs:117`-`:118`) converts admission into genuinely metered and
destroys the property outright, and seamless warmup carries session guards across
the phase boundary (`engine/execute/plan.rs:29`-`:34`), so profiling does not start
cold-empty.

Two derived readings must not be taken from such a run. `concurrency < N`
silently produces a Server-shaped closed loop with **no diagnostic**, so the
equality is load-bearing and is checked rather than assumed. And **schedule
adherence and queue delay are meaningless here**: `bounded_reanchor_target`
re-anchors the target to `now` once lag exceeds the catch-up window, default 10 ms
(`request_rate.rs:41`-`:43`, `timing/arrival.rs:88`-`:95`), so after roughly the
first 10 ms the schedule tracks the wall clock and derived lateness collapses to
approximately zero by construction. The honest Offline reading of such a run is
its completion rate, not its arrival statistics.

The one workload that genuinely pre-schedules the entire set with no admission
gate is `fixed_schedule`, which schedules every entry up front in one pass
(`rust/runtime/src/fixed_schedule.rs:204`-`:241`) and reports `concurrency() == None`
(`engine/execute/dataset_build.rs:271`-`:274`). It is also the only workload whose
arrival time is an authored fact rather than a dispatch observation — everywhere
else `arrival_ms` is stamped at coordinator dispatch as `clock.now_ns() - origin_ns`
(`engine/execute/capture.rs:366`), which is why `fixed_schedule` correspondingly
reports `has_credit_timestamps() == false` (`fixed_schedule.rs:183`-`:185`). An
all-equal-timestamp fixed schedule is therefore the more literal Offline analogue,
and is recorded here as the alternative to reach for if the burst approximation
proves insufficient.

### Measurement correctness

The normative statements are I12-I18. This section records only what those
invariants do not say: why a RAG task makes them non-trivial, and where the
seams are.

A RAG task is many requests deep, so quantities that coincide in a flat workload
come apart. One task spans a dozen model calls across four endpoints, holds
several concurrent requests during a fan-out hop and none between stages, and
mixes client-side work (parse, chunk, retrieval, driver) with wire time. Each of
I12-I18 marks a place where the obvious reuse of an existing single-request
mechanism produces a number that is plausible, reportable, and wrong.

Three of them are pre-existing defects rather than new work, and each is a bug in
the tree today independent of whether this design proceeds:

- I14 is live now. Any run with more than one model or endpoint loses TTFT, ITL,
  latency, ISL, and OSL from five exporters.
- I16 is a missing precondition: the per-record row carries no profile, model, or
  node identity, so no per-role number can be substantiated.
- I13 affects every existing graph workload, since the concurrency sweep-line
  counts requests and a fan-out graph's request curve is not its task curve.

They therefore lead the delivery order, ahead of any RAG capability.

The remaining two are new seams rather than fixes. I12's task-level origin and
terminal do not exist yet, and adding them must not disturb the request-level
origin, which is already correct for free. I18 is the discipline that would have
caught I5 and I7 at introduction: assert the aggregate against the raw records,
never against another aggregate.

### The stock configuration

The benchmark is one named thing or it is a pile of flags that each reader
assembles slightly differently, so `rag_e2e` ships as a stock config template
pinning the corpus source, chunker parameters, embedding and rerank and answer
profiles, index kind and parameters, hop and fan-out and `k` bounds, phase shape,
and artifact set — that is, exactly the closure `pipeline_digest` covers, which
is what makes I24's reference profile a shipped artifact rather than prose.

The template surface is **a compile-time frozen array, not a directory scan**:
`pub const TEMPLATES: &[Template]` (`rust/cli/src/config/templates_data.rs:5`),
28 entries, each with hand-written metadata and `content: include_str!(...)`.
Twenty-seven include out of the Python tree (`src/aiperf/config/templates/*.yaml`)
and one out of the native-only `rust/cli/templates/`. So the Python YAML *files*
are live — they are the compiled-in bytes — while the Python *discovery*
mechanism is dead: `src/aiperf/config/templates/discovery.py:1`-`:40` parses a
`# @template` sentinel block that nothing in `rust/` reads, and the metadata is
duplicated by hand into `templates_data.rs`. Lookup is exact-match; an unknown
name prints and exits 1 (`rust/cli/src/config/mod.rs:95`-`:98`). `init` rewrites
only `model` and `url` and strips the SPDX header (`:100`-`:105`, `:173`-`:189`).

Two consequences follow, and both are obligations on this work rather than
observations about it. **Adding the YAML alone does nothing** — the entry must be
hand-added to `templates_data.rs`, and the only existing test is
`assert!(TEMPLATES.len() >= 20)` (`config/mod.rs:235`), so there is no
directory-to-array parity guard and no check that the hand-copied metadata matches
the `# @template` block. This design adds that parity test rather than relying on
the discipline that currently holds the count at 28.

**And `config init` never validates its own output.** `config validate` is a
separate opt-in command, so a template can ship, be listed, be emitted, and fail
at run time. This is not hypothetical: the shipped `speed_bench_sweep.yaml`
sweeps `datasets.main.format` over `speed_bench_coding`, `speed_bench_rag`, and
siblings (`src/aiperf/config/templates/speed_bench_sweep.yaml:52`-`:63`, `:84`),
while the registered format id is the singular `speed_bench`
(`rust/runtime/src/dataset/loader/mod.rs:424`-`:427`) with category as a loader
*option* (`public.rs:890`), and `DatasetFormatRegistry::get` is exact-match
returning `LoaderNotFound` otherwise (`loader/mod.rs:491`-`:498`). A stock RAG
config that ships broken in the same way would be worse than no stock config, so
`rag_e2e` is covered by a test that emits it and runs `config validate` over the
emitted bytes.

Precedent for pinning a whole benchmark as data exists twice. The named stock
dataset catalog (`rust/runtime/resources/public_datasets.yaml`, loaded into a
`BTreeMap<String, PublicMeta>` at
`rust/runtime/src/config/model/public_catalog.rs:27`-`:30`) pins loader format,
source repo, subset, split, and revision per name — including a raw URL pinned to
a git SHA (`public_datasets.yaml:326`-`:333`) — and is the closest existing shape
to a pinned benchmark definition. SPEED-Bench is the only existing end-to-end
named benchmark: a stock sweep template plus a dedicated native report command
that reads `input_config` back out of each run's JSON
(`rust/cli/src/speed_bench.rs:69`, `:88`). It pins one model rather than a model
set, which is exactly the axis `rag_e2e` extends.

Adding the stock config touches: the template YAML, `templates_data.rs`, the
parity test, `public_datasets.yaml` if the corpus is a new stock dataset,
`register_builtin_formats` (`dataset/loader/mod.rs:379`) for the ingestion format,
`src/aiperf/config/schema/aiperf-config.schema.json` for any new enum value, and
then `llms.txt` (which states the embedded-template count at `:177`),
`docs/specs/README.md`, and both `tools/check_agent_files_sync.py` and
`tools/check_docs_current.py`.

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
  scored metric, and is reported as such.

  The identities are carried on the **passage-set graph channel** the retrieval
  node writes — the same channel `rerank` and `grade` already read, so no second
  source of truth exists — accumulated per task and sealed into a per-run
  retrieval-evidence artifact using the content-addressed shape T7 already reuses
  for the index (`rust/runtime/src/eval/native_graph/artifacts.rs:64`-`:84`,
  `:135`-`:147`, minus its quota model). This is uniform across local `flat`/`hnsw`
  hops and any future remote index, because it sits above the transport
  distinction. `ParsedResponse.sources` is explicitly **not** used: it is
  write-only and absent on the local path (see `## Built`).
- **Output-length compliance** (`aiperf rag compliance`) re-reads a pinned run's
  records and asserts the answer-role mean OSL falls inside an authored band around
  an authored reference (the MLPerf analogue is 273.81 tokens ±10%). The check is a
  single aggregate over one role. The report states that scope explicitly and names
  what it does not cover: the rewriter, grader, and sufficiency roles generate
  unbounded output that moves `rag_tasks_per_second` and is bounded by nothing here.
  Those roles are covered only by I20's per-role reporting. It depends on step 1 of the role work,
  since attributing OSL to the answer role requires the per-record profile id.

- **The validity gate.** A scored run is valid only if its accuracy reaches an
  authored fraction of an authored reference accuracy (the MLPerf analogue is
  99% of the reference, relaxed to 97% for the reasoning models in the first
  instantiation). `aiperf rag score` evaluates the gate, states the verdict in
  its output, and returns non-zero when the gate fails.

  **Nothing in AIPerf today turns a measured metric into a run verdict**, so this
  is genuinely new machinery and must not be mistaken for an extension of
  something existing. Every non-zero exit in the product is operational — bad
  flags, missing file, a run that did not complete
  (`rust/cli/src/main.rs:47`-`:54`; `rust/cli/src/validate.rs:174`, `:183`;
  `rust/cli/src/sweep/aggregate.rs:89`, `:111`, which counts runs that did not
  finish, not runs that scored badly). The accuracy plane has no reference value
  and no run threshold at all: `AccuracyRollup`
  (`rust/runtime/src/metrics_core/accuracy.rs:144`-`:158`) reports `n`,
  `correct_count`, `accuracy`, and a CI, and compares them to nothing;
  `LIGHTEVAL_CORRECTNESS_THRESHOLD` (`:20`) is the per-answer grader cut, not a
  run gate. `aiperf eval` ends in an unconditional `Ok(0)`
  (`rust/cli/src/eval.rs:226`, `:418`) with reward reported and never gated.

  Four near-misses are structurally similar and are **not** precedents, each for
  a specific reason worth recording so the gate is not built on one of them by
  mistake. Goodput SLOs (`accumulator.rs:1024`-`:1038`) evaluate authored
  thresholds per request, but the result is a 0/1 counter input — a run can
  attain 0% and still exit 0. `compare`'s `Verdict`
  (`rust/cli/src/compare.rs:195`-`:236`, `:265`) is a display column; a 90%
  regression prints "worse" and exits 0, and a missing result file returns
  `Ok(0)` (`:57`). `sla_breach_knee` (`rust/cli/src/search/sla_breach.rs:41`-`:56`)
  does real threshold arithmetic but is a post-process artifact writer whose
  *write failure* is downgraded to a warning (`rust/cli/src/profile.rs:623`-`:628`).
  The adaptive controller's `passed`/`sla_passed`
  (`rust/runtime/src/adaptive_core/controller.rs:105`,
  `adaptive_core/artifacts.rs:153`, `:434`) is the closest measured boolean in the
  codebase, but it is a control signal steering concurrency; failing to converge
  does not fail the run.

  What is reusable is the comparison arithmetic, which already exists three times
  over and should not be written a fourth: `passes_threshold`
  (`rust/runtime/src/metrics_core/definition.rs:101`) is the canonical
  direction-aware policy, with `Slo::passes` (`accumulator.rs:127`) and
  `SlaFilter::satisfied_by` (`rust/cli/src/search.rs:526`, which treats a missing
  observation as failure) as its two existing wrappers. What is new is exactly
  three things: an authored reference accuracy on the config surface, where no
  such field exists; a run-level assertion evaluated after the accuracy rollup;
  and a non-zero return plumbed through `dispatch::run` to `main.rs:47`. The last
  is mechanically cheap — every `run()` already returns `i32` — and its cost is
  entirely that no caller has ever used it for a measurement, so this becomes the
  first place in the product where a number can fail a run. It is therefore
  scoped to `aiperf rag score`, a post-run pass over recorded artifacts, and
  deliberately not to `aiperf profile`: a benchmark run that measured a system
  honestly should not exit non-zero because the system scored poorly.

Mock-server fixtures reuse the existing accuracy-fixture mechanism
(`rust/mock-server/src/accuracy.rs`, flags at `rust/mock-server/src/config.rs:753`-`:817`)
by adding a RAG format variant, rather than building a parallel canned-answer system.
Deterministic embeddings and rerank scores already exist as pure hash functions of the
request text, but the dimension is hardcoded at 768 and scores can only be *predicted*
by recomputing the hash, never *dictated*; a test that needs a specific ranking order
should add one score-by-index knob to `compute_mock_score` rather than fork it.

### Refusals

Every refusal below exists to prevent a **wrong measurement**, not to resist
misuse. Each names a configuration that would otherwise run to completion and
report a plausible number that does not mean what the report says it means, so
each fails before dispatch rather than degrading, and each names the invariant it
protects.

| Refused before dispatch | Protects |
|---|---|
| A QnA run whose index `corpus_digest` disagrees with its pinned plan or query set | I1 |
| An index artifact whose header dimension disagrees with the embed profile's returned vector width | I3 |
| A graph node bound to an endpoint kind whose request cannot be constructed from graph materialization | I5 |
| A `rag_qna` graph naming an unregistered endpoint profile | I6 |
| An authored hop, fan-out, or `k` bound that is zero or unbounded | I9 |
| A pinned plan whose hop count exceeds `stage_bound()` | I9, I19 |
| A pinned `rag_qna` run with a `cache_bust_target` other than `None` | I19 |
| An ingestion document whose chunk count exceeds the driver's `stage_bound()` | corpus completeness |
| A sufficiency verdict decoder that is absent or ambiguous | I11 |
| `--steady-state` on a `rag_qna` run, unless the task-level curve is available | I13 |
| `aiperf rag compliance` before per-record role attribution exists | I16 |
| `aiperf rag compliance` against a sketch-mode run | I17 |
| A scored `rag_qna` run under sketch mode (`--sketch-metrics`, `AIPERF_METRICS_SKETCH=1`) | I17, I20 |
| Retrieval integrity reported from a run that did not retain the retrieval-evidence artifact | I18 |
| A scored `rag_ingest` run with the parsed-corpus cache enabled | I12 |
| An ingestion format whose parse or chunk stage is implemented in a `DatasetLoader` or `Composer` | I12 |
| A staged `rag_*` run under a node failure policy that reports a failed hop as completed | I21 |
| A `rag_qna` loop whose verdict channel yields a value outside its declared verdict set | I11, I21 |
| A driver-authored stage plan that is not a projection of the authored source graph | I10, I22 |
| An index seal missing any worker or cell part, or whose passage ordinals are not a permutation of `0..N` | I1 |
| A sealed index exceeding the Kubernetes publication cap of 512 MiB, refused at seal rather than at publish | I1 |
| A run asserting reference-profile comparability whose resolved parameters diverge from that profile | I24 |
| A `rag_ingest` run declaring the Offline scenario whose phase is not single-turn with `concurrency == requests` and no ramp | Offline mapping |
| A `rag_qna` run reporting an Offline throughput without recording its admission bound | Offline mapping, I23 |
| `aiperf rag score --gate` without an authored reference accuracy | validity gate |
| A scored run that reports served-model provenance it did not observe | I25 |
| A scored run whose resolved plan omits any parameter `pipeline_digest` covers | I23 |
| A multi-endpoint `rag_qna` run over gRPC | transport limit (`grpc_execution.rs:130`) |

## Compared to the MLPerf reference implementation

The reference implementation is public: `mlcommons/inference`, directory
`e2e-rag/`, read at commit `cfb0df14b21a3898521891f021a1c6aadec2ab2c`
(2026-08-26), 61 files and roughly 15,000 lines of Python. Every claim in this
section is from that source, not from the announcement post, and every file
reference below is relative to `e2e-rag/`. Where the reference's own
documentation disagrees with its code, the code is what is recorded here and the
disagreement is noted, because a design that matches a stale doc matches nothing.

This section exists because the comparison changes the design in four places. It
is ordered by consequence: what the reference forces us to change, what it
confirms, what it validates, and what it binds less tightly than we do.

### What the reference forces us to change

**The index default is a comparability decision, not a performance one.** The
reference has exactly one index constructor — `faiss.IndexHNSWFlat(dimension, 32)`
with `efConstruction = 200` and `efSearch = 100`, all three hardcoded literals
(`retrieve/vectordb.py:263`-`:266`) — against `faiss-cpu` (`requirements.txt`).
There is no flat path and no IVF path in the code at all. The `--vector_index_method`
flag advertising `choices=["flat","hnsw","ivf"]`
(`reference_mlperf_datasetup.py:133`-`:138`) is forwarded into `VectorDB.__init__`,
absorbed by `**kwargs`, and never read (`retrieve/vectordb.py:152`-`:163`);
`measure_indexing_with_chunking.py:229` is honest and prints `HNSW (fixed)`. The
shipped manifest confirms the scored values: `IndexHNSWFlat`, dim 768,
`metric_type: 1`, efC 200, efS 100, M 32.

The consequence for `### The vector index seam` is that the T7 bandwidth gate
decides the right default for *AIPerf-internal A/B*, where exactness removes
approximate-recall as a confound, and decides nothing about *cross-harness*
comparability, where `hnsw` at M=32/efC=200/efS=100 is the only comparable
configuration. Those are two different questions and this record previously
conflated them. `flat` therefore remains the default and remains the exactness
reference, and a run asserting reference-profile comparability under I24 must
resolve to `hnsw` with those three parameters or declare the divergence. The
`faiss-cpu` fact also invalidates the bandwidth arithmetic's implicit
accelerator-memory framing: the reference's own retrieval is host-memory-bound,
so our ceiling estimate is the *right shape* against the wrong bandwidth number,
and T7 measures rather than assumes.

**The Offline mapping is per-workload, and the QnA half is not Offline-shaped.**
`user.conf` carries two LoadGen models with deliberately opposite admission:

```
e2e-rag-db.Offline.min_query_count  = 2515    max_async_queries = 2515
e2e-rag-qna.Offline.min_query_count = 824     max_async_queries = 10
```

Ingestion is genuine Offline — one query per frozen document, the whole set
admissible at once, which is exactly this record's `concurrency == requests == N`
mapping. QnA is a **concurrency-10 closed loop over 824 queries**, matched by a
SUT thread pool of the same size (`reference_mlperf_perf.sh:31`-`:32`,
`MAX_ASYNC_QUERIES=10` / `MAX_WORKERS=10`), and declared Offline anyway. By this
record's own language that is "a Server-shaped closed loop with no diagnostic" —
so the refusal row added for the Offline declaration would reject the reference
configuration as written.

The resolution is that the refusal is right and the *scope* was wrong. It applies
to `rag_ingest`, where the full-set mapping is exact. For `rag_qna` the honest
statement is that the reference number is a bounded-concurrency completion rate,
not an Offline throughput, and a comparable AIPerf run declares
`concurrency = 10` explicitly and reports it as such. `pipeline_digest` covers the
admission bound for exactly this reason: two runs at concurrency 10 and
concurrency 824 over the same corpus are not the same measurement, and nothing in
the reference's artifact set distinguishes them.

**The hop bound is load-bearing on the score, and that is measured rather than
argued.** `max_iterations` defaults to 10 in three signatures
(`multi_shot_retrieval.py:1186`, `reference_SUT.py:58`, `reference_mlperf.py:111`)
and is set to **5** by every submission script
(`reference_mlperf_perf.sh:41`, `config.template.sh:31`). At 5, **45.8% of the 824
queries hit the cap** (`CLAUDE.md:291`-`:300`). A run at 5 and a run at 10 share a
corpus bit-for-bit and are different benchmarks — which is the comparability
asymmetry this record identified, demonstrated on the reference's own numbers
rather than by construction. I23 keeps the hop bound inside `pipeline_digest`;
this is the evidence for why.

**Sub-query fan-out is prompt-enforced in the reference and code-enforced here.**
`max_sub_queries = 3` (`reference_mlperf_perf.sh:42`, `config.template.sh:32`)
reaches the model only as prompt text (`multi_shot_retrieval.py:126`); the returned
list is consumed without a clamp (`:1416`), under `temperature = 1.0`. I9 refuses
an unbounded fan-out, so we are stricter, and deliberately: an advisory bound
inside a sampled prompt is not a bound on the measured request count. The fallback
when query generation yields nothing is `[original_query]` (`:1474`-`:1477`), which
is the total-decoder behavior I11 requires, arrived at ad hoc.

### What the reference confirms

**Per-node endpoint-profile routing is the reference's literal topology.** Two
vLLM servers on distinct ports serve distinct models per role: `gpt-oss-20b-mxfp4`
on `:8192` grades document relevance at a hardcoded 4096 max tokens
(`multi_shot_retrieval.py:468`, `:479`-`:481`), and `gpt-oss-120b-mxfp4` on `:8123`
handles sufficiency (`:576`, `:600`-`:602`), query generation (`:799`,
`:812`-`:814`), and answering — where the answerer shares the *sufficiency*
endpoint and model rather than a separately configured one (`:707`, `:724`-`:726`).
The judge is a third model (`meta-llama/Llama-3.1-8B-Instruct`) run after the fact
and outside the SUT. So the `metadata["endpoint"]` per-node selector this record
bets on is not a generalization we invented; it is what the reference needs, and
the role→profile map that `pipeline_digest` binds is a map the reference also has
but records nowhere.

**The reference ships a tunable search space, which is the sharpest argument for
I23 and I24.** `params.py` carries `optuna_suggest` ranges on retrieval
parameters — `top_k_retriever` defaults to 10 and is tunable 5→100. A harness whose
own parameter file invites search over the pipeline cannot rely on convention to
keep two submissions comparable. `pipeline_digest` is the artifact that makes the
resulting divergence visible instead of silent.

**Ingestion is a scored phase with parse and chunk inside the timed window.**
`measure_indexing_with_chunking.py:387`-`:391` sums chunking + indexing + save and
reports `throughput_passages_per_second`; only post-hoc validation is excluded.
That is this record's `rag_ingest` window, including I12's insistence that parse
and chunk are measured stages rather than dataset-loader work. One divergence
worth keeping: the reference times with `time.time()` throughout
(`ingestion_monitor.py:99`, `:115`, `:141`, `:204`, `:250`), so its KPI is
vulnerable to a wall-clock step; AIPerf routes all measurement through `Clock` and
derives UTC from a single anchor, so ours is not.

### What the reference validates

**The reference contains a live instance of the failure I21 exists to prevent.**
If `kept_docs` is still empty at the final iteration, the sufficiency call is
skipped, no answer is generated, and the SUT reports an empty string that falls
through to `max(1, len(answer.split()))` — a **1-token** result
(`reference_SUT.py:318`-`:320`). A query that retrieved nothing contributes a
one-token success to the scored tokens/sec figure. This is exactly a failed hop
reported as a completed one, and it is in the reference, not hypothetical.

**Verdict totality is achieved by forcing one direction, which I11 permits only if
the direction is authored.** `check_sufficiency` returns `sufficient = True` on the
final iteration (`multi_shot_retrieval.py:683`-`:686`, marked `[OVERRIDE]`), on
empty output (`:655`), on missing JSON (`:673`), and on exception (`:701`). The
decoder is total, which I11 requires, but the default is "terminate and answer" and
it is implicit at four separate sites. Our verdict-set declaration makes the same
choice explicit and one-place, which is the difference between a design and a
convergent accident.

**The "mechanism exists, is cited correctly, and is inert exactly where it is
needed" pattern is not an AIPerf peculiarity.** This record found it five times in
our tree. The reference has at least five of its own:

- `--reasoning` / `reasoning_effort` is plumbed end-to-end
  (`reference_mlperf.py:188` → `reference_SUT.py:247` →
  `multi_shot_retrieval.py:1190`) and never read; all four call sites pass a
  literal `"medium"` (`:516`-`:518`, `:632`-`:634`, `:764`-`:766`, `:857`-`:859`).
  `CLAUDE.md:230`-`:231` documents temperatures of 0.0/0.1 while the code runs at
  1.0 — a stale doc over a live default.
- `evaluation.py:145` gates reranking on `rag_db._reranker_model`, an attribute no
  class defines (`retrieve/ragdb.py:27`, `:32` define `_reranker_model_name` and
  `_reranker_queue`), so the single-shot path silently never reranks.
- `--mlperf_conf` is parsed (`reference_mlperf.py:50`-`:54`) and never used; only
  `user.conf` reaches `settings.FromConfig`.
- `db_manifest.py` accepts `--cosine_threshold` and `--top_k_depth` and documents
  them as ignored (`:307`-`:310`) — while
  `reference_mlperf_datasetup_accuracy.sh:50`-`:52` still passes
  `--cosine_threshold 0.9999`, so the accuracy script appears to enforce a gate it
  does not.
- `llm_logger.py:216`-`:227` computes accuracy from a `judge_score >= 4` rubric
  threshold; nothing in the repo ever writes `judge_score`. The shipped
  `logs_result.json` accordingly reports `queries_correct: 0` alongside
  `accuracy: 0.317` — a summary that is internally inconsistent and should not be
  quoted as a result.

The lesson we take is procedural: I23's obligation is not "record the parameters"
but "record the parameters that were *resolved and used*," and the parity test in
`### The stock configuration` earns its place precisely because four of the five
cases above are a value that is authored, threaded, and then dropped.

### Where the reference binds less than we do

`db_manifest.py` is the direct `corpus_digest` analogue, and comparing it is the
most useful part of this exercise because it built the strict thing and then
shipped the loose one.

Its stated intent is behavioral equivalence, explicitly *not* byte identity
(`db_manifest.py:18`-`:51`). It computes `_corpus_set_sha256` (`:106`-`:127`) with a
construction worth adopting: SHA-256 per passage over **raw text with no
normalization**, then the sorted per-passage digests folded with a `\x00`
delimiter — order-independent by construction and cheap to maintain incrementally.
Our `corpus_digest` is BLAKE3 over a sorted document projection, which has the same
order-independence property; the reference's per-element-then-sort shape is the
better one for an incremental sealer and this record adopts it in
`### One corpus identity`.

Then `verify` never compares it. The corpus hash is **reported from the manifest
and never recomputed against the database** (`db_manifest.py:373`-`:377`, comment:
"informational only, never gated"). The four gated checks are `total_passages`
equality, `embedding_dim` equality, `index_params` dict equality, and a mean
top-K URL set-overlap ≥ 0.95 over 50 seeded probe queries (`:350`-`:408`). A
stricter path exists — `cmd_compare` (`:428`-`:477`) *does* gate on corpus-set
equality at `:456` — and is not the one on the submission path.

So the reference binds passage count, embedding dimension, exact index
parameters, and observable retrieval behavior on a fixed probe set. It leaves
unbound: the corpus content digest, the chunk size and overlap and text-boundary
mode (absent from the manifest entirely), the parser, the source HTML, passage
metadata, and the vectors. `total_passages` carries the load as a proxy for "same
chunking, same parsing, same corpus," which compensating changes defeat.

Two things follow. First, I1 and I2 are stronger than the reference and should
stay that way — a digest that is computed, recorded, and never checked is the
inert-mechanism pattern again, in the one place where it decides whether two
submissions mean the same thing. Second, the reference's probe-set retrieval gate
is a genuinely good idea we do not have: 50 seeded queries, deterministic sample
(`random.Random(0xC0FFEE)`, `:185`-`:189`), top-10 URL overlap with normalization
that strips scheme and anchors so metadata formatting cannot fail it. It catches
the class of divergence a content digest cannot — an index built from identical
passages that nonetheless retrieves differently, which HNSW's insertion-order
dependence makes real. This record adopts it as an index-integrity check
alongside the digest rather than instead of it.

Corpus and chunker constants, for the pinned `rag_e2e` profile: 2515 frozen
Wikipedia HTML pages shipped as a tarball with re-scraping explicitly forbidden
(`scripts/download_dataset_and_models.sh:53`-`:72`), 768-character chunks with
32-character overlap under word-boundary optimization
(`config.template.sh:17`-`:18`; note the `read_docs.py` CLI default is
`sentence`, so the scored config overrides it), yielding 108,711 passages in the
shipped manifest. Chunking is **character-based, not token-based**
(`text_splitter.py:143`-`:151`), which matters because our chunker is too and the
embedding model's 512-token limit is what makes 768 characters safe. Acquisition
is pinned by artifact with **no checksum on any downloaded asset** — our
acquire-once-into-a-private-snapshot path with a digest is stronger, and nothing
about the reference's approach recommends relaxing it.

### The scored metric and the validity gate

The headline number is LoadGen's, not the reference code's: the SUT returns a
zero-filled `int32` buffer sized to the answer call's real
`usage.completion_tokens` (`reference_SUT.py:309`-`:330`), so Offline reports
tokens/s alongside samples/s. Only the final `answer_generator` call is reported;
the three upstream roles that dominate token volume are invisible to LoadGen. Our
per-role attribution (I16) is therefore a superset of what the reference scores,
and the aggregate-only compliance scope in `### Accuracy and compliance` matches
the reference's actual granularity.

Accuracy is a binary LLM-as-judge verdict, with retrieval scored separately and
then **discarded**: `accuracy_eval.py:129`-`:150` computes precision/recall/F1 over
retrieved-vs-ground-truth URL sets and prints them, but the MLPerf-format
`accuracy.txt` written at `:353`-`:355` contains only the judge accuracy line.
Reference results are P@N 72%, R@N 67%, F1@N 66%, judge accuracy 36%, against an
oracle-context ceiling of 68% (`CLAUDE.md:207`-`:214`). Two different judges with
different prompts, schemas, and defaults exist in the tree (`accuracy_eval.py` for
the LoadGen path, `evaluate.py:127`-`:163` for the non-LoadGen path) and are not
consistent with each other; the judge model is not pinned, and three defaults
disagree — including one that self-judges with the SUT's own answer model. Judge
failures score `correct = False` (`:118`-`:119`, `:124`-`:126`), so a judge outage
depresses accuracy rather than erroring. Our design pins the judge inside
`pipeline_digest` and refuses rather than silently scoring zero, and this is the
clearest place where matching the reference exactly would be the wrong call.

**The 97% validity gate is not in `e2e-rag/`.** It is delegated by comment to
`tools/submission/submission_checker.py` (`accuracy_eval.py:348`-`:352`), which
parses the `Accuracy:` line out of `accuracy.txt`. Within `e2e-rag/`,
`accuracy_eval.py:284`-`:356` has no threshold, no comparison, and no `sys.exit` —
a 0% run exits 0, and `reference_mlperf.py:271` propagates only a crash. The one
real pass/fail gate in the tree is on ingestion:
`datasetup_accuracy_eval.py:406`-`:413` requires ≥99% file success **and** an MD5
match, prints `FAILED`, writes `Overall Result: PASS|FAIL`, and exits 1
(`:645`-`:648`). That MD5 is a self-consistency check — the SUT's reported hash
versus the file it wrote — never a comparison against a reference value, which it
could not be since the artifact is not reproducible across runs.

This confirms the scoping in `### Accuracy and compliance`: the gate belongs in
`aiperf rag score --gate` against an authored reference accuracy, not in
`aiperf profile`, and the reference's own separation of "run" from "check" is the
same shape. The 97% constant itself must come from the full `mlcommons/inference`
checkout; it is not quotable from `e2e-rag/` and this record does not assert it.

Output-length compliance is TEST09's, and `run_compliance_test09.sh` runs a
verification script that is not vendored in this directory. The mechanism is
clear from the SUT side: TEST09's `audit.config` makes LoadGen sample responses
into `mlperf_log_accuracy.json` during a *performance* run, and because the SUT
encodes `n_tokens` int32 slots, the verifier can recover generated lengths and
prove the SUT generated what it claimed while running at speed. The reference OSL
figure of 273.81 is the `answer_generator`-only mean of one specific logged run
(`ISL_OSL_statistics.txt:107`) — not a pipeline aggregate, which is 551.58 for the
same run, and not a mean across the five logged runs, which is about 235. Any
comparison we publish must name which of the three it is.

### Per-query reasoning-type labels

The label exists in the query set and never reaches the score. FRAMES ships a
`reasoning_types` column, pipe-delimited for multi-label queries
(`evaluation.py:615`, `:769`, `:840`). But `QSL.py:47`-`:61` loads only `Prompt`,
`Answer`, and the `wikipedia_link_*` columns, so the label cannot reach the SUT or
LoadGen; `accuracy_eval.py` never mentions it, and the scored path produces no
breakdown at all. Per-tag analysis exists only in `evaluation.py:565`-`:645`, a
retrieval-only non-LoadGen evaluator that never sees an answer or a judge score.

Two consequences for the reasoning-type work in `### Metrics`. The multi-valued
canonicalization this record specifies — sorted, `;`-joined, never `,`, never a
list type — is the right call and the reference's `|` delimiter is a compatible
input encoding to normalize from. And the free path through
`AccuracyAssociation.task`, which gives per-reasoning-type judge accuracy with
zero schema change, produces a breakdown the reference does not have on its scored
path at all. That is the strongest form of the argument for doing it early: it is
cheap, it is additive, and it is a capability rather than parity work.

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

Two further deferrals come from the reference comparison and are named because
they are divergences rather than oversights. **Sparse retrieval** — the reference
treats `bm25s` as a first-class alternative retrieval method (`params.py:449`,
`:569`) — has no analogue here; the `VectorIndex` category is shaped to accept a
sparse implementation later, and until one exists a BM25 configuration is outside
our comparability class. **In-process embedding and reranking**: the reference
embeds locally with sentence-transformers and reranks with a ColBERTv2.0 MaxSim
worker (`reranker_worker.py:126`-`:128`), while this design routes both through
endpoint profiles over HTTP. That is a deliberate difference — it puts embedding
and rerank latency on the measured request timeline, which is the property a
benchmarking tool wants — but it means embed and rerank cost is not directly
comparable to a reference number, and a run asserting I24 comparability declares
it.

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

MLPerf reference implementation, `mlcommons/inference`, directory `e2e-rag/` at
commit `cfb0df14b21a3898521891f021a1c6aadec2ab2c` (2026-08-26). Paths below are
relative to that directory.

- Index construction and its hardcoded parameters: `retrieve/vectordb.py:263`-`:266`;
  normalized embeddings `:201`; unread `vector_index_method` `:152`-`:163` against
  `reference_mlperf_datasetup.py:133`-`:138`.
- Chunker: `text_splitter.py:120`-`:155`; scored constants `config.template.sh:17`-`:18`;
  parser and boundary handling `read_docs.py:126`-`:133`, `:256`-`:320`, `:415`-`:417`.
- Corpus acquisition: `scripts/download_dataset_and_models.sh:53`-`:72`; floating
  alternative `download_docs.py:578`, `:412`-`:565`.
- Database manifest, digest construction, and the four gated checks:
  `db_manifest.py:106`-`:127`, `:185`-`:189`, `:350`-`:408`; ungated corpus hash
  `:373`-`:377`; stricter unused comparison `:428`-`:477`.
- Ingestion as a timed phase: `measure_indexing_with_chunking.py:387`-`:391`;
  wall-clock timing `ingestion_monitor.py:99`, `:115`, `:141`, `:204`, `:250`.
- Ingestion pass/fail gate: `datasetup_accuracy_eval.py:406`-`:413`, `:645`-`:648`.
- Multi-hop loop, its bounds, and the fallbacks:
  `multi_shot_retrieval.py:1252`, `:1271`-`:1308`, `:1310`-`:1424`, `:1464`-`:1477`,
  `:1495`-`:1569`; sufficiency forcing `:655`, `:673`, `:683`-`:686`, `:701`;
  unclamped fan-out `:1416`.
- Role-to-endpoint map: `multi_shot_retrieval.py:468`, `:479`-`:481`, `:576`,
  `:600`-`:602`, `:707`, `:724`-`:726`, `:799`, `:812`-`:814`.
- Scenario declaration and admission bounds: `user.conf`;
  `reference_mlperf_perf.sh:31`-`:32`, `:41`-`:42`; `reference_mlperf.py:159`-`:162`,
  `:204`-`:241`.
- Token accounting handed to LoadGen: `reference_SUT.py:261`-`:262`, `:309`-`:330`;
  empty-answer fallback `:318`-`:320`.
- Accuracy, judges, and the absent gate: `accuracy_eval.py:43`-`:65`, `:129`-`:150`,
  `:284`-`:356`; second judge `evaluate.py:127`-`:163`.
- Reasoning-type labels present in the data and absent from the score:
  `evaluation.py:565`-`:645`; `QSL.py:47`-`:61`.
- Reranker: `reranker_worker.py:111`-`:128`; sparse alternative `params.py:449`, `:569`.
