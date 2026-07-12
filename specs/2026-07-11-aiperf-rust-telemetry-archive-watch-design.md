<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AIPerf-Rust: durable telemetry archive and `aiperf watch`

**Date:** 2026-07-11
**Author:** Anthony Casagrande (Tech Lead) + Codex
**Status:** design — not built
**Decision:** adopt the useful operational ideas demonstrated by Tachometer—an always-on watch
surface, per-source cadence/failure isolation, topology enrichment, immutable columnar history,
and periodic object-store durability—without importing Tachometer's parser, row schema, filter
semantics, checkpoint algorithm, or metric authority.

**Grounding:**

- current native code: `aiperf-clock`, `aiperf-transport`, `aiperf-server-metrics`,
  `aiperf-gpu-telemetry`, `aiperf-network-latency`, `aiperf-metrics`, the runner telemetry
  adapters, `ScheduledPhaseSidecar`, and runner protocol/artifact ownership;
- current design authority:
  `2026-07-10-aiperf-rust-telemetry-accumulators-design.md`,
  `2026-07-11-aiperf-rust-exporters-overhaul-design.md`,
  `2026-07-11-python-orchestrator-rust-single-run-design.md`, and
  `2026-07-11-aiperf-runner-only-execution-surface-design.md`;
- Tachometer comparison target: `ai-dynamo/aiperf#1036` at
  `2ea84c88984ac1761d1f7f324570ca9089cb4ec5`, including the complete Rust scraper/writer,
  Python wrapper, configuration, tests, and compaction path;
- validated comparison and runtime receipts:
  `artifacts/code-review.md` and `artifacts/repro-runtime-20260711/`.
- accepted wire-format authorities: Prometheus
  [text exposition 0.0.4](https://prometheus.io/docs/instrumenting/exposition_formats/) and
  [OpenMetrics text 1.0.0](https://github.com/prometheus/OpenMetrics/blob/main/specification/OpenMetrics.md).
  OpenMetrics 2.0 remains draft/out of scope until a separately versioned parser factory is
  implemented and advertised.

---

## 0. Executive decision

Native AIPerf already has the stronger **measurement** pipeline. It owns request correlation,
token timing, endpoint usage, HTTP traces, exact phase barriers, counter deltas, histogram
semantics, GPU energy joins, network adjustment, SLO goodput, sweep lines, timeslices, and the
typed native-v2 report. Nothing in this design replaces or bypasses that authority.

What native AIPerf lacks is an **operational history plane**:

1. watch arbitrary telemetry sources independently of an inference benchmark;
2. give each source its own fixed-deadline cadence and failure domain;
3. attach operator-authored topology labels without changing source identity;
4. retain raw structured scrape history in queryable Parquet;
5. spill locally, survive process failure, resume safely, and synchronize immutable partitions
   to an object store during a long run;
6. mark benchmark phase boundaries in that history when the same machinery runs as a sidecar.

The product surface is the human-facing Python command `aiperf watch`. It launches the **same
sole Rust executable**, `aiperf-runner`, with the normal strict validate/execute envelope and a
registered `telemetry_watch` workload over the `online_http` backend. There is no second Rust
binary, PyO3 binding, raw argv parser, Python scraper, or alternate metrics engine.

The archive is deliberately **not a report cache**. Native accumulators consume the live source
records directly and remain authoritative. Archive records are forensic/query data and can be
reprocessed later, but they never feed results back into the run that created them.

---

## 1. Current code truth and the exact gap

### 1.1 What is already built

The current native pipeline has the required semantic foundations:

- `ServerMetricsSource` and `GpuTelemetrySource` are object-safe source seams over the injected
  `Clock` and native HTTP transport.
- `PrometheusTextParser` preserves escaped labels and per-label-set structured histograms,
  rejects malformed exposition, distinguishes classic/OpenMetrics, and keeps values in `f64`.
- GPU and server sidecars force start/end scrapes at phase barriers and continuously sample
  gauges on the `Clock`.
- GPU, server, and network accumulators implement the shared `Accumulator<Record>` contract,
  retain typed records, compute exact boundary deltas, and join results into native-v2.
- runner artifacts can retain GPU/server/network JSONL and a compatibility handoff for Python's
  server-metrics Parquet exporter.
- Python owns human CLI/configuration/presentation; `aiperf-runner` is the only native product
  executable and owns native run-time IO.

These paths are benchmark-bounded. They retain enough history for one report, then write local
per-run artifacts at finalization. They do not provide an always-on session, remote incremental
durability, crash recovery, source-specific schedules, or a stable raw columnar schema.

### 1.2 What is genuinely useful in Tachometer

The Tachometer PR demonstrates five product concepts worth adopting:

- an independently invocable `watch` experience;
- one polling loop and frequency per endpoint;
- configuration-authored hostname/GPU/worker placement metadata;
- local Arrow/Parquet spill with periodic remote synchronization;
- metric/time-clustered Parquet intended for direct analytical queries.

Those are operational ideas, not metric algorithms. This spec carries the ideas through native
AIPerf's existing seams and corrects their data model and durability contract.

### 1.3 What must not be copied

The following Tachometer mechanics are explicitly rejected:

- custom comma-splitting Prometheus parsing and silent parse-line drops;
- labels serialized into a `metric_name` string;
- Float32 values/bounds/sums/counts;
- histogram grouping without the complete non-`le` label identity;
- non-2xx response bodies treated as successful metrics;
- semantic name-collapsing node/DCGM filters;
- completion-paced `scrape; sleep(interval)` scheduling;
- one shared `Arc<Mutex>` writer state;
- mutable `current.arrow` snapshots combined with already-committed Parquet;
- periodic rewriting of all historical `incomplete-N` data;
- a global final compaction as the only authoritative dataset;
- a second Rust/PyO3 product surface.

The first five are not theoretical concerns. Runtime validation against the compiled PR binary
proved cross-label histogram contamination, duplicate checkpoint rows, Float32 counter loss,
quoted-label truncation, and HTTP-500 ingestion. The native design makes each impossible by
construction.

---

## 2. Scope, non-goals, and invariants

### 2.1 In scope

- a `telemetry_watch` runner workload and Python `aiperf watch` projection;
- reusable per-source fixed-deadline scrape drivers;
- generic Prometheus/OpenMetrics, native DCGM, and compile-time registered source factories;
- structured enrichment and redaction policies;
- an archive schema for scrape attempts, samples, lifecycle markers, and manifests;
- a bounded single-owner archive writer with local WAL and immutable Parquet partitions;
- local filesystem and object-store sinks behind traits;
- restart recovery, exact-resume validation, orphan handling, and terminal finalization;
- optional attachment to existing scheduled benchmark telemetry without a second scrape;
- native-v2 archive provenance/health metadata;
- query and process acceptance gates.

### 2.2 Deferred

- a web UI, dashboard server, or embedded query engine;
- automatic retention/garbage collection of remote archives;
- distributed multi-writer archives;
- arbitrary SQL-derived metrics in the Rust process;
- changing the authoritative GPU/server/network aggregate algorithms;
- MLflow/W&B upload of raw archive partitions;
- cross-archive joins or fleet catalog services.
- Graph-mode attachment until Graph owns a built workload-neutral lifecycle/phase observer,
  boundary commands, and telemetry/report join;
- product offline attachment until the in-process engine sink and deterministic external-progress
  integration are built. Virtual library tests use the inline strategy described in §5.

### 2.3 Non-negotiable invariants

1. **One native executable.** Python `aiperf watch` launches `aiperf-runner`.
2. **One measurement authority.** The native live accumulators and reporter remain authoritative.
3. **One scrape per source event.** Accumulator and archive projections fan out from one
   pre-projection attempt envelope, including failures; enabling archival never doubles source
   traffic.
4. **All sampling time through `Clock`.** No `Instant`, `SystemTime`, or `tokio::time` in the
   clock-aware source/scheduler path.
5. **No request-path backpressure.** Archive-only decode/projection, serialization, compression,
   filesystem, and upload work never run on or block the per-request/per-token local loop.
6. **One mutable owner.** One archive IO worker owns WAL, partition builders, and manifests; no
   shared writer mutex exists.
7. **Structured identity.** Metric family, semantic type, source, and labels are separate fields.
8. **Float64 preservation.** Every finite source number remains `f64` through Parquet.
9. **Non-finite values are explicit.** NaN/±Inf never cross a serialization boundary as an
   unclassified JSON/Parquet number.
10. **No silent loss.** Failed, empty, unchanged, missed, backpressured, and dropped observations
    are counted and surfaced in scrape records/manifest health.
11. **Exactly-once logical commit.** Every frame for which a caller observed a local-durable
    receipt has one logical identity and is recovered exactly once. A crash may also recover a
    complete durable frame whose receipt was not observed; persistence retry keeps its identity.
12. **Immutable incremental history.** Sync uploads content-addressed partitions and bounded
    manifest metadata; it never rewrites the whole archive or a flat full-history list per commit.
13. **Fail closed on identity.** Resume requires matching schema, archive ID, configuration digest,
    source descriptors, identity-key digest, and archive-writer compatibility ID. The exact runner
    distribution remains recorded provenance but may change only through an explicitly compatible
    writer implementation.
14. **Known credentials never become archive dimensions.** AIPerf-authored/provider credentials are
    removed by a non-disableable baseline; arbitrary endpoint metric content remains classified
    potentially sensitive source data and follows explicit sanitization/storage policy.
15. **Compile-time extensibility.** Source, enrichment, redaction, rotation, admission, recovery,
    and sink choices cross traits/factories, not closed implementation enums.

---

## 3. Product and ownership model

### 3.1 Human surface

Python owns:

```text
aiperf watch --config watch.yaml
```

Python performs normal Config-v2 YAML/Jinja/environment/CLI expansion, selects an exact runner,
checks capabilities/distribution identity, projects a strict authored request, forwards signals,
and presents the terminal archive summary. It does not scrape, parse, enrich, buffer, compact, or
upload telemetry.

Direct endpoint convenience flags may lower into the same Config-v2 model, but they are not a
second wire contract.

### 3.2 Runner composition

Watch is a registered workload, not a new operation family:

```text
RunnerEnvelopeV2 { operation: validate|execute }
  run.backend  = { type: "online_http", ... }
  run.workload = { type: "telemetry_watch", config: ... }
```

`telemetry_watch` requires a real clock and a control-plane HTTP transport but no inference model,
semantic response, dataset, phase list, or `RequestSink`. The preimplementation runner-v2 design's
outer DTO is therefore revised here: shared resource blocks are workload-scoped, not globally
required. Every workload factory declares each resource block as `required`, `optional`, or
`forbidden`; missing required or present forbidden resources fail before backend preparation.
This does not make validation permissive.

The `online_http` prepared backend exposes a LocalSet-compatible `ControlPlaneHttpProvider`
capability over its injected `Clock` and native transport. It validates a secret-free per-source
transport profile and resolves provider-held credentials/TLS material into a prepared
`ControlPlaneHttp` handle. Equivalent profiles may share a dedicated control-plane pool, but that
pool and its capacity/configuration are isolated from inference connections. The handle accepts an
owned request plus an absolute call deadline and returns exact entity bytes, allowlisted response
metadata, and native transport timings. Connection/TLS/proxy/reuse/connect-timeout policy is bound
at handle preparation; request/total lifetime is bounded again per call. A workload requirement
asks for the provider capability, so the compatibility matrix is derived rather than special-
casing the `telemetry_watch` ID. This reuses the native stack, not its hot-path pool, and is not a
special `reqwest` path.

The computed capability matrix adds:

| Backend | `telemetry_watch` |
|---|:---:|
| `online_http` | yes |
| `dynamo_offline` | no |

Replay/fault-injection tests may drive the library runtime with `SimClock`, but the product pair is
real-clock only. Virtual tests use a deterministic inline decoder plus `MemoryArchiveSink`, then
persist captured frames after the simulation. A future threaded virtual implementation must expose
pending external work through the DES quiescence source and sort same-instant completions by stable
attempt sequence; OS-worker latency never advances virtual time. Product offline and Graph
attachment remain deferred as stated in §2.2.

### 3.3 Attached benchmark mode

An ordinary scheduled run may request an archive target in addition to existing telemetry
summaries. The run owns exactly one fixed-deadline driver per physical telemetry source across all
phases. Phase sidecars subscribe to that driver and submit typed boundary commands; they do not own
cadence loops. Each all-outcome attempt envelope is projected twice where applicable:

```text
one physical source attempt
  +-- native domain projection/accumulator (when successful and supported)
  `-- archive attempt/exposition projection (every outcome)
```

Every physical attempt has a stable source-attempt ID and the active phase-membership set captured
at its snapshot instant. Native phase-local projections consume that membership; run-level source/
endpoint facts deduplicate by physical attempt ID, so seamless overlap neither loses one phase nor
counts two fetches. Exact old/new summary parity is a shipping gate.

Boundary coalescing never uses timestamp proximity. The phase orchestrator assigns a typed
`coalescing_group_id` to exactly those transition subscribers that share one physical snapshot;
their lifecycle markers remain distinct. Commands without the same group never coalesce. Exact
phase lifecycle markers come from a `PhaseObserver` tee, while boundary scrape capture time remains
a separate fact. Archive availability cannot change request timings or metric formulas. The
terminal report records archive completeness separately.

### 3.4 Why Rust owns this IO

The exporter design normally assigns post-run uploads to Python. Incremental archival is the
demonstrated exception: local WAL acknowledgment, bounded ingress, partition rotation, crash
recovery, and periodic remote durability occur while sources are live. Deferring them until after
the child exits defeats the feature and would require a second process to duplicate Rust-owned
records. Therefore the archive sink is runner-owned native IO; Python still owns presentation and
post-run analysis.

---

## 4. Crate and dependency shape

### 4.1 New crates

`aiperf-prometheus` is an IO-free leaf containing an exposition model and bounded parser seam for
exactly two advertised formats: Prometheus text 0.0.4 and OpenMetrics text 1.0.0. Content type
selects the strict archive parser; a strict failure is never silently reclassified as success under
a different format. It extracts and replaces—not merely wraps—the narrower lexical/state-machine core
currently embedded in server metrics:

```rust
pub trait ExpositionParser: Debug + Send + Sync {
    fn parse(
        &self,
        format: ExpositionFormat,
        exact_body: &[u8],
        limits: &ParseLimits,
    ) -> Result<Exposition, ParseError>;
}
```

The parser validates UTF-8 and enforces line, label, family, sample, bucket, and decoded-byte limits
while consuming the body. `Exposition` preserves HELP/TYPE/UNIT metadata, source family and emitted
sample names/roles, structured escaped labels, source numeric/timestamp lexemes, counter-created
facts, unknown/untyped, gauge, counter, info, stateset, histogram, gauge-histogram, summary,
per-point timestamps, and scalar/bucket exemplars with their label sets, values, and optional
timestamps. Prometheus `untyped` and OpenMetrics `unknown` remain distinguishable source tokens even
though they share the archive `unknown` semantic branch. Projection policies—not the lexer—decide
whether benchmark accumulation excludes summaries, `_created`, uptime, or unsupported domain
families. An accepted wire feature is never silently dropped or normalized into another semantic
type.

Current native server-metrics compatibility is a separate projection policy. For the pinned vLLM
case, one bounded fetched body may produce (a) a strict declared-format archive parse outcome and
(b) an explicitly named classic-text fallback entity used only by the native projection. The
attempt records declared media type, strict parser format/outcome, actual compatibility grammar,
and `native_compatibility_fallback=true`. The fallback never creates archive sample/family rows or
turns the strict outcome into success. Thus one network response may be parsed twice by one bounded
decode job without doubling source traffic or changing current native metrics.

`aiperf-telemetry-archive` owns domain-neutral archive DTOs, ingress/sink/policy traits, WAL,
Parquet partition writing, manifests, recovery, and object-store synchronization. It depends on
`aiperf-clock`, serialization/Arrow/Parquet/object-store libraries, and optionally
`aiperf-prometheus`; it does not depend on `aiperf-metrics`, GPU, server, runner, or an inference
backend.

### 4.2 Dependency direction

```text
aiperf-runner
  +-- aiperf-server-metrics --+--> aiperf-prometheus
  +-- aiperf-gpu-telemetry ---+
  +-- aiperf-network-latency
  +-- aiperf-telemetry-archive --> aiperf-prometheus
  `-- aiperf-extensions ---------> archive source-factory traits

aiperf-metrics remains IO-free and has no archive dependency.
```

Runner-local prepared source adapters own pure decoders plus two projections. They create a native
`ServerMetricsRecord`, `GpuScrape`, or network record only from a successful supported decoded
entity and create an archive attempt from every fetch outcome. Existing lossy domain records are
never treated as raw archive inputs. No cycle or heavy Parquet dependency is pushed into the
domain accumulators.

### 4.3 No hot-path dynamic dispatch

Factories and policies are selected during validation/preparation. Source drivers and the writer
may retain `dyn` seams because scrapes are low-rate control-plane events. Per-sample conversion is
batch-local. There is no registry lookup, allocation policy, lock, or archive callback per token.

---

## 5. Core trait seams

The signatures below are design-level; concrete error DTOs remain plain hand-written library
error enums. Generic `Entity`/`Record` types are monomorphized inside each prepared source factory;
the registry erases only startup construction and does not require a closed enum of source kinds.

```rust
pub trait ControlPlaneHttpProvider {
    fn prepare(
        &self,
        profile: ValidatedControlPlaneProfile,
        secrets: &dyn SecretProviderResolver,
    ) -> Result<Rc<dyn ControlPlaneHttp>, ControlPlanePrepareError>;
}

#[async_trait(?Send)]
pub trait ControlPlaneHttp {
    async fn execute(
        &self,
        request: ControlPlaneRequest,
        absolute_deadline_ns: i64,
    ) -> Result<ControlPlaneResponse, ControlPlaneHttpError>;
}

pub trait ArchiveSourceFactory: Debug + Send + Sync {
    fn descriptor(&self) -> &'static ArchiveSourceDescriptor;
    fn validate(&self, config: &RawValue) -> Result<ValidatedSourceConfig, SourceConfigError>;
    fn prepare(
        &self,
        config: ValidatedSourceConfig,
        context: &ArchiveSourceContext,
    ) -> Result<Box<dyn PreparedTelemetryDriver>, SourcePrepareError>;
}

pub trait PreparedTelemetryDriver {
    fn source_id(&self) -> &str;
    fn start(
        self: Box<Self>,
        context: PreparedDriverContext,
    ) -> Result<RunningTelemetryDriver, DriverStartError>;
}

#[async_trait(?Send)]
pub trait TelemetryFetcher {
    async fn fetch(&self, request: FetchRequest, deadline_ns: i64) -> FetchedAttempt;
    async fn shutdown(&self) -> Result<(), ArchiveSourceError>;
}

pub trait AttemptDecoder<ArchiveEntity, NativeEntity>: Debug + Send + Sync {
    fn decode(
        &self,
        fetched: FetchedAttempt,
        limits: &DecodeLimits,
    ) -> DecodedAttempt<ArchiveEntity, NativeEntity>;
}

pub struct DecodedAttempt<ArchiveEntity, NativeEntity> {
    pub facts: AttemptFacts,
    pub strict_archive_entity: Option<ArchiveEntity>,
    pub native_entity: Option<NativeEntity>,
    pub strict_parse_outcome: ParseOutcome,
    pub native_compatibility: Option<CompatibilityFallback>,
    pub exact_entity: Option<ExactEntityLease>,
}

pub trait NativeProjection<NativeEntity, Record>: Debug + Send + Sync {
    fn project(&self, entity: &NativeEntity) -> Result<Option<Record>, NativeProjectionError>;
}

pub trait ArchiveProjection<ArchiveEntity, NativeEntity>: Debug + Send + Sync {
    fn project(
        &self,
        attempt: &DecodedAttempt<ArchiveEntity, NativeEntity>,
        permit: ArchiveProjectionPermit,
        context: &ArchiveProjectionContext,
    ) -> Result<ArchiveFrameDraft, ArchiveProjectionError>;
}

pub trait TelemetryEnricher: Debug + Send + Sync {
    fn attributes(
        &self,
        sample: &ArchiveSampleView<'_>,
        source: &ArchiveSourceIdentity,
    ) -> Result<AttributePatch, EnrichmentError>;
}

pub trait ArchiveSanitizer: Debug + Send + Sync {
    fn sanitize_source(&self, source: SourceDescriptorView<'_>) -> SanitizedSourceDescriptor;
    fn sanitize_sample(&self, sample: ArchiveSampleView<'_>) -> SanitizedSample;
    fn sanitize_marker(&self, marker: ArchiveMarkerView<'_>) -> SanitizedMarker;
    fn sanitize_diagnostic(&self, diagnostic: &ArchiveDiagnostic) -> ArchiveDiagnostic;
}

pub trait SegmentRotationPolicy: Debug + Send {
    fn should_rotate(&self, state: &OpenSegmentState, now_ns: i64) -> bool;
}

pub trait ArchiveAdmissionPolicy {
    fn try_reserve(
        &self,
        state: ArchiveIngressState,
        upper_bound: ArchiveProjectionFootprint,
    ) -> Result<ArchiveProjectionPermit, AdmissionRejection>;
}

pub trait ArchiveRecoveryPolicy: Debug + Send {
    fn recover(&self, local: &LocalArchiveState, remote: Option<&RemoteArchiveState>)
        -> Result<RecoveryPlan, ArchiveRecoveryError>;
}

#[async_trait]
pub trait ArchiveSink: Send {
    async fn recover(&mut self) -> Result<RecoveredArchive, ArchiveSinkError>;
    async fn append_frame(&mut self, frame: ArchiveWalFrame)
        -> Result<AppendReceipt, ArchiveSinkError>;
    async fn checkpoint(&mut self) -> Result<CheckpointReceipt, ArchiveSinkError>;
    async fn finalize(&mut self, reason: TerminationReason)
        -> Result<FinalizedArchive, ArchiveSinkError>;
}

pub trait EpochAnchorProvider {
    fn anchor(&self, clock: &dyn Clock) -> Result<EpochAnchor, EpochAnchorError>;
}
```

`ArchiveSourceFactory::prepare` constructs a concrete
`TypedTelemetryDriver<Fetcher, Decoder, ArchiveEntity, NativeEntity, Record, NativeProjection,
ArchiveProjection>` and erases only that complete driver. The object-safe boundary starts/commands
the already composed pipeline; neither the runner nor a registry recovers entity types through
`Any`, source IDs, or a second lookup. For the threaded strategy, archive/native entities, returned
records, frame drafts, and job results are `Send + 'static`, and pure decoder/projection/enrichment/
sanitizer objects are `Send + Sync + 'static`. The `TelemetryFetcher`, Clock, admission policy,
running driver handle, and observer graph stay LocalSet-owned `?Send`. The virtual inline strategy
may use a separately prepared local pipeline and never crosses an OS-thread boundary.

`FetchedAttempt` owns bounded received encoded bytes, validated content-decoded exposition bytes,
transport facts, and typed allowlisted response metadata. A mandatory baseline strips all
AIPerf-authored/provider credential material before construction; arbitrary headers never enter the
DTO. If raw retention is enabled, the prepared policy receives an opaque reference-counted
`ExactEntityLease` with explicitly separate encoded/decoded byte handles. Only archive/raw
projection can open it; generic observers cannot.

Source fetch and Clock/lifecycle ownership stay on the LocalSet. Shared fetch/native-decode capacity
is independently bounded and guaranteed for an accepted attached telemetry attempt. The driver
sends exact owned bytes/facts to the ordered CPU pool, receives the decoded native result, and feeds
the native accumulator even when archive admission is unavailable. Only then does it
nonblockingly acquire `ArchiveProjectionPermit` from a validated worst-case footprint. The permit
owns byte/frame/WAL quota, is consumed by one draft, and refunds unused capacity. Denial records a
loss range without repeating parse or delaying native delivery. Primary watch may wait/fail before
fetch according to its durable admission policy because the archive is its product.

The CPU pool returns `ArchiveFrameDraft` without a global accepted sequence. The single archive
owner assigns the inclusive global `record_seq`, computes the final frame ID, and writes the
`ArchiveWalFrame`. The frame is a versioned persisted sum of attempt/family/sample batch,
lifecycle, receipt-range, raw-object descriptor, and coalesced loss-range payloads. It is a closed
wire schema, not a source extension point. Every frame has stable source/control identity and CRC,
and every successful `append_frame` has the same local-durable meaning.

Every wire-selected family—source, sink, rotation, admission, recovery, enrichment, sanitizer,
raw-body retention, and identity-key provider—has its own frozen descriptor/strict-validate/prepare
factory registry. Thread bounds follow the two explicit execution strategies above rather than a
blanket rule. Stable IDs never select a core string branch.

At least these concrete implementations ship:

- `PrometheusArchiveSource` and replay/fault-injection sources;
- `StaticLabelEnricher` and `NoopEnricher`;
- `BaselineCredentialSanitizer` plus allow/deny-key structured sanitizers; `NoopSanitizer` applies
  only after the non-disableable baseline and means no additional content policy;
- row/byte/Clock-age segment rotation policies composed by `AnyRotationPolicy`;
- primary-watch and attached-best-effort admission policies;
- create-new and exact-resume recovery policies;
- `ParquetArchiveSink` over a local spool plus optional `dyn ArchiveObjectStore`;
- `MemoryArchiveSink` for deterministic tests.

Stable wire IDs select factories/policies through frozen registries. A core string `match` does not
select implementations.

---

## 6. Source scheduling and isolation

### 6.1 One driver per source, one scrape in flight

Each physical source has one run-owned local driver task in standalone and attached modes. Drivers
await network operations independently; a slow endpoint cannot consume another endpoint's
in-flight slot. Bounded archive decode runs off-loop as specified in §5. A driver serializes
continuous and forced-boundary commands for its own source; two requests to the same endpoint never
overlap, even across seamless phases.

The driver owns:

- the prepared source;
- cadence anchor and tick index;
- source-record sequence (issued attempts and compact gap ranges share this namespace);
- consecutive/total failure counters;
- current source state (`active`, `degraded`, `disabled`, `stopped`);
- a command channel with reserved capacity for boundary/shutdown commands.

`scheduled_ns` is a cadence target/lateness fact, never the request-completion deadline. At launch:

```text
absolute_call_deadline_ns = min(
    request_start_ns + validated_source_request_timeout_ns,
    boundary_deadline_ns if present,
    run_duration_deadline_ns if present,
    shutdown_deadline_ns if stopping
)
```

The profile-bound control handle owns connect/TLS timeout; validation requires it not exceed the
source's authored connect ceiling. Its per-call request/total lifetime uses the absolute minimum
above. A call already beyond its boundary/run deadline emits a timeout without network IO. Deadline
cancellation must release the transport body/
connection state and returns one timeout attempt; it never detaches an unobserved request. Source
policy declares whether a boundary timeout merely degrades telemetry or fails a phase when the
sidecar is required.

Cross-source result ordering is never completion-order authority. Archive identity includes source
and sequence; reports/manifests sort by stable source ID.

### 6.2 Fixed-deadline cadence

Cadence is anchored once:

```text
deadline(n) = cadence_anchor_ns + n * interval_ns
```

After a scrape completes, the driver selects the first deadline strictly after `Clock::now_ns()`.
It does not sleep a full interval after completion, overlap requests, or burst to repay missed
ticks. Every skipped deadline increments `missed_ticks` and produces a compact gap record.

This yields an intended 3 Hz when scrapes are fast and an honest degraded cadence when they are
slow. `SimClock` tests pin same-instant ordering and overrun arithmetic.

### 6.3 Boundary priority

Attached phase barriers submit a forced `BoundaryStart` or `BoundaryEnd` command with phase ID,
boundary identity, orchestrator-issued optional `coalescing_group_id`, and absolute deadline. It
preempts the next continuous deadline but never interrupts an already issued HTTP request. Only
commands carrying the identical non-null coalescing group share a physical attempt; timestamp
proximity is irrelevant. All requesting boundaries are recorded on the attempt and receive the
same snapshot, while their lifecycle markers remain distinct. The phase waits for the forced result
under the same Clock deadline. Shared decode feeds native delivery first; archive projection uses
its independent permit. Continuous scheduling re-anchors from the original cadence, not the
boundary completion time.

### 6.4 Failure classification

Every attempt becomes one `ArchiveScrapeRecord`. `outcome` describes transport/parse disposition;
`body_unchanged` is an orthogonal success fact. V1 writes full sample rows for every successful
unchanged scrape rather than requiring readers to chase a prior sample batch. Outcomes include:

- success with samples;
- success with an empty exposition;
- HTTP status failure;
- transport/timeout failure;
- parse failure with line/category and a redacted bounded diagnostic;
- source-incompatible terminal disable;
- missed tick or admission skip;
- source shutdown failure.

Successful rows may carry `body_unchanged=true` and `same_body_as_attempt_seq`; health counts those
observations separately without classifying them as failures or empty scrapes.

The existing server `/prometheus/metrics` fallback and terminal-incompatible behavior remain
source semantics. Generic watch sources may use a separately configured failure policy, but they
never turn HTTP errors into samples or silently discard invalid lines.

---

## 7. Time model

### 7.1 Monotonic measurement time

Every scrape and lifecycle marker records the injected `Clock` timestamp. Attached archives use
the exact run/phase timeline, including virtual time where applicable. This field is the authority
for ordering and correlation with native-v2.

### 7.2 Cross-process wall-time anchor

An always-on archive also needs queryable time across process restarts. Extending `Clock` with wall
time would contaminate the scheduling seam, so watch prepares an independent anchor once:

```rust
pub struct EpochAnchor {
    pub clock_ns: i64,
    pub unix_epoch_ns: i128,
    pub capture_uncertainty_ns: u64,
}
```

The system provider captures `clock_before_ns`, reads wall time once, then captures
`clock_after_ns`. It rejects a reversed/overflowing bracket, chooses the monotonic midpoint as
`clock_ns`, and sets `capture_uncertainty_ns` to at least half the bracket span plus the declared
wall/monotonic clock-resolution allowance. Tests inject all three reads. The field measures anchor
acquisition uncertainty, not oscillator drift or later NTP steps.

For a real-clock session:

```text
unix_ns(t) = anchor.unix_epoch_ns + (clock_ns(t) - anchor.clock_ns)
```

The concrete system provider reads wall time only while creating the injected anchor. Every later
timestamp derives from the monotonic `Clock`, so NTP steps cannot reorder a session. Derived Unix
time is approximate placement after capture; it is not advertised as a continuously bounded UTC
clock. Optional later wall observations are diagnostic markers only and never remap samples. Each
restart creates a new `archive_session_id` and anchor; the manifest retains both. Virtual sessions
set `time_domain="virtual"` and omit Unix time.

### 7.3 Timestamp vocabulary

Scrape records distinguish:

- scheduled deadline;
- request sent;
- first response byte;
- snapshot/capture instant;
- parse complete;
- archive accepted;
- local durable acknowledgment;
- remote referenced acknowledgment.

No field called merely `time_since_start` substitutes for these meanings.

---

## 8. Archive schema v1

### 8.1 Accepted exposition surface and identity

The archive parser advertises Prometheus text 0.0.4 and OpenMetrics text 1.0.0. For those formats it
preserves every valid family type and point component listed in §4.1. Unknown content types,
OpenMetrics 2.0, protobuf/native histograms, or valid features outside an advertised parser
descriptor produce `unsupported_format`/`unsupported_feature` attempts; they are never silently
downgraded. The native benchmark projection remains its narrower current policy.

Every digest uses a versioned domain and length-prefixed fields, never raw concatenation:

```text
digest(domain, fields...) =
    BLAKE3(domain_utf8 || 0x00 || each(u64_be(length) || exact_bytes))
```

Domains include `aiperf.archive.config.v1`, `.batch.v1`, `.frame.v1`, `.series-display.v1`,
`.series-source.v1`, `.partition.v1`, `.manifest.v1`, and `.index-node.v1`. Canonical maps sort by
UTF-8 key bytes and reject duplicate keys. The configuration digest covers the fully validated
secret-free authored config, every selected factory ID plus normalized config, accepted-format
matrix, source descriptors, schema fingerprints, writer compatibility ID, and identity-key ID.

Two series identities prevent redaction from silently merging source series:

- `source_series_key`: keyed BLAKE3 over the pre-redaction source ID, family, semantic type, and
  canonical source labels. The key comes from `IdentityKeyProvider`; only its ID/digest is durable.
- `series_key`: ordinary BLAKE3 over the stored post-redaction identity plus
  `source_series_key`.

A many-to-one post-redaction mapping is visible because `source_series_key` differs. The default
sanitizer rejects it; intentional coalescing requires a named policy and a recorded outcome.
Enrichment attributes never enter either series key. Discovered attribute changes create a new
`attribute_epoch_id` and a topology-change marker.

### 8.2 Physical Arrow/Parquet contract

The repository contains one canonical UTF-8 JSON schema descriptor per table. It fixes field
order, names, nullability, Arrow logical/physical type, dictionary index width, child layout, and
schema metadata. Generated Arrow schemas are not the fingerprint authority. Each table stores:

```text
aiperf.archive.table = <attempts|samples|markers>
aiperf.archive.schema_version = 1.0
aiperf.archive.schema_fingerprint =
    BLAKE3("aiperf.archive.arrow-schema.v1\0" || exact_descriptor_bytes)
```

V1 uses these exact aliases:

| Alias | Arrow type and invariants |
|---|---|
| `Uuid` | `FixedSizeBinary(16)`, non-null |
| `Digest` | `FixedSizeBinary(32)`, non-null unless the field says nullable |
| `Enum8` | `Dictionary(Int8, Utf8)`, non-null; values are schema-defined strings |
| `EpochNs` | `Decimal128(38,0)`, nullable |
| `StringMap` | sorted `Map(entries: Struct<key: Utf8 non-null, value: Utf8 non-null>)` |
| `ArchiveNumber` | non-null `Struct<kind: Enum8, finite_value: Float64 nullable, exact_u64: UInt64 nullable>` |
| `Exemplar` | nullable `Struct<labels: StringMap, value: ArchiveNumber, timestamp_lexeme: Utf8 nullable, timestamp_unix_ns: EpochNs>` |

`ArchiveNumber.kind` is `finite`, `pos_inf`, `neg_inf`, `nan`, or `absent`. `finite_value` is
present only for `finite`; `exact_u64` may additionally preserve a source-native non-negative
integer and must convert to the identical finite f64 value. Every numeric leaf—including created
time, sums, counts, bounds, buckets, quantiles, states, and exemplars—uses this representation.
No raw non-finite Float64 is written. Maps are non-null (possibly empty); list elements and map keys
are non-null. Parquet dictionary encoding is an implementation choice, but the logical Arrow
dictionary index is always Int8 for frozen enums.

Increment 1 must check the canonical descriptors into the repository, generate schemas from them,
and freeze exact Arrow IPC/Parquet/DuckDB/Polars goldens before any archive capability is
advertised. “Binary or hex,” alternate decimal encodings, and unspecified exemplar layouts are not
conforming v1 writers.

### 8.3 Manifest graph and heads

Directory enumeration is never dataset authority. The local discovery authority is a small,
checksummed, atomically replaced and parent-directory-fsynced `LOCAL-LATEST` pointer. It contains
both current and preceding immutable head descriptors. Remote discovery uses a conditionally
updated `LATEST` with the same logical shape. A head descriptor contains:

```jsonc
{
  "archive_id": "uuid",
  "local_commit_seq": 7,
  "generation_key": "manifests/generation-7-blake3-....json",
  "generation_hash": "blake3:...",
  "index_root_key": "manifest-index/blake3-....json",
  "index_root_hash": "blake3:...",
  "parent_generation_hash": "blake3:...",
  "archive_state": "open"
}
```

Generation objects are immutable, content-addressed, and hash-linked. Generation zero is a full
genesis containing archive/schema/writer identity, the normalized config digest, identity-key
digest, secret-free source descriptors, session/anchor, exact runner distribution provenance, and
empty index root. Later generations are bounded transaction records containing parent, session,
added/removed partition IDs, exact WAL-frame/table coverage, health delta, state transition, and
termination reason. `unix_epoch_ns` is a decimal string in JSON.

The partition set is a persistent content-addressed B-tree with fanout 256. A commit copy-on-writes
only O(log₂₅₆ partitions) bounded index pages and the small generation object; the root hash
defines the complete logical file set. Leaves contain ordered partition descriptors with table,
key, content hash, byte/row counts, min/max Clock time, source IDs, schema fingerprint, and exact
covered `(frame_id, table)` projections. Readers walk the root; they do not glob. This supplies
bounded incremental metadata for an always-on archive without an unbounded delta chain or a flat
full-list rewrite.

`LOCAL-LATEST` and remote `LATEST` are discovery pointers, the immutable generation is transaction
authority, and its index root is dataset authority. Manifests never contain credentials, signed
URLs, response bodies, raw labels, or unredacted diagnostics.

### 8.4 Scrape-attempt table

One row exists per attempt or coalesced gap. Field order and nullability are normative:

| Field | Exact type | Null? |
|---|---|:---:|
| `archive_id`, `session_id` | `Uuid` | no |
| `source_id` | `Utf8` | no |
| `attempt_seq` | `UInt64` | no |
| `frame_id`, `batch_id` | `Digest` | no |
| `reason`, `outcome` | `Enum8` | no |
| `boundary_ids` | `List<Utf8 non-null>` | no |
| `scheduled_ns`, `request_start_ns`, `first_byte_ns`, `capture_ns`, `parse_done_ns`, `accepted_ns`, `local_durable_ns`, `remote_referenced_ns` | `Int64` | yes |
| `unix_epoch_ns` | `EpochNs` | yes |
| `http_status` | `UInt16` | yes |
| `latency_ns` | `Int64` | yes |
| `body_digest` | `Digest` | yes |
| `body_unchanged` | `Boolean` | no |
| `same_body_as_attempt_seq` | `UInt64` | yes |
| `sample_count` | `UInt64` | no |
| `error_kind`, `error_message` | `Utf8` | yes |
| `gap_first_seq`, `gap_last_seq`, `gap_first_deadline_ns`, `gap_last_deadline_ns`, `gap_count` | integer fields | yes |

`outcome` is one of `success`, `empty`, `http`, `transport`, `timeout`, `parse`,
`unsupported_format`, `unsupported_feature`, `missed`, `dropped`, `disabled`, or `shutdown`.
`body_unchanged` is valid only for successful/empty HTTP+parse observations; successful unchanged
attempts still have complete sample rows. Failed and empty scrapes are queryable rather than
inferred from absence.

### 8.5 Sample table

One row represents one metric family/base-label-set at one successful scrape:

| Field | Exact type | Null? |
|---|---|:---:|
| archive/session/source/frame/batch/attempt identity | same exact types as attempt table | no |
| `clock_ns`, `unix_epoch_ns` | `Int64`, `EpochNs` | no / yes |
| `metric_family`, `source_type_token` | `Utf8` | no |
| `semantic_type` | `Enum8` | no |
| `source_series_key`, `series_key` | `Digest` | no |
| `labels`, `attributes` | `StringMap` | no |
| `attribute_epoch_id` | `Digest` | no |
| `help`, `unit` | `Utf8` | yes |
| `source_timestamp_lexeme` | `Utf8` | yes |
| `source_timestamp_unix_ns` | `EpochNs` | yes |
| `payload` | structured value below | no |
| `wire_samples` | list of wire sample structs below | no |

`semantic_type` is `unknown`, `gauge`, `counter`, `stateset`, `info`, `histogram`,
`gauge_histogram`, or `summary`. `payload` is a non-null struct with nullable branches
`scalar`, `counter`, `stateset`, `info`, `histogram`, and `summary`; validation requires exactly
the branch selected by `semantic_type` (unknown/gauge use scalar, gauge-histogram uses histogram).
Branches use only `ArchiveNumber`, `StringMap`, lists, and these exact child structs:

- counter: `total`, `created`, and scalar exemplar;
- stateset: ordered list of `{state: Utf8, enabled: ArchiveNumber}`;
- info: its point label map;
- histogram/gauge-histogram: `sum`, `count`, `created`, and ordered buckets of
  `{upper_bound_lexeme: Utf8, upper_bound: ArchiveNumber, cumulative_count: ArchiveNumber,
  exemplar: Exemplar}`;
- summary: `sum`, `count`, `created`, and ordered quantiles of
  `{quantile_lexeme: Utf8, quantile: ArchiveNumber, value: ArchiveNumber}`.

Absent optional numeric components use `ArchiveNumber(kind="absent")`; the enclosing semantic
branch itself is nullable only for branch selection. `wire_samples` preserves every emitted sample
as `{emitted_name: Utf8, role: Enum8, labels: StringMap, value: ArchiveNumber,
source_timestamp_lexeme: Utf8?, source_timestamp_unix_ns: EpochNs?, exemplar: Exemplar?}`. This
retains the source name/role association rather than reconstructing it from suffixes later.
Histogram bounds sort numerically with `+Inf` last, but retain their lexemes; no lower bounds are
synthesized. Counts remain cumulative as emitted. Phase deltas belong to accumulators/views.

### 8.6 Lifecycle marker table

Markers connect history to runner facts without pretending they are samples. The exact schema is:

| Field | Exact type | Null? |
|---|---|:---:|
| `archive_id`, `session_id` | `Uuid` | no |
| `frame_id` | `Digest` | no |
| `marker_seq` | `UInt64` | no |
| `kind` | `Enum8` | no |
| `clock_ns`, `unix_epoch_ns` | `Int64`, `EpochNs` | no / yes |
| `run_id`, `phase_id`, `source_id` | `Utf8` | yes |
| `phase_state`, `completion_reason` | `Enum8` | yes |
| `phase_start_ns`, `sent_end_ns`, `requests_end_ns` | `Int64` | yes |
| `attribute_epoch_id` | `Digest` | yes |
| `attributes` | `StringMap` | no |

Kinds cover session/run lifecycle, exact phase `STARTED`/`SENDING_COMPLETE`/`COMPLETE`, source
state, topology change, archive degradation/recovery, local generation, and remote publication.
Phase fields are copied from one `PhaseObserver` snapshot; capture completion of a forced scrape is
a separate attempt timestamp.

### 8.7 Optional exact raw-body retention

The default stores only a keyed body digest. `RawBodyRetentionPolicy` may retain compressed exact
entity bytes for all or failed scrapes. Raw bodies do not pass through the structured sanitizer,
because doing so would destroy exactness. They are a separately classified artifact surface:

- configuration requires an explicit sensitive-data acknowledgment, restrictive local mode, and
  an `ArchiveRawKeyProvider` reference;
- remote retention requires authenticated encryption; object keys use a keyed plaintext digest,
  while nonce/algorithm/key ID and exact plaintext integrity digest live inside authenticated
  metadata;
- key material, plaintext digest, and bodies never appear in manifests/reports/logs;
- raw-body bytes count against receive, spool, transaction-reserve, and retention quotas.

Raw bodies are never embedded in Parquet rows. A report may state policy ID and retained-byte count,
not a signed access URL.

---

## 9. Ingress and writer isolation

### 9.1 Ownership topology

```text
current-thread LocalSet
    +-- source fetch drivers and native accumulator delivery
    +-- Clock maintenance driver
    `-- fixed-memory per-source loss ledger
              | bounded owned bytes + shared-decode credit
              v
bounded ordered decode CPU pool
              | native result, decoded archive entity, exact-entity lease
              v
LocalSet native delivery + nonblocking ArchiveProjectionPermit
              | ArchiveFrameDraft (unsequenced)
              v
single mutable archive-state owner
    +-- WAL and open-partition builders
    +-- immutable manifest/index pages and LOCAL-LATEST
    `-- bounded asynchronous immutable-object upload futures
              `-- verified receipts return to the same owner
```

Channels are per attempt/batch/control frame, never per token or individual metric. The archive
owner alone mutates WAL, partitions, heads, and remote publication state. Upload futures receive
only immutable files/bytes and cannot advance a head; a slow remote future is bounded/timed and is
polled without stopping local command/WAL progress. Parquet encoding, compression, filesystem
calls, and object-store clients do not run on the request `LocalSet`.

A LocalSet maintenance driver sleeps only through `Rc<dyn Clock>` and sends `Tick { now_ns }`,
rotate, checkpoint, sync, and retry commands. The worker never invents schedule time from wall time
or `tokio::time`. A small independent control channel is reserved for lifecycle markers,
maintenance, health snapshot, loss-ledger flush, and finalize commands so a full data queue cannot
deadlock shutdown. Control priority does not bypass the final accepted-sequence watermark in §10.

### 9.2 Admission modes

The same runtime supports two explicit policies:

- **primary watch:** the archive is the product. Before fetch/decode admission it reserves entity
  bytes, decode result, channel frame, WAL, worst-case temporary/open Parquet, manifest/index pages,
  raw CAS, file/inode, and emergency-finalization capacity. When any queue/spool/filesystem reserve
  is unavailable, the
  source driver does not issue unbounded new scrapes. It waits or skips future cadence deadlines
  according to the selected policy, records missed intervals, and fails the operation if local
  durability cannot progress within its budget. Default: fail rather than silently discard.
- **attached benchmark:** the benchmark is the product. Bounded shared/native decode is guaranteed
  once the telemetry attempt is accepted and native delivery occurs first. Archive projection then
  tries a nonblocking byte/frame/WAL reservation using its validated upper bound. Rejection
  updates a fixed-memory per-source loss ledger that coalesces contiguous attempt/deadline ranges
  by reason. The reserved control lane persists those ranges at checkpoint/finalize. If the writer
  itself is dead, the LocalSet-owned ledger reaches `ReportTelemetryArchive.health` on best-effort
  success; primary/required failures place it in the typed diagnostic artifact. The request
  path never waits. `archive.required=true` may convert archive degradation into a reporting-stage
  failed terminal after benchmark execution; it still cannot change measured request data or emit
  a partial result on the authoritative report path.

Boundary scrapes always reach their native accumulator even if archive admission fails.

### 9.3 Batch identity

`batch_id` uses the domain-separated, length-prefixed digest rule from §8.1 over archive/session/
source, source-record sequence, outcome, and the configured decoded-entity unchanged digest when
present. The archive owner stamps global `record_seq`; `frame_id` then includes frame schema/kind,
source/control identity, and that sequence. Marker and loss frames therefore share the same
persistence identity discipline as attempt/sample batches.

Retries of persistence retain `batch_id`/`frame_id`. Every issued request or compact gap range gets
a new source-record sequence; a persistence retry never does. Partitions and recovery deduplicate exact
`(frame_id, table)` projections, not a frame globally before all tables are covered.

---

## 10. Local durability and crash recovery

### 10.1 Durable genesis and sealed WAL segments

After all side-effect-free preparation succeeds, create/exact-resume acquires the exclusive archive
lock and commits generation zero before source activation, decode admission, or the first frame.
Genesis and its empty index root are written to content-addressed temporary files, file-fsynced,
renamed, directory-fsynced, then installed through a file- and parent-directory-fsynced
`LOCAL-LATEST`. A new session WAL header contains the verified genesis hash, schema fingerprints,
archive/session IDs, writer compatibility ID, and first accepted sequence. No frame is valid under
an unknown genesis.

WAL files are numbered immutable segments. One `.open` segment is append-only; complete segments
are footer-checksummed, file-fsynced, renamed `.wal`, and directory-fsynced. A frame is
length-delimited and contains wire version, frame/batch ID, accepted sequence, payload kind,
payload, and CRC. A corrupt/truncated final frame is never guessed past. A segment is never
front-truncated or rewritten to remove a prefix.

Acknowledgment vocabulary is exact:

1. **accepted:** the archive owner assigned `accepted_seq` and owns the frame, but it may exist only
   in memory;
2. **local durable:** the complete frame and required segment/directory metadata passed the
   configured fsync policy;
3. **receipt observed:** the producer received `AppendReceipt::LocalDurable`.

Every observed durable receipt is recovered exactly once. A crash after fsync but before response
may recover an uncertain local-durable frame that the producer did not observe; retrying
persistence uses the same frame ID. A new source scrape is never used to resolve receipt
uncertainty.

### 10.2 Immutable partition/index/head transaction

Physical table builders rotate independently, but logical coverage is per `(frame_id, table)`.
Each frame declares its required projections (`attempts`, zero-or-more `samples`, or `markers`/
loss). A partition footer carries exact frame/table coverage and row count; recovery never treats a
frame as globally committed merely because one table projection exists.

One local commit performs these ordered durability steps:

1. choose completed projections from a WAL prefix; keep any not-yet-rotated table projection
   pending;
2. write every due `part-<content-hash>.parquet.tmp`, finish footer, flush, file-fsync, rename to its
   content-addressed key, and directory-fsync;
3. copy-on-write the bounded manifest-index path adding partitions and coverage; file-fsync,
   rename, and directory-fsync every new content-addressed page;
4. write the immutable hash-linked generation transaction, file-fsync, rename, and
   directory-fsync;
5. write a new `LOCAL-LATEST.tmp` containing current and preceding valid heads, file-fsync,
   atomically rename it, and fsync its parent directory;
6. only now mark those exact projections committed;
7. unlink and directory-fsync only whole sealed WAL segments for which every required projection
   of every frame is covered by the authoritative index. A partially covered segment remains
   intact even if it repeats already covered projections.

The owner may group several ready table partitions into one generation, but does not force all
tables to share a physical rotation size. No mutable Arrow checkpoint participates in assembly.

### 10.3 Ordered stop and finalization watermark

Data and reserved control channels do not define commit order by receive order. Graceful stop:

1. stops source issuance;
2. atomically closes data admission and captures `final_accepted_seq`;
3. submits exact lifecycle and coalesced loss-ledger frames;
4. drains/locally-durably acknowledges every accepted sequence through the watermark, or persists
   an explicit loss frame for an attached-mode rejection that never became accepted;
5. rotates every open builder until all required frame/table projections are covered;
6. commits a `locally_finalized` generation and retires eligible whole WAL segments;
7. attempts requested remote publication under §11.

A finalize command received early on the control lane cannot bypass the watermark. Forced process
termination may interrupt any step; recovery resumes from the preceding authoritative head and
WAL.

### 10.4 Recovery cases and proof target

Recovery reads fixed `LOCAL-LATEST`, verifies its checksum, current/preceding head descriptors,
generation ancestry, genesis, index root/pages, and referenced partitions. It never repairs
identity by directory globbing. It deterministically handles:

- truncated/corrupt final open-WAL frame: discard only the incomplete non-durable suffix;
- complete durable frame with no observed receipt: recover it under its existing frame ID;
- frame projection absent from the index: replay only that missing table projection;
- repeated WAL frame whose subset of projections is indexed: skip only covered projections;
- temporary Parquet/index/generation files: delete after verification or leave as orphans;
- valid immutable unreferenced partition: adopt only when its exact coverage and content hash match
  pending WAL; otherwise leave for explicit GC;
- bad current generation/index: use the preceding descriptor retained in `LOCAL-LATEST`; if neither
  verifies, fail closed;
- referenced missing/corrupt local object: restore the exact hash from remote or fail closed;
- schema/config/source/identity-key/writer-compatibility mismatch: refuse resume before session or
  source activation.

Crash/property tests stop after every file flush, fsync, rename, directory fsync, head replacement,
WAL seal/unlink, watermark, local-finalization, and remote-publication edge. They prove:

```text
observed-durable frame projections ⊆ recovered projections exactly once
recovered projections = all complete local-durable frames (including uncertain receipts)
```

No frame/table pair appears twice in the resolved index.

### 10.5 Spool quota and transaction reserve

Remote targets require a local spool. Admission accounts separately for configured logical
bytes/files and actual filesystem free blocks/inodes. It preserves a conservative reserve for the
largest admitted WAL frame, every open/temporary Parquet builder, copy-on-write index path,
generation/head files, optional raw object, WAL seal, and emergency finalization. The reserve is
recomputed when limits/rotation state change and is unavailable to normal admission.

The manifest/head health records current/high-water use and reserve failures. Primary watch applies
its explicit policy before reserve violation; attached mode records visible loss. The writer never
deletes the only durable copy of a referenced partition or consumes finalization reserve merely to
extend normal collection.

---

## 11. Object-store synchronization

### 11.1 Required store capability seam

`object_store::ObjectStore` may back an adapter, but the archive depends on the narrower seam it can
actually prove:

```rust
#[async_trait]
pub trait ArchiveObjectStore: Debug + Send + Sync {
    fn capabilities(&self) -> ArchiveStoreCapabilities;
    async fn put_if_absent(&self, key: &str, body: Bytes, digest: Digest)
        -> Result<CreateReceipt, ArchiveStoreError>;
    async fn get_verified(&self, key: &str, expected: Digest)
        -> Result<Bytes, ArchiveStoreError>;
    async fn read_head(&self, key: &str) -> Result<VersionedHead, ArchiveStoreError>;
    async fn compare_and_swap_head(
        &self,
        key: &str,
        expected_version: &ObjectVersion,
        replacement: Bytes,
    ) -> Result<ObjectVersion, HeadConflict>;
}
```

Authoritative remote resume requires immutable create-if-absent, cryptographic exact-byte
verification, and linearizable versioned head CAS. GET followed by unconditional PUT is never a
CAS implementation. An ETag or size alone is not integrity; the default reads back and checks
BLAKE3, while a provider checksum is accepted only when the adapter proves a named cryptographic
algorithm covers the exact bytes. Credentials come from provider references/environment and never
serializable archive config.

The adapter declares named-object read-after-write visibility. A weaker store may be accepted only
with an authored consistency horizon: transient missing/unavailable referenced immutable objects
retry within that bound; hash mismatch fails immediately; expiry fails closed and reports
visibility lag separately from corruption.

### 11.2 Immutable publication protocol

Periodic sync drives bounded asynchronous uploads while the sole owner continues local WAL work:

1. create-if-absent every newly referenced partition and manifest-index page;
2. verify exact bytes by the §11.1 integrity contract;
3. create-if-absent every hash-linked generation from the remote head's descendant path;
4. verify the target generation and root reference only verified immutable objects;
5. conditionally replace `LATEST` from its exact object version/head hash with the new head;
6. only after CAS success mark covered frames remote-referenced.

Remote generation identities equal local generation identities; one CAS may advance over several
already uploaded ancestors but cannot coalesce or renumber them. `LATEST` contains archive ID,
local commit sequence, generation key/hash, index-root key/hash, parent hash, writer session ID, and
archive state. Manifest-generation keys are content-addressed/create-only, so a losing writer cannot
overwrite the winner's generation before losing CAS. Retries are physically and logically
idempotent.

### 11.3 Locking, ancestry, and reconciliation

An archive ID has one writer. Create-new uses a unique ID. Exact-resume takes a local exclusive lock
and conditions every remote head update on the exact version it read. A CAS conflict stops the
process; distributed merge is out of scope.

Before source activation, recovery compares verified hash-linked heads:

| Relationship | Action |
|---|---|
| equal | continue |
| remote is ancestor of local | upload the verified descendant path, then CAS forward |
| local is ancestor of remote, with no pending local WAL | verify/download exact remote objects and advance local head |
| local is ancestor of remote, with pending local WAL | fail closed as concurrent/stale history |
| neither is an ancestor | fail closed as divergence |
| remote absent | create only from validated local genesis |

No state is inferred from object listing. A remote descendant may be adopted only when archive,
schema, config, source, identity-key, and writer-compatibility identities match.

### 11.4 Lifecycle, sync-only resume, and partial availability

The operation lifecycle distinguishes `open`, `stop_requested`, `locally_finalized`,
`remotely_finalized`, and `failed`. `locally_finalized` is an immutable sealed generation;
`remotely_finalized` means verified remote `LATEST` references that exact sealed head and a local
publication receipt is durable. A locally finalized/remote-incomplete archive may start a fenced
sync-only resume. That mode may upload, verify, CAS, and record receipts, but cannot create a new
telemetry session, activate sources, admit frames, or reopen the sealed generation.

A network outage does not destroy locally durable history. Terminal status distinguishes local and
remote finalization, loss, and failure before local authoritative finalization. Reports return both
`head_uri` (mutable discovery) and the exact immutable generation URI/hash; no ambiguous
`archive-manifest.json` or global `final.parquet` is required.

### 11.5 Transactional compaction

Optional compaction runs only against a locally finalized archive under its exclusive lock. It
binds to an exact parent generation/root, writes and verifies immutable replacement partitions,
and proves complete `(frame_id, table)` coverage equality. One new generation atomically removes
old partition IDs and adds replacements in its copy-on-write index; then local head replacement and
remote CAS use that exact parent. A changed head leaves outputs unreferenced orphans. Old
partitions remain authoritative until the replacement head commits and are retained until a
separate GC policy proves no retained head references them.

---

## 12. Enrichment, redaction, and cardinality

### 12.1 Enrichment is additive

Static and discovered topology data goes into `attributes`, for example:

- hostname, cluster, namespace, pod;
- worker role/index/process;
- GPU UUID/index/model/PCI identity;
- deployment, region, availability zone;
- user-defined run/experiment tags.

Enrichers cannot rename metric families, remove source labels, aggregate CPU counters, collapse
devices, or change numeric values because their API receives an immutable sample view and returns
only an `AttributePatch`. Reserved `aiperf.*` attributes, duplicate keys, and post-patch limit
violations are errors. Derived/materialized views belong downstream. Discovered changes advance the
source's attribute epoch and emit a marker; they do not change series identity.

### 12.2 Redaction order

Pipeline order is:

```text
parse exact source
  -> canonicalize pre-redaction source identity
  -> keyed source_series_key
  -> additive attribute patch
  -> typed structured-surface sanitizer
  -> canonicalize stored identity/series_key
  -> encode
```

The sanitizer has capability-limited transformations for source descriptors, sample labels and
attributes, exemplar labels, marker attributes, diagnostics, and report/archive health fields. It
cannot change a numeric value or semantic role. Redaction affects stored labels and display
`series_key`, but protected keyed `source_series_key` preserves distinct source identity. The
manifest records sanitizer and identity-key provider IDs/config digests, never secret values.
Endpoint userinfo and configured secret headers are removed before any durable source descriptor
exists. Exact raw-body retention follows §8.7 instead of this structured pipeline.

### 12.3 Bounds

Validation/runtime enforce configured limits for:

- source count;
- label/attribute key and value byte length;
- labels per series;
- samples and histogram buckets per scrape;
- unique series per source/window;
- response body and retained raw-body size;
- compressed and decompressed entity bytes plus expansion ratio;
- diagnostic length.

Exceeding a bound produces an explicit failed/degraded scrape record. It never truncates labels or
silently merges series. Cardinality-limit implementations live behind a policy trait.

### 12.4 Prometheus HTTP security and negotiation

`prometheus_http` source config is strict and includes credential-provider ID, TLS trust roots and
optional mTLS provider references, redirect policy, proxy policy, accepted media versions,
content-encoding policy, connect/request/total deadline ceilings, and compressed/decompressed
limits. Raw secret values are never authored wire fields. Redirects and ambient proxies default to
disabled; enabling a named proxy requires an explicit provider/config and never changes the native
transport's loopback bypass silently. Cross-origin redirects require a separately acknowledged
policy and never forward credentials by default.

The request advertises only the two §8.1 formats, validates the response `Content-Type` and
`Content-Encoding`, applies compressed/decompressed limits while receiving, and rejects a metric-
looking non-2xx body before parse. TLS, DNS, connection reuse, byte capture, and Clock deadlines are
implemented by the prepared backend's `ControlPlaneHttp`, not a source-private client.

---

## 13. Native accumulator and phase integration

### 13.1 Preserve current semantics

The server/GPU/network formulas, boundary snapshots, reset-clamp, histogram learner, unit inference,
vLLM/SGLang atlas, GPU scaling, energy joins, and RTT delivery do not change. The delivery wrapper
gains physical attempt identity plus a phase-membership set. A phase-projection adapter presents the
record once to every active phase-local view; run-level source/endpoint fetch and update metadata is
ingested once per attempt ID. Duplicate attempt IDs are rejected. Exact parity tests pin every old
native summary across ordinary and seamless overlap before replacing phase-owned cadence loops.

The archive exposition projection may preserve families (for example summaries or `_created`)
that benchmark projection intentionally excludes. That is expected. One bounded decode job can
produce a strict archive entity and the separately named native-compatible entity described in §4;
archive completeness must not broaden benchmark metrics.

### 13.2 Attempt observation hook

Run-owned source composition gains an all-outcome observer before lossy domain projection rather
than file-writing branches on `ServerMetricsRecord`/`GpuScrape`:

```rust
pub trait TelemetryAttemptObserver<Record> {
    fn observe(
        &self,
        observation: &AttemptObservation<'_, Record>,
        context: &TelemetryObservationContext,
    );
}
```

`AttemptObservation` exposes attempt facts, strict/native parse dispositions, phase membership,
optional native record, and archive admission outcome—but neither decoded archive entity nor exact
bytes. Concrete consumers include native phase/run delivery, report health, and test recorders;
archive projection is already encapsulated inside the prepared driver. The runner assembles a small
generic/static tee at preparation. Every transport/parse/unsupported/success outcome is observed
once. Observation never occurs per token.

Boundary order remains:

1. submit a typed boundary command with an absolute Clock deadline;
2. fetch once and run one bounded decode job (strict plus named native fallback when applicable);
3. feed the supported native projection/boundary snapshot synchronously;
4. nonblockingly reserve/project/submit the archive frame draft or record an archive loss range;
5. return the phase barrier result or typed timeout/failure according to required-sidecar policy.

Archive remote durability is not awaited at a phase boundary.

### 13.3 Lifecycle markers

The archive installs a tee on the existing typed `PhaseObserver`. `STARTED`,
`SENDING_COMPLETE`, and `COMPLETE` markers copy the exact `PhaseConfig`/`PhaseStats` state,
`start_ns`, `sent_end_ns`, `requests_end_ns`, phase ID/kind, completion reason, and optional branch
facts delivered at the authoritative transition. Independently sampled
`ScheduledPhaseSidecar::on_phase_start/on_phase_end` timestamps are not lifecycle authority and are
not used for markers. Warmup/profiling names and run identity are typed fields. Forced-scrape
request/capture timestamps remain attempt facts, so a query can distinguish transition from sample
availability without post-hoc inference.

---

## 14. Protocol and configuration

### 14.1 Strict workload DTO

This design normatively amends the preimplementation runner-v2 authored model. Required common
fields remain `identity`, `artifact_target`, `backend`, and `workload`. Inference-scoped fields move
under a strict resource block:

```rust
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredRunSpecV2 {
    pub identity: RunIdentitySpecV2,
    pub artifact_target: PathBuf,
    pub backend: NamedRunnerComponentSpecV2,
    pub workload: NamedRunnerComponentSpecV2,
    #[serde(default)]
    pub resources: AuthoredRunResourcesV2,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthoredRunResourcesV2 {
    pub models: Option<ModelsSpec>,
    pub endpoints: Option<EndpointProfilesSpecV2>,
    pub metrics: Option<MetricsSpec>,
    pub artifacts: Option<ArtifactSpecV2>,
    pub sidecars: Option<SidecarSpecV2>,
}
```

Every `RunnerWorkloadFactory` returns `ResourceRequirementsV2`, classifying each field as
`required`, `optional`, or `forbidden`. Outer validation first checks common structure, the workload
factory strictly validates its raw config and requirements, then resource validation rejects absent
required and present forbidden blocks before backend validation. Scheduled/graph retain required
models/endpoints; standalone watch forbids models/endpoints/metrics/sidecars and allows optional
generic artifact policy. An empty resource block is therefore intentional, not permissive.

The complete authored projection below is deserializable by that revised DTO; omitted resource
fields are forbidden/unused rather than filled with dummy inference values:

```jsonc
{
  "protocol_version": 2,
  "operation": "execute",
  "expected_distribution_id": "blake3:<exact-runner>",
  "run": {
    "identity": {"benchmark_id": "watch-20260711-a"},
    "artifact_target": "/var/lib/aiperf/runs/watch-20260711-a",
    "backend": {
      "type": "online_http",
      "config": {"client": {"connect_timeout_ns": 10000000000}}
    },
    "workload": {
      "type": "telemetry_watch",
      "config": {
        "duration_ns": null,
        "shutdown_timeout_ns": 30000000000,
        "sources": [
          {
            "id": "node-a",
            "type": "prometheus_http",
            "interval_ns": 1000000000,
            "request_timeout_ns": 5000000000,
            "config": {
              "url": "https://node-a:9100/metrics",
              "credential_provider": "node-metrics",
              "tls": {"trust_provider": "cluster-ca", "mtls_provider": null},
              "redirects": "disabled",
              "proxy": "disabled",
              "accepted_formats": ["prometheus_text_0_0_4", "openmetrics_text_1_0_0"],
              "max_compressed_bytes": 8388608,
              "max_decompressed_bytes": 33554432
            },
            "attributes": {"role": "node", "cluster": "lab-a"}
          }
        ],
        "archive": {
          "target": "s3://benchmarks/watch/archive-id/",
          "local_spool": "/var/tmp/aiperf/archive-id",
          "spool_quota_bytes": 107374182400,
          "spool_quota_files": 100000,
          "required": true,
          "sink": {"type": "parquet_object_store", "config": {}},
          "rotation": {"type": "rows_bytes_age", "config": {}},
          "admission": {"type": "primary_durable", "config": {}},
          "recovery": {"type": "create_new", "config": {}},
          "identity_key": {"type": "secret_provider", "config": {"id": "archive-identity"}},
          "enrichers": [],
          "sanitizers": [],
          "raw_body": {"type": "none", "config": {}}
        }
      }
    },
    "resources": {}
  }
}
```

Python may accept friendly durations/paths; the runner wire uses normalized integer ns, absolute
local paths, and normalized target URIs. Unknown fields and unknown factory IDs fail validation.

### 14.2 Validation before side effects

Static validation covers:

- exact runner distribution/capabilities;
- workload resource requirements and typed `ControlPlaneHttp` backend capability;
- unique/valid source IDs and registered source types;
- positive intervals/absolute timeout budgets and bounded counts/sizes;
- credential/TLS/proxy/redirect/content-negotiation/compression policy without resolving secrets;
- target scheme/sink compatibility;
- local spool path safety, quota/transaction reserve, and artifact-target non-aliasing;
- policy IDs/configs and raw-retention acknowledgment;
- watch/backend compatibility;
- no benchmark-only fields/models/datasets/phases that would be inert.

Execution preparation then checks source-specific configuration, local target/spool state, remote
store capabilities/reachability where configured, exact-resume ancestry/identity, filesystem
reserve, and credential-provider availability without creating an authoritative archive. Only
after complete preparation does it acquire the archive lock, durably commit/verify genesis and
`LOCAL-LATEST`, start the IO/decode workers and Clock maintenance task, and activate sources.

### 14.3 Signals and stdout

Runner stdout retains exactly one terminal JSON line. Progress is structured stderr and/or an
optional local status artifact. Python forwards SIGINT/SIGTERM as a graceful stop request:

```text
PREPARED -> GENESIS_DURABLE -> RUNNING -> STOP_REQUESTED -> DRAINING
                                                       -> LOCALLY_FINALIZED
                                                       -> REMOTELY_FINALIZED
                                        any stage ----> FAILED
```

A second signal or expired shutdown budget may force termination; recovery must make the next
resume deterministic. The terminal response includes archive ID, local/remote head and immutable
generation locations/hashes, completeness, health counts, and failure stage. A sync-only resume
uses the same validate/execute envelope with recovery policy `finalize_remote`; it activates no
source.

---

## 15. Report and query contract

### 15.1 Native-v2

Every successful watch execution writes a minimal native-v2 outcome with common runner provenance,
empty request-metric maps, and a typed optional `ReportTelemetryArchive` block. It contains no
fabricated request distribution or benchmark duration. Attached runs add the same block to their
normal report:

```jsonc
{
  "telemetry_archive": {
    "schema_version": "1.0",
    "archive_id": "uuid",
    "session_id": "uuid",
    "state": "remotely_finalized",
    "local_head": {
      "head_uri": "file:///.../LOCAL-LATEST",
      "generation_uri": "file:///.../manifests/generation-7-blake3-....json",
      "generation_hash": "blake3:...",
      "index_root_hash": "blake3:..."
    },
    "remote_head": {
      "head_uri": "s3://.../LATEST",
      "generation_uri": "s3://.../manifests/generation-7-blake3-....json",
      "generation_hash": "blake3:...",
      "index_root_hash": "blake3:..."
    },
    "finalized_local": true,
    "finalized_remote": true,
    "lossy": false,
    "health": {}
  }
}
```

The block is an additive mode-specific extension expressly permitted by native-v2; its addition
does not change top-level schema version `2.0`. Implementation adds a typed DTO plus old-reader,
new-reader, absent-block, watch, and attached goldens. Credentials, raw labels, signed URLs, and
arbitrary diagnostics are excluded. Python presentation may link the archive and summarize source
health but cannot reinterpret it as native metrics.

If `archive.required=true` fails during the reporting/finalization stage, `RunTerminalV2` returns
`success=false`, `stage="reporting"`, and no authoritative `report_path`. Runner-v2 gains an
optional typed `diagnostic_artifacts` list with kind, relative path, and content hash. It may point
to an `archive_failure_diagnostic` and locally finalized immutable head, but that artifact is not a
`NativeReport`; Python outer-loop metric consumers must never load it as a result. This preserves
evidence without creating a partial authoritative report.

### 15.2 Query layout

Partition paths cluster by archive/session/table/source/time bucket without placing user labels in
object keys. Within each samples partition, rows sort by:

```text
(metric_family, series_key, clock_ns, attempt_seq)
```

Scrape partitions sort by `(source_id, attempt_seq)`. Parquet statistics and the manifest index's
min/max/source metadata enable pruning. The query resolver starts from a verified head/root and
walks the persistent index; it never globs. `metric_name_clean` is unnecessary because family
identity has its own column.

The first documentation examples use DuckDB/Polars/Arrow to:

- chart one family/label set over time;
- compute counter deltas with reset handling explicitly;
- inspect scrape failure/missed-tick intervals;
- join phase markers to telemetry;
- compare source cadence and response latency;
- find archive loss/degradation.

### 15.3 Schema evolution

Readers select the manifest/schema fingerprint before scanning partitions. A minor version may add
nullable columns or enum values only when v1 readers have defined unknown-value pass-through; it may
not change field order/type/nullability or reinterpret existing values. Every new fingerprint has
cross-reader goldens. An incompatible writer uses a new table/schema path and manifest major.
Compaction preserves original frame/batch IDs, source values, and declared input/output
fingerprints.

---

## 16. Failure semantics

### 16.1 Source failure

Source failures are data. They update health and scrape records. Configured retry/disable policy
decides future attempts. One source cannot cancel unrelated source tasks unless primary-watch
policy explicitly declares all sources required.

### 16.2 Archive failure

- WAL/local partition failure in primary watch: stop source issuance, attempt bounded finalization,
  return failed terminal.
- remote failure with healthy local spool: continue within quota, mark remote lag, retry; terminal
  may be locally finalized/remote-incomplete or fail if remote durability is required; sync-only
  resume can finish publication without reopening collection.
- attached best-effort archive failure: native benchmark continues; its successful report marks the
  archive degraded/lossy and includes persisted/coalesced loss ranges.
- attached required archive failure: benchmark execution may finish, but the runner returns a
  reporting-stage failure with no authoritative report path and only the §15.1 diagnostic-artifact
  surface.
- manifest identity/hash failure: fail closed; never guess or glob a replacement dataset.

### 16.3 Parse failure

Malformed exposition yields no partial successful sample batch by default. The scrape record keeps
the typed error. An optional parser policy may accept a standards-defined partial document only if
the outcome explicitly records partiality and exact rejected-line counts; the default is atomic.

### 16.4 Process crash

The latest valid local head, its verified immutable generation/index root, and sealed/open WAL
segments define recovery. Unreferenced remote objects are harmless orphans. A crash never makes
directory enumeration the logical dataset and never causes an old checkpoint to be concatenated
with its committed replacement.

---

## 17. Performance and capacity budgets

The archive is control-plane work, but attached mode must prove bounded data-plane impact. No scale
number is a product claim until a versioned `AcceptanceProfileV1` result is checked into the release
artifacts. The feature is currently unbuilt, so no supported profile is asserted by this document.

Each profile records exact runner distribution, commit/Cargo.lock, OS/kernel, CPU topology, RAM,
filesystem/object store, Arrow/Parquet/compression settings, query-reader versions, source count,
intervals, compressed/decompressed body sizes, points/buckets/labels/cardinality, duration,
request workload, remote fault schedule, and these measured outputs:

- samples and entity bytes per second plus unique series;
- archive CPU-core seconds, peak RSS, post-warmup RSS slope, queue/decode/writer lag;
- WAL/Parquet/manifest-index/spool growth and object-store throughput/requests;
- missed deadlines and source launch-lag percentiles;
- checkpoint, signal-to-admission-close, local-finalize, remote-finalize, and forced-recovery time;
- paired request throughput and p50/p95/p99 latency deltas with archival off/on.

At least seven randomized paired off/on trials run after an identical warmup. The gate reports the
median delta and a paired bootstrap 95% confidence interval. Attached archival passes only when:

- the confidence interval's lower bound for request-throughput delta is at least `-1.0%`;
- the upper bound for each p50/p95/p99 latency delta is at most `+2.0%` (and the absolute p99
  increase is at most 1 ms);
- no archive work appears in per-token callbacks and no request LocalSet queue grows unbounded;
- p99 source launch lag is at most `max(5 ms, 1% of interval)` for non-faulted sources;
- steady-state RSS slope after hour one is at most 1 MiB/hour and peak RSS stays below the profile's
  predeclared numeric budget;
- archive CPU, queue/writer lag, spool growth, and object-store request rates stay below the
  profile's predeclared numeric budgets;
- default graceful local finalization completes within 30 seconds after admission closes, unless
  the profile deliberately configures a larger bounded shutdown budget;
- a 24-hour accelerated/real soak has no missed acknowledged frames, no duplicate frame/table
  projections, no flat-history metadata rewrite, and no unbounded head/index-chain growth.

Before `telemetry_watch` or attached archival is advertised, the release must check in at least one
standalone and one attached profile with all numeric inputs/budgets/results populated and passing.
Documentation may state only the measured envelope of those profiles. A new dependency/runtime
version or material schema/writer change invalidates the profile until rerun.

---

## 18. Verification strategy

### 18.1 Parser/schema gates

1. exact content negotiation and golden corpora for Prometheus text 0.0.4 and OpenMetrics text
   1.0.0; unsupported format/version is typed, never retried under another grammar;
2. escaped labels, UTF-8, commas, quotes, backslashes, HELP/TYPE/UNIT, info, stateset, unknown/
   untyped, gauge histogram, summary, histogram, counter-created, source timestamps, and scalar/
   bucket exemplars retain emitted names/roles;
3. multiple histogram/gauge-histogram base label sets remain isolated and structured while native
   benchmark projection retains all intentional exclusions;
4. `100000001`, representative large counters, and exact UInt64 annotations survive every pinned
   Arrow/Parquet reader;
5. NaN/±Inf at every scalar/sum/count/bucket/quantile/exemplar leaf map to `ArchiveNumber` and no
   raw non-finite boundary value exists;
6. deterministic keyed pre-redaction identity, post-redaction identity, map order, digest domains,
   topology epochs, schema descriptors/fingerprints, manifest/index, and report goldens;
7. exact field/type/nullability/dictionary/metadata compatibility through pinned Arrow, Parquet,
   DuckDB, Polars, and pyarrow versions;
8. streaming size/cardinality limits, property parse/encode/decode round trips, and malformed-input
   atomic failure.

### 18.2 Scheduling gates

1. `SimClock` exact fixed deadlines with zero scrape time;
2. overrun skips debt without drift or catch-up bursts;
3. slow source A cannot shift source B deadlines;
4. one source never has two scrapes in flight;
5. absolute Clock timeout cancels/reclaims transport and yields one timeout attempt;
6. coincident seamless-phase boundaries coalesce while exact phase markers remain distinct;
7. bounded worst-case decode cannot stall unrelated source/request LocalSet work at the qualified
   profile;
8. failed/empty/unchanged/missed/dropped outcomes have exact counters/records and unchanged success
   retains full sample rows;
9. graceful signal closes admission and fixes the final accepted-sequence watermark before drain.

### 18.3 Durability/recovery gates

1. crash before/after genesis and every WAL append/fsync/seal, partition/index/generation write,
   file/directory fsync, head replacement, WAL unlink, watermark, and finalization edge;
2. receipt-observed durable projections are recovered exactly once; a complete fsynced but
   unobserved frame is recovered as an uncertain operation under the same ID;
3. independently rotated multi-table projections cannot make global dedup omit a table, and stale/
   repeated WAL frames cannot duplicate a frame/table pair;
4. finalize on the reserved control lane cannot overtake accepted data or loss-ledger frames;
5. corrupt/truncated files fall back through the preceding fixed head without deleting the last
   good generation or guessing from a directory;
6. transaction-reserve exhaustion and real ENOSPC/inode exhaustion fail before destroying the only
   durable copy;
7. create-if-absent retries, exact-byte verification, named-object visibility horizon, and
   conditional `LATEST` CAS are idempotent under competing writers;
8. every equal/ancestor/divergent local/remote reconciliation cell and sync-only finalization path;
9. exact-resume identity/writer mismatch fails before session/source activation;
10. exact-parent compaction failure leaves the old head authoritative and cannot expose duplicate
    replacement coverage;
11. the 24-hour profile produces immutable partitions and O(log₂₅₆ P) manifest-index update
    work rather than flat full-history rewrites.

### 18.4 Security gates

1. endpoint userinfo/auth headers/provider secrets/object-store credentials absent from every
   descriptor, sample, exemplar, marker, manifest, report, diagnostic artifact, log, and error;
2. sanitization covers every structured durable surface; keyed pre-redaction identity prevents
   silent series merge and defeats low-entropy dictionary tests;
3. compressed/decompressed body, label, exemplar, marker, attribute, diagnostic, series, and bucket
   bounds reject adversarial input during receive/parse rather than after unbounded allocation;
4. redirects/proxies/content negotiation/TLS/mTLS/credential forwarding obey strict defaults and
   a non-2xx metric-looking body never parses;
5. raw-body retention is off by default; opt-in requires classification, key provider, restrictive
   permissions, authenticated encryption for remote, and artifact-wide secret scanning;
6. path traversal and unsafe artifact/spool/target aliasing fail validation;
7. partition/index/generation/head hashes detect tampering/corruption.

### 18.5 Product subprocess gates

1. the complete §14.1 envelope with empty resources validates/deserializes, while missing required
   scheduled resources and present forbidden watch resources fail before preparation;
2. Python `aiperf watch` -> exact packaged runner -> in-process HTTP Prometheus mock -> durable
   genesis/WAL/Parquet/index/head -> terminal response;
3. multiple endpoint cadences including one slow/failing/oversized source and distinct per-call
   deadlines over the shared native transport;
4. HTTP 500 with metric-looking body remains an HTTP failure record, not a sample;
5. SIGINT/SIGTERM graceful finalization, forced-crash exact resume, and local-final/sync-only remote
   completion;
6. object-store emulator visibility lag, outage, conflicting CAS, restart, and finalization;
7. ordinary scheduled benchmark with attached server/GPU archive proves one physical run-owned
   driver/source feeds report and archive across seamless phases, `PhaseObserver` markers align
   exactly, and native metrics are unchanged;
8. required attached archive failure yields failed reporting terminal, no `report_path`, and only a
   typed diagnostic artifact; best-effort yields a successful typed archive block;
9. online-only runner capability advertises watch only after qualified profiles; unsupported
   distribution/backend capability fails before IO.

### 18.6 Query compatibility gates

Golden archives are read by pinned Arrow, DuckDB, Polars, and pyarrow versions. Queries begin at
local/remote heads, verify immutable generation/root hashes, walk the persistent index, and prove
structured label filtering, every semantic payload, phase joins, failure/loss-range discovery,
unchanged-success continuity, and partition pruning. No directory glob supplies file lists.

### 18.7 Performance/capacity gates

The versioned §17 harness validates schema completeness and all required numeric profile fields,
runs the paired bootstrap method, enforces every threshold, records dependency/hardware identities,
and rejects stale profiles after relevant lock/schema/writer changes. Capability generation checks
for passing standalone and attached profile artifacts rather than trusting documentation text.

---

## 19. Implementation increments

### Increment 1 — exposition and archive schema

1. implement the bounded Prometheus 0.0.4/OpenMetrics 1.0.0 `aiperf-prometheus` model/parser seam;
2. preserve current server/DCGM projection semantics with parity tests;
3. check in canonical Arrow descriptors; implement every semantic/numeric/exemplar/source-time
   payload, digest/identity rule, sanitizer surface, and golden Parquet/index/manifest/report;
4. add the five Tachometer regression fixtures as mandatory tests.

### Increment 2 — local writer and recovery

1. implement bounded ordered decode/projection, fixed-memory loss ledger, Clock maintenance driver,
   and single mutable archive owner;
2. add durable genesis, sealed framed WAL segments, independent table projection coverage,
   immutable Parquet, persistent manifest index, generation objects, and durable local head;
3. add every-step crash/property recovery matrix and transaction-reserved spool quotas;
4. ship no product command until exact-once recovery gates pass.

### Increment 3 — source runtime and watch product path

1. revise runner-v2 resources/requirements and implement the prepared `ControlPlaneHttp` capability;
2. implement strict secured source factories and one run-owned fixed-deadline driver per physical
   source;
3. register `telemetry_watch`, add strict Python Config-v2 projection and `aiperf watch` command;
4. add local archive, signal, failure, and query subprocess gates.

### Increment 4 — benchmark attachment

1. replace phase-owned cadence loops with run-owned source subscriptions and pre-projection
   all-outcome attempt tees;
2. emit exact lifecycle markers through a `PhaseObserver` tee;
3. add typed archive provenance/health and failure diagnostic-artifact protocol;
4. prove no extra scrapes, no metric drift, and no request-path backpressure.

### Increment 5 — object-store durability and resume

1. implement capability-gated archive-store adapters, bounded immutable uploads, strong
   verification, and conditional heads;
2. implement create-new/exact/sync-only policies and hash-ancestry reconciliation;
3. add visibility/CAS/outage emulator matrix, finalization lifecycle, and §17 profiles;
4. document operational recovery and orphan/GC procedures.

### Increment 6 — optional compaction and ecosystem docs

1. add manifest-transactional compaction without in-place history mutation;
2. publish Arrow/DuckDB/Polars examples and schema compatibility policy;
3. qualify and publish numeric standalone/attached scale profiles;
4. document sync-only recovery, key rotation constraints, orphan inspection, and external GC.

Each increment lands behind capabilities and tests. A library implementation is not product support
until the exact Python-to-runner subprocess gate passes.

---

## 20. Rejected alternatives

### Vendor the Tachometer crates

Rejected. Their useful ideas are small; their parser/schema/checkpoint/compaction semantics violate
native invariants and have reproduced corruption bugs.

### Add a `tachometer-scraper` or native `aiperf watch` binary

Rejected. It recreates a second native product/configuration surface. Python remains the human CLI
and `aiperf-runner` remains the only Rust executable.

### Run the watcher in Python

Rejected. It would duplicate native parser/source records, lose the shared Clock/transport/phase
seams, and require cross-process telemetry reconstruction.

### Make the archive the native metric input

Rejected. Persistence failures, queue admission, schema evolution, and remote lag must not alter the
authoritative live measurement path. Later offline analysis may consume archives explicitly.

### One flat row with labels inside `metric_name`

Rejected. It is lossy, hard to validate, vulnerable to escaping errors, and prevents typed family
semantics. Labels and values remain structured.

### Float32 to reduce archive size

Rejected. Prometheus values are f64 and common counters exceed Float32 consecutive-integer range.
Parquet compression is the size mechanism.

### Mutable Arrow checkpoint plus final concat

Rejected. It cannot prove non-overlap with committed Parquet and reproduced duplicate rows. WAL and
manifest generations provide explicit commit identity.

### Rewrite all history on every sync

Rejected. It is quadratic over a long watch and makes remote durability depend on increasingly
large transactions. Upload immutable new partitions once and copy-on-write only a bounded path in
the persistent manifest index.

### Use one overwritten flat manifest or a monolithic front-truncated WAL

Rejected. An overwritten manifest cannot supply a preceding valid generation, a flat full
partition array is quadratic for an always-on writer, and prefix mutation can destroy a pending WAL
suffix. Use immutable generation/index objects, a directory-durable head, and sealed whole-segment
retirement.

### Fire overlapping scrape tasks to maintain frequency

Rejected. It creates per-source races, unbounded work under slow endpoints, and ambiguous ordering.
Use one in-flight scrape per source and explicit missed ticks.

### Use wall clock for each sample

Rejected. Wall-clock steps destroy monotonic ordering. Bracket one injected epoch anchor, report its
capture uncertainty, and derive approximate cross-process placement from `Clock` deltas without
later remapping.

### Archive existing `ServerMetricsRecord`/`GpuScrape` values

Rejected. Those domain records intentionally discard unsupported families and failures. Archive an
all-outcome pre-projection envelope; feed native and archive projections from the same decoded
entity.

### Dynamic wide columns for arbitrary metadata

Rejected. Endpoint-dependent schemas complicate unions/evolution and invite name collisions. Use
structured attributes with stable core columns.

---

## 21. Completion criteria

This design is complete only when:

- Python exposes `aiperf watch` and no second Rust executable exists;
- the revised workload-scoped runner-v2 DTO deserializes the complete watch envelope and rejects
  required/forbidden resource violations;
- the exact runner derives and validates `online_http + telemetry_watch` from frozen factories and
  the typed `ControlPlaneHttp` capability;
- every physical source uses the injected Clock/native transport, one run-owned fixed-deadline
  driver, absolute deadline, and bounded ordered decode path;
- Prometheus text 0.0.4/OpenMetrics text 1.0.0 parsing preserves every advertised semantic role,
  source timestamp, exemplar, metadata, and numeric leaf without changing native benchmark
  projection semantics;
- canonical Arrow descriptors, keyed/pre-post redaction identities, tagged numbers, manifest index,
  and native-v2 DTO are deterministic and readable by the pinned query ecosystem;
- durable genesis, sealed WAL, independent table coverage, immutable partitions/generations/index,
  and local head recover every complete durable frame/table pair exactly once across all injected
  crash points, including uncertain receipts and finalization races;
- remote sync verifies create-only immutable objects and advances a linearizable conditional head
  without flat history rewrites; exact/sync-only resume reconciles ancestry fail-closed;
- exact resume fails closed on identity/config/schema/key/writer mismatch and concurrent writers;
- failures, gaps, unchanged bodies, misses, drops/loss ranges, local durability, visibility lag, and
  remote publication are observable;
- enrichment is API-limited to attributes, sanitization covers every structured surface, source
  identity survives redaction, and raw retention is separately protected;
- attached mode reuses one source attempt across active phases, emits exact `PhaseObserver` markers,
  leaves native results unchanged, and passes the numeric §17 regression profile;
- primary watch and attached modes have real Python-to-runner subprocess proofs;
- the five reproduced Tachometer defects are permanent regression tests;
- native-v2 2.0 additively identifies archive provenance/completeness without treating archived
  samples as native metrics, while required reporting failure emits no authoritative report path;
- checked-in passing standalone and attached acceptance profiles gate capability advertisement.

Until these gates pass, the existing phase-bounded native telemetry pipeline remains code truth and
no `watch` capability should be advertised.
