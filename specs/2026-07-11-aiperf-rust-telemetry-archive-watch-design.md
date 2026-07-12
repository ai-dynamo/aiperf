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

- current native code: `aiperf-clock`, `aiperf-transport-http`, `aiperf-server-metrics`,
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
- cross-archive joins or fleet catalog services;
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
8. **Exact numeric-token preservation.** Every accepted source number retains its exact source
   lexeme. A finite analytical `f64` is stored when representable, with an explicit exact/rounded/
   unavailable status; eligible integer tokens may additionally retain an exact `u64`.
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
    source descriptors, archive-key digest, and archive-writer compatibility ID. The exact runner
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

Every physical attempt has a stable source-attempt ID and the continuous active-phase-membership
set captured at its snapshot instant. Native phase-local projections consume that membership;
explicit boundary subscribers carried by `BoundaryStart`/`BoundaryEnd` commands receive the
snapshot regardless of whether their phase is currently active. Run-level source/
endpoint facts deduplicate by physical attempt ID, so seamless overlap neither loses one phase nor
counts two fetches. Archive-off versus archive-on parity on the run-owned driver is exact; comparison
to the former completion-paced loop separates formula parity from intentional cadence differences.

Boundary coalescing never uses timestamp proximity. The phase orchestrator assigns a typed
`coalescing_group_id` to exactly those transition subscribers that share one physical snapshot;
their lifecycle markers remain distinct. Commands without the same group never coalesce. Exact
phase lifecycle markers come from a `PhaseObserver` tee and persist the matching structured
boundary reference, while boundary scrape capture time remains
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
        context: &SequencedArchiveProjectionContext,
    ) -> Result<SequencedArchiveFrameDraft, ArchiveProjectionError>;
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
        -> Result<DurabilityCompletion, ArchiveSinkError>;
    async fn record_receipt(&mut self, event: ReceiptEventDraft)
        -> Result<AppendReceipt, ArchiveSinkError>;
    async fn checkpoint(&mut self) -> Result<CheckpointCompletion, ArchiveSinkError>;
    async fn finalize(&mut self, reason: TerminationReason)
        -> Result<FinalizeCompletion, ArchiveSinkError>;
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
owns byte/frame/WAL quota and is `Send`; the driver moves it, the decoded archive entity, and exact-
entity lease into a nonblocking `ProjectionReservation` sent to the archive owner. Denial records a
loss range without repeating parse or delaying native delivery. Primary watch may wait/fail before
fetch according to its durable admission policy because the archive is its product.

The single owner assigns the next inclusive global `record_seq` and final `frame_id` before any
authoritative row digest exists, then dispatches a `SequencedProjectionJob` carrying those values,
the permit, entity, lease, and the source's preceding attribute-epoch state to the second bounded
CPU pool. That job performs family/point construction, enrichment, sanitization, attribute-epoch
transition/marker insertion, canonical logical-row hashing, and final draft allocation off the
LocalSet. Hashes produced by decode or pre-reservation normalization are explicitly provisional and
cannot enter WAL coverage. The job returns `SequencedArchiveFrameDraft` plus the next epoch state.

Each source has a bounded FIFO projection strand with at most one active job; strands run in
parallel across sources. The permit covers both its queued slot and active footprint. The owner
starts source record N+1 only after N returns and updates the epoch chain, and commits returned jobs
through a bounded global `record_seq` reorder buffer. If a sequenced projection fails, the owner
persists a loss frame under that already reserved sequence. Thus there is no epoch fork, reordered
topology marker, missing global sequence, or pre-stamp coverage digest.

A per-driver drain tracker owns every outstanding reservation/job/permit; owner completion resolves
it. Finalization closes reservations and waits until each becomes a final draft or explicit loss
before the frame fence. The owner writes the resulting `ArchiveWalFrame`, a versioned persisted sum
of attempt/family/sample batch,
lifecycle, raw-object material/reference, and coalesced loss-range payloads. Receipt events use
their separately indexed journal and are never self-attesting WAL payloads. The frame is a closed
wire schema, not a source extension point. Every frame has stable source/control identity and CRC,
and every successful `append_frame` completion token has the same local-durable meaning. The token
travels through the LocalSet Clock bridge and `record_receipt` before the caller receives an
`AppendReceipt`; checkpoint/finalize publication completions use the same handshake.

Every wire-selected family—source, sink, rotation, admission, recovery, enrichment, sanitizer,
raw-body retention, and archive-key provider—has its own frozen descriptor/strict-validate/prepare
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
- source-record sequence for issued source events, plus independent loss/tick sequences;
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

Attached phase barriers submit a forced `BoundaryStart` or `BoundaryEnd` command with one typed
`BoundaryReference` (phase ID, boundary identity, start/end role, orchestrator-issued optional
`coalescing_group_id`) and absolute deadline. It
preempts the next continuous deadline but never interrupts an already issued HTTP request. Only
commands carrying the identical non-null coalescing group share a physical attempt; timestamp
proximity is irrelevant. All requesting structured boundary references are recorded on the attempt and receive the
same snapshot, while their lifecycle markers remain distinct. The phase waits for the forced result
under the same Clock deadline. Shared decode feeds native delivery first; archive projection uses
its independent permit. Continuous scheduling re-anchors from the original cadence, not the
boundary completion time.

### 6.4 Failure classification

Every attempt becomes one `ArchiveScrapeRecord`. `outcome` describes transport/parse disposition;
`body_unchanged` is an orthogonal success fact. V1 writes full family/MetricPoint rows for every successful
unchanged scrape rather than requiring readers to chase a prior sample batch. Issued-attempt
outcomes include:

- success with samples;
- success with an empty exposition;
- HTTP status failure;
- transport/timeout failure;
- parse failure with line/category and a redacted bounded diagnostic;
- source-incompatible terminal disable;
- source shutdown failure.

Missed ticks and archive admission/projection skips become typed §8.8 loss ranges. They are not
fabricated HTTP attempts.

Successful rows may carry `body_unchanged=true` and `same_body_as_source_record_seq`; health counts those
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
- archive enqueue observation;
- local durable receipt observation (receipt relation, not the attempt row);
- remote referenced receipt observation (receipt relation, not the attempt row).

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
`.series-source.v1`, `.attribute-epoch.v1`, `.body-encoded.v1`, `.body-decoded.v1`,
`.logical-row.v1`, `.projection-multiset.v1`, `.projection-coverage.v1`, `.raw-object.v1`,
`.receipt-target.v1`, `.receipt-event.v1`, `.partition.v1`, `.manifest.v1`, and `.index-node.v1`.
Canonical maps sort by UTF-8 key bytes and reject duplicate
keys. The configuration digest covers the fully validated secret-free authored config, every
selected factory ID plus normalized config, accepted-format/role-validity matrix, source
descriptors, schema/index fingerprints, writer compatibility ID, and archive-key provider ID.

Two series identities prevent redaction from silently merging source series:

- `source_series_key`: keyed BLAKE3 over the pre-redaction source ID, family, semantic type, and
  canonical source labels. The key comes from `ArchiveKeyProvider`; only its ID/digest is durable.
- `series_key`: ordinary BLAKE3 over the stored post-redaction identity plus
  `source_series_key`.

A many-to-one post-redaction mapping is visible because `source_series_key` differs. The default
sanitizer rejects it; intentional coalescing requires a named policy and a recorded outcome.
Enrichment attributes never enter either series key. `ArchiveKeyProvider` derives independent
keyed-BLAKE3 subkeys for source series, encoded body, decoded body, and raw-object naming from
frozen context strings; one output is never reused in another domain.

Discovered attribute changes increment a source-local `attribute_epoch_seq` and compute:

```text
attribute_epoch_id = digest(
  "aiperf.archive.attribute-epoch.v1",
  archive_id, session_id, source_id, attribute_epoch_seq,
  previous_attribute_epoch_id, canonical_post_sanitization_attributes
)
```

The topology marker and first sample referencing a new epoch are projections of one WAL frame; the
marker sorts first within that frame. Resume verifies the sequence/previous-ID chain. Unsanitized
attributes never enter an exposed digest.

Parsing has separate syntax and advertised-format semantic gates. A checked-in v1 role matrix
freezes the OpenMetrics constraints for counter, info, stateset, histogram, gauge-histogram,
summary, unknown, and gauge points (including integer/boolean, sign, cumulative bucket, required
`+Inf`, created, timestamp, exemplar, NaN/Inf, and metadata rules) plus the distinct Prometheus
0.0.4 rules. Any OpenMetrics role violation rejects the entire exposition atomically. Tagged
storage represents valid non-finite values only where the selected format/role permits them; it is
not a license to archive semantically invalid points.

### 8.2 Physical Arrow/Parquet contract

The repository contains one canonical UTF-8 JSON schema descriptor per table. It fixes field
order, names, nullability, Arrow logical/physical type, dictionary index width, child layout, and
schema metadata. Generated Arrow schemas are not the fingerprint authority. Each table stores:

```text
aiperf.archive.table = <attempts|families|samples|markers|losses|raw_references>
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
| `ArchiveNumber` | non-null `Struct<kind: Enum8, source_lexeme: Utf8 nullable, finite_value: Float64 nullable, exact_u64: UInt64 nullable, f64_status: Enum8>` |
| `SourceTimestamp` | non-null `Struct<lexeme: Utf8 nullable, normalized_unix_ns: EpochNs, status: Enum8>` |
| `CreatedTimestamp` | non-null `Struct<lexeme: Utf8 nullable, normalized_unix_ns: EpochNs, status: Enum8>` |
| `Exemplar` | nullable `Struct<labels: StringMap, value: ArchiveNumber, timestamp: SourceTimestamp>` |
| `BoundaryReference` | non-null `Struct<boundary_id: Utf8 non-null, phase_id: Utf8 non-null, role: Enum8 non-null, coalescing_group_id: Utf8 nullable>` |

`ArchiveNumber.kind` is `finite`, `pos_inf`, `neg_inf`, `nan`, or `absent`. Source tokens always
retain their exact lexeme; only synthetic `absent` has a null lexeme. `finite_value` is the
analytical IEEE-754 projection when representable. `exact_u64` independently preserves any
non-negative integer lexeme fitting UInt64, including values above 2⁵³; it does not require exact f64
conversion. `f64_status` is `exact`, `rounded`, `unavailable`, or `not_applicable`. Every numeric
leaf—including sums, counts, bounds, buckets, quantiles, states, and exemplars—uses this
representation. Semantic OpenMetrics Created values use `CreatedTimestamp`; an arbitrary classic
sample merely named `_created` remains an `ArchiveNumber` unless the selected semantic parser
assigned the Created role. No raw non-finite Float64 is written.

`SourceTimestamp.status` is `absent`, `exact_ns`, `sub_ns_precision`, or `out_of_range` and never
rounds silently. Prometheus 0.0.4 accepts its specified integer Unix milliseconds and checked-
multiplies by 1,000,000. OpenMetrics 1.0 parses decimal Unix seconds and normalizes only when exact
integer nanoseconds fit `EpochNs`; otherwise the lexeme remains authoritative and normalized value
is null. The attempt records declared media type and actual strict parser grammar/version, so a
reader never infers timestamp units from a metric type.

`CreatedTimestamp` has the same physical child types and status vocabulary but a distinct logical
alias/schema field so readers cannot treat a creation epoch as a sample observation timestamp.
Its grammar and units come from the selected exposition format's Created production.

Maps are non-null (possibly empty); list elements and map keys are non-null. Parquet dictionary
encoding is an implementation choice, but the logical Arrow dictionary index is always Int8 for
frozen enums.

Increment 1 must check the canonical descriptors into the repository, generate schemas from them,
and freeze exact Arrow IPC/Parquet/DuckDB/Polars goldens before any archive capability is
advertised. “Binary or hex,” alternate decimal encodings, and unspecified exemplar layouts are not
conforming v1 writers.

### 8.3 Manifest graph and heads

#### 8.3.1 Canonical logical-row bytes

Compaction and recovery equality use `aiperf.archive.logical-row-encoding.v1`; Arrow builders,
Parquet encoders, dictionary choices, and JSON rendering are never digest authority. A canonical
row begins with the fixed ASCII magic/version, table ID, 32-byte schema fingerprint, and schema
field count, then encodes fields in descriptor order. Every nullable value begins with `0x00` for
null or `0x01` for present. Present values use this exact schema-directed encoding:

- booleans are one byte `0x00`/`0x01`; signed and unsigned integers are their fixed-width
  big-endian representation;
- finite `Float64` is the exact big-endian IEEE-754 bit pattern, preserving negative zero;
  non-finite values exist only through the tagged `ArchiveNumber` struct;
- `Decimal128(38,0)` is the signed two's-complement 16-byte big-endian integer;
- UTF-8 and variable binary are `u64_be(byte_length) || exact_bytes`; fixed binary is exact bytes;
- a dictionary/enum value is its logical UTF-8 string, never its Arrow dictionary index;
- a struct encodes children in descriptor order, including each nullable child's null/present tag; a list
  encodes `u64_be(element_count)` followed by ordered elements; a map first rejects duplicate keys,
  sorts v1's UTF-8 keys by their exact UTF-8 byte sequence, and then encodes its count and ordered
  key/value pairs;
- `ArchiveNumber`, timestamp, exemplar, payload, and wire-sample values are ordinary nested structs
  under these same rules; no implementation-specific padding or validity bitmap enters the bytes.

The row digest is `digest("aiperf.archive.logical-row.v1", schema_fingerprint, table_id,
canonical_row_bytes)`. Independent Rust and Python verifier fixtures freeze every primitive,
negative zero, maximum integer/decimal, null nesting, enum, list, map-order, semantic payload, and
full row. Python is a conformance reader/verifier, not a second product writer.

#### 8.3.2 Indexed object graph

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
genesis containing archive/schema/writer identity, the normalized config digest, archive-key
digest, canonical-spool ID, secret-free source descriptors, session/anchor, exact runner
distribution provenance, and empty index root. Later generations are bounded transaction records containing parent, session,
added/removed partition/raw-object IDs, exact logical projection evidence, health delta, state transition, and
termination reason. `unix_epoch_ns` is a decimal string in JSON.

Head, generation, and index-node JSON each have a checked-in canonical descriptor/version/
fingerprint. Canonical bytes are UTF-8, object keys in byte order, no insignificant whitespace,
integers in minimal decimal, lowercase fixed-width hex digests, and no JSON floats. Each envelope
stores magic/type/version, payload byte length, payload, and BLAKE3 checksum; its content key hashes
those exact canonical bytes.

The partition/raw-object/coverage descriptor set is a persistent content-addressed B-tree. Its composite search key
is `(object_kind_u8, table_id_u8, source_id_utf8, min_clock_ns_sortable_i64,
logical_object_id_digest)` with lexicographic byte comparison. A leaf stores 1–256 sorted partition
raw-object, or projection-coverage descriptors. An internal page stores 2–256 ordered children, each with inclusive
min/max composite key, child key/hash/byte length, exact sorted source IDs, table/object-kind mask,
and per-table min/max Clock time for pruning. Insert overflow at 257 entries deterministically
splits 128 left/129 right; deletion does not merge during normal append. Pages are at most 1 MiB;
validated source/cardinality limits make 256-entry worst cases fit. The root descriptor carries
height, logical entry count, and the same aggregate pruning summary.

Partition descriptors contain table, key, physical content hash/bytes/rows, min/max Clock time,
source IDs, schema fingerprint, and per-projection evidence:

```text
(frame_id, table, logical_row_count, logical_multiset_digest)
```

Each logical row has a domain-separated digest over the canonical schema-level value (not Arrow/
Parquet bytes). A projection multiset digest sorts its row digests lexicographically and hashes the
length-prefixed sequence.

Every frame declares its required table projections. Each required `(frame_id, table)` has a
persistent `ProjectionCoverage` index entry containing frame ID, table, source ID, min/max Clock
time, `row_count`, logical multiset digest, and ordered `fragment_ids`. V1 never splits one
frame/table projection across physical partitions: a builder appends or rotates the entire
projection atomically, so `fragment_ids` has zero or one element. Zero-row projections have
`row_count=0`, the digest of the empty row-digest sequence, and an empty fragment list; they are
still committed evidence rather than inferred from absence. Parser/cardinality limits and the
validated maximum frame footprint must fit one hard partition bound. A configuration whose worst-
case projection cannot fit is rejected before activation, and a runtime bounds violation becomes a
loss frame rather than a partially accepted projection.

A commit touching K descriptors copy-on-writes O(K log₂₅₆ P) bounded
pages and one bounded generation transaction. The root hash defines the complete logical object
set. Readers walk and prune this exact contract; they do not glob. Independent-reader goldens build,
split, delete, verify, and range-scan trees and bound page reads for source/time predicates.

`LOCAL-LATEST` and remote `LATEST` are discovery pointers, the immutable generation is transaction
authority, and its index root is dataset authority. Manifests never contain credentials, signed
URLs, response bodies, raw labels, or unredacted diagnostics.

### 8.4 Scrape-attempt table

One row exists per issued scrape event; missed cadence and admission-loss ranges use §8.8 instead
of impersonating requests. Field order and nullability are normative:

| Field | Exact type | Null? |
|---|---|:---:|
| `archive_id`, `session_id` | `Uuid` | no |
| `source_id` | `Utf8` | no |
| `record_seq`, `source_record_seq` | `UInt64` | no |
| `request_attempt_seq` | `UInt64` | yes |
| `frame_id`, `batch_id` | `Digest` | no |
| `reason`, `outcome` | `Enum8` | no |
| `boundary_refs` | `List<BoundaryReference non-null>` | no |
| `declared_media_type`, `strict_parser_format`, `native_compatibility_format` | `Utf8` | yes |
| `native_compatibility_fallback` | `Boolean` | no |
| `scheduled_ns`, `request_start_ns`, `first_byte_ns`, `capture_ns`, `parse_done_ns`, `archive_enqueue_ns` | `Int64` | yes |
| `unix_epoch_ns` | `EpochNs` | yes |
| `http_status` | `UInt16` | yes |
| `latency_ns` | `Int64` | yes |
| `decoded_body_digest`, `encoded_body_digest` | `Digest` | yes |
| `raw_object_id` | `Digest` | yes |
| `body_unchanged` | `Boolean` | no |
| `same_body_as_source_record_seq` | `UInt64` | yes |
| `family_count`, `metric_point_count`, `wire_sample_count` | `UInt64` | no |
| `error_kind`, `error_message` | `Utf8` | yes |

`outcome` is one of `success`, `empty`, `http`, `transport`, `timeout`, `parse`,
`unsupported_format`, `unsupported_feature`, `disabled`, or `shutdown`.
`body_unchanged` is valid only for successful/empty HTTP+parse observations; successful unchanged
attempts still have complete family/sample rows. `source_record_seq` advances once for an issued
source event; `request_attempt_seq` advances only when network IO is issued and may be null for a
pre-IO timeout/disable. `record_seq` is the archive-owner global frame sequence. Failed and empty
scrapes are queryable rather than inferred from absence.

The keyed decoded-exposition digest drives unchanged detection and batch identity, so two different
gzip encodings of identical decoded exposition are unchanged. Encoded-body digest is stored only
when a configured raw/integrity policy uses it. Both use distinct `ArchiveKeyProvider` subkeys.
Declared media type, strict actual grammar, and native compatibility grammar are normalized frozen
IDs, not free-form response header copies.

### 8.5 Metric-family metadata table

One row exists for every parsed family, including a valid family with HELP/TYPE/UNIT but zero
metrics/points:

| Field | Exact type | Null? |
|---|---|:---:|
| `archive_id`, `session_id` | `Uuid` | no |
| `source_id` | `Utf8` | no |
| `frame_id`, `batch_id` | `Digest` | no |
| `record_seq` | `UInt64` | no |
| `family_seq` | `UInt64` | no |
| `metric_family`, `source_type_token` | `Utf8` | no |
| `semantic_type` | `Enum8` | no |
| `help_present`, `type_present`, `unit_present` | `Boolean` | no |
| `help`, `unit` | `Utf8` | yes |
| `help_line_seq`, `type_line_seq`, `unit_line_seq` | `UInt64` | yes |
| `metric_count`, `metric_point_count`, `wire_sample_count` | `UInt64` | no |

Presence bits distinguish missing metadata from a present empty value. A completely empty
MetricSet has no family rows but its successful attempt has all three counts zero; a metadata-only
family has a family row and point/sample counts zero. Families are required projections for every
strictly successful parsed attempt and participate in WAL/index/compaction evidence.

### 8.6 Metric-point sample table

One row represents one ordered MetricPoint, not merely one family/label set. OpenMetrics metrics
with multiple points therefore produce multiple rows:

| Field | Exact type | Null? |
|---|---|:---:|
| `archive_id`, `session_id` | `Uuid` | no |
| `source_id` | `Utf8` | no |
| `frame_id`, `batch_id` | `Digest` | no |
| `record_seq` | `UInt64` | no |
| `family_seq`, `metric_point_seq` | `UInt64` | no |
| `clock_ns`, `unix_epoch_ns` | `Int64`, `EpochNs` | no / yes |
| `metric_family`, `source_type_token` | `Utf8` | no |
| `semantic_type` | `Enum8` | no |
| `source_series_key`, `series_key` | `Digest` | no |
| `labels`, `attributes` | `StringMap` | no |
| `attribute_epoch_id` | `Digest` | no |
| `point_time_status` | `Enum8` | no |
| `source_timestamp` | `SourceTimestamp` | no |
| `payload` | structured value below | no |
| `wire_samples` | ordered list of point-owned wire sample structs below | no |

`semantic_type` is `unknown`, `gauge`, `counter`, `stateset`, `info`, `histogram`,
`gauge_histogram`, or `summary`. `payload` is a non-null struct with nullable branches
`scalar`, `counter`, `stateset`, `info`, `histogram`, and `summary`; validation requires exactly
the branch selected by `semantic_type` (unknown/gauge use scalar, gauge-histogram uses histogram).
Branches use only `ArchiveNumber`, `StringMap`, lists, and these exact child structs:

- counter: `total`, `created: CreatedTimestamp`, and scalar exemplar;
- stateset: ordered list of `{state: Utf8, enabled: ArchiveNumber}`;
- info: its point label map;
- histogram/gauge-histogram: `sum`, `count`, `created: CreatedTimestamp`, and ordered buckets of
  `{upper_bound_lexeme: Utf8, upper_bound: ArchiveNumber, cumulative_count: ArchiveNumber,
  exemplar: Exemplar}`;
- summary: `sum`, `count`, `created: CreatedTimestamp`, and ordered quantiles of
  `{quantile_lexeme: Utf8, quantile: ArchiveNumber, value: ArchiveNumber}`.

Absent optional numeric components use `ArchiveNumber(kind="absent")`; absent Created components
use `CreatedTimestamp(status="absent")`. The enclosing semantic branch itself is nullable only for
branch selection. `wire_samples` preserves every emitted sample
as `{emitted_name: Utf8, role: Enum8, labels: StringMap, value: ArchiveNumber,
source_timestamp: SourceTimestamp, exemplar: Exemplar?}`. Each wire sample belongs to exactly this
MetricPoint; `metric_point_seq` is source order by the first contributing wire sample. This
retains the source name/role association rather than reconstructing it from suffixes later.
Histogram bounds sort numerically with `+Inf` last, but retain their lexemes; no lower bounds are
synthesized. Counts remain cumulative as emitted. Phase deltas belong to accumulators/views.

`point_time_status` is `all_absent`, `uniform_explicit`, `mixed_components`, or
`partial_components`. `source_timestamp` carries the common value only for `uniform_explicit` and
is `absent` for `all_absent`, `mixed_components`, and `partial_components`; every component's exact
timestamp remains on its `wire_samples` entry. A classic histogram/summary assembled from unequal
or partly present component timestamps remains a structured point, but readers may not interpret it
as one source-time snapshot. `clock_ns` remains the authoritative capture timeline in all cases.

### 8.7 Lifecycle marker table

Markers connect history to runner facts without pretending they are samples. The exact schema is:

| Field | Exact type | Null? |
|---|---|:---:|
| `archive_id`, `session_id` | `Uuid` | no |
| `frame_id` | `Digest` | no |
| `record_seq` | `UInt64` | no |
| `marker_seq` | `UInt64` | no |
| `kind` | `Enum8` | no |
| `clock_ns`, `unix_epoch_ns` | `Int64`, `EpochNs` | no / yes |
| `run_id`, `phase_id`, `source_id` | `Utf8` | yes |
| `phase_state`, `completion_reason` | `Enum8` | yes |
| `boundary_id`, `coalescing_group_id` | `Utf8` | yes |
| `boundary_role` | `Enum8` | yes |
| `phase_start_ns`, `sent_end_ns`, `requests_end_ns` | `Int64` | yes |
| `attribute_epoch_id` | `Digest` | yes |
| `attributes` | `StringMap` | no |

Kinds cover session/run lifecycle, exact phase `STARTED`/`SENDING_COMPLETE`/`COMPLETE`, source
state, topology change, and archive degradation/recovery. A marker never claims successful
durability/publication of the generation containing itself. Those later facts use §8.9 receipts and
head state. Phase fields and optional boundary reference are copied from one `PhaseObserver`
snapshot. `boundary_id`, `phase_id`, `boundary_role`, and `coalescing_group_id` must equal the
corresponding attempt `BoundaryReference`; lifecycle transitions with no forced snapshot leave all
three boundary fields null. Capture completion of a forced scrape is a separate attempt timestamp. A topology marker and the first point/family rows for
its epoch share one frame, with marker logical row order first.

### 8.8 Loss-range table

Missed cadence and rejected already-issued work have different identities and never share attempt
columns. The exact loss schema is:

| Field | Exact type | Null? |
|---|---|:---:|
| `archive_id`, `session_id` | `Uuid` | no |
| `source_id` | `Utf8` | yes |
| `frame_id` | `Digest` | no |
| `record_seq`, `loss_seq`, `count` | `UInt64` | no |
| `loss_kind`, `reason` | `Enum8` | no |
| `first_source_record_seq`, `last_source_record_seq` | `UInt64` | yes |
| `first_request_attempt_seq`, `last_request_attempt_seq` | `UInt64` | yes |
| `first_tick`, `last_tick` | `UInt64` | yes |
| `first_deadline_ns`, `last_deadline_ns` | `Int64` | yes |
| `boundary_refs` | `List<BoundaryReference non-null>` | no |
| `boundary_overflow_count` | `UInt64` | no |
| `boundary_overflow_digest` | `Digest` | yes |

`loss_kind` distinguishes at least `missed_cadence` (nothing issued), `archive_rejected` (native
work issued/delivered but archive projection denied), `projection_failed`, `writer_failed`, and
`shutdown_abandoned`. The role-validity matrix fixes which range pairs are required or forbidden:
missed cadence uses tick/deadline ranges and has no source/request range; issued-work loss uses its
source sequence and optional request sequence. Ranges coalesce only when kind/reason/source and all
present identities are contiguous. Boundary references are retained up to a validated bound;
overflow is counted and digested over canonical `BoundaryReference` bytes, never silently
truncated. The reserved control lane persists loss frames even
when ordinary archive admission is exhausted.

### 8.9 Non-self-referential durability receipts

Attempt/family/sample rows never predict their own local durability or later remote reference.
These observations live in a separate append-only receipt relation discovered through
`LOCAL-RECEIPTS`. Receipt targets are a closed discriminated union:

- `wal_range` targets archive/session, WAL segment ID, exact durable-prefix hash, inclusive first/
  last global `record_seq`, and the digest of those frames' declared projection coverage;
- `publication` targets the sealed generation hash, index-root hash, installed head hash, verified
  CAS object version, resulting archive state, and resulting writer-claim state. A terminal
  publication target requires `writer_claim_state=absent` because the same final head CAS clears
  the active claim; there is no second claim-release CAS.

`receipt_target_id` is a domain-separated digest of that immutable semantic target and never
contains an observation time. A separate immutable `ReceiptEvent` contains `receipt_seq`, target
ID, observation kind (`response_observed` or `recovery_verified`), exactly one of
`response_observed_ns`/`recovery_verified_ns`, and an `event_id` that hashes target, kind, and time.
Both times come from the injected `Clock`. Recovery never invents or backfills a response-observed
time; it may append a distinct recovery-verification event after independently proving the target.

Owner/worker completion crosses a Clock bridge before becoming a receipt:

1. the archive owner proves a WAL durable extent or verified publication and sends an immutable
   completion token to the source/run LocalSet;
2. that LocalSet observes the token, stamps `response_observed_ns = Clock::now_ns()`, and returns a
   `ReceiptEventDraft` to the receipt owner;
3. the receipt owner durably indexes the event, then resolves the caller's `AppendReceipt`.

A crash between steps 1 and 2 leaves durable data but no response observation. Recovery verifies
the target and may create `recovery_verified`; OS-worker completion time is never substituted.

Receipts are stored in immutable canonical batches of at most 1,024 events and at most 1 MiB.
Contiguous WAL targets from one session/segment may coalesce before batching when their record
ranges and declared-coverage digests compose exactly. A separate content-addressed persistent
B-tree uses the §8.3 page format and deterministic split rule, keyed lexicographically by
`(target_kind_u8, session_id_or_zero_16, first_record_seq_or_zero_u64,
generation_hash_or_zero_32, receipt_seq_u64, event_id_32)`. The unused discriminator fields must be
all-zero bytes, so two writers cannot choose different encodings. A checksummed
`LOCAL-RECEIPTS` pointer contains current and preceding receipt-head descriptors; each head binds
the receipt root hash, logical event count, and last receipt sequence. Batch, copied index pages,
receipt head, pointer replacement, and parent directory are flushed/fsynced in the same order as
the primary manifest protocol. Readers start at this pointer and never glob receipt files, so
lookup and restart work remain bounded as the archive grows.

Each receipt attests only an earlier WAL range or sealed generation, never its own durability.
Loss of a receipt cannot remove covered data; it makes response observation unknown. If remote CAS
succeeds but its response is uncertain, sync-only rereads and verifies the exact head/version/
absent-claim state, then appends a `recovery_verified` publication event. `remotely_finalized`
requires a durable local publication event. Receipt heads may advance after the primary generation
is sealed and do not reopen frame admission. Query adapters expose targets/events as the bounded,
joinable `durability_receipts` relation.

### 8.10 Optional exact raw-body retention

The default stores only the keyed decoded-exposition digest used for unchanged detection.
`RawBodyRetentionPolicy` may retain the exact received encoded entity for all or failed scrapes.
Raw bodies do not pass through the structured sanitizer,
because doing so would destroy exactness. They are a separately classified artifact surface:

- configuration requires an explicit sensitive-data acknowledgment, restrictive local mode, and
  an `ArchiveRawKeyProvider` reference;
- every retained body uses an AEAD envelope. The equality ID uses the raw-object subkey over exact
  encoded bytes. Its public header contains envelope version, algorithm, key ID, and random nonce;
  exact plaintext digest/length/content-encoding are inside encrypted plaintext metadata, not AAD;
- key material, plaintext digest, and bodies never appear in manifests/reports/logs;
- raw-body bytes count against receive, spool, transaction-reserve, and retention quotas.

Because exact endpoint bytes can echo an otherwise known credential, this opt-in encrypted
artifact is the explicit exception to structured known-credential absence. It never becomes a
dimension, descriptor, diagnostic, report field, object key, or plaintext log; classification,
encryption, restrictive access, retention, and secret-scanning policy govern the exact artifact.

Randomized encryption is performed exactly once per equality ID, not once per referencing frame.
The projection worker returns `RawObjectCandidate { raw_object_id, exact_entity_lease }`; it never
chooses a nonce. The single archive owner serially consults its committed-plus-pending raw registry.
For a new ID it consumes the lease, creates one random-nonce envelope, and writes the complete
ciphertext/nonce/digest into the accepting WAL frame before acknowledging it. For an indexed or
pending ID it reuses the exact existing physical descriptor and drops the redundant lease without
decrypting/re-encrypting. Recovery rebuilds the registry from the verified index plus WAL before
accepting drafts. Thus concurrent duplicates cannot create different ciphertext for the same
content-addressed key, and every create-if-absent retry supplies byte-identical envelope bytes.
Nonce derivation from plaintext or equality ID is forbidden.

Every retained frame has one row in the `raw_references` table:

| Field | Exact type | Null? |
|---|---|:---:|
| `archive_id`, `session_id` | `Uuid` | no |
| `source_id` | `Utf8` | no |
| `frame_id`, `batch_id`, `raw_object_id` | `Digest` | no |
| `record_seq`, `source_record_seq` | `UInt64` | no |
| `retention_reason` | `Enum8` | no |

The frame declares `raw_references` as a required table projection with normal §8.3 coverage. The
physical raw-object index descriptor is shared and contains only object ID/key, ciphertext digest/
bytes, envelope algorithm/key ID, and required local/remote coverage—never a frame ID, plaintext
digest, or key material. The attempt row repeats the opaque object ID for convenient joins. WAL
retirement, requested finalization, and remote head CAS wait for every reference projection plus
the one applicable verified physical object. Compaction preserves references and descriptors;
head reachability drives GC. Crash tests cover first-object creation, duplicate frames before and
after indexing, concurrent candidates, and every envelope/register boundary. Raw bodies are never
embedded in other Parquet rows. A report may state policy ID and retained-byte count, not a signed
access URL.

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
              | ProjectionReservation (entity/lease/permit)
              v
single mutable archive-state owner
    +-- assign record_seq/frame_id
    `-- schedule bounded per-source FIFO strands
              | SequencedProjectionJob
              v
bounded archive-projection CPU pool (one active job/source)
              | SequencedArchiveFrameDraft or typed failure
              v
same archive owner: ordered result/failure commit
    +-- WAL and open-partition builders
    +-- immutable manifest/index pages and LOCAL-LATEST
    `-- bounded asynchronous immutable-object upload futures
              | fsync/CAS completion token
              v
LocalSet Clock bridge
              | ReceiptEventDraft
              `---------------------> same archive owner: receipt journal
                                      | durable AppendReceipt
                                      `----------------------> LocalSet caller
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
deadlock shutdown. Control priority does not bypass the final owner-assigned record-sequence fence
in §10.

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
present. On accepting a projection reservation, the archive owner stamps global `record_seq`;
`frame_id` then includes frame schema/kind, source/control identity, and that sequence. Both enter
the immutable sequenced projection context before canonical row hashing. Marker and loss frames
therefore share the same persistence identity discipline as attempt/sample batches.

Retries of persistence retain `batch_id`/`frame_id`. Every issued source event gets a new source-
record sequence; compact loss ranges use their independent `loss_seq`. A persistence retry advances
neither. Partitions and recovery deduplicate exact
`(frame_id, table)` projections, not a frame globally before all tables are covered.

---

## 10. Local durability and crash recovery

### 10.1 Durable genesis and sealed WAL segments

Exact durability is advertised only on a qualified local spool: same-filesystem temporary/final
paths, atomic replacement rename, durable file and parent-directory fsync, stable inode/link
semantics, accurate blocks/inodes, and a crash-released exclusive lock held by an open descriptor.
Preparation probes/allowlists the filesystem and rejects unqualified network/FUSE mounts. The lock
is held across authoritative recovery, collection, sync-only resume, and compaction. Heads/WAL are
always reread and recovery/reconciliation rerun after acquiring it, closing pre-lock TOCTOU.

Create-new writes generation zero exactly once before source activation, decode admission, or the
first frame. Genesis and its empty index root are written to content-addressed temporary files,
file-fsynced, renamed, directory-fsynced, then installed through a file- and parent-directory-
fsynced `LOCAL-LATEST`.

Exact-resume verifies—not rewrites—genesis. Under the lock and any remote writer fence, it rereads/
reconciles heads, completes recovery of every prior WAL, creates a new session/anchor, and commits a
hash-linked `session_started` generation before opening that session's WAL or activating sources.
Sync-only resume creates no session or WAL. A new session WAL header contains the verified current
head/genesis hashes, schema fingerprints, archive/session IDs, writer compatibility ID, and first
global record sequence. No frame is valid under an unknown authoritative session.

WAL files are numbered immutable segments. One `.open` segment is append-only; complete segments
are footer-checksummed, file-fsynced, renamed `.wal`, and directory-fsynced. A frame is
length-delimited and contains wire version, frame/batch ID, global record sequence, payload kind,
payload, and CRC. A corrupt/truncated final frame is never guessed past. A segment is never
front-truncated or rewritten to remove a prefix.

Acknowledgment vocabulary is exact:

1. **accepted:** the archive owner assigned `record_seq` and owns the frame, but it may exist only
   in memory;
2. **local durable:** the complete frame and required segment/directory metadata passed the
   configured fsync policy;
3. **completion observed:** the LocalSet Clock bridge received the owner's immutable durability
   token and stamped `response_observed_ns`;
4. **receipt observed:** the corresponding event/index/head is durable and the producer received
   `AppendReceipt::LocalDurable`.

Every observed durable receipt is recovered exactly once. A crash after fsync but before response
may recover an uncertain local-durable frame that the producer did not observe; retrying
persistence uses the same frame ID. A new source scrape is never used to resolve receipt
uncertainty.

### 10.2 Immutable partition/index/head transaction

Physical table builders rotate independently, but logical coverage uses the persistent §8.3
`ProjectionCoverage` entry for every declared table. A scrape frame declares the applicable
attempt/family/sample/marker/raw-reference projections; control frames declare marker or loss
projections. A retained raw reference additionally requires its shared physical raw object.
Partition footers, coverage entries, and the index carry the exact evidence; recovery never treats
a frame as globally committed merely because one projection or raw object exists. Zero-row family/
sample projections still require their explicit empty coverage entry.

One local commit performs these ordered durability steps:

1. choose completed whole-frame/table projections from a WAL prefix; keep any not-yet-rotated
   projection pending and never split one projection across partitions;
2. write every due `part-<content-hash>.parquet.tmp`, finish footer, flush, file-fsync, rename to its
   content-addressed key, and directory-fsync;
3. write each newly registered raw AEAD envelope once to its opaque keyed temporary object, flush/
   file-fsync, rename to its final key, directory-fsync, and verify its ciphertext digest; a reused
   raw ID must resolve to the byte-identical existing descriptor;
4. copy-on-write the bounded manifest-index path adding partition descriptors, shared raw-object
   descriptors, and every nonzero or zero-row `ProjectionCoverage`; file-fsync, rename, and
   directory-fsync every new content-addressed page;
5. write the immutable hash-linked generation transaction, file-fsync, rename, and
   directory-fsync;
6. write a new `LOCAL-LATEST.tmp` containing current and preceding valid heads, file-fsync,
   atomically rename it, and fsync its parent directory;
7. only now mark those exact projections committed;
8. retain WAL for every projection first covered by the current head. Unlink and directory-fsync
   only whole sealed segments whose required projections are covered by the *preceding* retained
   head (or an independently verified exact remote copy). A partially/newly covered segment remains
   intact even if it repeats indexed projections.

The owner may group several ready table partitions into one generation, but does not force all
tables to share a physical rotation size. No mutable Arrow checkpoint participates in assembly.

### 10.3 Ordered stop and finalization watermark

Data and reserved control channels do not define commit order by receive order. Graceful stop:

1. stops source issuance and closes new shared/archive reservations;
2. waits for every outstanding permit/job to resolve into native delivery plus a frame draft or a
   coalesced loss entry;
3. submits exact terminal lifecycle and loss-ledger frame drafts;
4. closes persisted-frame admission entirely; the sole archive owner assigns the remaining
   sequences and captures inclusive `final_record_seq`;
5. drains/locally-durably acknowledges every sequence through that fence;
6. rotates every open builder until all declared table coverage entries and referenced shared raw
   objects through the fence are covered;
7. commits a `locally_finalized` generation, then a bounded retention-checkpoint generation with
   the same verified index root so the finalized generation is the preceding valid head; only then
   retires eligible whole WAL segments;
8. attempts requested remote publication under §11.

A finalize command received early on the control lane cannot bypass the watermark. Forced process
termination may interrupt any step; recovery resumes from the preceding authoritative head and
WAL.

### 10.4 Recovery cases and proof target

Recovery reads fixed `LOCAL-LATEST`, verifies its checksum, current/preceding head descriptors,
generation ancestry, genesis, index root/pages, and referenced partitions. It never repairs
identity by directory globbing. It deterministically handles:

- EOF before a declared complete final frame/footer: discard only that physically incomplete tail;
- complete-length frame with CRC/hash failure, or any sealed-segment corruption: restore the exact
  independently verified bytes from remote/redundant storage or fail closed; checksum failure never
  proves non-durability;
- complete durable frame with no observed receipt: recover it under its existing frame ID;
- declared frame/table coverage absent from the index: replay only that missing projection,
  including an explicit zero-row entry where declared;
- repeated WAL frame whose subset of coverage entries is indexed: skip only those exact covered
  projections; a raw reference never substitutes for its shared physical object or vice versa;
- temporary Parquet/index/generation files: delete after verification or leave as orphans;
- valid immutable unreferenced partition: adopt only when its exact coverage and content hash match
  pending WAL; otherwise leave for explicit GC;
- bad current generation: use the preceding descriptor only when its index plus retained WAL/
  verified remote coverage includes every durable projection; one-generation-lag retirement makes
  this true for ordinary head corruption. Shared index-page corruption restores or fails closed;
- referenced missing/corrupt local object: restore the exact hash from remote or fail closed;
- schema/config/source/archive-key/writer-compatibility mismatch: refuse resume before session or
  source activation.

Crash/property tests stop after every file flush, fsync, rename, directory fsync, head replacement,
WAL seal/unlink, watermark, local-finalization, and remote-publication edge. They prove:

```text
for every successful recovery with intact or restored durable media:
  observed-durable frame projections ⊆ recovered projections exactly once
  recovered projections = all complete local-durable frames (including uncertain receipts)
```

No projection evidence appears twice in the resolved index. Detected unrecoverable media
corruption fails closed rather than silently weakening this equality.

### 10.5 Spool quota and transaction reserve

Remote targets require a local spool. Admission accounts separately for configured logical
bytes/files and actual filesystem free blocks/inodes. It preserves a conservative reserve for the
largest admitted WAL frame, the one-generation fallback WAL window, every open/temporary Parquet
builder, copy-on-write index path, generation/head/receipt files, optional raw object, WAL seal, and
emergency finalization. The reserve is
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
    ) -> Result<ObjectVersion, HeadUpdateError>;
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
2. create-if-absent every required encrypted raw object and verify its ciphertext bytes;
3. verify exact bytes by the §11.1 integrity contract;
4. create-if-absent every hash-linked generation from the remote head's descendant path;
5. verify the target generation and root reference only verified immutable objects;
6. conditionally replace `LATEST` from its exact object version/head hash/active writer claim with
   the new head. For terminal publication that single replacement also sets the archive state to
   `remotely_finalized` and the writer claim to absent;
7. after CAS success, durably write the §8.9 local publication receipt before reporting remote
   finalization.

Remote generation identities equal local generation identities; one CAS may advance over several
already uploaded ancestors but cannot coalesce or renumber them. `LATEST` contains archive ID,
local commit sequence, generation key/hash, index-root key/hash, parent hash, writer session ID, and
archive state. Manifest-generation keys are content-addressed/create-only, so a losing writer cannot
overwrite the winner's generation before losing CAS. Retries are physically and logically
idempotent. `HeadUpdateError` distinguishes verified conflict from uncertain transport outcome.
After either, the owner rereads through the visibility horizon: the exact desired archive/
generation/root/claim hash is idempotent success; a verified different head follows ancestry/
conflict rules; persistent uncertainty fails without guessing.

### 11.3 Locking, ancestry, and reconciliation

An archive ID has one writer. Genesis binds a random `canonical_spool_id`. Source-activating exact resume
requires that exact spool identity and its qualified open-descriptor lock; copying only the remote
head to a different spool cannot reopen collection. Before a remote-backed resumed session activates
sources, it conditionally installs `WriterClaim { claim_epoch, writer_session_id,
canonical_spool_id, session_started_generation_hash }` in `LATEST`. Every later head update is
conditioned on that claim. Clean remote finalization clears it atomically in the same terminal head
CAS; a second release write is forbidden. The durable publication receipt binds the resulting
object version, terminal head hash, and absent-claim state. A claim left by a crash has no
wall-clock expiry: takeover requires the canonical spool/lock plus an explicitly authored prior-
claim ID, or a separate operator-mediated fencing action. A different host/spool may run sync-only
but cannot acknowledge new collection. Distributed merge is out of scope.

A create-new operation generates an unguessable archive ID and may begin locally when an optional
remote target is temporarily unavailable; its first remote publication conditionally creates
`LATEST` with the same claim. Exact-resume of an existing remote archive never activates sources
without acquiring the claim. `CreateReceipt` includes the created object's version for later CAS.

Before source activation, recovery compares verified hash-linked heads:

| Relationship | Action |
|---|---|
| equal | acquire/verify the writer claim, then continue |
| remote is ancestor of local | upload the verified descendant path, then CAS forward |
| local is ancestor of remote, with no pending local WAL | verify/download exact remote objects and advance local head |
| local is ancestor of remote, with pending local WAL | fail closed as concurrent/stale history |
| neither is an ancestor | fail closed as divergence |
| remote absent | create only from validated local genesis |

No state is inferred from object listing. A remote descendant may be adopted only when archive,
schema, config, source, archive-key, canonical-spool, and writer-compatibility identities match.

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

Optional compaction runs only against a locally finalized archive under its exclusive lock. Each
transaction is bounded by configured maximum input partitions, output bytes, logical rows, and
index subtrees; larger work becomes a sequence of exact-parent generations. It binds to one parent
generation/root, writes and verifies immutable replacements, and proves per-projection logical row
count/multiset-digest equality. The generation atomically swaps only those bounded descriptors in
its copy-on-write index. A changed local head leaves outputs as unreferenced orphans.

Remote publication first verifies that remote `LATEST` is an ancestor of the compaction parent,
uploads the complete immutable descendant path, and may CAS directly to the compaction generation;
it need not publish each intermediate head separately. Divergence fails. Old partitions remain
authoritative until the replacement head commits and are retained until a separate GC policy proves
no retained head references them.

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
manifest records sanitizer and archive-key provider IDs/config digests, never secret values.
Endpoint userinfo and configured secret headers are removed before any durable source descriptor
exists. `BaselineCredentialSanitizer` is non-disableable; an empty optional sanitizer list means no
additional content policy, not raw passthrough of known credentials. AIPerf cannot identify every
arbitrary secret an endpoint may place in a metric label, so all exact source content is classified
potentially sensitive and operators must select storage access/sanitization accordingly. Exact raw-
body retention follows §8.10 instead of this structured pipeline.

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

The prepared control profile binds connect/TLS/mTLS/proxy/reuse ceilings per distinct policy;
equivalent source profiles may share the isolated control pool. The per-call absolute deadline from
§6.1 supplies request/total lifetime and cannot relax the prepared connect ceiling. The request
advertises only the two §8.1 formats, validates the response `Content-Type` and
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
formula and boundary result over identical input records.

Byte parity uses a frozen typed `NativeMeasurementParityV1` projection constructed before mode-
specific archive reporting. It contains every authoritative request, phase, server/GPU/network,
accuracy/adaptive, and native metric value plus their measurement provenance/order, while excluding
`ReportTelemetryArchive`, archive configuration/capabilities, archive health, receipt/head IDs,
archive artifact paths/hashes, and other archive-only provenance. It is not an ad hoc JSON field-
deletion helper; a canonical descriptor and golden serialization define it. After archive-off and
archive-on both use the run-owned driver, this projection is byte-for-byte equal. Their complete
native-v2 reports are intentionally different because only archive-on carries the additive archive
block. Comparison with the former completion-paced loop explicitly permits and characterizes time-
series differences caused solely by fixed-deadline sample instants.

The archive exposition projection may preserve families (for example summaries or `_created`)
that benchmark projection intentionally excludes. That is expected. One bounded decode job can
produce a strict archive entity and the separately named native-compatible entity described in §4;
archive completeness must not broaden benchmark metrics.

### 13.2 Attempt observation hook

Run-owned source composition gains a post-native all-outcome observer rather than file-writing
branches on `ServerMetricsRecord`/`GpuScrape`:

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
bytes. Authoritative native delivery is a direct prepared-driver callback that occurs exactly once
before archive reservation. Observer consumers are report health and test recorders;
archive projection is already encapsulated inside the prepared driver. The runner assembles a small
generic/static tee at preparation. Every transport/parse/unsupported/success outcome is observed
once. Observation never occurs per token.

Boundary order remains:

1. submit a typed boundary command with an absolute Clock deadline;
2. fetch once and run one bounded decode job (strict plus named native fallback when applicable);
3. deliver the supported native projection/boundary snapshot directly and synchronously exactly once;
4. nonblockingly reserve and enqueue archive projection, or record an archive loss range;
5. emit one post-native factual observation containing the admission outcome;
6. return the phase barrier result or typed timeout/failure according to required-sidecar policy.

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

The observer contract is extended with `PhaseTransitionContext { boundary:
Option<BoundaryReference> }`. The orchestrator allocates that reference before issuing a forced
snapshot and passes the same value to both the driver command and the lifecycle transition; it is
not reconstructed by a sidecar. Coalesced phase-A-end/phase-B-start transitions therefore produce
two marker rows with distinct boundary/phase/role values and one shared group, while the physical
attempt carries both references. `SENDING_COMPLETE` or any transition without a forced snapshot
uses `None`. Exact joins use the structured IDs, never timestamp proximity.

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

The `archive` object is one reusable strict DTO rather than two nearly equivalent configuration
shapes:

```rust
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryArchiveSpecV2 {
    pub target: NormalizedArchiveUri,
    pub local_spool: PathBuf,
    pub spool_quota_bytes: u64,
    pub spool_quota_files: u64,
    pub required: bool,
    pub sink: NamedRunnerComponentSpecV2,
    pub rotation: NamedRunnerComponentSpecV2,
    pub admission: NamedRunnerComponentSpecV2,
    pub recovery: NamedRunnerComponentSpecV2,
    pub archive_key: NamedRunnerComponentSpecV2,
    #[serde(default)]
    pub enrichers: Vec<NamedRunnerComponentSpecV2>,
    #[serde(default)]
    pub sanitizers: Vec<NamedRunnerComponentSpecV2>,
    pub raw_body: NamedRunnerComponentSpecV2,
}
```

Every named component strictly decodes its own `config`; the reusable DTO does not accept an
untyped option bag. `TelemetryWatchConfigV2.archive` and the attached resource below both use these
exact bytes and validation rules.

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
              "connect_timeout_ns": 3000000000,
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
          "archive_key": {"type": "secret_provider", "config": {"id": "archive-identity"}},
          "enrichers": [],
          "sanitizers": [], // baseline known-credential sanitizer is always active
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

### 14.2 Attached scheduled resource

`SidecarSpecV2` gains one optional resource that attaches an archive to already-authored telemetry
sources without duplicating their URL, credentials, transport, cadence, or failure policy:

```rust
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TelemetryArchiveAttachmentSpecV2 {
    pub source_ids: Vec<TelemetrySourceId>,
    pub archive: TelemetryArchiveSpecV2,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SidecarSpecV2 {
    // Existing strict source resources remain here.
    pub gpu_telemetry: Option<Box<RawValue>>,
    pub network_latency: Option<Box<RawValue>>,
    pub server_metrics: Option<Box<RawValue>>,
    pub live_streaming: Option<Box<RawValue>>,
    pub telemetry_archive: Option<TelemetryArchiveAttachmentSpecV2>,
}
```

Each selected server/GPU/network sidecar factory must expose stable prepared source IDs before the
attachment prepares. `source_ids` is non-empty and unique; every ID must resolve to exactly one
prepared physical telemetry source in the same run. An omitted source is still available to its
native accumulator but is not archived. Unknown IDs, duplicate IDs, source configuration repeated
inside the attachment, `primary_durable` admission on an attached archive, or an attachment without
at least one prepared source fail validation. V1 allows attachment only to `online_http +
scheduled`; `graph` and `dynamo_offline` reject it before backend/source preparation until their
deferred lifecycle integrations are built.

This normalized envelope is a complete attached-run example; no field below is an illustrative
ellipsis:

```jsonc
{
  "protocol_version": 2,
  "operation": "execute",
  "expected_distribution_id": "blake3:<exact-runner>",
  "run": {
    "identity": {"benchmark_id": "scheduled-archive-20260711-a", "random_seed": 7},
    "artifact_target": "/var/lib/aiperf/runs/scheduled-archive-20260711-a",
    "backend": {"type": "online_http", "config": {}},
    "workload": {
      "type": "scheduled",
      "config": {
        "worker_count": 1,
        "dataset": {"type": "synthetic", "entries": 100},
        "tokenizer": {"name": "builtin"},
        "phases": [
          {
            "name": "profiling",
            "type": "concurrency",
            "exclude_from_results": false,
            "concurrency": 1,
            "requests": 100
          }
        ]
      }
    },
    "resources": {
      "models": {"strategy": "round_robin", "items": [{"name": "model"}]},
      "endpoints": {
        "profiles": [
          {
            "id": "default",
            "type": "chat",
            "urls": ["http://127.0.0.1:8000"],
            "streaming": true,
            "use_server_token_count": true,
            "wait_for_model_timeout": 0.0,
            "wait_for_model_interval": 5.0,
            "wait_for_model_mode": "inference"
          }
        ]
      },
      "metrics": {},
      "artifacts": {},
      "sidecars": {
        "server_metrics": {
          "sources": [
            {
              "id": "server-primary",
              "type": "prometheus_http",
              "interval_ns": 1000000000,
              "request_timeout_ns": 5000000000,
              "config": {
                "url": "http://127.0.0.1:8000/metrics",
                "redirects": "disabled",
                "proxy": "disabled",
                "accepted_formats": ["prometheus_text_0_0_4", "openmetrics_text_1_0_0"],
                "max_compressed_bytes": 8388608,
                "max_decompressed_bytes": 33554432
              }
            }
          ]
        },
        "telemetry_archive": {
          "source_ids": ["server-primary"],
          "archive": {
            "target": "file:///var/lib/aiperf/archives/scheduled-archive-20260711-a",
            "local_spool": "/var/lib/aiperf/spool/scheduled-archive-20260711-a",
            "spool_quota_bytes": 10737418240,
            "spool_quota_files": 10000,
            "required": false,
            "sink": {"type": "parquet_local", "config": {}},
            "rotation": {"type": "rows_bytes_age", "config": {}},
            "admission": {"type": "attached_best_effort", "config": {}},
            "recovery": {"type": "create_new", "config": {}},
            "archive_key": {"type": "secret_provider", "config": {"id": "archive-identity"}},
            "enrichers": [],
            "sanitizers": [],
            "raw_body": {"type": "none", "config": {}}
          }
        }
      }
    }
  }
}
```

### 14.3 Validation before side effects

Static validation covers:

- exact runner distribution/capabilities;
- workload resource requirements and typed `ControlPlaneHttpProvider` backend capability;
- unique/valid source IDs and registered source types;
- positive intervals/absolute timeout budgets and bounded counts/sizes;
- credential/TLS/proxy/redirect/content-negotiation/compression policy without resolving secrets;
- target scheme/sink compatibility;
- local spool path safety, quota/transaction reserve, and artifact-target non-aliasing;
- policy IDs/configs and raw-retention acknowledgment;
- watch/backend compatibility;
- attached source-reference resolution and `online_http + scheduled` compatibility;
- no benchmark-only fields/models/datasets/phases that would be inert.

Execution preparation then checks source-specific configuration, local target/spool path candidate,
remote store capabilities/reachability where configured, filesystem reserve/capability probes, and
credential-provider availability without creating an authoritative archive. Only after complete
preparation does it acquire the qualified lifetime archive lock and reread authoritative state
under that lock. Create-new commits genesis; exact-resume reconciles/replays prior WAL and commits
`session_started`; sync-only creates no session. Remote exact-resume acquires its conditional writer
claim before sources; create-new follows the unique-ID first-publication rule in §11.3. Only after
the applicable transactions does it start IO/decode workers and Clock maintenance and activate
sources.

### 14.4 Signals and stdout

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

The IO-free reporter seam is revised to accept `NativeReportInput { metrics:
Option<&AccumulatorSummary>, outcome }`; its existing accumulator-based helper delegates with
`Some`. Standalone watch passes `None`, which serializes empty metric maps and absent metric-derived
summary times rather than constructing a fake accumulator. Common provenance and the typed mode
block remain required.

```jsonc
{
  "telemetry_archive": {
    "schema_version": "1.0",
    "archive_id": "uuid",
    "session_id": "uuid",
    "state": "remotely_finalized",
    "publication_receipts_uri": "file:///.../LOCAL-RECEIPTS",
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
    "health": {"loss_ranges": [], "writer_alive": true}
  }
}
```

The block is an additive mode-specific extension expressly permitted by native-v2; its addition
does not change top-level schema version `2.0`. Implementation adds a typed DTO plus old-reader,
new-reader, absent-block, watch, and attached goldens. Credentials, raw labels, signed URLs, and
arbitrary diagnostics are excluded. Python presentation may link the archive and summarize source
health but cannot reinterpret it as native metrics.

For attached best-effort mode, `health` is assembled from both the writer snapshot and the
LocalSet-owned fixed-memory loss ledger. Writer death therefore still yields a successful native
benchmark report with `writer_alive=false`, `lossy=true`, and structured ranges. Primary watch or
required attachment instead follows the failed diagnostic path below.

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
(metric_family, series_key, clock_ns, record_seq, metric_point_seq)
```

Other partitions use total orders: attempts `(source_id, source_record_seq, record_seq)`, families
`(source_id, record_seq, family_seq)`, markers `(clock_ns, record_seq, marker_seq)`, losses
`(source_id, record_seq, loss_seq)`, and raw references `(source_id, record_seq)`. Parquet statistics
and the manifest index's min/max/source metadata enable pruning. The query resolver starts from a
verified head/root and walks the persistent index; it never globs. `metric_name_clean` is
unnecessary because family identity has its own column.

The first documentation examples use DuckDB/Polars/Arrow to:

- chart one family/label set over time;
- inspect metadata-only families and repeated MetricPoints;
- compute counter deltas with reset handling explicitly;
- inspect scrape failure/missed-tick intervals;
- join phase markers to telemetry;
- join optional local/remote durability receipts to their covered frame ranges;
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
- a 24-hour accelerated/real soak has no missed acknowledged frames, no duplicate logical
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
   untyped, gauge histogram, summary, histogram, semantic Created timestamps, source timestamps,
   and scalar/bucket exemplars retain emitted names/roles; arbitrary classic `_created` samples are
   not retyped without a semantic role;
3. zero-point metadata-only families, empty MetricSets, repeated MetricPoints for one label set, and
   point-owned wire samples/timestamps round-trip distinctly; classic all-absent/uniform/mixed/
   partial component timestamps produce the exact point status without losing component lexemes;
4. multiple histogram/gauge-histogram base label sets remain isolated and structured while native
   declared-format compatibility fallback retains pinned old semantics without creating archive
   success rows;
5. `100000001`, values around 2⁵³, `u64::MAX`, values outside analytical f64 range, exact numeric
   lexemes, and exact/rounded/unavailable f64 status survive every pinned reader; Prometheus-ms/
   OpenMetrics-seconds exact/sub-ns/out-of-range timestamp cases retain format and normalization
   status;
6. the per-format/per-role semantic matrix accepts every legal NaN/Inf case, atomically rejects
   every illegal count/sum/bucket/state/info/metadata case, and emits no raw non-finite boundary;
7. deterministic keyed pre-redaction identity, post-redaction identity, map order, digest domains,
   topology epochs, schema descriptors/fingerprints, manifest/index, and report goldens; independent
   Rust/Python logical-row fixtures cover every scalar/nested type, null, negative zero, map order,
   and full semantic rows;
8. exact field/type/nullability/dictionary/metadata compatibility through pinned Arrow, Parquet,
   DuckDB, Polars, and pyarrow versions;
9. streaming size/cardinality limits, property parse/encode/decode round trips, and malformed-input
   atomic failure.

### 18.2 Scheduling gates

1. `SimClock` exact fixed deadlines with zero scrape time;
2. overrun skips debt without drift or catch-up bursts;
3. slow source A cannot shift source B deadlines;
4. one source never has two scrapes in flight;
5. absolute Clock timeout cancels/reclaims transport and yields one timeout attempt;
6. only an orchestrator-issued shared coalescing group reuses a boundary attempt while exact phase
   markers remain distinct; structured boundary references join every marker to that physical
   attempt without timestamps, including coalesced end/start transitions;
7. one physical attempt with multi-phase membership preserves phase and run-level native parity
   across seamless overlap;
8. virtual inline capture/post-simulation persistence cannot advance virtual time; a future external
   progress source blocks quiescence deterministically;
9. bounded worst-case decode cannot stall unrelated source/request LocalSet work at the qualified
   profile, and archive family/point construction runs only on the second bounded CPU stage after a
   nonblocking permit;
10. deliberately inverted worker latency preserves source-record FIFO attribute epochs/markers,
    while distinct source strands still run in parallel and final WAL order follows global
    `record_seq`;
11. failed/empty/unchanged attempt outcomes and missed/rejected loss ranges have exact counters/
   records, and unchanged success retains full family/MetricPoint rows;
12. graceful signal resolves permits/terminal frames, closes frame admission, and fixes the final
    owner-assigned record-sequence watermark before drain;
13. archive-off/archive-on runs over the run-owned fixed-deadline driver have byte-exact
    `NativeMeasurementParityV1` bytes while their archive blocks differ as expected; comparison with the retired completion-paced loop pins formula/boundary parity while
    explicitly characterizing cadence-caused sample-time differences.

### 18.3 Durability/recovery gates

1. filesystem/lock capability rejection plus crash before/after create genesis, resume
   `session_started`, every WAL append/fsync/seal, raw/partition/index/generation/receipt-batch/
   receipt-index/receipt-head write, file/directory fsync, pointer replacement, lagged WAL unlink,
   watermark, and finalization edge;
2. receipt-observed durable projections are recovered exactly once; a complete fsynced but
   unobserved frame is recovered as an uncertain operation under the same ID. Crashes between
   fsync/CAS completion, LocalSet Clock observation, receipt-draft return, and receipt-head
   durability preserve absent response time and a distinct recovery-verification event;
3. independently rotated multi-table projections cannot make global dedup omit a table, and stale/
   repeated WAL frames cannot duplicate logical projection evidence; empty exposition and metadata-
   only cases persist zero-row coverage with the empty multiset digest;
4. finalize on the reserved control lane cannot overtake accepted data or loss-ledger frames;
5. only incomplete physical WAL tails discard; complete checksum failures restore/fail closed, and
   one-generation-lag WAL makes preceding-head rollback complete without directory guessing;
6. transaction-reserve exhaustion and real ENOSPC/inode exhaustion fail before destroying the only
   durable copy;
7. raw-reference coverage, one randomized physical envelope per equality ID, duplicate/concurrent
   candidate reuse, create-if-absent retries with byte-identical ciphertext, exact-byte verification,
   named-object visibility horizon, pre-activation writer claims, and uncertain/success/conflict
   `LATEST` CAS reconciliation are idempotent under competing writers;
8. every equal/ancestor/divergent local/remote reconciliation cell and sync-only finalization path;
9. exact-resume identity/writer mismatch fails before session/source activation;
10. bounded exact-parent compaction verifies per-projection logical row counts/multiset digests;
    failure leaves the old head authoritative and cannot expose duplicate/missing replacement rows;
11. the 24-hour profile produces immutable partitions and O(K log₂₅₆ P) manifest-index update
    work rather than flat full-history rewrites; receipt batches/index pages remain bounded with the
    same asymptotic property;
12. terminal publication clears the writer claim in the same CAS, and its receipt binds the exact
    resulting object version, head hash, state, and absent claim.

### 18.4 Security gates

1. endpoint userinfo/AIPerf-authored auth headers/provider secrets/object-store credentials are
   removed by the non-disableable baseline and absent from every structured durable metadata,
   dimension, report, log, diagnostic, and error surface; an opt-in encrypted exact raw object may
   contain source-echoed bytes only under the explicit §8.10 exception;
2. sanitization covers every structured durable surface; keyed pre-redaction identity prevents
   silent series merge and defeats low-entropy dictionary tests;
3. compressed/decompressed body, label, exemplar, marker, attribute, diagnostic, series, and bucket
   bounds reject adversarial input during receive/parse rather than after unbounded allocation;
4. redirects/proxies/content negotiation/TLS/mTLS/credential forwarding obey strict defaults and
   a non-2xx metric-looking body never parses;
5. raw-body retention is off by default; opt-in requires classification, key provider, restrictive
   permissions, authenticated encryption locally and remotely, and artifact-wide secret scanning;
6. path traversal and unsafe artifact/spool/target aliasing fail validation;
7. partition/index/generation/head hashes detect tampering/corruption;
8. arbitrary endpoint labels are treated as potentially sensitive source data; configured
   sanitizers/access policy—not an impossible universal secret detector—govern them.

### 18.5 Product subprocess gates

1. the complete §14.1 watch envelope and §14.2 attached scheduled envelope validate/deserialize;
   missing required scheduled resources, forbidden watch resources, empty/duplicate/unknown
   attachment source IDs, and repeated source configuration fail before preparation;
2. Python `aiperf watch` -> exact packaged runner -> in-process HTTP Prometheus mock -> durable
   genesis/WAL/Parquet/index/head -> terminal response;
3. multiple endpoint cadences including one slow/failing/oversized source and distinct per-call
   deadlines over profile-bound native control handles isolated from inference capacity;
4. HTTP 500 with metric-looking body remains an HTTP failure record, not a sample;
5. SIGINT/SIGTERM graceful finalization, forced-crash exact resume, and local-final/sync-only remote
   completion;
6. object-store emulator visibility lag, outage, conflicting CAS, restart, and finalization;
7. ordinary scheduled benchmark with attached server/GPU archive proves one physical run-owned
   driver/source feeds report and archive across seamless phases, `PhaseObserver` markers align
   exactly, and archive-off/archive-on `NativeMeasurementParityV1` bytes are equal on the new
   cadence while only archive-on has the typed archive block;
8. required attached archive failure yields failed reporting terminal, no `report_path`, and only a
   typed diagnostic artifact; best-effort yields a successful typed archive block;
9. online-only runner capability advertises watch only after qualified profiles; unsupported
   distribution/backend capability plus attached Graph and Dynamo-offline pairs fail before IO.

### 18.6 Query compatibility gates

Golden archives are read by pinned Arrow, DuckDB, Polars, and pyarrow versions. Queries begin at
local/remote heads, verify immutable generation/root hashes, walk the persistent index, and prove
bounded source/time page pruning, metadata-only family discovery, repeated-point ordering,
structured label filtering, every semantic payload, phase joins, durability-receipt joins,
failure/loss-range discovery, unchanged-success continuity, and partition pruning. No directory
glob supplies file lists.

### 18.7 Performance/capacity gates

The versioned §17 harness validates schema completeness and all required numeric profile fields,
runs the paired bootstrap method, enforces every threshold, records dependency/hardware identities,
and rejects stale profiles after relevant lock/schema/writer changes. Capability generation checks
for passing standalone and attached profile artifacts rather than trusting documentation text.

---

## 19. Implementation increments

### Increment 1 — exposition and archive schema

1. implement the bounded Prometheus 0.0.4/OpenMetrics 1.0.0 `aiperf-prometheus` model/parser seam;
2. implement the frozen role-validity matrix, strict archive/native-fallback split, metadata-only
   families, repeated MetricPoints, lexemes, timestamps, and preserve current server/DCGM parity;
3. check in canonical Arrow/head/generation/index descriptors; implement logical-row evidence,
   every digest/identity/epoch rule, sanitizer surface, and golden Parquet/index/manifest/report;
4. add the five Tachometer regression fixtures as mandatory tests.

### Increment 2 — local writer and recovery

1. implement erased prepared drivers, bounded shared decode and archive-projection CPU stages,
   fixed-memory loss ledger, Clock maintenance/virtual-inline strategies, the fsync/CAS-to-LocalSet
   receipt Clock bridge, and the single mutable archive owner;
2. add qualified lifetime lock, create-only genesis/resumed-session transaction, sealed WAL with
   lagged retirement, persistent zero/nonzero table coverage, shared raw-object registry plus raw-
   reference rows, immutable Parquet/index/generations/head, and the bounded indexed non-self-
   referential receipt journal;
3. add every-step crash/property/corruption recovery matrix and transaction-reserved spool quotas;
4. ship no product command until exact-once recovery gates pass.

### Increment 3 — source runtime and watch product path

1. revise runner-v2 resources/requirements and implement the profile-bound
   `ControlPlaneHttpProvider` capability over isolated control capacity;
2. implement strict secured source factories and one run-owned fixed-deadline driver per physical
   source;
3. register `telemetry_watch`, add strict Python Config-v2 projection and `aiperf watch` command;
4. add local archive, signal, failure, and query subprocess gates.

### Increment 4 — benchmark attachment

1. replace phase-owned cadence loops with run-owned source subscriptions, orchestrator coalescing
   groups, physical attempt/phase-membership delivery, and all-outcome tees;
2. emit exact lifecycle markers through a `PhaseObserver` tee;
3. add typed archive provenance/health and failure diagnostic-artifact protocol;
4. prove no extra scrapes, no metric drift, and no request-path backpressure.

### Increment 5 — object-store durability and resume

1. implement capability-gated archive-store adapters, bounded partition/raw uploads, strong
   verification, conditional heads, writer claims, and uncertain-CAS reconciliation;
2. implement create-new/exact/sync-only policies, hash ancestry, and durable publication receipts;
3. add visibility/CAS/outage emulator matrix, finalization lifecycle, and §17 profiles;
4. document operational recovery and orphan/GC procedures.

### Increment 6 — optional compaction and ecosystem docs

1. add bounded manifest-transactional compaction proving logical multiset equality without in-place
   history mutation;
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
all-outcome pre-projection envelope; derive strict archive and native-compatible projections from
the same fetched bytes.

### Put durability/publication timestamps in the row they describe

Rejected. A frame cannot predict its own fsync or later remote CAS, and immutable rows cannot be
patched afterward. Separate receipt objects attest earlier frame ranges/generations without
self-reference.

### Dynamic wide columns for arbitrary metadata

Rejected. Endpoint-dependent schemas complicate unions/evolution and invite name collisions. Use
structured attributes with stable core columns.

---

## 21. Completion criteria

This design is complete only when:

- Python exposes `aiperf watch` and no second Rust executable exists;
- the revised workload-scoped runner-v2 DTO deserializes the complete watch envelope and rejects
  required/forbidden resource violations; the strict attached scheduled DTO resolves unique
  existing telemetry source IDs without duplicating their configuration;
- the exact runner derives and validates `online_http + telemetry_watch` from frozen factories and
  the typed `ControlPlaneHttpProvider` capability;
- every physical source uses the injected Clock/native transport, one run-owned fixed-deadline
  driver, absolute deadline, and bounded ordered decode path;
- Prometheus text 0.0.4/OpenMetrics text 1.0.0 parsing preserves strict grammar, every valid role,
  zero-point families, repeated MetricPoints, numeric/timestamp lexemes, exemplars, and a separately
  named native fallback without changing benchmark semantics;
- canonical Arrow/head/generation/index/logical-row descriptors, keyed pre/post-redaction/body
  identities, exact lexemes with optional analytical numbers, component timestamp status,
  attribute epochs, and native-v2 DTO are deterministic and readable/prunable by the pinned query
  ecosystem;
- qualified lock, create-only genesis, resumed-session generation, sealed/lag-retained WAL,
  persistent zero/nonzero projection coverage, shared encrypted raw objects with per-frame
  references, immutable partitions/generations/index/head, and bounded indexed receipt journal
  recover every complete durable projection exactly once across all injected crash/finalization
  points;
- remote sync verifies create-only partition/raw/index/generation objects, owns a pre-activation
  writer claim, and advances a linearizable conditional head without flat rewrites; uncertain CAS
  and exact/sync-only resume reconcile ancestry fail-closed;
- exact resume fails closed on identity/config/schema/key/writer mismatch and concurrent writers;
- failures, gaps, unchanged bodies, misses, drops/loss ranges, local durability, visibility lag, and
  remote publication are observable;
- enrichment is API-limited to attributes, sanitization covers every structured surface, source
  identity survives redaction, and raw retention is separately protected;
- attached mode reuses one source attempt across continuous phase membership and explicit boundary
  subscribers, emits exactly joined `PhaseObserver` markers,
  produces byte-identical archive-off/archive-on `NativeMeasurementParityV1` bytes on the run-owned
  cadence, emits the archive block only when enabled, and passes the numeric §17 regression profile;
- primary watch and attached modes have real Python-to-runner subprocess proofs;
- the five reproduced Tachometer defects are permanent regression tests;
- native-v2 2.0 additively identifies archive provenance/completeness without treating archived
  samples as native metrics, while required reporting failure emits no authoritative report path;
- checked-in passing standalone and attached acceptance profiles gate capability advertisement.

Until these gates pass, the existing phase-bounded native telemetry pipeline remains code truth and
no `watch` capability should be advertised.
