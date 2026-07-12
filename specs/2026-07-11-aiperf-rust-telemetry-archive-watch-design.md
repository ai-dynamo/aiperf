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
- optional attachment to existing benchmark telemetry sidecars without a second scrape;
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
14. **Secrets never become archive dimensions.** URLs are credential-free; credentials remain in
    environment/secret providers; diagnostics and manifests are redacted.
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

The `online_http` prepared backend exposes a LocalSet-compatible `ControlPlaneHttp` capability
over its injected `Clock` and native transport. It accepts an owned request plus an absolute
Clock deadline and returns exact entity bytes, response status/headers, and native transport
timings. Per-call deadlines override only request/total lifetime; connection/TLS/reuse policy
remains backend-owned. A workload requirement asks for this typed capability, so the compatibility
matrix is derived rather than special-casing the `telemetry_watch` ID. This is not a second HTTP
client or a special `reqwest` path.

The computed capability matrix adds:

| Backend | `telemetry_watch` |
|---|:---:|
| `online_http` | yes |
| `dynamo_offline` | no |

Replay/fault-injection tests may drive the library runtime with `SimClock`, but the product pair is
real-clock only. Offline benchmarks may still archive their in-process telemetry/event records
through the attachment seam; that does not make standalone HTTP watch an offline workload.

### 3.3 Attached benchmark mode

An ordinary scheduled/graph run may request an archive target in addition to existing telemetry
summaries. The run owns exactly one fixed-deadline driver per physical telemetry source across all
phases. Phase sidecars subscribe to that driver and submit typed boundary commands; they do not own
cadence loops. Each all-outcome attempt envelope is projected twice where applicable:

```text
one physical source attempt
  +-- native domain projection/accumulator (when successful and supported)
  `-- archive attempt/exposition projection (every outcome)
```

Continuous samples are attributed to every active phase window by marker-time joins; a seamless
phase overlap never causes a second source request. Coincident boundary commands coalesce into one
physical scrape whose envelope names every requesting boundary. Exact phase lifecycle markers come
from a `PhaseObserver` tee, while boundary scrape capture time remains a separate fact. Archive
availability cannot change request timings or metric formulas. The terminal report records archive
completeness separately.

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
selects the parser; a body is never silently reinterpreted under a different format after a parse
failure. It extracts and replaces—not merely wraps—the narrower lexical/state-machine core
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
    ) -> Result<Box<dyn ArchiveSource>, SourcePrepareError>;
}

#[async_trait(?Send)]
pub trait ArchiveSource {
    fn source_id(&self) -> &str;
    async fn fetch(&self, reason: ScrapeReason, deadline_ns: i64) -> FetchedAttempt;
    async fn shutdown(&self) -> Result<(), ArchiveSourceError>;
}

pub trait AttemptDecoder<Entity>: Debug + Send + Sync {
    fn decode(
        &self,
        fetched: FetchedAttempt,
        limits: &DecodeLimits,
    ) -> DecodedAttempt<Entity>;
}

pub struct DecodedAttempt<Entity> {
    pub facts: AttemptFacts,
    pub entity: Option<Entity>,
    pub parse_outcome: ParseOutcome,
}

pub trait NativeProjection<Entity, Record>: Debug + Send + Sync {
    fn project(&self, entity: &Entity) -> Result<Option<Record>, NativeProjectionError>;
}

pub trait ArchiveProjection<Entity>: Debug + Send + Sync {
    fn project(
        &self,
        attempt: &DecodedAttempt<Entity>,
        context: &ArchiveProjectionContext,
    ) -> Result<ArchiveBatch, ArchiveProjectionError>;
}

pub trait TelemetryEnricher {
    fn attributes(
        &self,
        sample: &ArchiveSampleView<'_>,
        source: &ArchiveSourceIdentity,
    ) -> Result<AttributePatch, EnrichmentError>;
}

pub trait ArchiveSanitizer {
    fn sanitize_source(&self, source: SourceDescriptorView<'_>) -> SanitizedSourceDescriptor;
    fn sanitize_sample(&self, sample: ArchiveSampleView<'_>) -> SanitizedSample;
    fn sanitize_marker(&self, marker: ArchiveMarkerView<'_>) -> SanitizedMarker;
    fn sanitize_diagnostic(&self, diagnostic: &ArchiveDiagnostic) -> ArchiveDiagnostic;
}

pub trait SegmentRotationPolicy: Debug + Send {
    fn should_rotate(&self, state: &OpenSegmentState, now_ns: i64) -> bool;
}

pub trait ArchiveAdmissionPolicy {
    fn admit(&self, state: ArchiveIngressState, batch: &ArchiveBatch) -> AdmissionDecision;
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

`FetchedAttempt` owns bounded exact entity bytes plus transport status/headers/timestamps and never
contains credentials. `ArchiveWalFrame` is a versioned persisted sum of attempt/sample batch,
lifecycle marker, and coalesced loss-range frames. It is a closed wire schema, not a source
extension point. Every frame has a stable frame ID, source/control sequence, and CRC, and every
successful `append_frame` has the same local-durable meaning.

Source fetch and Clock/lifecycle ownership stay on the LocalSet. After a driver reserves bounded
decode and archive-ingress credits, it sends owned bytes/facts to an ordered bounded CPU pool. The
pure decoder creates one `DecodedAttempt`; native and archive projections consume that same value.
Results return in source-attempt order so a boundary can synchronously feed its native accumulator
before the phase barrier returns. A factory may keep decode on the LocalSet only when its advertised
maximum body/CPU budget passes the attached regression profile; the generic Prometheus path uses
the pool.

Every wire-selected family—source, sink, rotation, admission, recovery, enrichment, sanitizer,
raw-body retention, and identity-key provider—has its own frozen descriptor/strict-validate/prepare
factory registry. Only sink-owned objects crossing to the IO worker require `Send`; LocalSet-owned
sources, admission, and observation graphs remain `?Send`. Stable IDs never select a core string
branch.

At least these concrete implementations ship:

- `PrometheusArchiveSource` and replay/fault-injection sources;
- `StaticLabelEnricher` and `NoopEnricher`;
- allow/deny-key structured sanitizers and `NoopSanitizer`;
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
- attempt sequence;
- consecutive/total failure counters;
- current source state (`active`, `degraded`, `disabled`, `stopped`);
- a command channel with reserved capacity for boundary/shutdown commands.

Each request has `deadline_ns = min(scheduled_or_boundary_budget, shutdown_budget)` and passes that
absolute value through `ControlPlaneHttp`. Deadline cancellation must release the transport body/
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
boundary identity, and absolute deadline. It preempts the next continuous deadline but never
interrupts an already issued HTTP request. Coincident commands coalesce and all requesting
boundaries are recorded on the one attempt. The phase waits for the forced result under the same
Clock deadline. The same `DecodedAttempt` feeds the supported native projection and archive
projection. Continuous scheduling re-anchors from the original cadence, not the boundary
completion time.

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
              | bounded owned bytes + reserved credits
              v
bounded ordered decode/projection CPU pool
              | ArchiveWalFrame, accepted_seq
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
- **attached benchmark:** the benchmark is the product. Source/native accumulator work proceeds;
  archive-only decode/projection first tries a nonblocking byte/CPU/ingress reservation. Rejection
  updates a fixed-memory per-source loss ledger that coalesces contiguous attempt/deadline ranges
  by reason. The reserved control lane persists those ranges at checkpoint/finalize. If the writer
  itself is dead, the same structured ranges appear in the failed terminal diagnostic. The request
  path never waits. `archive.required=true` may convert archive degradation into a reporting-stage
  failed terminal after benchmark execution; it still cannot change measured request data or emit
  a partial result on the authoritative report path.

Boundary scrapes always reach their native accumulator even if archive admission fails.

### 9.3 Batch identity

`batch_id` uses the domain-separated, length-prefixed digest rule from §8.1 over archive/session/
source, attempt sequence, outcome, and keyed exact-body digest when present. `frame_id` similarly
includes frame schema/kind and control/source sequence. Marker and loss frames therefore share the
same persistence identity discipline as attempt/sample batches.

Retries of persistence retain `batch_id`/`frame_id`. A new source request always gets a new
`attempt_seq`; a persistence retry never does. Partitions and recovery deduplicate exact
`(frame_id, table)` projections, not a frame globally before all tables are covered.

---

## 10. Local durability and crash recovery

### 10.1 WAL, not a mutable full-buffer checkpoint

Each admitted batch is encoded as a length-delimited, schema-versioned frame with batch ID, CRC,
and payload in a session WAL. Under the default durable policy, `AppendReceipt::LocalDurable` is
returned only after the frame and containing directory metadata meet the configured fsync policy.

The Parquet builder consumes WAL frames. A crash may lose unacknowledged memory, but it cannot turn
an acknowledged batch into an unreported disappearance or duplicate.

### 10.2 Immutable partition commit protocol

For each rotated partition:

1. select a complete prefix of WAL frames;
2. write `part-<content-hash>.parquet.tmp`;
3. finish Parquet footer, flush, and fsync the file;
4. atomically rename to its content-addressed final key and fsync the directory;
5. write a new local manifest generation to a temporary file, flush/fsync, then atomically rename;
6. only after the manifest references the partition, advance/delete the committed WAL prefix.

No `current.arrow` participates in final assembly. The authoritative logical dataset is the set of
immutable partitions referenced by the latest valid manifest generation.

### 10.3 Recovery cases

Recovery deterministically handles every crash point:

- truncated/corrupt final WAL frame: discard only that unacknowledged suffix;
- complete WAL frame absent from manifest: replay into a partition;
- temporary Parquet without valid footer/hash: delete locally;
- valid content-addressed Parquet absent from manifest: verify its batch IDs, then either adopt it
  if it exactly covers pending WAL or leave it as an orphan for GC;
- manifest generation with bad checksum/schema: fall back to the preceding valid generation;
- partition referenced by manifest but missing/corrupt locally: recover from remote or fail closed;
- batch already referenced plus repeated in WAL: skip by batch ID and advance WAL;
- configuration/source identity mismatch: refuse resume before source activation.

Property tests inject a crash after every numbered commit step and prove the recovered logical batch
set equals the acknowledged set with no duplicates.

### 10.4 Spool quota

Remote targets require a local spool. Byte/file quotas are validated before execution and monitored
continuously. The manifest reports current/high-water usage. Primary watch fails or applies its
explicit admission policy before exhaustion; attached mode degrades with visible loss accounting.
The writer never deletes the only durable copy of a referenced partition.

---

## 11. Object-store synchronization

### 11.1 Existing abstraction

Use the `object_store::ObjectStore` trait (or a smaller AIPerf wrapper around it) for local/S3-like
implementations. Credentials come from the normal provider/environment chain and never from
serializable archive config.

### 11.2 Immutable upload protocol

Periodic sync:

1. uploads each locally committed content-addressed partition not yet remote-durable;
2. verifies size/hash or provider checksum;
3. uploads an immutable manifest generation referencing only verified remote objects;
4. conditionally advances `LATEST` from generation N to N+1;
5. marks batches remote-referenced only after the pointer succeeds.

Retries may create the same physical object again, but content addressing and manifest identity
make the logical commit idempotent. Historical partitions are never merged/reuploaded merely
because a new partition arrived.

### 11.3 Concurrency

An archive ID has one writer. Create-new uses a unique ID. Exact-resume obtains a local exclusive
lock and advances the remote generation with conditional compare-and-set. A failed CAS means
another writer or stale state; the process stops rather than forking history. Distributed
multi-writer merge is out of scope.

### 11.4 Finalization and partial availability

Graceful finalization drains accepted batches, rotates open segments, commits local and requested
remote manifests, writes `finalized=true` with termination reason, and returns the manifest URI.

A network outage does not destroy locally durable history. Terminal status distinguishes:

- finalized locally and remotely;
- finalized locally, remote incomplete;
- degraded/lossy;
- failed before authoritative finalization.

No global `final.parquet` is required. An optional post-run compactor may create larger replacement
partitions and a new manifest generation, but old partitions remain referenced until the new
generation is completely committed. Garbage collection is a separate policy/tool.

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
devices, or change numeric values. Derived/materialized views belong downstream.

### 12.2 Redaction order

Pipeline order is:

```text
parse exact structured source -> enrich -> redact -> canonicalize -> hash/encode
```

Redaction therefore affects both stored labels and `series_key`. The manifest records the redaction
policy ID/config digest, never secret values. Endpoint userinfo and configured secret headers are
removed before source identity exists.

### 12.3 Bounds

Validation/runtime enforce configured limits for:

- source count;
- label/attribute key and value byte length;
- labels per series;
- samples and histogram buckets per scrape;
- unique series per source/window;
- response body and retained raw-body size;
- diagnostic length.

Exceeding a bound produces an explicit failed/degraded scrape record. It never truncates labels or
silently merges series. Cardinality-limit implementations live behind a policy trait.

---

## 13. Native accumulator and phase integration

### 13.1 Preserve current semantics

The server/GPU/network accumulator inputs, boundary snapshots, reset-clamp, histogram learner,
unit inference, vLLM/SGLang atlas, GPU scaling, energy joins, and RTT delivery do not change.

The archive exposition projection may preserve families (for example summaries or `_created`)
that benchmark projection intentionally excludes. That is expected. One parsed `Exposition` can
feed two explicit projection policies; archive completeness must not broaden benchmark metrics.

### 13.2 Sidecar hook

Sidecar composition gains an optional batch observer rather than file-writing branches:

```rust
pub trait TelemetryBatchObserver<Record> {
    fn observe(&self, record: &Record, context: &TelemetryObservationContext);
}
```

Concrete observers include accumulator ingestion, archive encoding/ingress, and test recorders.
The runner assembles a small static tee at preparation. Observation occurs once per completed
scrape and never per token.

Boundary order remains:

1. force/decode source scrape;
2. feed accumulator/boundary snapshot synchronously;
3. submit archive batch/attempt record according to admission policy;
4. return phase barrier result.

Archive remote durability is not awaited at a phase boundary.

### 13.3 Lifecycle markers

The phase driver already owns authoritative phase timestamps. It emits markers using the same
values passed to `ScheduledPhaseSidecar::on_phase_start/on_phase_end`. Warmup/profiling names and
run identity are typed attributes. No post-hoc timestamp inference assigns phases.

---

## 14. Protocol and configuration

### 14.1 Strict workload DTO

The wire uses factory-owned raw configs under the runner-v2 workload registry. An illustrative
authored projection is:

```jsonc
{
  "backend": {
    "type": "online_http",
    "config": {"client": {"connect_timeout_ns": 10000000000}}
  },
  "workload": {
    "type": "telemetry_watch",
    "config": {
      "duration_ns": null,
      "sources": [
        {
          "id": "node-a",
          "type": "prometheus_http",
          "interval_ns": 1000000000,
          "request_timeout_ns": 5000000000,
          "config": {"url": "http://node-a:9100/metrics"},
          "attributes": {"role": "node", "cluster": "lab-a"}
        }
      ],
      "archive": {
        "target": "s3://benchmarks/watch/archive-id/",
        "local_spool": "/var/tmp/aiperf/archive-id",
        "required": true,
        "sink": {"type": "parquet_object_store", "config": {}},
        "rotation": {"type": "rows_bytes_age", "config": {}},
        "admission": {"type": "primary_durable", "config": {}},
        "recovery": {"type": "create_new", "config": {}},
        "enrichers": [],
        "redactors": [],
        "raw_body": {"type": "none", "config": {}}
      }
    }
  }
}
```

Python may accept friendly durations/paths; the runner wire uses normalized integer ns, absolute
local paths, and normalized target URIs. Unknown fields and unknown factory IDs fail validation.

### 14.2 Validation before side effects

Static validation covers:

- exact runner distribution/capabilities;
- unique/valid source IDs and registered source types;
- positive intervals/timeouts and bounded counts/sizes;
- target scheme/sink compatibility;
- local spool path safety and quota policy;
- policy IDs/configs and raw-retention acknowledgment;
- watch/backend compatibility;
- no benchmark-only fields/models/datasets/phases that would be inert.

Execution preparation then checks source-specific configuration, local target/spool state, remote
reachability where configured, exact-resume manifest identity, and credentials without creating an
authoritative archive. Only after complete preparation does it create/lock the archive session,
start the IO worker, and activate sources.

### 14.3 Signals and stdout

Runner stdout retains exactly one terminal JSON line. Progress is structured stderr and/or an
optional local status artifact. Python forwards SIGINT/SIGTERM as a graceful stop request:

```text
PREPARED -> RUNNING -> STOP_REQUESTED -> DRAINING -> FINALIZED
                                          `-------> FAILED
```

A second signal or expired shutdown budget may force termination; recovery must make the next
resume deterministic. The terminal response includes archive ID, local/remote manifest URIs,
completeness, health counts, and failure stage.

---

## 15. Report and query contract

### 15.1 Native-v2

Every successful watch execution still writes a minimal native-v2 outcome with common runner
provenance and a typed `telemetry_archive` block. It contains no fabricated request metrics or
benchmark duration. Attached runs add the same block to their normal report:

```jsonc
{
  "telemetry_archive": {
    "schema_version": "1.0",
    "archive_id": "uuid",
    "manifest_uri": ".../archive-manifest.json",
    "local_manifest_path": "...",
    "finalized_local": true,
    "finalized_remote": true,
    "lossy": false,
    "health": {}
  }
}
```

Credentials, raw labels, signed URLs, and arbitrary diagnostics are excluded. Python presentation
may link the archive and summarize source health but cannot reinterpret it as native metrics.

### 15.2 Query layout

Partition paths cluster by archive/session/table/source/time bucket without placing user labels in
object keys. Within each samples partition, rows sort by:

```text
(metric_family, series_key, clock_ns, attempt_seq)
```

Scrape partitions sort by `(source_id, attempt_seq)`. Parquet statistics and manifest min/max/source
metadata enable pruning. `metric_name_clean` is unnecessary because family identity has its own
column.

The first documentation examples use DuckDB/Polars/Arrow to:

- chart one family/label set over time;
- compute counter deltas with reset handling explicitly;
- inspect scrape failure/missed-tick intervals;
- join phase markers to telemetry;
- compare source cadence and response latency;
- find archive loss/degradation.

### 15.3 Schema evolution

Readers select the manifest schema version before scanning partitions. A minor version may add
nullable columns/value variants. It may not reinterpret existing fields. A major writer change
uses a new table/schema path and manifest major. Compaction preserves original batch IDs and source
values and declares its input/output schema versions.

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
  may be local-only or fail if remote durability is required.
- attached archive failure: native benchmark continues; report marks archive degraded/lossy unless
  explicitly required.
- manifest identity/hash failure: fail closed; never guess or glob a replacement dataset.

### 16.3 Parse failure

Malformed exposition yields no partial successful sample batch by default. The scrape record keeps
the typed error. An optional parser policy may accept a standards-defined partial document only if
the outcome explicitly records partiality and exact rejected-line counts; the default is atomic.

### 16.4 Process crash

The latest valid manifest plus WAL defines recovery. Unreferenced remote objects are harmless
orphans. A crash never makes directory enumeration the logical dataset and never causes an old
checkpoint to be concatenated with its committed replacement.

---

## 17. Performance and capacity budgets

The archive is control-plane work, but attached mode must prove no data-plane regression.

Required budgets:

- no per-request or per-token archive work;
- one allocation-owned batch per scrape after decode;
- bounded source-to-writer bytes and source cardinality;
- one IO owner and no global writer mutex;
- Parquet compression/flush outside request `LocalSet`;
- configurable row/byte/Clock-age partition rotation;
- bounded shutdown/checkpoint duration;
- no full-history rewrite during periodic sync;
- manifest memory proportional to partitions, with later hierarchical manifests if scale demands;
- stable memory under a 24-hour high-cardinality watch soak.

Benchmark acceptance requires statistically indistinguishable request throughput/latency with
attached archival disabled versus enabled at supported telemetry rates. Standalone watch must
publish measured maximum samples/sec, sources, series cardinality, spool growth, and object-store
bandwidth for its supported profile; no unmeasured throughput claim enters documentation.

---

## 18. Verification strategy

### 18.1 Parser/schema gates

1. classic and strict OpenMetrics escaped labels, UTF-8, commas, quotes, backslashes, HELP/TYPE/UNIT;
2. multiple histogram label sets remain isolated and structured;
3. summaries and supported exemplars survive archive projection while benchmark projection keeps
   its intentional exclusions;
4. `100000001` and representative large counters survive Float64 Parquet round-trip;
5. NaN/±Inf map to tagged value kinds and never serialize as invalid JSON/non-finite Parquet values;
6. deterministic labels, series hashes, schema fingerprint, and exact manifest golden;
7. property-based parse/encode/decode round trips and malformed-input atomic failure.

### 18.2 Scheduling gates

1. `SimClock` exact fixed deadlines with zero scrape time;
2. overrun skips debt without drift or catch-up bursts;
3. slow source A cannot shift source B deadlines;
4. one source never has two scrapes in flight;
5. boundary command priority and rejoin to the original cadence;
6. failed/empty/duplicate/missed outcomes have exact counters/records;
7. graceful signal stops issuance before writer drain.

### 18.3 Durability/recovery gates

1. crash injection after every WAL/partition/manifest commit step;
2. acknowledged batch set equals recovered referenced batch set exactly once;
3. stale/repeated WAL frames cannot duplicate rows;
4. corrupt/truncated files are rejected without deleting the last good generation;
5. remote upload retries and conditional-manifest CAS are idempotent;
6. remote outage preserves local durability and respects spool policy;
7. exact-resume mismatch fails before source activation;
8. compaction failure leaves the old manifest authoritative;
9. 24-hour accelerated soak produces immutable incremental partitions, not quadratic rewrites.

### 18.4 Security gates

1. endpoint userinfo/auth headers/object-store credentials absent from every artifact/log/error;
2. redaction precedes series hashing and persists through resume;
3. label/body/diagnostic bounds reject adversarial cardinality/size;
4. raw-body retention is off by default and requires explicit config acknowledgment;
5. path traversal and unsafe local spool/target aliasing fail validation;
6. manifest/partition hashes detect tampering/corruption.

### 18.5 Product subprocess gates

1. Python `aiperf watch` -> exact packaged runner -> in-process HTTP Prometheus mock -> local
   manifest/Parquet -> terminal response;
2. multiple endpoint cadences including one slow/failing source;
3. HTTP 500 with metric-looking body remains an HTTP failure record, not a sample;
4. SIGINT/SIGTERM graceful finalization and forced-crash exact resume;
5. local object-store emulator periodic sync/restart/finalization;
6. ordinary scheduled benchmark with attached server/GPU archive proves one scrape feeds report and
   archive, phase markers align exactly, and native metrics are unchanged;
7. online-only runner capability advertises watch; unsupported distribution/pair fails before IO.

### 18.6 Query compatibility gates

Golden archives are read by pinned Arrow, DuckDB, and Polars versions. Queries prove structured
label filtering, histogram reconstruction, phase joins, failure-gap discovery, and partition
pruning. The manifest, not an implementation-specific directory glob, supplies file lists.

---

## 19. Implementation increments

### Increment 1 — exposition and archive schema

1. extract the standards-correct `aiperf-prometheus` lexical/model seam;
2. preserve current server/DCGM projection semantics with parity tests;
3. define archive DTOs, schema fingerprint, series/batch identity, and golden Parquet/manifest;
4. add the five Tachometer regression fixtures as mandatory tests.

### Increment 2 — local writer and recovery

1. implement the single-owner IO worker and bounded ingress/control channels;
2. add framed WAL, rotation policy, immutable Parquet, local manifest generations, and fsync order;
3. add crash-point/property recovery matrix and spool quotas;
4. ship no product command until exact-once recovery gates pass.

### Increment 3 — source runtime and watch product path

1. implement factory-backed source registry and per-source fixed-deadline drivers;
2. register `telemetry_watch` with runner-v2 and advertise the computed backend/workload pair;
3. add strict Python Config-v2 projection and `aiperf watch` command;
4. add local archive, signal, failure, and query subprocess gates.

### Increment 4 — benchmark attachment

1. add batch-observer tees to existing server/GPU/network sidecars;
2. emit exact lifecycle/phase markers;
3. add archive provenance/health to native-v2;
4. prove no extra scrapes, no metric drift, and no request-path backpressure.

### Increment 5 — object-store durability and resume

1. implement object-store target factory, immutable uploads, conditional generations, and retry;
2. implement create-new/exact-resume policies and remote/local reconciliation;
3. add emulator failure matrix, local-only terminal states, and long soak;
4. document operational recovery and orphan/GC procedures.

### Increment 6 — optional compaction and ecosystem docs

1. add manifest-transactional compaction without in-place history mutation;
2. publish Arrow/DuckDB/Polars examples and schema compatibility policy;
3. measure supported scale profiles;
4. consider hierarchical manifests only after observed partition-count evidence.

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
large transactions. Upload immutable new partitions once.

### Fire overlapping scrape tasks to maintain frequency

Rejected. It creates per-source races, unbounded work under slow endpoints, and ambiguous ordering.
Use one in-flight scrape per source and explicit missed ticks.

### Use wall clock for each sample

Rejected. Wall-clock steps destroy monotonic ordering. Capture one injected epoch anchor and derive
cross-process time from `Clock` deltas.

### Dynamic wide columns for arbitrary metadata

Rejected. Endpoint-dependent schemas complicate unions/evolution and invite name collisions. Use
structured attributes with stable core columns.

---

## 21. Completion criteria

This design is complete only when:

- Python exposes `aiperf watch` and no second Rust executable exists;
- the exact runner advertises and validates `online_http + telemetry_watch` from frozen factories;
- every source uses the injected Clock/native transport and independent fixed-deadline driver;
- generic Prometheus parsing is standards-correct, structured, Float64, and shared without changing
  existing benchmark projection semantics;
- the archive schema and manifest are versioned, deterministic, and readable by the pinned query
  ecosystem;
- local WAL/immutable partitions recover every acknowledged batch exactly once across all injected
  crash points;
- remote sync uploads immutable partitions and advances conditional manifest generations without
  full-history rewrites;
- exact resume fails closed on identity/config/schema mismatch and concurrent writers;
- failures, gaps, duplicates, misses, drops, local durability, and remote lag are observable;
- enrichment is additive, redaction precedes hashing, and secrets never enter artifacts;
- attached mode reuses existing scrapes, emits exact phase markers, leaves native results unchanged,
  and shows no supported-profile request-path regression;
- primary watch and attached modes have real Python-to-runner subprocess proofs;
- the five reproduced Tachometer defects are permanent regression tests;
- native-v2 identifies archive provenance/completeness without treating archived samples as native
  benchmark metrics.

Until these gates pass, the existing phase-bounded native telemetry pipeline remains code truth and
no `watch` capability should be advertised.
