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
3. **One scrape per source event.** Accumulator and archive projections fan out from the same
   successful scrape; enabling archival never doubles source traffic.
4. **All sampling time through `Clock`.** No `Instant`, `SystemTime`, or `tokio::time` in the
   clock-aware source/scheduler path.
5. **No request-path backpressure.** Archive serialization, compression, and upload never run on
   or block the per-request/per-token local loop.
6. **One mutable owner.** One archive IO worker owns WAL, partition builders, and manifests; no
   shared writer mutex exists.
7. **Structured identity.** Metric family, semantic type, source, and labels are separate fields.
8. **Float64 preservation.** Every finite source number remains `f64` through Parquet.
9. **Non-finite values are explicit.** NaN/±Inf never cross a serialization boundary as an
   unclassified JSON/Parquet number.
10. **No silent loss.** Failed, empty, duplicate, missed, backpressured, and dropped observations
    are counted and surfaced in scrape records/manifest health.
11. **Exactly-once logical commit.** Every durably acknowledged batch has one logical identity and
    appears at most once in the manifest's referenced partitions after any recovery.
12. **Immutable remote history.** Sync uploads content-addressed partitions; it never rewrites the
    whole archive.
13. **Fail closed on identity.** Resume requires matching schema, archive ID, configuration digest,
    source descriptors, and runner distribution compatibility.
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
semantic response, dataset, phase list, or `RequestSink`. The `online_http` prepared backend must
expose its injected `Clock` and reusable control transport through a trait-backed preparation
capability. This is not a special `reqwest` path.

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
summaries. Existing sidecars retain ownership of source activation and boundary scrapes. Each
decoded source record is projected twice:

```text
one source scrape
  +-- native accumulator (authoritative for this run)
  `-- archive ingress (forensic, bounded, independently fallible)
```

Phase start/end callbacks also emit archive lifecycle markers. Archive availability cannot change
request timings or metric formulas. The terminal report records archive completeness separately.

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

`aiperf-prometheus` is an IO-free leaf containing a standards-correct exposition model and parser
seam. It extracts the lexical/state-machine core currently embedded in server/GPU parsers:

```rust
pub trait ExpositionParser {
    fn parse_classic(&self, body: &str) -> Result<Exposition, ParseError>;
    fn parse_openmetrics(&self, body: &str) -> Result<Exposition, ParseError>;
}
```

`Exposition` preserves HELP/TYPE/UNIT metadata, structured escaped labels, counters, gauges,
untyped samples, summaries, histograms grouped by the complete base label set, and supported
OpenMetrics exemplars/timestamps. Projection policies—not the lexer—decide whether benchmark
accumulation excludes summaries, `_created`, or uptime families.

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

Runner-local adapters map existing `ServerMetricsRecord`, `GpuScrape`, and network samples into
archive batches. A later second consumer may justify extracting those adapters, but no cycle or
heavy Parquet dependency is pushed into the domain accumulators.

### 4.3 No hot-path dynamic dispatch

Factories and policies are selected during validation/preparation. Source drivers and the writer
may retain `dyn` seams because scrapes are low-rate control-plane events. Per-sample conversion is
batch-local. There is no registry lookup, allocation policy, lock, or archive callback per token.

---

## 5. Core trait seams

The signatures below are design-level; concrete error DTOs remain plain hand-written library
error enums.

```rust
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
    async fn scrape(&self, reason: ScrapeReason) -> ArchiveScrapeOutcome;
    async fn shutdown(&self) -> Result<(), ArchiveSourceError>;
}

pub trait ArchiveEncoder<Record> {
    fn encode(&self, record: &Record, context: &EncodeContext)
        -> Result<ArchiveBatch, ArchiveEncodeError>;
}

pub trait TelemetryEnricher {
    fn enrich(&self, sample: &mut ArchiveSample, source: &ArchiveSourceIdentity);
}

pub trait TelemetryRedactor {
    fn redact(&self, sample: &mut ArchiveSample, source: &ArchiveSourceIdentity);
    fn redact_error(&self, error: &ArchiveDiagnostic) -> ArchiveDiagnostic;
}

pub trait SegmentRotationPolicy {
    fn should_rotate(&self, state: &OpenSegmentState, now_ns: i64) -> bool;
}

pub trait ArchiveAdmissionPolicy {
    fn admit(&self, state: ArchiveIngressState, batch: &ArchiveBatch) -> AdmissionDecision;
}

pub trait ArchiveRecoveryPolicy {
    fn recover(&self, local: &LocalArchiveState, remote: Option<&RemoteArchiveState>)
        -> Result<RecoveryPlan, ArchiveRecoveryError>;
}

#[async_trait]
pub trait ArchiveSink: Send {
    async fn recover(&mut self) -> Result<RecoveredArchive, ArchiveSinkError>;
    async fn append_batch(&mut self, batch: ArchiveBatch) -> Result<AppendReceipt, ArchiveSinkError>;
    async fn append_marker(&mut self, marker: ArchiveMarker) -> Result<(), ArchiveSinkError>;
    async fn checkpoint(&mut self) -> Result<CheckpointReceipt, ArchiveSinkError>;
    async fn finalize(&mut self, reason: TerminationReason)
        -> Result<FinalizedArchive, ArchiveSinkError>;
}

pub trait EpochAnchorProvider {
    fn anchor(&self, clock: &dyn Clock) -> Result<EpochAnchor, EpochAnchorError>;
}
```

At least these concrete implementations ship:

- `PrometheusArchiveSource` and replay/fault-injection sources;
- `StaticLabelEnricher` and `NoopEnricher`;
- allow/deny-key redactors and `NoopRedactor`;
- row/byte/Clock-age segment rotation policies composed by `AnyRotationPolicy`;
- primary-watch and attached-best-effort admission policies;
- create-new and exact-resume recovery policies;
- `ParquetArchiveSink` over a local spool plus optional `dyn ObjectStore`;
- `MemoryArchiveSink` for deterministic tests.

Stable wire IDs select factories/policies through frozen registries. A core string `match` does not
select implementations.

---

## 6. Source scheduling and isolation

### 6.1 One driver per source, one scrape in flight

Each source has one local driver task. Drivers are independent, so a slow endpoint cannot delay a
different endpoint. A driver serializes continuous and forced-boundary commands for its own
source; two requests to the same endpoint never overlap.

The driver owns:

- the prepared source;
- cadence anchor and tick index;
- attempt sequence;
- consecutive/total failure counters;
- current source state (`active`, `degraded`, `disabled`, `stopped`);
- a command channel with reserved capacity for boundary/shutdown commands.

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

Attached phase barriers submit a forced `BoundaryStart` or `BoundaryEnd` command. It preempts the
next continuous deadline but never interrupts an already issued HTTP request. The phase waits for
the forced result under its configured Clock deadline. The same decoded record feeds accumulator
and archive projection. Continuous scheduling re-anchors from the original cadence, not the
boundary completion time.

### 6.4 Failure classification

Every attempt becomes one `ArchiveScrapeRecord`, including:

- success with samples;
- success with an empty exposition;
- unchanged body/duplicate;
- HTTP status failure;
- transport/timeout failure;
- parse failure with line/category and a redacted bounded diagnostic;
- source-incompatible terminal disable;
- missed tick or admission skip;
- source shutdown failure.

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
    pub uncertainty_ns: u64,
}
```

For a real-clock session:

```text
unix_ns(t) = anchor.unix_epoch_ns + (clock_ns(t) - anchor.clock_ns)
```

The concrete system provider reads wall time only while creating the injected anchor. Every later
timestamp derives from the monotonic `Clock`, so NTP steps cannot reorder a session. Tests inject a
fixed provider. Each restart creates a new `archive_session_id` and anchor; the manifest retains
both. Virtual sessions set `time_domain="virtual"` and omit Unix time.

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

### 8.1 Design rules

- Arrow/Parquet types are explicit and versioned.
- Labels remain `map<string,string>` in deterministic key order.
- `series_key` is BLAKE3 over source ID, metric family, semantic type, and canonical labels.
- Histograms and summaries stay one structured sample per base label set.
- Numeric source values use Float64; source-native integer identity may additionally use UInt64.
- non-finite values use a tagged representation, not raw NaN/Inf at the boundary;
- source and scrape identity are repeated where needed so partitions are independently readable;
- arbitrary enrichment stays in a structured attributes map, not dynamically added columns;
- schemas add nullable fields within a major version; incompatible changes increment the major.

### 8.2 Manifest

`archive-manifest.json` is the only authority over referenced partitions:

```jsonc
{
  "archive_schema_version": "1.0",
  "archive_id": "uuid",
  "archive_sessions": [
    {
      "session_id": "uuid",
      "time_domain": "real_monotonic",
      "epoch_anchor": {"clock_ns": 0, "unix_epoch_ns": "...", "uncertainty_ns": 1000},
      "runner_distribution_id": "blake3:..."
    }
  ],
  "config_digest": "blake3:...",
  "sources": [],
  "partitions": [
    {
      "table": "samples",
      "key": "parts/samples/.../blake3-....parquet",
      "content_hash": "blake3:...",
      "row_count": 10000,
      "min_clock_ns": 0,
      "max_clock_ns": 1000000000,
      "source_ids": ["server-a"]
    }
  ],
  "health": {
    "attempted": 0, "succeeded": 0, "failed": 0, "empty": 0,
    "duplicates": 0, "missed_ticks": 0, "admission_skips": 0,
    "dropped_batches": 0, "local_durable_batches": 0, "remote_referenced_batches": 0
  },
  "finalized": false,
  "termination_reason": null,
  "generation": 7
}
```

`unix_epoch_ns` is a decimal string because JSON cannot exactly represent i128 nanoseconds.
Manifests never contain credentials, signed URLs, response bodies, or unredacted diagnostics.

### 8.3 Scrape-attempt table

One row exists per attempt/gap:

| Field | Type | Meaning |
|---|---|---|
| `archive_id`, `session_id`, `source_id` | UTF-8 | stable identity |
| `attempt_seq` | UInt64 | monotone within `(session,source)` |
| `batch_id` | fixed binary/hex | hash of session/source/attempt |
| `reason` | dictionary UTF-8 | cadence, boundary-start/end, manual, retry |
| `outcome` | dictionary UTF-8 | success, empty, duplicate, HTTP, transport, parse, missed, dropped, disabled |
| scheduling/request/capture timestamps | Int64 nullable | Clock timeline |
| `unix_epoch_ns` | fixed decimal/UTF-8 nullable | anchored real time |
| `http_status` | UInt16 nullable | exact response status |
| `latency_ns` | Int64 nullable | request duration |
| `content_hash` | fixed binary nullable | BLAKE3 of exact response bytes |
| `duplicate_of_attempt_seq` | UInt64 nullable | prior identical body |
| `sample_count` | UInt64 | parsed structured sample count |
| `error_kind`, `error_message` | UTF-8 nullable | bounded redacted diagnostic |
| gap/drop counts | UInt64 nullable | compact missed-range representation |

Failed and empty scrapes are therefore queryable rather than inferred from missing rows.

### 8.4 Sample table

One row represents one metric family/base-label-set at one successful scrape:

| Field | Type |
|---|---|
| archive/session/source/batch/attempt identity | same as scrape table |
| `clock_ns`, anchored `unix_epoch_ns` | Int64 / decimal nullable |
| `metric_family` | UTF-8 |
| `semantic_type` | dictionary UTF-8 (`counter`, `gauge`, `histogram`, `summary`, `untyped`) |
| `series_key` | fixed 32-byte BLAKE3 |
| `labels` | map UTF-8 → UTF-8 |
| `attributes` | map UTF-8 → UTF-8 |
| `help`, `unit` | UTF-8 nullable |
| `value_kind` | dictionary UTF-8 (`finite`, `pos_inf`, `neg_inf`, `nan`, `absent`) |
| `scalar_value` | Float64 nullable; present only for finite scalar values |
| `histogram_sum`, `histogram_count` | Float64 nullable |
| `histogram_buckets` | list of `{upper_bound: UTF-8, cumulative_count: Float64}` nullable |
| `summary_sum`, `summary_count` | Float64 nullable |
| `summary_quantiles` | list of `{quantile: Float64, value_kind, value: Float64?}` nullable |
| supported exemplar fields | nullable struct/list |

Histogram bucket bounds stay textual because `+Inf` is a valid bound and source spelling can
matter. Buckets are sorted numerically with `+Inf` last, but no lower bounds are synthesized.
Counts remain cumulative as emitted; phase-delta interpretation belongs to accumulators/views.

### 8.5 Lifecycle marker table

Markers connect operational history to runner events without pretending they are samples:

- watch/session prepared, started, stop requested, draining, finalized;
- benchmark run start/end;
- phase start/end and phase identity;
- archive degraded/recovered;
- source activated/disabled;
- local checkpoint and remote manifest generations.

Each marker has stable identity, Clock time, optional Unix time, typed kind, and a bounded
structured attributes map.

### 8.6 Optional raw-body CAS

The default stores only `content_hash`. `RawBodyRetentionPolicy` may retain compressed exact bodies
under content-addressed keys for all scrapes or failed scrapes. This is explicit opt-in because raw
exposition may contain sensitive labels that bypass structured redaction. The manifest records
policy and retained-byte counts. Raw bodies are never embedded in Parquet rows.

---

## 9. Ingress and writer isolation

### 9.1 Ownership topology

```text
current-thread LocalSet source drivers
    |
    | owned ArchiveBatch, bounded batch-level channel
    v
single archive IO worker thread
    +-- WAL owner
    +-- open segment builders
    +-- local manifest generations
    `-- remote sync state
```

The channel is per batch/scrape, never per token or individual metric. The IO worker alone mutates
state. Parquet encoding, compression, filesystem calls, and object-store clients do not run on the
request `LocalSet`.

A small independent control channel is reserved for lifecycle markers, checkpoint, health
snapshot, and finalize commands so a full data queue cannot deadlock shutdown.

### 9.2 Admission modes

The same runtime supports two explicit policies:

- **primary watch:** the archive is the product. When the data queue/spool is saturated, the
  source driver does not issue unbounded new scrapes. It waits or skips future cadence deadlines
  according to the selected policy, records missed intervals, and fails the operation if local
  durability cannot progress within its budget. Default: fail rather than silently discard.
- **attached benchmark:** the benchmark is the product. Source/accumulator work proceeds; archive
  ingress uses a nonblocking batch admission. Rejected batches increment a loss counter and create
  a terminal gap summary. The request path never waits. `archive.required=true` may convert archive
  degradation into a failed terminal operation after preserving a non-authoritative failure report,
  but it still cannot change measured request data.

Boundary scrapes always reach their native accumulator even if archive admission fails.

### 9.3 Batch identity

```text
batch_id = BLAKE3(archive_id || session_id || source_id || attempt_seq || content_hash/outcome)
```

Retries of persistence retain `batch_id`. A new source request always gets a new `attempt_seq`.
Partitions and recovery use batch IDs to prevent logical duplication.

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
