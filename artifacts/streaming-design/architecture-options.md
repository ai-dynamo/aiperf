<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Architecture options

## Decision

Adopt an additive, composable streaming dataset plane selected by a registered
`shadow_replay` workload. Register source and format factories independently,
retain current finite dataset and Graph-IR contracts, and reuse the existing
clock, scheduled runtime, transport, observer, and phase seams at execution.

The architecture unifies two input lifecycles:

- **finite streaming**: a large immutable dataset is processed shard by shard
  with bounded memory and ends only after a sealed snapshot reaches EOF;
- **following streaming**: an open source publishes immutable partitions over
  time and remains pending between arrivals until stop or an explicit seal.

They share acquisition, decode, backpressure, cursor, checkpoint, and consumer
contracts. Event-time replay is an optional downstream policy, not a property
of every streaming dataset.

Source partitions are acquisition boundaries only. A run-scoped session
coordinator routes every fragment by stable session key, so one multi-turn,
agentic, or graph session may span arbitrarily many partitions without being
closed or reconstructed at a chunk boundary. Checkpoint barriers also drive a
checkpointed results sink: immutable result segments and a canonical manifest
make partial progress queryable and restart-safe without keeping the complete
run in memory.

## Option A: retrofit `DatasetLoader::load`

Change `DatasetLoader::load -> Vec<RawRow>` to return an async row stream and
teach existing composers and `Dataset` to ingest incrementally.

### Advantages

- Reuses existing format IDs and loader registration.
- Appears to minimize new public traits.

### Rejection

This changes the meaning of nearly every downstream dataset contract while
still failing to solve session closure, event-time watermarks, run-time source
arrival, checkpointing, and cellular placement. Existing samplers require a
stable population, composers consume a complete vector, `Dataset` owns a
resident segment arena, and multi-phase runs expect repeatable enumeration.

An async iterator hidden beneath a caller that ultimately collects the entire
dataset is not streaming support. It would also make current finite callers
handle `Pending`/follow lifecycle they do not need. Preserve the finite
contracts and extract reusable decoders instead.

## Option B: one monolithic S3/Dynamo replay workload

Implement a new workload that lists S3, parses Dynamo trace rows, reorders them,
and dispatches requests.

### Advantages

- Fastest route to the first demonstration.
- Few initial types.

### Rejection

It binds source authority, object discovery, wire format, session assembly,
watermarks, replay timing, and target execution into one implementation. It
cannot naturally support large HF/Baseten datasets, local live feeds, other
object stores, alternative trace formats, or non-replay streaming consumers.
It also encourages Dynamo request construction inside the loader, bypassing
`RequestMaterializer` and endpoint/transport neutrality.

This option creates a feature, not a platform.

## Option C: mutable Graph-IR during execution

Continuously compile arriving trace chunks into nodes appended to the currently
executing `GraphRecord`/`GraphInputBundle`.

### Advantages

- Reuses the graph executor for dependency-shaped traces.
- Could eventually support streaming tool/agent graphs.

### Rejection for generation 1

The current graph contract depends on complete topology validation, root
selection, cycle/deadlock analysis, stable segment identity, and a frozen
bundle. Cross-object parents make missing data indistinguishable from roots
until a watermark closes the relevant time/key space. Mutation would invalidate
inspection and scheduler invariants and is unnecessary for request-level shadow
replay or ordinary large dataset streaming.

The source/acquisition substrate is intentionally reusable by a future
`streaming_graph` consumer, but that consumer needs its own explicit graph
fragment, closure, and incremental-validation design.

## Option D: general process-external broker

Define Kafka-like RPC/IPC and run acquisition/decoding in a separate service.

### Advantages

- Independent scaling and crash isolation.
- Existing streaming systems could provide retention and replay.

### Rejection as the native baseline

It introduces serialization, deployment, authority, compatibility, and failure
surfaces before AIPerf has a correct in-process contract. It would make S3 and HF
support dependent on another service and violate the pure-Rust single-product
requirement. A later source adapter may consume an external broker, but the
broker is not the AIPerf architecture.

## Option E: composable native streaming dataset plane

Add registered source and format factory categories, compose them inside a
registered workload, and connect the output to existing scheduled execution.

### Advantages

- Source, format, ordering, replay, placement, and checkpoint policies have
  independent ownership.
- Supports both multi-GB finite data and perpetual live feeds.
- Preserves existing finite dataset and Graph-IR meanings.
- Reuses transport-neutral request materialization, phase shutdown, metrics,
  real/sim clocks, and worker-local dispatch.
- Provides explicit boundedness and restart semantics.
- Lets S3/NVCF, HF/Baseten, local files, and future sources share the runtime.

### Costs

- Requires new registry categories and protocol-v2 configuration.
- Requires a new canonical streaming-unit and watermark contract.
- Cellular execution needs incremental bounded transfer rather than only fixed
  startup snapshots.
- Some current format logic must be separated from global collection/grouping.

### Resolution

Selected. The detailed specification defines its invariants and migration.

## Additive now, convergence later

The first implementation is additive. Once both paths are proven, finite
datasets may be represented internally as a sealed streaming source followed by
an explicit `collect` consumer, but the public finite adapters keep their
completion semantics. That migration is optional and evidence-driven; it is not
a prerequisite for shadow replay.

## Registry boundary decision

`StreamingDatasetSourceFactory` and `StreamingDatasetFormatFactory` are named
registry categories because config selects them and independent implementations
must compose. `EventTimePolicy`, `LateRecordPolicy`, `ReplayAdmissionPolicy`,
checkpoint writers, and placement are initially host-owned validated policies
or constructor-injected traits. They are seams for testing and future
replacement, but not every seam needs global name-based discovery.

The native-runtime-plugin exemplar currently limits dynamically loadable
generation-1 categories. This design adds the new factories to the statically
composed `AIPerfRegistry`; exposing them through shared-library manifests would
require a separately reviewed plugin API generation, ownership table, and
performance contract. No hidden dynamic-plugin promise is made here.
