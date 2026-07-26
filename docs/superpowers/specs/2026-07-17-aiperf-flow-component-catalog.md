# AIPerf Flow Component Catalog

## Status

Proposed capability and symbol catalog for the approved AIPerf Flow design.
This document converts the visual concepts found in the Rust implementation
into a prioritized, reusable Flow vocabulary. It does not change the language
or runtime contracts in the main design record.

The companion
[`2026-07-17-aiperf-flow-core-components-design.md`](2026-07-17-aiperf-flow-core-components-design.md)
specifies the top 25 shareable stdlib components, their compositional contracts,
and the minimal runtime leaf kernel.

The catalog is intentionally larger than the first implementation. It defines
the vocabulary the architecture must be able to acquire without adding
document-specific React, TypeScript, JavaScript, or CSS.

All entries are governed by the AIPerf Flow north star: a generic,
deterministic scene compiler and animator whose live browser rendering has the
composition, typography, motion, lighting, and detail of a professionally
produced high-resolution explainer while remaining interactive and narrated.
High-resolution video is a fidelity metaphor, not the primary output.

Capabilities contribute to one evaluated semantic scene. Canvas 2D is the
preferred cinematic visual backend; a synchronized semantic HTML twin is
always available; simplified SVG/HTML is the required fallback; and future
WebGPU acceleration must consume the same semantics, timing, layout, and hit
contracts.

## Research basis

The catalog was derived from the Rust workspace rather than from existing slide
content. The review covered the six Rust workspace crates and their tests,
examples, and benchmarks:

- `rust/loadgen-core/`;
- `rust/runtime/`;
- `rust/cli/`;
- `rust/mock-server/`;
- `rust/e2e-tests/`;
- `rust/pyext/`.

The most useful visual semantics live in runtime, transport, dataset, graph,
cellular, metrics, telemetry, and mock-server code. The CLI mostly composes
those systems and therefore contributes end-to-end stories, state transitions,
and boundaries rather than a separate rendering vocabulary. `pyext` is
packaging metadata and contributes no domain visualization.

Executable code and manifests remain authoritative. This catalog names source
areas as evidence; it does not treat the catalog itself as a runtime claim.

## Catalog rules

### Three implementation levels

Every entry must begin at the lowest sufficient level:

1. **Core primitive:** generic geometry, text, chart, motion, layout, or
   interaction behavior useful outside AIPerf.
2. **Flow symbol:** a reusable `.flow` definition assembled only from
   capabilities and other symbols.
3. **Capability package:** trusted runtime code for an algorithm, specialized
   renderer, large-cardinality path, or behavior that cannot be implemented
   efficiently by declarative composition.

An AIPerf-specific visual metaphor is not sufficient reason to create runtime
code. Repetition promotes a composition to a symbol. Algorithmic behavior,
performance constraints, or specialized accessibility requirements promote a
symbol to a capability.

### Semantic data before geometry

Domain inputs retain identities and relationships through Flow IR. A compiler
must not lower a token span, prompt segment, graph dependency, cell partition,
or metric lineage into unrelated rectangles before the runtime can inspect the
relationship.

Every domain component must expose:

- stable entity and relationship IDs;
- the source semantic model;
- generated layout and geometry as overridable outputs;
- timeline anchors and transition identities;
- deterministic visual display commands, paint bounds, hit regions, and quality
  tiers independent of a specific rendering API;
- inspectable evidence and provenance;
- a persistent semantic HTML twin with focus/selection synchronization;
- a simplified SVG/HTML fallback plus textual or tabular equivalent;
- reduced-motion and high-contrast behavior;
- pause-to-explore and exact-beat resume behavior;
- deterministic serialization and hashing;
- a declared cost and cardinality model.

### Capability naming

The IDs below are proposed namespaces:

- `core.*` for generic runtime primitives;
- `viz.*` for reusable technical visualization algorithms;
- `aiperf.*` for AIPerf semantic components.

Final IDs must remain globally unique and versioned. Symbol names are shown in
PascalCase; capability IDs use dot-separated lowercase names.

### Runtime binding requirement

The foundation runtime currently renders `group`, `rect`, `text`, and
`connector` nodes through SVG, with initial timeline and inspector behavior.
That path is a pipeline proof and future fallback, not the final visual
architecture. Domain capabilities require dispatch by the capability
identifier authored in Flow IR, not only by a synthesized
`core.${node.kind}` name. The registry, compiler, scene evaluator, Canvas
display-list builder, semantic twin, and fallback renderer must share one
descriptor-derived binding contract before the domain catalog is considered
implemented.

## Core visual substrate

These capabilities are generic prerequisites for most AIPerf components.

### Semantic identity and continuity

- `core.semantic-entity`: stable identity, aliases, labels, descriptions,
  evidence, and state across scenes.
- `core.semantic-relation`: typed directed or undirected relationship with
  source, target, ports, multiplicity, and provenance.
- `core.semantic-morph`: deterministic correspondence between source and target
  entities, including split, merge, reorder, replacement, and disappearance.
- `core.focus-context`: focus one entity or region while retaining spatial
  context and an accessible outline.
- `core.compare`: aligned before/after, side-by-side, overlay, and difference
  states with synchronized inspection.

### Text, spans, and structured content

- `core.glyph-run`: measured glyph sequence with grapheme boundaries and stable
  span IDs.
- `core.span-map`: one-to-one, one-to-many, many-to-one, and overlapping maps
  between source spans and semantic values.
- `core.segment-strip`: ordered, sized, nested, and labeled spans with clipping,
  truncation, and continuation markers.
- `core.structured-payload`: collapsible object, list, message, and byte-range
  inspection without embedding a custom component.
- `core.code-diff`: stable line and token identity across code or configuration
  transformations.

### Queues, resources, and ownership

- `viz.queue`: FIFO, priority, continuation-priority, and bounded queues with
  arrivals, waits, service, rejection, and cancellation.
- `viz.slot-pool`: capacity, checked-out slots, debt after shrink, waiters, and
  release order.
- `viz.resource-ledger`: credits acquired and returned across independently
  owned resources.
- `viz.ownership-map`: entities partitioned among workers, cells, processes, or
  services with transfer and merge boundaries.
- `viz.barrier`: parties, generations, arrivals, leader action, release, abort,
  and timeout.

### Time and event order

- `viz.event-lane`: timestamped events with causal and wall-clock ordering.
- `viz.waterfall`: nested intervals, points, open intervals, and derived spans.
- `viz.phase-lifecycle`: prepare, prewarm, warmup, profile, stop, grace, cancel,
  drain, force, finalize, and failure transitions.
- `viz.dual-clock`: authored, simulated, monotonic, wall, and observer-relative
  times with explicit conversion edges.
- `viz.event-frontier`: the next event across multiple event sources and the
  rule that advances the selected clock.
- `viz.ramp`: linear, exponential, and stochastic control curves with target,
  update cadence, actuator state, and deterministic RNG lineage.

### Graphs, trees, and high-cardinality views

- `viz.compound-graph`: nested nodes, typed ports, routed edges, edge bundles,
  collision handling, and level of detail.
- `viz.execution-graph`: static topology overlaid with node readiness,
  execution, first-token, terminal, and failure state.
- `viz.prefix-trie`: content-addressed prefixes, longest-common-prefix paths,
  split points, cache blocks, and reconstruction.
- `viz.partition-grid`: modulo, range, hash, and hierarchical partitions with
  exact coverage and overlap diagnostics.
- `viz.reduction-tree`: fanout, tiers, partial aggregates, barriers, failures,
  and the final root value.
- `viz.metric-dag`: record, aggregate, derived, rate, timeslice, and SLO metric
  dependencies with missing-value propagation.
- `viz.column-store`: sparse columns, row ordinals, append/merge boundaries,
  exact and sketch storage, and retained versus dropped data.

### Technical charts

- `viz.distribution`: histogram, density, percentile ruler, box plot, and
  tail-focused views from one semantic distribution.
- `viz.sweep-line`: event deltas, active count, threshold crossings, and
  selected windows.
- `viz.timeseries-stack`: aligned metrics with gaps, phases, annotations,
  thresholds, and selectable scales.
- `viz.capacity-surface`: workload dimensions mapped to latency, throughput,
  goodput, saturation, and failure regions.
- `viz.parity`: exact, tolerance-based, aggregate-equivalent, and unsupported
  comparisons with field-level evidence.

## AIPerf domain packages

### Prompt, segment, and token semantics

Evidence: `rust/runtime/src/dataset/`, `rust/runtime/src/body_plan.rs`,
`rust/runtime/src/multiturn.rs`, endpoint request builders, and the server and
local tokenizer paths.

#### `aiperf.prompt-composition`

Represents an ordered prompt as typed segments: system, user, assistant, tool,
image, audio, video, generated, reused, truncated, and synthetic. It preserves
segment IDs, source handles, token counts, byte ranges, roles, nesting, and
materialization order.

Primary symbols:

- `PromptSegmentComposer`;
- `SegmentHandleChip`;
- `PromptLengthRuler`;
- `TruncationWindow`;
- `PrefixReuseBand`;
- `MultimodalPartStrip`.

The first implementation should be a `.flow` symbol over
`core.segment-strip`. A capability is justified only for large prompt
cardinality or specialized text measurement.

#### `aiperf.tokenization`

Maps source graphemes and bytes to token spans, token IDs, decoded text, and
special tokens while preserving reverse traceability.

Primary symbols:

- `TokenSpanMorph`;
- `TokenIdRail`;
- `TokenizerRoundTrip`;
- `SpecialTokenMarker`;
- `ServerLocalTokenizerCompare`.

This package depends on `core.glyph-run`, `core.span-map`, and
`core.semantic-morph`. It must support non-bijective tokenization, Unicode
graphemes, byte fallback, and reduced motion without losing correspondence.

#### `aiperf.segment-store`

Shows `load → compose → store → sample → materialize`, content-addressed
segment handles, BLAKE3 prefix-dependent identity, shared segments, and
materialized request bodies.

Primary symbols:

- `DatasetPipeline`;
- `SegmentStoreMap`;
- `ContentAddressChain`;
- `MaterializationTrace`;
- `SamplerDrawDeck`.

#### `aiperf.prefix-cache`

Relates prompt blocks and prefix tries to cache lookup, reuse, admission,
eviction, and observed reuse metrics.

Primary symbols:

- `PromptBlockReuseStrip`;
- `PrefixCacheTrie`;
- `CacheAdmissionTimeline`;
- `IdealRealizedReuseCompare`;
- `KvBlockResidencyMap`.

This is a capability package when it performs trie layout, large block
virtualization, or animated longest-common-prefix matching.

### Endpoint and request construction

Evidence: `rust/runtime/src/endpoints/`, `rust/runtime/src/body_plan.rs`, and
`rust/runtime/src/content_server/`.

#### `aiperf.endpoint-registry`

Shows endpoint registration, identifier lookup, descriptor validation,
configuration preparation, worker-local endpoint tables, and fail-closed
unknown identifiers.

Primary symbols:

- `EndpointRegistryCatalog`;
- `EndpointDescriptorCard`;
- `PreparedEndpointTable`;
- `EndpointDialectMatrix`;
- `CapabilityFreezeBoundary`.

#### `aiperf.body-plan`

Explains how authored content becomes a protocol request through literal bytes,
segment splices, JSON insertion, media URLs, headers, parameters, chat
templates, and endpoint-specific dialect rules.

Primary symbols:

- `BodyPlanSplice`;
- `RequestAssemblyTree`;
- `PartShapeMorph`;
- `ChatTemplateInjection`;
- `RawPayloadBypass`;
- `MediaUrlPublication`.

#### `aiperf.content-publication`

Shows synthetic or file-backed media entering the content store, receiving a
fingerprinted URL, being served, and becoming a multimodal request part.

Primary symbols:

- `ContentAddressedAsset`;
- `MediaPublicationPath`;
- `AssetLifetimeTimeline`;
- `UrlToRequestPart`.

### Transport and measurement

Evidence: `rust/runtime/src/transport/` and `rust/loadgen-core/`.

#### `aiperf.request-lifecycle`

The canonical request story from authored turn through arrival, admission,
connection acquisition, request send, response headers, streaming chunks,
tokens, usage, terminal classification, and record finalization.

Primary symbols:

- `RequestLifecycleWaterfall`;
- `ObserverEventRail`;
- `SendCompleteHeaderCompare`;
- `TokenArrivalTrain`;
- `TerminalClassificationGate`;
- `RequestRecordInspector`.

This component must distinguish:

- arrival from admission;
- send completion from response headers;
- first visible token from terminal usage and `[DONE]`;
- reasoning from visible output tokens;
- endpoint usage from client tokenization;
- transport failure from application-level failure.

#### `aiperf.connection-pool`

Shows endpoint selection, connection reuse, HTTP/1 versus h2c, UDS, TLS,
pre-send connect retries, linear clock-driven backoff, and retry exhaustion.

Primary symbols:

- `ConnectionPoolAcquire`;
- `ProtocolNegotiationPath`;
- `ConnectRetryLadder`;
- `EndpointRoundRobin`;
- `LoopbackProxyBypass`.

#### `aiperf.streaming-parser`

Shows byte chunks entering an incremental parser, complete SSE lines, UTF-8
boundaries, event payload reduction, token classification, and terminal frames.

Primary symbols:

- `ChunkBoundaryLens`;
- `SseLineAssembler`;
- `Utf8CarryBuffer`;
- `ResponseReductionPipeline`;
- `PrefillReleaseFilter`.

The parser is a capability because byte-boundary simulation and high-volume
stream rendering are algorithmic.

#### `aiperf.timeout-stack`

Compares connect, request, stream-idle, cancellation, phase grace, and force
timeouts, including which layer owns the terminal result.

Primary symbols:

- `TimeoutNesting`;
- `CancellationRace`;
- `GraceDrainEscalation`;
- `FailureOwnershipTrace`.

### Scheduling, phases, and control

Evidence: clock, timing, scheduled, request-rate, user-centric,
fixed-schedule, phase-runtime, adaptive, and workload modules.

#### `aiperf.workload-scheduler`

Shows request-rate, concurrency, user-centric, and fixed-schedule workload
families over shared admission and clock seams.

Primary symbols:

- `WorkloadFamilySelector`;
- `ArrivalGeneratorPlot`;
- `ContinuationPriorityQueue`;
- `VirtualUserHistory`;
- `FixedScheduleReplay`;
- `RequestBudgetPartition`.

#### `aiperf.phase-orchestrator`

Shows authored phase order, resource preparation, prewarm, barriers, controller
start/stop, seamless handoff, grace, cancellation, drain, final statistics, and
sidecar activation.

Primary symbols:

- `PhaseLifecycleOrchestrator`;
- `WarmupProfilingHandoff`;
- `PhaseResourceLedger`;
- `GraceEscalationTimeline`;
- `SidecarPhaseOverlay`.

#### `aiperf.adaptive-control`

Shows a closed feedback loop from completed records through a measurement
window and SLA filters to a step policy and actuator.

Primary symbols:

- `AdaptiveAssessmentLoop`;
- `SlaFilterStack`;
- `ControlVariableGauge`;
- `SlaMarginStep`;
- `SustainGate`;
- `RampAdaptiveConflict`.

The package must distinguish closed-loop runtime control from CLI-level
open-loop search and sweeps.

#### `aiperf.rng-lineage`

Shows root seeds and deterministic namespaces for dataset, sampler, arrival,
cancellation, phase, actuator, curve, media, model selection, and cells.

Primary symbols:

- `RngDerivationTree`;
- `SeedParityCompare`;
- `SamplerPermutationDeck`;
- `CellSeedCoherence`.

### Graph execution and trajectory handling

Evidence: `rust/runtime/src/graph/` and
`rust/runtime/src/engine/graph_phase_runtime.rs`.

#### `aiperf.graph-program`

Shows graph records, traces, nodes, channels, static and timing edges, segment
items, initial state, and compiled execution plans.

Primary symbols:

- `GraphProgramExplorer`;
- `ChannelRequirementPort`;
- `TracePlanOverlay`;
- `SegmentItemInspector`;
- `GraphSourceFormatCompare`.

#### `aiperf.graph-execution`

Animates channel availability, node readiness, dispatch, first token, terminal
records, output publication, downstream firing, and trace completion.

Primary symbols:

- `AsyncDataflowScene`;
- `NodeReadinessGate`;
- `FirstTokenDependency`;
- `GraphFailurePropagation`;
- `TraceCompletionLedger`.

This requires a capability over `viz.execution-graph` because readiness and
firing are algorithmic and must remain usable at high cardinality.

#### `aiperf.trajectory-window`

Shows recorded traces compiled into an LCP trie, sampled `t*`, warmup
reconstruction before the cut, profiling execution after the cut, pressure
lanes, and per-lane timing.

Primary symbols:

- `LcpTrieCompiler`;
- `TStarCut`;
- `WarmupReprime`;
- `PressureLaneMatrix`;
- `TrajectoryParityOverlay`;

#### `aiperf.graph-phase`

Combines graph execution with phase accounting: session and prefill credits,
first-token release, terminal return, cancellation, failure ledgers, adaptive
sampling, and warmup abort policy.

Primary symbols:

- `GraphPhaseOrchestrator`;
- `GraphCreditLedger`;
- `FirstTokenReleaseGate`;
- `WarmupFailureLedger`;
- `LaneReturnWall`.

### Cellular and distributed execution

Evidence: `rust/runtime/src/cellular/`, `rust/runtime/src/hub/`, and cellular
engine modules.

#### `aiperf.cell-partition`

Shows the two-level `(cell × worker)` partition, modulo ownership, phase-local
budgets, global ordinal bases, session ownership, trace ownership, and exact
coverage.

Primary symbols:

- `CellPartitionGrid`;
- `TwoLevelOrdinalTiling`;
- `ConversationOwnershipMap`;
- `TraceOwnershipMap`;
- `BudgetSliceInspector`;

#### `aiperf.cellular-control-plane`

Shows controller, hub, cells, aggregator tiers, registration, start barriers,
heartbeats, terminal partitions, artifact streams, and failure detection.

Primary symbols:

- `CellularTopology`;
- `RegistrationBarrier`;
- `HeartbeatSketchTree`;
- `ArtifactShippingPath`;
- `HubPluginSurface`;
- `SlurmRankTopology`.

#### `aiperf.merge-paths`

Compares retained-record, exact-fold, and sketch paths from worker-local capture
through cell shipping and controller reduction.

Primary symbols:

- `RetentionDecisionTree`;
- `RetainedRecordMerge`;
- `ExactFoldStoreMerge`;
- `SketchDigestMerge`;
- `GlobalOrdinalJoin`;
- `ArtifactLaneConcat`.

This package must visualize semantic differences precisely:

- retained records merge in global dispatch order;
- exact-fold uses dense local stores and append/merge boundaries;
- sketch mode retains bounded approximate distributions;
- some cellular knobs are aggregate-equivalent rather than byte-identical;
- cross-host sidecars and artifact fidelity have explicit limitations.

#### `aiperf.reduction-topology`

Shows multi-tier aggregator placement, fanout, parent coordinates, barriers,
partial store reduction, and root delivery.

Primary symbols:

- `AggregatorTierTree`;
- `StorePartitionFlow`;
- `ReductionBarrier`;
- `PartialFailureCut`;

### Metrics, telemetry, and artifacts

Evidence: metrics core, report, export, accuracy, server metrics, GPU telemetry,
network latency, and engine record finalization.

#### `aiperf.metrics-pipeline`

Shows `RequestObserver → RecordIngest → ColumnStore → MetricsAccumulator →
AccumulatorSummary → NativeReport`.

Primary symbols:

- `MetricsPipeline`;
- `RecordIngestInspector`;
- `MetricCatalogDag`;
- `ColumnStoreExplorer`;
- `SummaryProjection`;

#### `aiperf.metric-semantics`

Explains TTFT, TTST, ITL, TPOT, request latency, throughput, goodput, visible
token counts, endpoint usage, cache reuse, error counts, and missing values.

Primary symbols:

- `LatencyMetricDerivation`;
- `TokenMetricLedger`;
- `UsageVisibleTokenCompare`;
- `GoodputSloOverlay`;
- `MissingMetricPropagation`.

#### `aiperf.steady-state`

Shows the in-flight sweep line, target fraction, first up-crossing, last
down-crossing, excluded ramp and drain, and window-scoped summary.

Primary symbols:

- `InflightSweepLine`;
- `SteadyStateWindow`;
- `RampDrainExclusion`;
- `WindowSummaryCompare`.

#### `aiperf.telemetry-plane`

Aligns benchmark phases with GPU samples, server metrics, network RTT probes,
request metrics, timeslices, and sidecar availability.

Primary symbols:

- `TelemetryPlaneStack`;
- `GpuSampleTimeline`;
- `NetworkRttCalibration`;
- `ServerMetricScrapeRail`;
- `SidecarCoverageMatrix`.

#### `aiperf.export-plane`

Shows one canonical result projected into report JSON, records JSONL, CSV,
Parquet, console, timeslice, OTLP, MLflow, W&B, and accuracy artifacts.

Primary symbols:

- `ExporterFanout`;
- `ArtifactContractMap`;
- `StreamingBatchParity`;
- `FeatureGatedArtifact`;
- `CrossHostArtifactBoundary`.

#### `aiperf.accuracy-plane`

Shows evaluator loading, problem association, request execution, captured
response, grading, evaluator shutdown, and merged performance plus accuracy
results.

Primary symbols:

- `AccuracyLifecycle`;
- `ProblemResponseJoin`;
- `GradeTally`;
- `AccuracyPerformanceOverlay`;
- `ShutdownErrorReconciliation`.

### Mock-server laboratories

Evidence: `rust/mock-server/`.

These are reference-data and fixture capabilities, not production-server
emulation inside Flow.

#### `aiperf.latency-lab`

- `LatencyModelCurve`;
- `TtftItlTimeline`;
- `JitterEnvelope`;
- `AnalyticScheduleCompare`;
- `ErrorInjectionControl`.

#### `aiperf.multimodal-lab`

- `SyntheticImageEncoder`;
- `SeededDetectionLayout`;
- `AudioWaveformFixture`;
- `VideoFrameClock`;
- `ImageGenerationStack`.

#### `aiperf.protocol-lab`

- `OpenAiRequestResponsePair`;
- `StreamingFrameSequence`;
- `UsageAccountingCompare`;
- `PrefixCacheFixture`;
- `RecordedRequestInspector`.

## Flagship scenes

The catalog is accepted only when the following `.flow`-only scenes work
without document-specific runtime code.

### 1. Text to token IDs

Characters and words split into tokenizer spans, morph to token IDs, and remain
traceable backward. The scene includes Unicode, a many-to-one span, a
one-to-many span, a special token, and server-versus-local tokenization.

Required components:

- `TokenSpanMorph`;
- `TokenIdRail`;
- `TokenizerRoundTrip`;
- `ServerLocalTokenizerCompare`.

### 2. Multi-segment multimodal prompt

A system segment, reused prefix, user text, image, tool call, generated reply,
and truncated tail assemble into one endpoint body. Segment and token lengths
remain inspectable before and after chat-template injection.

Required components:

- `PromptSegmentComposer`;
- `MultimodalPartStrip`;
- `BodyPlanSplice`;
- `ChatTemplateInjection`;
- `MediaPublicationPath`.

### 3. Request and token flow

Token IDs move through endpoint preparation, scheduling, admission, transport,
stream parsing, model prefill/decode, KV cache, usage, and output decoding.
Arrival, admission, first token, terminal, and request record timelines remain
distinct.

Required components:

- `RequestLifecycleWaterfall`;
- `ContinuationPriorityQueue`;
- `ConnectionPoolAcquire`;
- `SseLineAssembler`;
- `PrefixCacheTrie`;
- `ObserverEventRail`.

### 4. Graph trajectory

A recorded trace becomes an LCP trie, splits at `t*`, primes warmup state, and
runs multiple profiling pressure lanes. First-token edges and terminal edges
must teach different dependency semantics.

Required components:

- `LcpTrieCompiler`;
- `TStarCut`;
- `PressureLaneMatrix`;
- `AsyncDataflowScene`;
- `GraphPhaseOrchestrator`.

### 5. Cellular execution

One authored workload partitions across cells and worker threads, executes, and
merges through retained, exact-fold, and sketch alternatives. The scene
identifies byte-exact, tolerance-based, aggregate-equivalent, and unsupported
claims.

Required components:

- `TwoLevelOrdinalTiling`;
- `CellularTopology`;
- `RetentionDecisionTree`;
- `AggregatorTierTree`;
- `ArtifactShippingPath`;
- `HeartbeatSketchTree`.

### 6. Metrics truth

One request's observer events become a record, derived metrics, distributions,
steady-state summary, sidecars, and exported artifacts. Missing usage and
client-versus-server token-count policy remain visible rather than normalized
away.

Required components:

- `MetricsPipeline`;
- `LatencyMetricDerivation`;
- `UsageVisibleTokenCompare`;
- `InflightSweepLine`;
- `ExporterFanout`.

## Delivery priority

### P0: expressiveness proof

Implement first:

- `core.glyph-run`, `core.span-map`, `core.segment-strip`, and
  `core.semantic-morph`;
- `viz.queue`, `viz.slot-pool`, `viz.event-lane`, and `viz.waterfall`;
- `TokenSpanMorph`, `PromptSegmentComposer`, `BodyPlanSplice`, and
  `RequestLifecycleWaterfall`;
- the first three flagship scenes.

This set directly proves the original bespoke-visualization goal.
The proof is incomplete unless all three scenes run under deterministic
narrated playback, pause for semantic exploration, resume from the same beat,
and preserve meaning across Canvas, semantic HTML, and SVG/HTML fallback.

### P1: architecture flagship

Implement next:

- `viz.execution-graph`, `viz.prefix-trie`, `viz.partition-grid`,
  `viz.reduction-tree`, `viz.metric-dag`, and `viz.sweep-line`;
- graph, trajectory, cellular, metrics, steady-state, and telemetry packages;
- flagship scenes four through six.

### P2: analysis and operational depth

Implement after the architecture flagship:

- adaptive control, RNG lineage, accuracy, complete export-plane, connection
  protocol details, timeout races, and mock-server laboratories;
- high-cardinality Canvas renderers and virtualization;
- advanced parity and comparison modes.

### P3: broad reference vocabulary

Add remaining endpoint dialect matrices, synthetic media laboratories,
bench-derived capacity surfaces, and specialized exporter views when reference
content demonstrates repeated use.

## Descriptor requirements

The current descriptor shape is a foundation. Domain capabilities additionally
need descriptor fields for:

- input semantic schema and emitted semantic schema;
- supported Flow IR node kinds and version range;
- layout-plan and geometry override contracts;
- interaction events and actions;
- timeline anchors and morph identity policy;
- deterministic display-list schema, paint bounds, hit regions, damage regions,
  and draw-order policy;
- semantic-twin projection, focus mapping, selection mapping, and transcript
  linkage;
- exploration-safe beats and authored-camera restoration policy;
- cardinality thresholds and degradation strategy;
- worker, Canvas, and optional accelerator eligibility;
- accessibility outline, semantic HTML twin, SVG/HTML simplification, textual
  fallback, and data-table fallback;
- reduced-motion, high-contrast, no-depth, and missing-capability behavior;
- deterministic seed inputs and serialization;
- base, per-entity, per-edge, per-frame, and memory cost;
- reference and degraded quality budgets, including decorative effects that may
  be reduced without semantic loss;
- reference fixtures and conformance assertions.

Descriptor expansion must remain schema-driven so compiler validation, runtime
binding, generated references, and authoring skills cannot drift.

## Verification matrix

Every catalog entry that becomes a capability or supported symbol needs:

- source and Flow IR schema tests;
- deterministic normal-versus-packed IR parity;
- stable content-hash tests;
- explicit geometry override tests;
- desktop and mobile visual snapshots;
- 3840×2160 reference-fidelity snapshots for typography, composition, lighting,
  captions, and effects;
- high-contrast, reduced-motion, no-depth, and missing-capability snapshots;
- semantic outline, reading-order, keyboard, and transcript assertions;
- Canvas hit-region and semantic-twin focus/selection parity assertions;
- simplified SVG/HTML fallback semantic parity;
- direct-seek versus continuous-playback equality;
- pause-to-explore and resume-from-the-same-beat assertions;
- textual or tabular fallback assertions;
- cardinality fixtures at representative low, medium, and high counts;
- frame-time, memory, and damage-region budget measurements under reference and
  degraded profiles;
- source-map diagnostics for invalid relationships and unsupported variants.

AIPerf domain scenes additionally need semantic assertions against deterministic
Rust or mock-server fixtures. Visual similarity alone does not prove a metric,
token, partition, timing, or merge claim.

## Architectural conclusion

The Rust implementation does not imply that Flow needs hundreds of hard-coded
AIPerf primitives. It implies a smaller reusable substrate centered on:

- semantic identity and morphs;
- spans and ordered segments;
- queues, resources, and barriers;
- event order and multiple time bases;
- execution graphs and prefix tries;
- partition and reduction topology;
- metric lineage, distributions, and parity.

Most named AIPerf components can be `.flow` symbols over that substrate. Trusted
capability packages are reserved for token-span layout, streaming byte
simulation, graph firing, trie operations, high-cardinality rendering, and
other genuinely algorithmic behavior. This boundary preserves `.flow` as the
only authored scene format while leaving the system expressive enough for the
bespoke technical visualizations found throughout AIPerf.
