<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `docs/specs/` — AIPerf design records

This folder is the design record for AIPerf. Each spec describes one subsystem or
seam: what it is, the contract it holds, and where the code lives. A spec states
current built behavior; a `## Future requirements` section, where present, states
explicitly planned but unbuilt work. The code in `rust/` is authoritative — when a
spec and the code disagree, fix the spec.

Every spec follows the same shape: `## Purpose`, `## Built`, an optional
`## Future requirements`, and `## Source anchors` that point at the files that
realize it.

Start with [architecture.md](architecture.md) for the whole-system picture, then
read the record for the subsystem you are touching.

## Index

### Whole system

| Spec | Purpose |
|---|---|
| [architecture.md](architecture.md) | Process model, crate topology, and the three orthogonal seams (time, transport, workload) every subsystem composes over. |
| [repository-layout.md](repository-layout.md) | Cargo workspace topology, package identity (`aiperf-e2e-tests` at `rust/e2e-tests`), and naming rules for new packages, enforced by `tools/check_crate_layout.py`. |
| [wheel-packaging.md](wheel-packaging.md) | The one installable artifact: the three-step `bundle-cli`→hatchling→`wheel_repack.py` build, the `.data/scripts/` binary interface with no launcher shim, the `py3-none-<platform>` tag derived from the injected binary's own glibc floor, and the packaged payload census hatchling's `packages = ["src/aiperf"]` sweep produces. |
| [extension-registry.md](extension-registry.md) | Static link-time extensibility: the `AIPerfRegistry`/`AIPerfExtension` composition seam, its capability categories, and the frozen bootstrap object graph. |
| [runner-protocol.md](runner-protocol.md) | The Config-v2 front end ↔ execution boundary: the protocol-v2 stdio envelope, the `BenchmarkRun` vocabulary, dataset/transport path selection, and the in-process linked-capability catalog. |
| [config-model-unification.md](config-model-unification.md) | Planned convergence of the Config-v2 front end onto one typed `BenchmarkConfig`/`BenchmarkRun` shared by CLI producer and runtime consumer (mirroring the Python `AIPerfConfig`→`build_benchmark_plan`→`BenchmarkRun` pipeline), retiring the untyped projection seam and the per-workload `*WorkloadConfigV2` DTOs. |
| [typed-factory-runner.md](typed-factory-runner.md) | Planned completion of config-model-unification: the runtime consumes the typed `BenchmarkRun` directly and selects a component by an open, normalized string id (`RegistryId`) whose config is a typed struct for built-ins and an opaque `RawValue` (plugin-decoded) only for the runtime-plugin tail — deleting the `AuthoredRunSpecV2` projection while keeping the discriminant a string (as Python's `ExtensibleStrEnum` is) and reserving enums for closed knobs (`DispatchMode`/`HopRouting`). |
| [2026-07-24-rust-port-flow-deck-design.md](2026-07-24-rust-port-flow-deck-design.md) | Design for `apps/aiperf-flow`'s `rust-port-flow` deck: an interactive, zoomable, playable request-lifecycle explainer of the whole Rust port (nine stages, semantic-zoom drill-down, animated request particle, clock/transport seam toggles) plus the shared `src/interactive/` primitives it introduces. |
| [2026-07-24-rust-port-flow-v2-swimlane-timeline.md](2026-07-24-rust-port-flow-v2-swimlane-timeline.md) | Swimlane-timeline redesign of the `rust-port-flow` deck: the request as one continuous line riding a time axis (the Clock seam) through subsystem lanes grouped in nested seam frames; supersedes the node-graph rendering while keeping the nine-stage content and the `ZoomStage`/`useFlowPlayer`/`SeamToggle` primitives. |
| [2026-07-23-aiperf-flow-elk-layout-engine.md](2026-07-23-aiperf-flow-elk-layout-engine.md) | App-wide ELK (`elkjs`) auto-layout seam for `apps/aiperf-flow`: replaces every diagram's hand-picked `{x,y}` node positions with a measure→layout→apply cycle driven by real React Flow node sizes, fixing overlapping ("smooshed") boxes and doubled-back edges; opt-in per diagram via `PipelineCanvas`'s `layout` prop / `AutoLayoutFlow`, adopted fully by `rust-port-flow` first then rolled out to the rest. |
| [2026-08-01-aiperf-flow-slide-clock.md](2026-08-01-aiperf-flow-slide-clock.md) | Planned continuous virtual-time seam for `apps/aiperf-flow`: a narration-driven `src/clock/` store with word-anchored cue points, delivered to charts through `useSyncExternalStore` rather than `node.data` (which would recreate React Flow's node array every frame), retrofitting `sweep`/`timeline`/`slices` with playheads and self-drawing curves; mirrors the runtime's own `Clock`/`SimClock` seam and makes reduced motion fall out as "the clock at end of span". |
| [2026-07-25-hf-generic-datasets-design.md](2026-07-25-hf-generic-datasets-design.md) | Built generic HuggingFace-dataset-by-ID input: a `--hf-dataset <id>` CLI/YAML passthrough that bypasses the hard-coded public catalog, plus a source-agnostic `hf` composer whose `infer_row_layout` auto-detects prompt/completion fields (message-array / flat / `context`+`input`) with explicit overrides and `/info` split-config resolution — the arbitrary-ID + auto-detect UX vLLM's Rust bench has, on top of AIPerf's existing (superior) HF transport. |

### Execution and scheduling

| Spec | Purpose |
|---|---|
| [execution-model.md](execution-model.md) | The single thread-per-core hot path, the two-trait transport seam, worker-local accumulation, and the shared reduce/measure layers. |
| [flatgraph-fast-path.md](flatgraph-fast-path.md) | Built `FlatGraphActor` fast path: eligible local and worker-backed production graph placement routes one-node/no-fan-in traces through the shared sink without the general graph context, proven byte-identical to the general executor through the real `aiperf` binary; later scheduled-workload and multi-node work remain future. |
| [scheduling.md](scheduling.md) | The scheduled workload shapes (request-rate, concurrency, user-centric, fixed-schedule) over one `Clock`-backed runtime, and how each partitions across sub-cells. |
| [global-exact-dispatch.md](global-exact-dispatch.md) | The built `runtime.dispatch` selector (`sharded` \| `global` \| `global-hop` \| `global-push`): `sharded` is a static per-thread partition, `global` (the `workers>1` default) admits from a shared per-cell global admission gate, and `global-hop`/`global-push` are single-coordinator dispatchers, so `workers>1` runs are byte-exact against Python's single global concurrency/rate limiter; also records which modes deliver exact `admit_ns` issuance ordering (measured) and the `global-push` vs `global-hop` throughput comparison. |
| [phase-orchestration.md](phase-orchestration.md) | One `Clock`-native lifecycle for warmup→profiling phases: the escalation ladder, cancellation latch, and the shared seam scheduled and graph runs both use. |
| [ancillary-timing.md](ancillary-timing.md) | The three knobs that ride on a running phase: ramping, seeded request cancellation, and sticky round-robin URL selection. |
| [adaptive-scale.md](adaptive-scale.md) | The closed-loop SLA controller (`ramp_until_fail`) layered over a running load phase, its actuators, and its schema-v2 artifacts. |
| [cellular.md](cellular.md) | Partitioning one run across cell processes and merging records or folded metric stores, the multi-process and velo cross-host topologies, and the fidelity guards. |
| [velo-hub.md](velo-hub.md) | The per-experiment control-plane hub: a plugin trait whose plugins each contribute an HTTP router and velo handlers backed by one shared handler function, the discovery connect-by-endpoint anchor, the cell↔controller, `/artifact`, `/dataset`, and `/phaser` plugins that fold every standalone velo plane onto the hub, and the `AIPERF_CELLULAR_HUB` bootstrap toggle. |
| [slurm-native.md](slurm-native.md) | Running a cellular benchmark natively under a SLURM allocation: the `aiperf slurm run` rank dispatch, the `SLURM_*` → controller/cell topology mapping and nodelist expansion, the `SlurmLauncher`, and the `aiperf slurm generate` sbatch script generator. |

### Transports

| Spec | Purpose |
|---|---|
| [http-transport.md](http-transport.md) | The Clock-injected hyper HTTP stack: wire/protocol support, SSE streaming, endpoint binding, and post-send cancellation. |
| [grpc-transport.md](grpc-transport.md) | The Clock-injected Tonic gRPC stack: the binding registry, the KServe OIP v2 and Riva families, the protoc-free codec, and the worker-local sink. |
| [python-native-transport.md](python-native-transport.md) | Planned abi3 PyO3 transport for the Python product, so sub-millisecond ITL is timestamped at the socket rather than behind an event-loop hop: a vendored copy of the wire path (no build edge to `aiperf-runtime`) whose census shows foreign coupling concentrated entirely in the worker-sink binding — `core/dispatch.rs` plus `http/sink*`, ~2,450 lines the PyO3 layer replaces — leaving ~20,500 lines including the `Clock` seam, `endpoints/`, and the hyper client to vendor verbatim and stay diffable against this tree; plus the widened `RequestRecord` contract (reduced outcome + pre-serialized blob) that keeps `AioHttpTransport` an equal conforming implementation of the existing `PluginType.TRANSPORT` seam, and the `cp311-abi3-<platform>` retag it forces. |
| [websocket-transport.md](websocket-transport.md) | Native Clock-injected WebSocket transport: Responses WS and separate Realtime, pre-dispatch message lowering, bounded worker-local full-duplex driver/control progress, pooled connection admission and rotation, fallback/TLS/proxy policy, and application-event lag metrics. |
| [websocket-mock-server.md](websocket-mock-server.md) | Planned deterministic WS/WSS mock target for serialized turns and duplex Realtime, including analytic application-event timing, fragmentation/control traffic, replay-boundary failures, bounded sanitized captures, and product-level metric verification. |
| [sagemaker-runtime-endpoint.md](sagemaker-runtime-endpoint.md) | AWS SageMaker Runtime dialect: `InvokeEndpoint`/`InvokeEndpointWithResponseStream` mock-server routes, the AWS `application/vnd.amazon.eventstream` binary frame codec, and the single-factory client endpoint dialect (selected via `--streaming`) it composes over, verified e2e and against a genuine `boto3` client. |
| [offline-cosimulation.md](offline-cosimulation.md) | Socket-free Dynamo co-simulation behind the `dynosim` feature: the steppable clocked engine boundary and the observer contract feeding AIPerf's own measurement. |
| [dry-run-virtual-workers.md](dry-run-virtual-workers.md) | Planned opt-in virtual worker placement for the analytic `dry_run` transport: deterministic multi-worker timing, session affinity, and routing assertions over one `SimClock`, without recreating Dynosim. |
| [dry-run-virtual-workers-sharded.md](dry-run-virtual-workers-sharded.md) | Planned virtual-clock `sharded` scheduled runner: production-equivalent workload/admission partitioning across logical workers on one `LocalSet` and `SimClock`. |
| [dry-run-virtual-workers-graph.md](dry-run-virtual-workers-graph.md) | Planned virtual-worker graph placement: deterministic per-node assignment, graph-causality preservation, cancellation cleanup, and per-node record attribution. |

### Inputs, endpoints, and graph

| Spec | Purpose |
|---|---|
| [content-to-wire.md](content-to-wire.md) | The end-to-end request dataplane: nine stages from dataset fetch to outbound bytes, the two freeze boundaries, the serialization budget (where each byte is produced, once), per-dispatch cost accounting, and the invariants that span stages. Start here to find *where* something happens. |
| [dataset.md](dataset.md) | The input-resolution plane: the content-addressed segment store, its pool/freeze/thaw write side, opaque raw-payload handling, and the loader→compose→store→sampler→materializer pipeline. |
| [prompt-corpus.md](prompt-corpus.md) | The shared `prompts.corpus` seam for synthetic, count/hash-based trace, and recorded-graph prompt synthesis over `sonnet`, `coding`, and exact-length `random`. |
| [endpoint-body-construction.md](endpoint-body-construction.md) | How an endpoint declares its request shape (`format_payload → BodyPlan`) and how that shape becomes wire bytes: content lowered once at load and carried as inline pre-serialized wires, so live assistant replies and static content splice through one path. |
| [endpoints.md](endpoints.md) | The `Endpoint` dialect adapter: the trait, every native dialect, endpoint identity, and the registry consumed by validation and execution. |
| [content-server.md](content-server.md) | The run-owned HTTP delivery sidecar that serves generated media by URL, its publication seam, and request-correlated media-fetch metrics (`rid`/`mi`/`td` URL tagging, streaming drain into `SidecarMetric` distributions). |
| [rng.md](rng.md) | The hash-derived randomness substrate: order-independent BLAKE3 stream derivation, generators, and sampling distributions. |
| [graph-runtime.md](graph-runtime.md) | The Graph-IR runtime: deterministic async dataflow, the `dag_jsonl`/`weka_trace`/`dynamo_trace` compilers, and the trajectory-snapshot/warmup-priming subsystem. |
| [recorded-agent-replay-rust-port.md](recorded-agent-replay-rust-port.md) | Native recorded-agent replay: canonical manifest-ordered trace programs, strict request lowering, heterogeneous LLM/tool Graph-IR, trace-local local/Docker environments and sandboxes, controller-folded metrics/artifacts, safe resume, cellular preflight, and pinned-reference parity coverage. |
| [agentic-eval-platform.md](agentic-eval-platform.md) | Planned agentic SWE evaluation substrate: immutable task/dataset/trial identities, sandbox and verifier contracts, semantic-graph lowering, task-health governance, paired graph-variant experiments, and the separation of replay fidelity, system performance, and task quality. |
| [dag-v3-graph-ir-extraction.md](dag-v3-graph-ir-extraction.md) | Decision record for extracting durable graph semantics and behavioral tests from the historical DAG-v3 branch while rejecting its universal Python node executor. |
| [semantic-agent-graph.md](semantic-agent-graph.md) | Native Rust semantic graph, lowering, transform-closure, and source-fidelity contract for captured and live agent/application workflows. |
| [native-harbor-agentic-benchmarking.md](native-harbor-agentic-benchmarking.md) | Native-Rust Harbor episode benchmark architecture: built bounded NativeGraph live rollouts plus the terminal-only `externally_driven` single-task compatibility path, whose exact `terminal_v1` capability preserves verifier score authority while emitting only Missing-fidelity digest evidence; capture proxies and cross-host episode transport remain planned. |
| [harbor-replacement-platform.md](harbor-replacement-platform.md) | Pure native-Rust Harbor replacement bar: owned source snapshots, v2 executable package identity, compatibility import tiers, task/agent/verifier contracts, the built ordered and sealed-rollout benchmark subsets, and graph-aware differentiators. |
| [harbor-native-rust-implementation.md](harbor-native-rust-implementation.md) | Native-Rust Harbor delivery architecture: canonical source capture, normalized-plan plus executable-source identity, snapshot-only local/Docker execution, directional verifier workdir safety, and single-/multi-step evidence. |
| [benchmark-compose-environments.md](benchmark-compose-environments.md) | Built strict Docker Compose benchmark sidecars: generated-main authority, validated public-only overlays, final-step service evidence, frozen separate-verifier transfer, and labelled cleanup. |
| [graph-ir-system-idle-gap.md](graph-ir-system-idle-gap.md) | Graph-IR system idle-gap cap at the centralized node firing gate. |
| [conditional-graph-lowering.md](conditional-graph-lowering.md) | The model-independent-branching contract: how pinned/recorded/weighted conditional branching and recorded non-LLM content resolve and fold into the flat `LlmNode`/`StaticEdge` substrate at lowering, the eager-vs-forbidden (branch-on-live-output) line, and the built `conditional_graph` eager-conditional compiler. |
| [otlp-genai-graph-input.md](otlp-genai-graph-input.md) | Native OTLP/HTTP JSON OpenInference and GenAI input: strict source decoding, trace topology reconstruction, span classification, and replay-span folding into flat Graph-IR state and timing. |

### Measurement and output

| Spec | Purpose |
|---|---|
| [metrics.md](metrics.md) | The IO-free metrics engine: the column-store accumulator, the metric catalog, sweep curves, and the typed report; exact vs sketch modes. |
| [eval-node-metrics-artifact.md](eval-node-metrics-artifact.md) | Built host-owned `aiperf eval --records-output` JSONL sidecar: streams one canonical record per completed NativeGraph model node while keeping reward JSON and package trust boundaries unchanged. |
| [definition-registry.md](definition-registry.md) | Built single shared definition layer: a lookup-only `Definition` (header, units, `larger_is_better`, `value_type`, group, order) for metrics and dataset-analysis outputs (server/GPU telemetry still seam-only) — split out of `MetricSpec`, keyed by namespaced id, made compile-time complete via the const-array `[MetricSpec; COUNT]` length, and used for SLA comparison, table rendering, and the `aiperf metrics` command. |
| [telemetry.md](telemetry.md) | Side-channel measurement: GPU telemetry, server metrics, and network latency, feeding values into the metrics seam. |
| [exporters.md](exporters.md) | The native output plane: the typed report core and the static set of `Exporter` sinks behind one trait. |
| [dataset-analysis.md](dataset-analysis.md) | Built `--dry-run` analytical report: dataset shape, turn-by-turn ISL/OSL, prefix/KV-cache reuse (ideal and finite-capacity), and the real execution timeline (concurrency, throughput, backlog) distilled from a dry run's records, emitted as `dataset_analysis.{txt,json,csv,html}`. |
| [accuracy.md](accuracy.md) | The Rust dispatch/capture vs pinned-Python grading split, the injected evaluator seam, and sharded capture with a single grade. |

### Targets

| Spec | Purpose |
|---|---|
| [mock-server.md](mock-server.md) | `aiperf-mock-server`: the standalone HTTP/gRPC inference target with deterministic generation, latency and error models, telemetry, and request recording. |
