/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Content ground truth for the cellular-architecture deck, ported verbatim from
//! `docs/canvases/cellular-architecture.canvas.tsx`. Every node/edge/story/ability label,
//! id, status, path, and proof string is preserved exactly as authored in the source canvas.

export type RecipeId = "t0" | "t1" | "t2" | "t3";
export type WorkloadKind = "scheduled" | "graph";
export type StorageMode = "retain" | "exact-fold" | "sketch";
export type StartMode = "synchronized" | "phaser" | "barrier-free";
export type TruthMode = "built-only" | "full";
export type Status = "built" | "partial" | "planned" | "rejected";
export type Lane = "control" | "data" | "execution" | "results";

export type AtlasNode = {
  id: string;
  lane: Lane;
  label: string;
  detail: string;
  status: Status;
  symbol: string;
  path: string;
  proof: string;
  x: number;
  y: number;
  w: number;
  h: number;
};

export type AtlasEdge = {
  id: string;
  from: string;
  to: string;
  lane: Lane;
  payload: string;
  status: Status;
};

export type RouteModel = {
  nodeIds: Set<string>;
  edgeIds: Set<string>;
  orderedEdges: string[];
  memory: string;
  percentiles: string;
  exactAggregates: string;
  artifacts: string;
  topology: string;
  warning?: string;
};

export type Ability = {
  dimension: string;
  built: string;
  boundary: string;
  status: "Built" | "Partial" | "Planned" | "Rejected" | "Approximation";
};

export type StoryChapter = "Launch" | "Distribute" | "Execute" | "Reduce" | "Scale";
export type ReductionMode = "retain" | "exact-fold" | "sketch";

export type StoryStep = {
  page: number;
  chapter: StoryChapter;
  title: string;
  thesis: string;
  addedNodeIds: readonly string[];
  addedEdgeIds: readonly string[];
  invariant: string;
  symbol: string;
  path: string;
  proof: string;
  change: string;
  simulation?: ReductionMode;
  fullAtlas?: true;
};

export const LANE_LABEL: Record<Lane, string> = {
  control: "CONTROL PLANE",
  data: "DATA PLANE",
  execution: "EXECUTION PLANE",
  results: "RESULTS PLANE",
};

export const NODES: AtlasNode[] = [
  { id: "config", lane: "control", label: "Authored run", detail: "Config v2 / --cells N", status: "built", symbol: "ProfileFlags → BenchmarkRun", path: "rust/cli/src/flags.rs", proof: "rust/cli/tests/online_v2_stdio.rs", x: 34, y: 94, w: 142, h: 58 },
  { id: "execute", lane: "control", label: "Unified aiperf", detail: "self-exec --execute", status: "built", symbol: "execute_mode::dispatch", path: "rust/cli/src/execute_mode.rs", proof: "rust/cli/tests/protocol_v2_stdio.rs", x: 218, y: 94, w: 154, h: 58 },
  { id: "controller", lane: "control", label: "Controller", detail: "validate · slice · launch", status: "built", symbol: "run_cellular", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", x: 418, y: 94, w: 160, h: 58 },
  { id: "start-barrier", lane: "control", label: "Synchronized START", detail: "all cells registered", status: "built", symbol: "all_registered → trigger", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "test_cellular_matches_single_cell + synchronized_start_releases_all_cells_together", x: 626, y: 72, w: 166, h: 50 },
  { id: "phaser", lane: "control", label: "Monotonic phaser", detail: "built opt-in START", status: "built", symbol: "Phaser", path: "rust/runtime/src/cellular/phaser.rs", proof: "test_cellular_phaser_start_matches_event_start", x: 626, y: 130, w: 166, h: 50 },
  { id: "barrier-free", lane: "control", label: "Barrier-free", detail: "k6-class start", status: "built", symbol: "AIPERF_CELL_BARRIER_FREE", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "test_cellular_barrier_free_matches_synchronized", x: 818, y: 94, w: 150, h: 58 },
  { id: "k8s-roles", lane: "control", label: "Native K8s roles", detail: "controller · cell; aggregator refusal", status: "partial", symbol: "run_controller / run_cell / run_aggregator", path: "rust/cli/src/cellular_role.rs", proof: "native role wiring + k8s.rs unit tests; no cluster e2e", x: 1012, y: 72, w: 166, h: 50 },
  { id: "shared-origin", lane: "control", label: "Shared timing origin", detail: "zero at START barrier", status: "built", symbol: "AIPERF_CELL_SHARED_ORIGIN", path: "rust/runtime/src/engine/cell_origin.rs", proof: "test_cellular_shared_origin_zeroes_at_the_barrier", x: 1202, y: 130, w: 170, h: 50 },
  { id: "dataset", lane: "data", label: "Dataset", detail: "synthetic · file · graph", status: "built", symbol: "DatasetInputAdapterResolver", path: "rust/runtime/src/engine/dataset_input.rs", proof: "rust/e2e-tests/tests/test_cellular_dataset_shipping.rs", x: 218, y: 240, w: 154, h: 58 },
  { id: "partition", lane: "data", label: "Ownership", detail: "i % cells == cell_id", status: "built", symbol: "ModuloCellPartition", path: "rust/runtime/src/cellular/partition.rs", proof: "cellular::partition unit tests", x: 418, y: 240, w: 160, h: 58 },
  { id: "dataset-fanout", lane: "data", label: "Dataset fan-out", detail: "opt-in verification overlay", status: "built", symbol: "DatasetIndex", path: "rust/runtime/src/cellular/dataset_session.rs", proof: "test_cellular_dataset_fanout_matches_baseline", x: 626, y: 218, w: 166, h: 50 },
  { id: "cell-index", lane: "data", label: "Cell-local index", detail: "owned request ids only", status: "built", symbol: "DatasetIndex", path: "rust/runtime/src/cellular/dataset_session.rs", proof: "dataset_session unit tests", x: 818, y: 240, w: 150, h: 58 },
  { id: "infinite", lane: "data", label: "Infinite scheduled", detail: "fails closed", status: "rejected", symbol: "validate_cellular_phase_budgets", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "cellular phase-budget unit tests", x: 1212, y: 240, w: 158, h: 58 },
  { id: "dataset-serve", lane: "data", label: "Dataset serving", detail: "Stage G · HTTP + zstd", status: "built", symbol: "build_dataset_serve_plan", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "test_cellular_dataset_shipping.rs + graph dataset shipping e2e", x: 1012, y: 218, w: 166, h: 50 },
  { id: "cell", lane: "execution", label: "Cell k / N", detail: "autonomous process or pod", status: "built", symbol: "fetch_cell_envelope", path: "rust/runtime/src/engine/cellular_cell.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", x: 626, y: 386, w: 166, h: 58 },
  { id: "shard", lane: "execution", label: "Worker shards", detail: "thread-per-core · !Send", status: "built", symbol: "run_sharded_scheduled", path: "rust/runtime/src/engine/sharded_scheduled.rs", proof: "thread_per_core_product.rs + worker_local_accumulation_parity.rs", x: 818, y: 386, w: 150, h: 58 },
  { id: "dispatch", lane: "execution", label: "Online dispatch", detail: "HTTP · gRPC hot path", status: "built", symbol: "RequestSink::dispatch", path: "rust/loadgen-core/src/sink.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", x: 1012, y: 386, w: 158, h: 58 },
  { id: "grpc-cellular", lane: "execution", label: "gRPC cellular", detail: "shared executor · partition ship", status: "built", symbol: "GrpcExecutionFactory", path: "rust/runtime/src/engine/grpc_turn_execution.rs", proof: "scheduled: test_grpc_cellular.rs; graph+gRPC cellular unproven", x: 1212, y: 354, w: 158, h: 50 },
  { id: "offline-cellular", lane: "execution", label: "DynoSim cellular", detail: "offline / online fail closed", status: "rejected", symbol: "validate_cellular_run_shape", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "dynosim_* uses an unwired SimClock executor", x: 1212, y: 414, w: 158, h: 50 },
  { id: "retained-records", lane: "results", label: "Retained rows", detail: "O(records) · exact", status: "built", symbol: "ShardRecords::Retained", path: "rust/runtime/src/engine/execute.rs", proof: "test_cellular_exact_fold_matches_retain", x: 436, y: 520, w: 166, h: 50 },
  { id: "exact-store", lane: "results", label: "Exact store", detail: "fold each completion", status: "built", symbol: "ShardRecords::Folded", path: "rust/runtime/src/engine/execute.rs", proof: "test_cellular_exact_fold_matches_retain", x: 436, y: 578, w: 166, h: 50 },
  { id: "tag-sketch", lane: "results", label: "TagSketch", detail: "t-digest + exact moments", status: "built", symbol: "MetricsStorageMode::Sketch", path: "rust/runtime/src/metrics_core/store.rs", proof: "test_cellular_sketch_matches_single_cell", x: 436, y: 636, w: 166, h: 50 },
  { id: "partition-wire", lane: "results", label: "Terminal partition", detail: "Partition | StorePartition", status: "built", symbol: "CellMessage", path: "rust/runtime/src/cellular/transport/mod.rs", proof: "cellular transport integration tests", x: 628, y: 578, w: 150, h: 58 },
  { id: "hierarchy-refusal", lane: "results", label: "Hierarchy refusal", detail: "fanout rejected pre-startup", status: "rejected", symbol: "run_aggregator", path: "rust/runtime/src/engine/cellular_aggregator.rs", proof: "test_cellular_hierarchy_is_refused", x: 822, y: 520, w: 158, h: 50 },
  { id: "controller-merge", lane: "results", label: "Controller merge", detail: "global · concat · append", status: "built", symbol: "merge_store_partitions / merge_records_*", path: "rust/runtime/src/cellular/shard.rs", proof: "cellular shard merge tests", x: 822, y: 590, w: 158, h: 58 },
  { id: "external-sink", lane: "results", label: "External sink", detail: "no central merge", status: "planned", symbol: "T3 stream-only", path: "specs/cellular.md", proof: "per-cell OTLP exists; mode is planned", x: 822, y: 658, w: 158, h: 50 },
  { id: "artifacts", lane: "results", label: "Record artifact lane", detail: "per-record files + concat", status: "built", symbol: "RecordArtifactLane", path: "rust/runtime/src/engine/record_lane.rs", proof: "test_cellular_emits_per_record_artifacts_matching_single_cell", x: 1022, y: 520, w: 158, h: 50 },
  { id: "artifact-shipping", lane: "results", label: "Cross-host shipping", detail: "Stage E · HTTP + zstd", status: "built", symbol: "ArtifactUploadServer", path: "rust/runtime/src/engine/artifact_shipping.rs", proof: "rust/e2e-tests/tests/test_cellular_http_shipping.rs", x: 1202, y: 520, w: 170, h: 50 },
  { id: "report", lane: "results", label: "One report", detail: "native-v2.json + exports", status: "built", symbol: "NativeReport", path: "rust/runtime/src/metrics_core/report.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", x: 1202, y: 590, w: 158, h: 58 },
];

export const EDGES: AtlasEdge[] = [
  { id: "config-execute", from: "config", to: "execute", lane: "control", payload: "protocol-v2 envelope", status: "built" },
  { id: "execute-controller", from: "execute", to: "controller", lane: "control", payload: "self-exec stdin", status: "built" },
  { id: "execute-k8s", from: "execute", to: "k8s-roles", lane: "control", payload: "native argv role", status: "partial" },
  { id: "controller-start", from: "controller", to: "start-barrier", lane: "control", payload: "registration + EventHandle", status: "built" },
  { id: "controller-phaser", from: "controller", to: "phaser", lane: "control", payload: "distributed START generation", status: "built" },
  { id: "controller-barrier-free", from: "controller", to: "barrier-free", lane: "control", payload: "immediate START", status: "built" },
  { id: "start-origin", from: "start-barrier", to: "shared-origin", lane: "control", payload: "barrier timing epoch", status: "built" },
  { id: "controller-dataset", from: "controller", to: "dataset", lane: "data", payload: "sliced envelope / dataset plan", status: "built" },
  { id: "dataset-partition", from: "dataset", to: "partition", lane: "data", payload: "stable instances", status: "built" },
  { id: "partition-cell", from: "partition", to: "cell", lane: "execution", payload: "owned positions k,k+N,…", status: "built" },
  { id: "partition-fanout", from: "partition", to: "dataset-fanout", lane: "data", payload: "request-id chunks", status: "built" },
  { id: "partition-serve", from: "partition", to: "dataset-serve", lane: "data", payload: "manifest + routable URLs", status: "built" },
  { id: "fanout-index", from: "dataset-fanout", to: "cell-index", lane: "data", payload: "replay + live tail", status: "built" },
  { id: "index-cell", from: "cell-index", to: "cell", lane: "execution", payload: "Indexed → InFlight", status: "built" },
  { id: "serve-cell", from: "dataset-serve", to: "cell", lane: "execution", payload: "HTTP fetch + decompress", status: "built" },
  { id: "origin-cell", from: "shared-origin", to: "cell", lane: "control", payload: "shared zero_ns", status: "built" },
  { id: "cell-shard", from: "cell", to: "shard", lane: "execution", payload: "two-level owned_positions", status: "built" },
  { id: "shard-dispatch", from: "shard", to: "dispatch", lane: "execution", payload: "RequestSink + Clock", status: "built" },
  { id: "dispatch-retain", from: "dispatch", to: "retained-records", lane: "results", payload: "CapturedRecord", status: "built" },
  { id: "dispatch-exact", from: "dispatch", to: "exact-store", lane: "results", payload: "RunCapture::fold_streaming", status: "built" },
  { id: "dispatch-sketch", from: "dispatch", to: "tag-sketch", lane: "results", payload: "finite metric values", status: "built" },
  { id: "retain-wire", from: "retained-records", to: "partition-wire", lane: "results", payload: "CellMessage::Partition", status: "built" },
  { id: "exact-wire", from: "exact-store", to: "partition-wire", lane: "results", payload: "StorePartition exact", status: "built" },
  { id: "sketch-wire", from: "tag-sketch", to: "partition-wire", lane: "results", payload: "StorePartition sketch", status: "built" },
  { id: "wire-merge", from: "partition-wire", to: "controller-merge", lane: "results", payload: "flat star fan-in", status: "built" },
  { id: "wire-external", from: "partition-wire", to: "external-sink", lane: "results", payload: "bounded aggregates", status: "planned" },
  { id: "merge-artifacts", from: "controller-merge", to: "artifacts", lane: "results", payload: "artifact completion barrier", status: "built" },
  { id: "artifacts-shipping", from: "artifacts", to: "artifact-shipping", lane: "results", payload: "Stage E upload", status: "built" },
  { id: "merge-report", from: "controller-merge", to: "report", lane: "results", payload: "ColumnStore → NativeReport", status: "built" },
  { id: "external-report", from: "external-sink", to: "report", lane: "results", payload: "no authoritative report", status: "planned" },
];

export const STORY_STEPS = [
  { page: 1, chapter: "Launch", title: "One run. Many cells. One report.", thesis: "Cellular scale preserves one benchmark identity while autonomous processes share the work.", addedNodeIds: ["config", "cell", "report"], addedEdgeIds: [], invariant: "Scaling changes placement, not the measurement contract.", symbol: "run_cellular", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", change: "Meet the authored run, an autonomous cell, and the authoritative report." },
  { page: 2, chapter: "Launch", title: "Author the Config v2 run", thesis: "The human-facing command resolves one strict benchmark request before any child exists.", addedNodeIds: ["config"], addedEdgeIds: [], invariant: "Every cell derives from the same resolved request.", symbol: "ProfileFlags", path: "rust/cli/src/flags.rs", proof: "rust/cli/tests/protocol_v2_stdio.rs", change: "Focus on the single authored source of truth." },
  { page: 3, chapter: "Launch", title: "Re-exec the unified binary", thesis: "The entry point launches the same aiperf image in hidden execution mode.", addedNodeIds: ["execute"], addedEdgeIds: ["config-execute"], invariant: "Process isolation survives without a second product executable.", symbol: "exec_bin::resolve", path: "rust/cli/src/exec_bin.rs", proof: "rust/cli/tests/protocol_v2_stdio.rs", change: "Add the protocol-v2 self-exec boundary." },
  { page: 4, chapter: "Launch", title: "Promote to controller at cells > 1", thesis: "Single-cell execution stays direct; multi-cell execution promotes one process to coordinator.", addedNodeIds: ["controller", "k8s-roles"], addedEdgeIds: ["execute-controller", "execute-k8s"], invariant: "The controller coordinates but does not dispatch inference load.", symbol: "run_cellular", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", change: "Introduce local controller promotion and the native Kubernetes role entry points." },
  { page: 5, chapter: "Launch", title: "Validate and fail closed", thesis: "Eligibility is checked before cells launch, preventing unsupported partial runs.", addedNodeIds: ["controller"], addedEdgeIds: [], invariant: "Unsupported transport, budget, storage, and workload combinations never degrade silently.", symbol: "validate_cellular_run_shape", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "cellular_controller.rs validation unit tests", change: "Expose the eligibility gate around the controller." },
  { page: 6, chapter: "Distribute", title: "Slice the global budget", thesis: "Request and conversation budgets are divided without overlap or omission.", addedNodeIds: ["dataset", "partition"], addedEdgeIds: ["controller-dataset", "dataset-partition"], invariant: "The union of cell-owned positions exactly tiles the global budget.", symbol: "owned_positions", path: "rust/runtime/src/engine/cell_launcher.rs", proof: "rust/e2e-tests/tests/test_cellular_multiturn.rs", change: "Add the deterministic workload source and modulo partition." },
  { page: 7, chapter: "Distribute", title: "Register, then release START", thesis: "Children become ready before the controller releases synchronized execution.", addedNodeIds: ["start-barrier"], addedEdgeIds: ["controller-start"], invariant: "No synchronized cell dispatches before the registration barrier opens.", symbol: "await_all_registered → start_event.trigger", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "test_cellular_matches_single_cell + velo synchronized-start tests", change: "Add the default lifecycle gate." },
  { page: 8, chapter: "Distribute", title: "Choose START policy and timing origin", thesis: "Phaser and barrier-free START are opt-in policies; shared origin can zero cell clocks at the synchronized barrier.", addedNodeIds: ["phaser", "barrier-free", "shared-origin"], addedEdgeIds: ["controller-phaser", "controller-barrier-free", "start-origin"], invariant: "Start policy and timing epoch change coordination, never ownership.", symbol: "PhaserServer / PhaserClient", path: "rust/runtime/src/cellular/transport/phaser_velo.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", change: "Reveal two alternate START paths and the e2e-proven shared timing origin." },
  { page: 9, chapter: "Distribute", title: "Distribute datasets by overlay or HTTP", thesis: "The opt-in Velo overlay verifies request fan-out, while Stage G serves cross-host file and graph inputs over HTTP+zstd.", addedNodeIds: ["dataset-fanout", "dataset-serve"], addedEdgeIds: ["partition-fanout", "partition-serve"], invariant: "Every cell receives only its owned input identities, independent of delivery mechanism.", symbol: "build_dataset_serve_plan", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "test_cellular_dataset_shipping.rs + test_cellular_dataset_fanout_matches_baseline", change: "Add both built data-delivery paths without replacing canonical cell-local execution." },
  { page: 10, chapter: "Distribute", title: "Index stable cell ownership", thesis: "Each cell tracks indexed, in-flight, and completed request identities.", addedNodeIds: ["cell-index"], addedEdgeIds: ["fanout-index"], invariant: "A request identity belongs to exactly one cell.", symbol: "DatasetIndex", path: "rust/runtime/src/cellular/dataset_session.rs", proof: "rust/e2e-tests/tests/test_cellular.rs::test_cellular_dataset_fanout_matches_baseline", change: "Add the cell-local ownership index." },
  { page: 11, chapter: "Execute", title: "Enter one autonomous cell", thesis: "A cell owns its runtime, transport state, metrics, and local lifecycle.", addedNodeIds: ["cell"], addedEdgeIds: ["partition-cell", "index-cell", "serve-cell", "origin-cell"], invariant: "Cells share no hot-path collector lock.", symbol: "fetch_cell_envelope", path: "rust/runtime/src/engine/cellular_cell.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", change: "Connect deterministic ownership to the autonomous process." },
  { page: 12, chapter: "Execute", title: "Partition again across worker shards", thesis: "Each cell applies the same ownership rule across thread-per-core workers.", addedNodeIds: ["shard"], addedEdgeIds: ["cell-shard"], invariant: "Nested ownership remains deterministic and disjoint.", symbol: "run_sharded_scheduled", path: "rust/runtime/src/engine/sharded_scheduled.rs", proof: "rust/cli/tests/thread_per_core_product.rs", change: "Add the second partition level inside the cell." },
  { page: 13, chapter: "Execute", title: "Stamp global ordinals", thesis: "Local progress maps back to one stable global request order.", addedNodeIds: ["shard"], addedEdgeIds: [], invariant: "ordinal = phase_base + local × cell_count + cell_id.", symbol: "global_ordinal", path: "rust/runtime/src/cellular/issuance.rs", proof: "rust/cli/tests/worker_local_accumulation_parity.rs", change: "Expose the identity formula carried by every worker." },
  { page: 14, chapter: "Execute", title: "Dispatch online; reject unwired simulation", thesis: "Every shard drives the same clock-injected HTTP or gRPC seam; DynoSim cellular fails closed.", addedNodeIds: ["dispatch", "grpc-cellular", "offline-cellular"], addedEdgeIds: ["shard-dispatch"], invariant: "Only the execution child dispatches inference requests.", symbol: "RequestSink", path: "rust/loadgen-core/src/sink.rs", proof: "scheduled HTTP + gRPC cellular e2e; graph+gRPC cellular unproven", change: "Add built HTTP and scheduled gRPC dispatch, then expose proof and DynoSim boundaries." },
  { page: 15, chapter: "Execute", title: "Observe and finalize each record", thesis: "Arrival, admission, token, usage, and terminal observations become one CapturedRecord.", addedNodeIds: ["retained-records"], addedEdgeIds: ["dispatch-retain"], invariant: "Storage mode begins only after the record is finalized.", symbol: "CapturedRecord", path: "rust/runtime/src/engine/records.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", change: "Cross from execution into the results plane." },
  { page: 16, chapter: "Reduce", title: "Retain rows for exact artifacts", thesis: "The retain path preserves per-record data and merges it in a deterministic topology order.", addedNodeIds: ["retained-records", "partition-wire"], addedEdgeIds: ["dispatch-retain", "retain-wire"], invariant: "Retain costs O(records) and keeps raw record artifacts available.", symbol: "CellMessage::Partition", path: "rust/runtime/src/cellular/transport/mod.rs", proof: "rust/e2e-tests/tests/test_cellular.rs::test_cellular_exact_fold_matches_retain", change: "Carry retained records over the cell wire.", simulation: "retain" },
  { page: 17, chapter: "Reduce", title: "Exact-fold into ColumnStore", thesis: "Each completed record folds into an exact mergeable store and the row is dropped.", addedNodeIds: ["exact-store"], addedEdgeIds: ["dispatch-exact", "exact-wire"], invariant: "Exact-fold retains exact record-derived aggregates without retaining rows.", symbol: "RunCapture::fold_streaming", path: "rust/runtime/src/engine/execute.rs", proof: "rust/e2e-tests/tests/test_cellular.rs::test_cellular_exact_fold_matches_retain", change: "Add the exact bounded-retention alternative.", simulation: "exact-fold" },
  { page: 18, chapter: "Reduce", title: "Sketch into t-digest plus exact moments", thesis: "Finite metric values stream into bounded sketches while counts and moments stay exact.", addedNodeIds: ["tag-sketch"], addedEdgeIds: ["dispatch-sketch", "sketch-wire"], invariant: "Percentiles are approximate; counts, sums, extrema, mean, std, and rates remain exact.", symbol: "TagSketch", path: "rust/runtime/src/metrics_core/store.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", change: "Add the k6-style bounded-memory reduction path.", simulation: "sketch" },
  { page: 19, chapter: "Scale", title: "Merge, publish, and understand the boundary", thesis: "Flat controller merge is built; hierarchical aggregation is refused before startup and external stream-only aggregation remains planned.", addedNodeIds: ["hierarchy-refusal", "controller-merge", "external-sink", "artifacts", "artifact-shipping", "report", "infinite"], addedEdgeIds: ["wire-merge", "wire-external", "merge-artifacts", "artifacts-shipping", "merge-report", "external-report"], invariant: "One authoritative report exists unless a future external sink explicitly replaces it.", symbol: "merge_store_partitions", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "rust/e2e-tests/tests/test_cellular.rs::test_cellular_hierarchy_is_refused", change: "Complete flat merge, record-lane, cross-host shipping, report, hierarchy refusal, and roadmap boundaries." },
  { page: 20, chapter: "Scale", title: "Full cellular system atlas", thesis: "Every plane, recipe, inspector, body plan, and ability boundary in one view.", addedNodeIds: [], addedEdgeIds: [], invariant: "The complete atlas preserves the measurement contract.", symbol: "FullAtlasPage", path: "rust/runtime/src/engine/cellular_controller.rs", proof: "rust/e2e-tests/tests/test_cellular.rs", change: "All previously introduced layers are visible together.", fullAtlas: true },
] as const satisfies readonly StoryStep[];

export function storyVisibility(page: number): { nodeIds: Set<string>; edgeIds: Set<string> } {
  const steps = STORY_STEPS.slice(0, Math.min(page, 19));
  return {
    nodeIds: new Set<string>(steps.flatMap((step) => [...step.addedNodeIds])),
    edgeIds: new Set<string>(steps.flatMap((step) => [...step.addedEdgeIds])),
  };
}

export const ABILITIES: Ability[] = [
  { dimension: "Online transport", built: "HTTP scheduled + graph; gRPC scheduled", boundary: "The shared cellular seam is transport-neutral, but graph+gRPC cellular has no product e2e proof", status: "Partial" },
  { dimension: "DynoSim transport", built: "Single-process offline and online simulation", boundary: "Cellular issuance and partition shipping are not wired into the SimClock executor", status: "Rejected" },
  { dimension: "Work unit", built: "Request · conversation · whole graph trace", boundary: "Multi-turn requires eligible exact-fold sampler", status: "Built" },
  { dimension: "Scheduled fixed budget", built: "Requests or exact-fold sessions ≥ cell count", boundary: "Requests and sessions tile deterministically across cells", status: "Built" },
  { dimension: "Scheduled dynamic budget", built: "None", boundary: "Duration, adaptive scale, and unbounded execution fail closed", status: "Rejected" },
  { dimension: "Graph sessions", built: "Whole traces partition by session", boundary: "Static requests budget fails closed", status: "Built" },
  { dimension: "Graph duration", built: "Partitioned source + duration stop are wired", boundary: "Unit-tested runtime path; no cellular duration e2e proof", status: "Partial" },
  { dimension: "Graph adaptive", built: "No cross-cell scaling consensus", boundary: "Validator does not reject it, but cellular behavior has no implementation proof", status: "Planned" },
  { dimension: "Retain", built: "Raw RecordsShardPartition", boundary: "O(records); graph concatenates instead of global-order merge", status: "Built" },
  { dimension: "Exact-fold", built: "Exact ColumnStorePartition", boundary: "Exact distributions; floating sums may differ by ULP", status: "Built" },
  { dimension: "Sketch", built: "t-digest + exact moments", boundary: "Approximate percentiles; no per-record artifacts", status: "Approximation" },
  { dimension: "Heartbeat lane", built: "Cells ship sketches; controller emits a live cross-cell aggregate", boundary: "Enabling the lane forces retain", status: "Built" },
  { dimension: "Python live stream", built: "Cell-local consumer forces retain", boundary: "No merged cross-cell per-record live stream", status: "Partial" },
  { dimension: "Artifacts", built: "Same-host concat + HTTP/zstd shipping", boundary: "Sketch has no per-record files; cross-host k8s needs routable shipping or shared storage", status: "Built" },
  { dimension: "Hierarchy", built: "Unavailable", boundary: "Hierarchy requests are refused before controller startup", status: "Rejected" },
  { dimension: "K8s hierarchy", built: "Unavailable", boundary: "Operator DNS cannot enable unavailable hierarchy", status: "Rejected" },
  { dimension: "k6-class start", built: "Barrier-free START", boundary: "Looser cross-cell start correlation", status: "Approximation" },
  { dimension: "Phaser START", built: "Monotonic generation broadcast with replay-on-attach over Velo", boundary: "Opt-in; broader phase orchestration is separate", status: "Built" },
  { dimension: "Shared timing origin", built: "Cells zero timestamps at the synchronized START barrier", boundary: "Opt-in through AIPERF_CELL_SHARED_ORIGIN", status: "Built" },
  { dimension: "External sink", built: "Cell-local OTLP sink exists in scratch execution", boundary: "No authoritative no-central-merge stream-only mode", status: "Planned" },
  { dimension: "Dataset fan-out", built: "Velo replay/live index + controlled dispatch", boundary: "Built and e2e-proven as an opt-in verification overlay; canonical execution still runs separately", status: "Built" },
  { dimension: "Cross-host dataset", built: "Stage G HTTP+zstd serving for file and graph inputs", boundary: "Requires routable controller address across hosts", status: "Built" },
  { dimension: "Native Kubernetes roles", built: "aiperf controller / cell entry points", boundary: "The aggregator entry point refuses hierarchy; no full cluster e2e", status: "Partial" },
  { dimension: "Side telemetry", built: "Collectors run on cell 0", boundary: "GPU/server/network sidecars are omitted from the merged report", status: "Partial" },
  { dimension: "Build image", built: "Cellular works with velo/full features", boundary: "Lean default CLI build rejects cells > 1", status: "Partial" },
];

export const NODE_BY_ID = new Map(NODES.map((node) => [node.id, node]));
export const EDGE_BY_ID = new Map(EDGES.map((edge) => [edge.id, edge]));

export function statusLabel(status: Status): string {
  return status === "built"
    ? "BUILT"
    : status === "partial"
      ? "PARTIAL"
      : status === "rejected"
        ? "REJECTED"
        : "PLANNED";
}

export function statusTone(status: Status): "success" | "warning" | "neutral" {
  return status === "built" ? "success" : status === "partial" ? "warning" : "neutral";
}

export function deriveRoute(
  recipe: RecipeId,
  workload: WorkloadKind,
  storage: StorageMode,
  start: StartMode,
): RouteModel {
  if (recipe === "t2") {
    return {
      nodeIds: new Set(["config", "execute"]),
      edgeIds: new Set(["config-execute"]),
      orderedEdges: ["config-execute"],
      memory: "No execution",
      percentiles: "No execution",
      exactAggregates: "No execution",
      artifacts: "No execution",
      topology: "Hierarchy request → refusal",
      warning: "Hierarchical aggregation is unavailable and refused before controller startup.",
    };
  }

  const nodeIds = new Set(["config", "execute", "controller", "dataset", "partition", "cell", "shard", "dispatch"]);
  const edgeIds = new Set(["config-execute", "execute-controller", "controller-dataset", "dataset-partition", "partition-cell", "cell-shard", "shard-dispatch"]);
  const orderedEdges = ["config-execute", "execute-controller"];

  const startNode =
    start === "synchronized" ? "start-barrier" : start === "phaser" ? "phaser" : "barrier-free";
  const startEdge =
    start === "synchronized" ? "controller-start" : start === "phaser" ? "controller-phaser" : "controller-barrier-free";
  nodeIds.add(startNode);
  edgeIds.add(startEdge);
  orderedEdges.push(startEdge, "controller-dataset", "dataset-partition", "partition-cell", "cell-shard", "shard-dispatch");

  let resultNode = "retained-records";
  let resultEdge = "dispatch-retain";
  let wireEdge = "retain-wire";
  if (storage === "exact-fold") {
    resultNode = "exact-store";
    resultEdge = "dispatch-exact";
    wireEdge = "exact-wire";
  } else if (storage === "sketch") {
    resultNode = "tag-sketch";
    resultEdge = "dispatch-sketch";
    wireEdge = "sketch-wire";
  }
  nodeIds.add(resultNode);
  nodeIds.add("partition-wire");
  edgeIds.add(resultEdge);
  edgeIds.add(wireEdge);
  orderedEdges.push(resultEdge, wireEdge);

  let warning: string | undefined;
  if (recipe === "t3") {
    nodeIds.add("external-sink");
    edgeIds.add("wire-external");
    orderedEdges.push("wire-external");
    warning = "T3 no-central-merge external streaming remains planned; choose the built START policy independently.";
  } else {
    nodeIds.add("controller-merge");
    edgeIds.add("wire-merge");
    orderedEdges.push("wire-merge");
  }

  if (recipe !== "t3") {
    nodeIds.add("artifacts");
    nodeIds.add("report");
    edgeIds.add("merge-artifacts");
    edgeIds.add("merge-report");
    orderedEdges.push("merge-report");
  }

  if (start === "phaser") warning = "Phaser START is built and e2e-proven; select it explicitly with AIPERF_CELL_PHASER_START.";
  if (workload === "graph" && storage === "retain") {
    warning = "Graph retain concatenates by cell and renumbers densely; it is deterministic per topology, not byte-identical.";
  }

  return {
    nodeIds,
    edgeIds,
    orderedEdges,
    memory:
      storage === "retain"
        ? "O(records)"
        : storage === "sketch"
          ? "O(shards × sketch + concurrency)"
          : "O(shards × exact store + concurrency)",
    percentiles: storage === "sketch" ? "Approximate · t-digest" : "Exact",
    exactAggregates:
      storage === "sketch"
        ? "Counts · sums · extrema · mean · std · rates"
        : storage === "exact-fold"
          ? "Exact order stats · float moments within tolerance"
          : "All record-derived metrics",
    artifacts: storage === "sketch" ? "Per-record artifacts unavailable" : "Lane, concat, or HTTP+zstd",
    topology:
      recipe === "t3"
        ? "Cells → external ingest (planned)"
        : "Cells → controller",
    warning,
  };
}

export const REDUCTION_COPY: Record<ReductionMode, readonly [string, string, string, string]> = {
  retain: ["CapturedRecord × N", "retain rows", "Partition", "O(records) · exact percentiles"],
  "exact-fold": ["CapturedRecord × N", "RunCapture::fold_streaming", "StorePartition · exact", "bounded rows · exact percentiles"],
  sketch: ["finite metric values", "t-digest + moments", "StorePartition · sketch", "bounded · approximate percentiles"],
};
