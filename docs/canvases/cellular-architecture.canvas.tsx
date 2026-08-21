import {
  Button,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Select,
  Spacer,
  Stack,
  Text,
  Toggle,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type RecipeId = "t0" | "t1" | "t2" | "t3";
type WorkloadKind = "scheduled" | "graph";
type StorageMode = "retain" | "exact-fold" | "sketch";
type StartMode = "synchronized" | "phaser" | "barrier-free";
type TruthMode = "built-only" | "full";
type Status = "built" | "partial" | "planned" | "rejected";
type Lane = "control" | "data" | "execution" | "results";

type AtlasNode = {
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

type AtlasEdge = {
  id: string;
  from: string;
  to: string;
  lane: Lane;
  payload: string;
  status: Status;
  d: string;
};

type RouteModel = {
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

type Ability = {
  dimension: string;
  built: string;
  boundary: string;
  status: "Built" | "Partial" | "Planned" | "Rejected" | "Approximation";
};

type StoryChapter = "Launch" | "Distribute" | "Execute" | "Reduce" | "Scale";
type ReductionMode = "retain" | "exact-fold" | "sketch";

type StoryStep = {
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

const CSS = `
@keyframes cellular-flow { to { stroke-dashoffset: -24; } }
.cellular-active-edge { animation: cellular-flow 1.8s linear infinite; }
.cellular-node { cursor: pointer; outline: none; }
.cellular-node:focus rect:first-of-type { stroke-width: 3; }
.cellular-recipe { grid-template-columns: minmax(280px,1.35fr) repeat(3,minmax(130px,.65fr)) auto; }
.cellular-readouts { grid-template-columns: repeat(5,minmax(0,1fr)); }
.cellular-lower { grid-template-columns: minmax(260px,.8fr) minmax(520px,1.6fr); }
.cellular-ability-scroll, .cellular-cell-scroll, .cellular-worker-scroll { overflow-x: auto; }
.cellular-ability-grid { min-width: 760px; }
.cellular-cell-grid { min-width: 700px; }
.cellular-worker-grid { min-width: 560px; }
@media (max-width: 1180px) {
  .cellular-recipe { grid-template-columns: repeat(2,minmax(0,1fr)); }
  .cellular-recipe-presets { grid-column: 1 / -1; }
  .cellular-readouts { grid-template-columns: repeat(2,minmax(0,1fr)); }
}
@media (max-width: 900px) {
  .cellular-lower { grid-template-columns: 1fr; }
  .cellular-atlas-scroll { overflow-x: auto; }
  .cellular-atlas-scroll svg { min-width: 1120px; }
}
@media (max-width: 620px) {
  .cellular-recipe, .cellular-readouts { grid-template-columns: 1fr; }
  .cellular-recipe-presets { grid-column: auto; }
}
@media (prefers-reduced-motion: reduce) {
  .cellular-active-edge { animation: none; }
}
`;

const STORY_CSS = `
.cellular-story-shell { min-height: 100%; outline: none; }
.cellular-story-rail { display: grid; grid-template-columns: repeat(5,minmax(0,1fr)); gap: 10px; }
.cellular-story-stage { display: grid; grid-template-columns: minmax(0,1fr) minmax(260px,340px); gap: 22px; align-items: start; }
.cellular-story-svg { min-width: 980px; }
.cellular-story-scroll { overflow-x: auto; }
.cellular-story-reduction { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); }
@media (max-width: 900px) {
  .cellular-story-stage { grid-template-columns: 1fr; }
  .cellular-story-rail { grid-template-columns: repeat(2,minmax(0,1fr)); }
}
@media (max-width: 620px) {
  .cellular-story-rail, .cellular-story-reduction { grid-template-columns: 1fr; }
}
`;

const LANE_Y: Record<Lane, number> = {
  control: 54,
  data: 200,
  execution: 346,
  results: 492,
};

const LANE_H: Record<Lane, number> = {
  control: 132,
  data: 132,
  execution: 132,
  results: 220,
};

const NODES: AtlasNode[] = [
  {
    id: "config",
    lane: "control",
    label: "Authored run",
    detail: "Config v2 / --cells N",
    status: "built",
    symbol: "ProfileFlags → BenchmarkRun",
    path: "rust/cli/src/flags.rs",
    proof: "rust/cli/tests/online_v2_stdio.rs",
    x: 34,
    y: 94,
    w: 142,
    h: 58,
  },
  {
    id: "execute",
    lane: "control",
    label: "Unified aiperf",
    detail: "self-exec --execute",
    status: "built",
    symbol: "execute_mode::dispatch",
    path: "rust/cli/src/execute_mode.rs",
    proof: "rust/cli/tests/protocol_v2_stdio.rs",
    x: 218,
    y: 94,
    w: 154,
    h: 58,
  },
  {
    id: "controller",
    lane: "control",
    label: "Controller",
    detail: "validate · slice · launch",
    status: "built",
    symbol: "run_cellular",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    x: 418,
    y: 94,
    w: 160,
    h: 58,
  },
  {
    id: "start-barrier",
    lane: "control",
    label: "Synchronized START",
    detail: "all cells registered",
    status: "built",
    symbol: "all_registered → trigger",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "test_cellular_matches_single_cell + synchronized_start_releases_all_cells_together",
    x: 626,
    y: 72,
    w: 166,
    h: 50,
  },
  {
    id: "phaser",
    lane: "control",
    label: "Monotonic phaser",
    detail: "built opt-in START",
    status: "built",
    symbol: "Phaser",
    path: "rust/runtime/src/cellular/phaser.rs",
    proof: "test_cellular_phaser_start_matches_event_start",
    x: 626,
    y: 130,
    w: 166,
    h: 50,
  },
  {
    id: "barrier-free",
    lane: "control",
    label: "Barrier-free",
    detail: "k6-class start",
    status: "built",
    symbol: "AIPERF_CELL_BARRIER_FREE",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "test_cellular_barrier_free_matches_synchronized",
    x: 818,
    y: 94,
    w: 150,
    h: 58,
  },
  {
    id: "k8s-roles",
    lane: "control",
    label: "Native K8s roles",
    detail: "controller · cell; aggregator refusal",
    status: "partial",
    symbol: "run_controller / run_cell / run_aggregator",
    path: "rust/cli/src/cellular_role.rs",
    proof: "native role wiring + k8s.rs unit tests; no cluster e2e",
    x: 1012,
    y: 72,
    w: 166,
    h: 50,
  },
  {
    id: "shared-origin",
    lane: "control",
    label: "Shared timing origin",
    detail: "zero at START barrier",
    status: "built",
    symbol: "AIPERF_CELL_SHARED_ORIGIN",
    path: "rust/runtime/src/engine/cell_origin.rs",
    proof: "test_cellular_shared_origin_zeroes_at_the_barrier",
    x: 1202,
    y: 130,
    w: 170,
    h: 50,
  },
  {
    id: "dataset",
    lane: "data",
    label: "Dataset",
    detail: "synthetic · file · graph",
    status: "built",
    symbol: "DatasetInputAdapterResolver",
    path: "rust/runtime/src/engine/dataset_input.rs",
    proof: "rust/e2e-tests/tests/test_cellular_dataset_shipping.rs",
    x: 218,
    y: 240,
    w: 154,
    h: 58,
  },
  {
    id: "partition",
    lane: "data",
    label: "Ownership",
    detail: "i % cells == cell_id",
    status: "built",
    symbol: "ModuloCellPartition",
    path: "rust/runtime/src/cellular/partition.rs",
    proof: "cellular::partition unit tests",
    x: 418,
    y: 240,
    w: 160,
    h: 58,
  },
  {
    id: "dataset-fanout",
    lane: "data",
    label: "Dataset fan-out",
    detail: "opt-in verification overlay",
    status: "built",
    symbol: "DatasetIndex",
    path: "rust/runtime/src/cellular/dataset_session.rs",
    proof: "test_cellular_dataset_fanout_matches_baseline",
    x: 626,
    y: 218,
    w: 166,
    h: 50,
  },
  {
    id: "cell-index",
    lane: "data",
    label: "Cell-local index",
    detail: "owned request ids only",
    status: "built",
    symbol: "DatasetIndex",
    path: "rust/runtime/src/cellular/dataset_session.rs",
    proof: "dataset_session unit tests",
    x: 818,
    y: 240,
    w: 150,
    h: 58,
  },
  {
    id: "infinite",
    lane: "data",
    label: "Infinite scheduled",
    detail: "fails closed",
    status: "rejected",
    symbol: "validate_cellular_phase_budgets",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "cellular phase-budget unit tests",
    x: 1212,
    y: 240,
    w: 158,
    h: 58,
  },
  {
    id: "dataset-serve",
    lane: "data",
    label: "Dataset serving",
    detail: "Stage G · HTTP + zstd",
    status: "built",
    symbol: "build_dataset_serve_plan",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "test_cellular_dataset_shipping.rs + graph dataset shipping e2e",
    x: 1012,
    y: 218,
    w: 166,
    h: 50,
  },
  {
    id: "cell",
    lane: "execution",
    label: "Cell k / N",
    detail: "autonomous process or pod",
    status: "built",
    symbol: "fetch_cell_envelope",
    path: "rust/runtime/src/engine/cellular_cell.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    x: 626,
    y: 386,
    w: 166,
    h: 58,
  },
  {
    id: "shard",
    lane: "execution",
    label: "Worker shards",
    detail: "thread-per-core · !Send",
    status: "built",
    symbol: "run_sharded_scheduled",
    path: "rust/runtime/src/engine/sharded_scheduled.rs",
    proof: "thread_per_core_product.rs + worker_local_accumulation_parity.rs",
    x: 818,
    y: 386,
    w: 150,
    h: 58,
  },
  {
    id: "dispatch",
    lane: "execution",
    label: "Online dispatch",
    detail: "HTTP · gRPC hot path",
    status: "built",
    symbol: "RequestSink::dispatch",
    path: "rust/loadgen-core/src/sink.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    x: 1012,
    y: 386,
    w: 158,
    h: 58,
  },
  {
    id: "grpc-cellular",
    lane: "execution",
    label: "gRPC cellular",
    detail: "shared executor · partition ship",
    status: "built",
    symbol: "GrpcExecutionFactory",
    path: "rust/runtime/src/engine/grpc_turn_execution.rs",
    proof: "scheduled: test_grpc_cellular.rs; graph+gRPC cellular unproven",
    x: 1212,
    y: 354,
    w: 158,
    h: 50,
  },
  {
    id: "offline-cellular",
    lane: "execution",
    label: "DynoSim cellular",
    detail: "offline / online fail closed",
    status: "rejected",
    symbol: "validate_cellular_run_shape",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "dynosim_* uses an unwired SimClock executor",
    x: 1212,
    y: 414,
    w: 158,
    h: 50,
  },
  {
    id: "retained-records",
    lane: "results",
    label: "Retained rows",
    detail: "O(records) · exact",
    status: "built",
    symbol: "ShardRecords::Retained",
    path: "rust/runtime/src/engine/execute.rs",
    proof: "test_cellular_exact_fold_matches_retain",
    x: 436,
    y: 520,
    w: 166,
    h: 50,
  },
  {
    id: "exact-store",
    lane: "results",
    label: "Exact store",
    detail: "fold each completion",
    status: "built",
    symbol: "ShardRecords::Folded",
    path: "rust/runtime/src/engine/execute.rs",
    proof: "test_cellular_exact_fold_matches_retain",
    x: 436,
    y: 578,
    w: 166,
    h: 50,
  },
  {
    id: "tag-sketch",
    lane: "results",
    label: "TagSketch",
    detail: "t-digest + exact moments",
    status: "built",
    symbol: "MetricsStorageMode::Sketch",
    path: "rust/runtime/src/metrics_core/store.rs",
    proof: "test_cellular_sketch_matches_single_cell",
    x: 436,
    y: 636,
    w: 166,
    h: 50,
  },
  {
    id: "partition-wire",
    lane: "results",
    label: "Terminal partition",
    detail: "Partition | StorePartition",
    status: "built",
    symbol: "CellMessage",
    path: "rust/runtime/src/cellular/transport/mod.rs",
    proof: "cellular transport integration tests",
    x: 628,
    y: 578,
    w: 150,
    h: 58,
  },
  {
    id: "hierarchy-refusal",
    lane: "results",
    label: "Hierarchy refusal",
    detail: "fanout rejected pre-startup",
    status: "rejected",
    symbol: "run_aggregator",
    path: "rust/runtime/src/engine/cellular_aggregator.rs",
    proof: "test_cellular_hierarchy_is_refused",
    x: 822,
    y: 520,
    w: 158,
    h: 50,
  },
  {
    id: "controller-merge",
    lane: "results",
    label: "Controller merge",
    detail: "global · concat · append",
    status: "built",
    symbol: "merge_store_partitions / merge_records_*",
    path: "rust/runtime/src/cellular/shard.rs",
    proof: "cellular shard merge tests",
    x: 822,
    y: 590,
    w: 158,
    h: 58,
  },
  {
    id: "external-sink",
    lane: "results",
    label: "External sink",
    detail: "no central merge",
    status: "planned",
    symbol: "T3 stream-only",
    path: "specs/cellular.md",
    proof: "per-cell OTLP exists; mode is planned",
    x: 822,
    y: 658,
    w: 158,
    h: 50,
  },
  {
    id: "artifacts",
    lane: "results",
    label: "Record artifact lane",
    detail: "per-record files + concat",
    status: "built",
    symbol: "RecordArtifactLane",
    path: "rust/runtime/src/engine/record_lane.rs",
    proof: "test_cellular_emits_per_record_artifacts_matching_single_cell",
    x: 1022,
    y: 520,
    w: 158,
    h: 50,
  },
  {
    id: "artifact-shipping",
    lane: "results",
    label: "Cross-host shipping",
    detail: "Stage E · HTTP + zstd",
    status: "built",
    symbol: "ArtifactUploadServer",
    path: "rust/runtime/src/engine/artifact_shipping.rs",
    proof: "rust/e2e-tests/tests/test_cellular_http_shipping.rs",
    x: 1202,
    y: 520,
    w: 170,
    h: 50,
  },
  {
    id: "report",
    lane: "results",
    label: "One report",
    detail: "native-v2.json + exports",
    status: "built",
    symbol: "NativeReport",
    path: "rust/runtime/src/metrics_core/report.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    x: 1202,
    y: 590,
    w: 158,
    h: 58,
  },
];

const EDGES: AtlasEdge[] = [
  { id: "config-execute", from: "config", to: "execute", lane: "control", payload: "protocol-v2 envelope", status: "built", d: "M176 123 H218" },
  { id: "execute-controller", from: "execute", to: "controller", lane: "control", payload: "self-exec stdin", status: "built", d: "M372 123 H418" },
  { id: "execute-k8s", from: "execute", to: "k8s-roles", lane: "control", payload: "native argv role", status: "partial", d: "M372 105 C610 34 840 34 1012 91" },
  { id: "controller-start", from: "controller", to: "start-barrier", lane: "control", payload: "registration + EventHandle", status: "built", d: "M578 116 C600 110 602 98 626 97" },
  { id: "controller-phaser", from: "controller", to: "phaser", lane: "control", payload: "distributed START generation", status: "built", d: "M578 132 C600 138 602 150 626 155" },
  { id: "controller-barrier-free", from: "controller", to: "barrier-free", lane: "control", payload: "immediate START", status: "built", d: "M578 104 C680 38 748 64 818 108" },
  { id: "start-origin", from: "start-barrier", to: "shared-origin", lane: "control", payload: "barrier timing epoch", status: "built", d: "M792 97 C960 97 1040 155 1202 155" },
  { id: "controller-dataset", from: "controller", to: "dataset", lane: "data", payload: "sliced envelope / dataset plan", status: "built", d: "M498 152 V202 H295 V240" },
  { id: "dataset-partition", from: "dataset", to: "partition", lane: "data", payload: "stable instances", status: "built", d: "M372 269 H418" },
  { id: "partition-cell", from: "partition", to: "cell", lane: "execution", payload: "owned positions k,k+N,…", status: "built", d: "M578 269 H606 V415 H626" },
  { id: "partition-fanout", from: "partition", to: "dataset-fanout", lane: "data", payload: "request-id chunks", status: "built", d: "M578 258 H626" },
  { id: "partition-serve", from: "partition", to: "dataset-serve", lane: "data", payload: "manifest + routable URLs", status: "built", d: "M578 278 C760 322 860 250 1012 243" },
  { id: "fanout-index", from: "dataset-fanout", to: "cell-index", lane: "data", payload: "replay + live tail", status: "built", d: "M792 243 H818" },
  { id: "index-cell", from: "cell-index", to: "cell", lane: "execution", payload: "Indexed → InFlight", status: "built", d: "M893 298 V330 H709 V386" },
  { id: "serve-cell", from: "dataset-serve", to: "cell", lane: "execution", payload: "HTTP fetch + decompress", status: "built", d: "M1095 268 V322 H742 V386" },
  { id: "origin-cell", from: "shared-origin", to: "cell", lane: "control", payload: "shared zero_ns", status: "built", d: "M1287 180 V326 H775 V386" },
  { id: "cell-shard", from: "cell", to: "shard", lane: "execution", payload: "two-level owned_positions", status: "built", d: "M792 415 H818" },
  { id: "shard-dispatch", from: "shard", to: "dispatch", lane: "execution", payload: "RequestSink + Clock", status: "built", d: "M968 415 H1012" },
  { id: "dispatch-retain", from: "dispatch", to: "retained-records", lane: "results", payload: "CapturedRecord", status: "built", d: "M1091 444 V472 H519 V520" },
  { id: "dispatch-exact", from: "dispatch", to: "exact-store", lane: "results", payload: "RunCapture::fold_streaming", status: "built", d: "M1104 444 V484 H550 V578" },
  { id: "dispatch-sketch", from: "dispatch", to: "tag-sketch", lane: "results", payload: "finite metric values", status: "built", d: "M1117 444 V496 H580 V636" },
  { id: "retain-wire", from: "retained-records", to: "partition-wire", lane: "results", payload: "CellMessage::Partition", status: "built", d: "M602 545 H615 V595 H628" },
  { id: "exact-wire", from: "exact-store", to: "partition-wire", lane: "results", payload: "StorePartition exact", status: "built", d: "M602 603 H628" },
  { id: "sketch-wire", from: "tag-sketch", to: "partition-wire", lane: "results", payload: "StorePartition sketch", status: "built", d: "M602 661 H615 V619 H628" },
  { id: "wire-merge", from: "partition-wire", to: "controller-merge", lane: "results", payload: "flat star fan-in", status: "built", d: "M778 607 H822" },
  { id: "wire-external", from: "partition-wire", to: "external-sink", lane: "results", payload: "bounded aggregates", status: "planned", d: "M778 625 C800 638 800 683 822 683" },
  { id: "merge-artifacts", from: "controller-merge", to: "artifacts", lane: "results", payload: "artifact completion barrier", status: "built", d: "M980 606 C1000 598 1000 552 1022 545" },
  { id: "artifacts-shipping", from: "artifacts", to: "artifact-shipping", lane: "results", payload: "Stage E upload", status: "built", d: "M1180 545 H1202" },
  { id: "merge-report", from: "controller-merge", to: "report", lane: "results", payload: "ColumnStore → NativeReport", status: "built", d: "M980 619 H1202" },
  { id: "external-report", from: "external-sink", to: "report", lane: "results", payload: "no authoritative report", status: "planned", d: "M980 683 C1080 675 1100 640 1202 626" },
];

const STORY_STEPS = [
  {
    page: 1,
    chapter: "Launch",
    title: "One run. Many cells. One report.",
    thesis: "Cellular scale preserves one benchmark identity while autonomous processes share the work.",
    addedNodeIds: ["config", "cell", "report"],
    addedEdgeIds: [],
    invariant: "Scaling changes placement, not the measurement contract.",
    symbol: "run_cellular",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    change: "Meet the authored run, an autonomous cell, and the authoritative report.",
  },
  {
    page: 2,
    chapter: "Launch",
    title: "Author the Config v2 run",
    thesis: "The human-facing command resolves one strict benchmark request before any child exists.",
    addedNodeIds: ["config"],
    addedEdgeIds: [],
    invariant: "Every cell derives from the same resolved request.",
    symbol: "ProfileFlags",
    path: "rust/cli/src/flags.rs",
    proof: "rust/cli/tests/protocol_v2_stdio.rs",
    change: "Focus on the single authored source of truth.",
  },
  {
    page: 3,
    chapter: "Launch",
    title: "Re-exec the unified binary",
    thesis: "The entry point launches the same aiperf image in hidden execution mode.",
    addedNodeIds: ["execute"],
    addedEdgeIds: ["config-execute"],
    invariant: "Process isolation survives without a second product executable.",
    symbol: "exec_bin::resolve",
    path: "rust/cli/src/exec_bin.rs",
    proof: "rust/cli/tests/protocol_v2_stdio.rs",
    change: "Add the protocol-v2 self-exec boundary.",
  },
  {
    page: 4,
    chapter: "Launch",
    title: "Promote to controller at cells > 1",
    thesis: "Single-cell execution stays direct; multi-cell execution promotes one process to coordinator.",
    addedNodeIds: ["controller", "k8s-roles"],
    addedEdgeIds: ["execute-controller", "execute-k8s"],
    invariant: "The controller coordinates but does not dispatch inference load.",
    symbol: "run_cellular",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    change: "Introduce local controller promotion and the native Kubernetes role entry points.",
  },
  {
    page: 5,
    chapter: "Launch",
    title: "Validate and fail closed",
    thesis: "Eligibility is checked before cells launch, preventing unsupported partial runs.",
    addedNodeIds: ["controller"],
    addedEdgeIds: [],
    invariant: "Unsupported transport, budget, storage, and workload combinations never degrade silently.",
    symbol: "validate_cellular_run_shape",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "cellular_controller.rs validation unit tests",
    change: "Expose the eligibility gate around the controller.",
  },
  {
    page: 6,
    chapter: "Distribute",
    title: "Slice the global budget",
    thesis: "Request and conversation budgets are divided without overlap or omission.",
    addedNodeIds: ["dataset", "partition"],
    addedEdgeIds: ["controller-dataset", "dataset-partition"],
    invariant: "The union of cell-owned positions exactly tiles the global budget.",
    symbol: "owned_positions",
    path: "rust/runtime/src/engine/cell_launcher.rs",
    proof: "rust/e2e-tests/tests/test_cellular_multiturn.rs",
    change: "Add the deterministic workload source and modulo partition.",
  },
  {
    page: 7,
    chapter: "Distribute",
    title: "Register, then release START",
    thesis: "Children become ready before the controller releases synchronized execution.",
    addedNodeIds: ["start-barrier"],
    addedEdgeIds: ["controller-start"],
    invariant: "No synchronized cell dispatches before the registration barrier opens.",
    symbol: "await_all_registered → start_event.trigger",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "test_cellular_matches_single_cell + velo synchronized-start tests",
    change: "Add the default lifecycle gate.",
  },
  {
    page: 8,
    chapter: "Distribute",
    title: "Choose START policy and timing origin",
    thesis: "Phaser and barrier-free START are opt-in policies; shared origin can zero cell clocks at the synchronized barrier.",
    addedNodeIds: ["phaser", "barrier-free", "shared-origin"],
    addedEdgeIds: ["controller-phaser", "controller-barrier-free", "start-origin"],
    invariant: "Start policy and timing epoch change coordination, never ownership.",
    symbol: "PhaserServer / PhaserClient",
    path: "rust/runtime/src/cellular/transport/phaser_velo.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    change: "Reveal two alternate START paths and the e2e-proven shared timing origin.",
  },
  {
    page: 9,
    chapter: "Distribute",
    title: "Distribute datasets by overlay or HTTP",
    thesis: "The opt-in Velo overlay verifies request fan-out, while Stage G serves cross-host file and graph inputs over HTTP+zstd.",
    addedNodeIds: ["dataset-fanout", "dataset-serve"],
    addedEdgeIds: ["partition-fanout", "partition-serve"],
    invariant: "Every cell receives only its owned input identities, independent of delivery mechanism.",
    symbol: "build_dataset_serve_plan",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "test_cellular_dataset_shipping.rs + test_cellular_dataset_fanout_matches_baseline",
    change: "Add both built data-delivery paths without replacing canonical cell-local execution.",
  },
  {
    page: 10,
    chapter: "Distribute",
    title: "Index stable cell ownership",
    thesis: "Each cell tracks indexed, in-flight, and completed request identities.",
    addedNodeIds: ["cell-index"],
    addedEdgeIds: ["fanout-index"],
    invariant: "A request identity belongs to exactly one cell.",
    symbol: "DatasetIndex",
    path: "rust/runtime/src/cellular/dataset_session.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs::test_cellular_dataset_fanout_matches_baseline",
    change: "Add the cell-local ownership index.",
  },
  {
    page: 11,
    chapter: "Execute",
    title: "Enter one autonomous cell",
    thesis: "A cell owns its runtime, transport state, metrics, and local lifecycle.",
    addedNodeIds: ["cell"],
    addedEdgeIds: ["partition-cell", "index-cell", "serve-cell", "origin-cell"],
    invariant: "Cells share no hot-path collector lock.",
    symbol: "fetch_cell_envelope",
    path: "rust/runtime/src/engine/cellular_cell.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    change: "Connect deterministic ownership to the autonomous process.",
  },
  {
    page: 12,
    chapter: "Execute",
    title: "Partition again across worker shards",
    thesis: "Each cell applies the same ownership rule across thread-per-core workers.",
    addedNodeIds: ["shard"],
    addedEdgeIds: ["cell-shard"],
    invariant: "Nested ownership remains deterministic and disjoint.",
    symbol: "run_sharded_scheduled",
    path: "rust/runtime/src/engine/sharded_scheduled.rs",
    proof: "rust/cli/tests/thread_per_core_product.rs",
    change: "Add the second partition level inside the cell.",
  },
  {
    page: 13,
    chapter: "Execute",
    title: "Stamp global ordinals",
    thesis: "Local progress maps back to one stable global request order.",
    addedNodeIds: ["shard"],
    addedEdgeIds: [],
    invariant: "ordinal = phase_base + local × cell_count + cell_id.",
    symbol: "global_ordinal",
    path: "rust/runtime/src/cellular/issuance.rs",
    proof: "rust/cli/tests/worker_local_accumulation_parity.rs",
    change: "Expose the identity formula carried by every worker.",
  },
  {
    page: 14,
    chapter: "Execute",
    title: "Dispatch online; reject unwired simulation",
    thesis: "Every shard drives the same clock-injected HTTP or gRPC seam; DynoSim cellular fails closed.",
    addedNodeIds: ["dispatch", "grpc-cellular", "offline-cellular"],
    addedEdgeIds: ["shard-dispatch"],
    invariant: "Only the execution child dispatches inference requests.",
    symbol: "RequestSink",
    path: "rust/loadgen-core/src/sink.rs",
    proof: "scheduled HTTP + gRPC cellular e2e; graph+gRPC cellular unproven",
    change: "Add built HTTP and scheduled gRPC dispatch, then expose proof and DynoSim boundaries.",
  },
  {
    page: 15,
    chapter: "Execute",
    title: "Observe and finalize each record",
    thesis: "Arrival, admission, token, usage, and terminal observations become one CapturedRecord.",
    addedNodeIds: ["retained-records"],
    addedEdgeIds: ["dispatch-retain"],
    invariant: "Storage mode begins only after the record is finalized.",
    symbol: "CapturedRecord",
    path: "rust/runtime/src/engine/records.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    change: "Cross from execution into the results plane.",
  },
  {
    page: 16,
    chapter: "Reduce",
    title: "Retain rows for exact artifacts",
    thesis: "The retain path preserves per-record data and merges it in a deterministic topology order.",
    addedNodeIds: ["retained-records", "partition-wire"],
    addedEdgeIds: ["dispatch-retain", "retain-wire"],
    invariant: "Retain costs O(records) and keeps raw record artifacts available.",
    symbol: "CellMessage::Partition",
    path: "rust/runtime/src/cellular/transport/mod.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs::test_cellular_exact_fold_matches_retain",
    change: "Carry retained records over the cell wire.",
    simulation: "retain",
  },
  {
    page: 17,
    chapter: "Reduce",
    title: "Exact-fold into ColumnStore",
    thesis: "Each completed record folds into an exact mergeable store and the row is dropped.",
    addedNodeIds: ["exact-store"],
    addedEdgeIds: ["dispatch-exact", "exact-wire"],
    invariant: "Exact-fold retains exact record-derived aggregates without retaining rows.",
    symbol: "RunCapture::fold_streaming",
    path: "rust/runtime/src/engine/execute.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs::test_cellular_exact_fold_matches_retain",
    change: "Add the exact bounded-retention alternative.",
    simulation: "exact-fold",
  },
  {
    page: 18,
    chapter: "Reduce",
    title: "Sketch into t-digest plus exact moments",
    thesis: "Finite metric values stream into bounded sketches while counts and moments stay exact.",
    addedNodeIds: ["tag-sketch"],
    addedEdgeIds: ["dispatch-sketch", "sketch-wire"],
    invariant: "Percentiles are approximate; counts, sums, extrema, mean, std, and rates remain exact.",
    symbol: "TagSketch",
    path: "rust/runtime/src/metrics_core/store.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    change: "Add the k6-style bounded-memory reduction path.",
    simulation: "sketch",
  },
  {
    page: 19,
    chapter: "Scale",
    title: "Merge, publish, and understand the boundary",
    thesis: "Flat controller merge is built; hierarchical aggregation is refused before startup and external stream-only aggregation remains planned.",
    addedNodeIds: ["hierarchy-refusal", "controller-merge", "external-sink", "artifacts", "artifact-shipping", "report", "infinite"],
    addedEdgeIds: ["wire-merge", "wire-external", "merge-artifacts", "artifacts-shipping", "merge-report", "external-report"],
    invariant: "One authoritative report exists unless a future external sink explicitly replaces it.",
    symbol: "merge_store_partitions",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs::test_cellular_hierarchy_is_refused",
    change: "Complete flat merge, record-lane, cross-host shipping, report, hierarchy refusal, and roadmap boundaries.",
  },
  {
    page: 20,
    chapter: "Scale",
    title: "Full cellular system atlas",
    thesis: "Every plane, recipe, inspector, body plan, and ability boundary in one view.",
    addedNodeIds: [],
    addedEdgeIds: [],
    invariant: "The complete atlas preserves the measurement contract.",
    symbol: "FullAtlasPage",
    path: "rust/runtime/src/engine/cellular_controller.rs",
    proof: "rust/e2e-tests/tests/test_cellular.rs",
    change: "All previously introduced layers are visible together.",
    fullAtlas: true,
  },
] as const satisfies readonly StoryStep[];

function storyVisibility(page: number) {
  const steps = STORY_STEPS.slice(0, Math.min(page, 19));
  return {
    nodeIds: new Set<string>(steps.flatMap((step) => [...step.addedNodeIds])),
    edgeIds: new Set<string>(steps.flatMap((step) => [...step.addedEdgeIds])),
  };
}

function validateStorySteps() {
  const steps: readonly StoryStep[] = STORY_STEPS;
  if (steps.length !== 20) throw new Error("cellular storyboard must contain exactly 20 pages");
  const nodeIds = new Set(NODES.map((node) => node.id));
  const edgeIds = new Set(EDGES.map((edge) => edge.id));
  steps.forEach((step, index) => {
    if (step.page !== index + 1) throw new Error(`story page ${index + 1} is out of order`);
    if (!step.invariant || !step.symbol || !step.path || !step.proof || !step.change) {
      throw new Error(`story page ${step.page} is missing evidence`);
    }
    step.addedNodeIds.forEach((id) => {
      if (!nodeIds.has(id)) throw new Error(`story page ${step.page} references missing node ${id}`);
    });
    step.addedEdgeIds.forEach((id) => {
      if (!edgeIds.has(id)) throw new Error(`story page ${step.page} references missing edge ${id}`);
    });
    if (step.page < 20) {
      const visibility = storyVisibility(step.page);
      EDGES.filter((edge) => visibility.edgeIds.has(edge.id)).forEach((edge) => {
        if (!visibility.nodeIds.has(edge.from) || !visibility.nodeIds.has(edge.to)) {
          throw new Error(`story page ${step.page} exposes edge ${edge.id} without both endpoints`);
        }
      });
    }
  });
  if (!steps[19]?.fullAtlas) throw new Error("story page 20 must render FullAtlasPage");
}

validateStorySteps();

const ABILITIES: Ability[] = [
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

const NODE_BY_ID = new Map(NODES.map((node) => [node.id, node]));
const EDGE_BY_ID = new Map(EDGES.map((edge) => [edge.id, edge]));

function validateCatalog() {
  for (const edge of EDGES) {
    if (!NODE_BY_ID.has(edge.from) || !NODE_BY_ID.has(edge.to)) {
      throw new Error(`Cellular atlas edge ${edge.id} references an unknown node`);
    }
  }
  for (const node of NODES) {
    if (node.status === "built" && !node.path.startsWith("rust/")) {
      throw new Error(`Built node ${node.id} lacks current source evidence`);
    }
    if (node.status === "planned" && !node.detail) {
      throw new Error(`Planned node ${node.id} lacks a visible status description`);
    }
  }
}

validateCatalog();

function deriveRoute(
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

function statusLabel(status: Status) {
  return status === "built"
    ? "BUILT"
    : status === "partial"
      ? "PARTIAL"
      : status === "rejected"
        ? "REJECTED"
        : "PLANNED";
}

function statusTone(status: Status): "success" | "warning" | "neutral" {
  return status === "built" ? "success" : status === "partial" ? "warning" : "neutral";
}

function RecipeStrip({
  recipe,
  workload,
  storage,
  start,
  truth,
  onRecipe,
  onWorkload,
  onStorage,
  onStart,
  onTruth,
  onPulse,
}: {
  recipe: RecipeId;
  workload: WorkloadKind;
  storage: StorageMode;
  start: StartMode;
  truth: TruthMode;
  onRecipe: (recipe: RecipeId) => void;
  onWorkload: (value: string) => void;
  onStorage: (value: string) => void;
  onStart: (value: string) => void;
  onTruth: (value: boolean) => void;
  onPulse: () => void;
}) {
  return (
    <div className="cellular-recipe" style={{ display: "grid", gap: 10, alignItems: "end" }}>
      <div className="cellular-recipe-presets">
        <Stack gap={5}>
          <Text size="small" tone="tertiary" weight="semibold">FIDELITY RECIPE</Text>
          <Row gap={6} wrap>
            {([
              ["t0", "T0 Exact"],
              ["t1", "T1 Bounded"],
              ["t2", "T2 Hierarchical"],
              ["t3", "T3 External sink"],
            ] as const).map(([id, label]) => (
              <span key={id} style={{ display: "contents" }}>
                <Pill active={recipe === id} onClick={() => onRecipe(id)}>{label}</Pill>
              </span>
            ))}
          </Row>
        </Stack>
      </div>
      <Stack gap={5}>
        <Text size="small" tone="tertiary">Work unit</Text>
        <Select value={workload} onChange={onWorkload} options={[
          { value: "scheduled", label: "Scheduled" },
          { value: "graph", label: "Graph trace" },
        ]} />
      </Stack>
      <Stack gap={5}>
        <Text size="small" tone="tertiary">Storage</Text>
        <Select value={storage} onChange={onStorage} options={[
          { value: "retain", label: "Retain rows" },
          { value: "exact-fold", label: "Exact fold" },
          { value: "sketch", label: "Sketch" },
        ]} />
      </Stack>
      <Stack gap={5}>
        <Text size="small" tone="tertiary">Start</Text>
        <Select value={start} onChange={onStart} options={[
          { value: "synchronized", label: "Synchronized" },
          { value: "phaser", label: "Phaser · opt-in" },
          { value: "barrier-free", label: "Barrier-free" },
        ]} />
      </Stack>
      <Stack gap={7}>
        <Row gap={7} align="center">
          <Toggle checked={truth === "full"} onChange={onTruth} />
          <Text size="small">Roadmap</Text>
        </Row>
        <Button variant="secondary" onClick={onPulse}>Follow signal</Button>
      </Stack>
    </div>
  );
}

function SystemAtlas({
  route,
  truth,
  selectedId,
  pulseEdgeId,
  onSelect,
}: {
  route: RouteModel;
  truth: TruthMode;
  selectedId: string;
  pulseEdgeId: string;
  onSelect: (id: string) => void;
}) {
  const t = useHostTheme();
  const visibleNodes = NODES.filter((node) => truth === "full" || node.status === "built");
  const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
  const visibleEdges = EDGES.filter(
    (edge) => visibleNodeIds.has(edge.from) && visibleNodeIds.has(edge.to) && (truth === "full" || edge.status === "built"),
  );

  const laneColor = (lane: Lane) =>
    lane === "control"
      ? t.category.yellow
      : lane === "data"
        ? t.category.blue
        : lane === "execution"
          ? t.category.green
          : t.category.purple;

  return (
    <div className="cellular-atlas-scroll" style={{ border: `1px solid ${t.stroke.secondary}`, borderRadius: 8, background: t.bg.editor }}>
      <svg viewBox="0 0 1400 760" style={{ display: "block", width: "100%", minWidth: 1040 }}>
        <defs>
          {(["control", "data", "execution", "results"] as Lane[]).map((lane) => (
            <marker key={lane} id={`arrow-${lane}`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill={laneColor(lane)} />
            </marker>
          ))}
        </defs>

        {(["control", "data", "execution", "results"] as Lane[]).map((lane) => (
          <g key={lane}>
            <rect x={10} y={LANE_Y[lane]} width={1380} height={LANE_H[lane]} rx={8} fill={t.fill.quaternary} />
            <text x={26} y={LANE_Y[lane] + 22} fill={laneColor(lane)} fontSize={10} fontWeight={700} letterSpacing="0.12em">
              {lane.toUpperCase()} PLANE
            </text>
          </g>
        ))}

        {[194, 394, 602, 804, 994, 1190].map((x, index) => (
          <g key={x}>
            <line x1={x} y1={54} x2={x} y2={734} stroke={t.stroke.tertiary} strokeDasharray="2 8" />
            <text x={x + 10} y={750} fill={t.text.quaternary} fontSize={9}>
              {["compose", "coordinate", "distribute", "execute", "reduce", "publish"][index]}
            </text>
          </g>
        ))}

        {visibleEdges.map((edge) => {
          const active = route.edgeIds.has(edge.id);
          const pulsing = pulseEdgeId === edge.id;
          const color = laneColor(edge.lane);
          return (
            <g key={edge.id} opacity={active ? 1 : 0.16} onClick={() => onSelect(edge.id)} style={{ cursor: "pointer" }}>
              <path d={edge.d} fill="none" stroke="transparent" strokeWidth={12} />
              <path
                d={edge.d}
                fill="none"
                stroke={color}
                strokeWidth={pulsing ? 3.8 : active ? 2 : 1}
                strokeDasharray={edge.status === "built" ? (pulsing ? "8 5" : undefined) : "4 5"}
                markerEnd={`url(#arrow-${edge.lane})`}
                className={pulsing ? "cellular-active-edge" : undefined}
              />
            </g>
          );
        })}

        {visibleNodes.map((node) => {
          const active = route.nodeIds.has(node.id);
          const selected = selectedId === node.id;
          const color = laneColor(node.lane);
          return (
            <g
              key={node.id}
              className="cellular-node"
              role="button"
              tabIndex={0}
              aria-label={`${node.label}. ${statusLabel(node.status)}. ${node.detail}`}
              onClick={() => onSelect(node.id)}
              onKeyDown={(event: { key: string; preventDefault: () => void }) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(node.id);
                }
              }}
              opacity={active || selected ? 1 : 0.34}
            >
              <rect
                x={node.x}
                y={node.y}
                width={node.w}
                height={node.h}
                rx={node.id === "cell" ? 22 : 7}
                fill={selected ? t.fill.secondary : t.bg.elevated}
                stroke={selected || active ? color : t.stroke.secondary}
                strokeWidth={selected ? 2.6 : active ? 1.5 : 1}
                strokeDasharray={node.status === "built" ? undefined : "4 3"}
              />
              {node.id === "cell" ? (
                <>
                  <circle cx={node.x + 20} cy={node.y + node.h / 2} r={8} fill={t.fill.tertiary} stroke={color} />
                  <circle cx={node.x + node.w - 20} cy={node.y + node.h / 2} r={8} fill={t.fill.tertiary} stroke={color} />
                </>
              ) : null}
              <text x={node.x + node.w / 2} y={node.y + 23} textAnchor="middle" fill={t.text.primary} fontSize={11.5} fontWeight={650}>
                {node.label}
              </text>
              <text x={node.x + node.w / 2} y={node.y + 40} textAnchor="middle" fill={t.text.tertiary} fontSize={9.2}>
                {node.detail}
              </text>
              {node.status !== "built" ? (
                <text x={node.x + node.w - 6} y={node.y - 6} textAnchor="end" fill={t.text.secondary} fontSize={8.2} fontWeight={700}>
                  {statusLabel(node.status)}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function FidelityStrip({ route }: { route: RouteModel }) {
  const t = useHostTheme();
  const readouts = [
    { label: "PER-CELL RETENTION", value: route.memory },
    { label: "PERCENTILES", value: route.percentiles },
    { label: "EXACT AGGREGATES", value: route.exactAggregates },
    { label: "RECORD ARTIFACTS", value: route.artifacts },
    { label: "RESULT TOPOLOGY", value: route.topology },
  ];
  return (
    <div className="cellular-readouts" style={{ display: "grid", borderTop: `1px solid ${t.stroke.secondary}`, borderBottom: `1px solid ${t.stroke.secondary}` }}>
      {readouts.map((readout, index) => (
        <div key={readout.label} style={{ minWidth: 0, padding: "10px 12px", borderLeft: index === 0 ? "none" : `1px solid ${t.stroke.tertiary}` }}>
          <Text size="small" tone="quaternary" weight="semibold">{readout.label}</Text>
          <Text size="small" weight="semibold" style={{ marginTop: 4 }}>{readout.value}</Text>
        </div>
      ))}
    </div>
  );
}

function CellCrossSection({ storage, workload }: { storage: StorageMode; workload: WorkloadKind }) {
  const t = useHostTheme();
  const positions = Array.from({ length: 12 }, (_, index) => index);
  const result =
    storage === "retain"
      ? "Vec<CapturedRecord>"
      : storage === "exact-fold"
        ? "Exact ColumnStore"
        : "TagSketch · t-digest";
  return (
    <Stack gap={10}>
      <Row align="center">
        <H2>Inside one cell</H2>
        <Spacer />
        <Pill size="sm">{workload === "graph" ? "whole traces" : "global request slots"}</Pill>
      </Row>
      <div className="cellular-cell-scroll">
        <div className="cellular-cell-grid" style={{ display: "grid", gridTemplateColumns: "130px repeat(12,minmax(24px,1fr))", gap: 4, alignItems: "center" }}>
          <Text size="small" tone="tertiary">Global position</Text>
          {positions.map((position) => (
            <span key={position} style={{ textAlign: "center" }}>
              <Code>{position}</Code>
            </span>
          ))}
          {[0, 1, 2].map((cell) => (
            <div key={cell} style={{ display: "contents" }}>
              <Text size="small" weight="semibold">Cell {cell} owns</Text>
              {positions.map((position) => {
                const owns = position % 3 === cell;
                return (
                  <div
                    key={`${cell}-${position}`}
                    style={{
                      height: 22,
                      borderRadius: 5,
                      border: `1px solid ${owns ? t.category.green : t.stroke.tertiary}`,
                      background: owns ? t.fill.secondary : "transparent",
                      color: owns ? t.text.primary : t.text.quaternary,
                      textAlign: "center",
                      fontSize: 10,
                      lineHeight: "20px",
                    }}
                  >
                    {owns ? position : "·"}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
      <Divider />
      <div className="cellular-worker-scroll">
        <div className="cellular-worker-grid" style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr auto 1fr", gap: 10, alignItems: "center" }}>
          {["worker 0 · local 0,2,…", "worker 1 · local 1,3,…", result].map((label, index) => (
            <div key={label} style={{ display: "contents" }}>
              <div style={{ padding: "10px 12px", border: `1px solid ${index === 2 ? t.category.purple : t.stroke.secondary}`, borderRadius: index === 2 ? 18 : 6, background: t.bg.elevated }}>
                <Text size="small" weight={index === 2 ? "semibold" : "normal"}>{label}</Text>
              </div>
              {index < 2 ? (
                <span style={{ display: "contents" }}>
                  <Text tone="tertiary">→</Text>
                </span>
              ) : null}
            </div>
          ))}
        </div>
      </div>
      <Text size="small" tone="tertiary">
        Cell ownership is interleaved across the global space; worker ownership is a second partition inside the cell.
        Each worker folds at record completion when exact-fold or sketch mode is active.
      </Text>
    </Stack>
  );
}

function EngineerInspector({ selectedId }: { selectedId: string }) {
  const dispatch = useCanvasAction();
  const node = NODE_BY_ID.get(selectedId);
  const edge = EDGE_BY_ID.get(selectedId);
  if (edge) {
    return (
      <Card>
        <CardHeader trailing={<Pill size="sm" tone={statusTone(edge.status)}>{statusLabel(edge.status)}</Pill>}>Selected flow</CardHeader>
        <CardBody>
          <Stack gap={10}>
            <Text weight="semibold">{NODE_BY_ID.get(edge.from)?.label} → {NODE_BY_ID.get(edge.to)?.label}</Text>
            <Text size="small" tone="secondary">{edge.payload}</Text>
            <Code>{edge.id}</Code>
          </Stack>
        </CardBody>
      </Card>
    );
  }
  const selected = node ?? NODES[2];
  return (
    <Card>
      <CardHeader trailing={<Pill size="sm" tone={statusTone(selected.status)}>{statusLabel(selected.status)}</Pill>}>Engineer inspector</CardHeader>
      <CardBody>
        <Stack gap={10}>
          <div>
            <Text weight="semibold">{selected.label}</Text>
            <Text size="small" tone="secondary" style={{ marginTop: 3 }}>{selected.detail}</Text>
          </div>
          <Divider />
          <Stack gap={4}>
            <Text size="small" tone="tertiary">SYMBOL</Text>
            <Code>{selected.symbol}</Code>
          </Stack>
          <Stack gap={4}>
            <Text size="small" tone="tertiary">SOURCE</Text>
            <Text size="small" truncate="start">{selected.path}</Text>
          </Stack>
          <Stack gap={4}>
            <Text size="small" tone="tertiary">PROOF / BOUNDARY</Text>
            <Text size="small">{selected.proof}</Text>
          </Stack>
          <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path: selected.path })}>
            Open source
          </Button>
        </Stack>
      </CardBody>
    </Card>
  );
}

function AbilityMap({ truth }: { truth: TruthMode }) {
  const t = useHostTheme();
  const rows = ABILITIES.filter((row) => truth === "full" || (row.status !== "Planned" && row.status !== "Partial"));
  const tone = (status: Ability["status"]) =>
    status === "Built"
      ? t.category.green
      : status === "Rejected"
        ? t.category.red
        : status === "Approximation"
          ? t.category.yellow
          : t.text.tertiary;
  return (
    <Stack gap={10}>
      <Row align="center">
        <H2>Ability map</H2>
        <Spacer />
        <Text size="small" tone="tertiary">capability ≠ fidelity</Text>
      </Row>
      <div className="cellular-ability-scroll" style={{ border: `1px solid ${t.stroke.secondary}`, borderRadius: 8 }}>
        <div className="cellular-ability-grid">
          <div style={{ display: "grid", gridTemplateColumns: "140px minmax(180px,.8fr) minmax(280px,1.3fr) 100px", gap: 0, background: t.fill.tertiary }}>
            {["Dimension", "What works", "Boundary", "Status"].map((header) => (
              <span key={header} style={{ display: "contents" }}>
                <Text size="small" weight="semibold" style={{ padding: "8px 10px" }}>{header}</Text>
              </span>
            ))}
          </div>
          {rows.map((row, index) => (
            <div key={row.dimension} style={{ display: "grid", gridTemplateColumns: "140px minmax(180px,.8fr) minmax(280px,1.3fr) 100px", borderTop: `1px solid ${t.stroke.tertiary}`, background: index % 2 ? t.fill.quaternary : "transparent" }}>
              <Text size="small" weight="semibold" style={{ padding: "8px 10px" }}>{row.dimension}</Text>
              <Text size="small" style={{ padding: "8px 10px" }}>{row.built}</Text>
              <Text size="small" tone="secondary" style={{ padding: "8px 10px" }}>{row.boundary}</Text>
              <Text size="small" weight="semibold" style={{ padding: "8px 10px", color: tone(row.status) }}>{row.status}</Text>
            </div>
          ))}
        </div>
      </div>
    </Stack>
  );
}

function FullAtlasPage() {
  const t = useHostTheme();
  const [recipe, setRecipe] = useCanvasState<RecipeId>("cellular.recipe.v2", "t1");
  const [workload, setWorkload] = useCanvasState<WorkloadKind>("cellular.workload.v2", "scheduled");
  const [storage, setStorage] = useCanvasState<StorageMode>("cellular.storage.v2", "sketch");
  const [start, setStart] = useCanvasState<StartMode>("cellular.start.v2", "synchronized");
  const [truth, setTruth] = useCanvasState<TruthMode>("cellular.truth.v2", "full");
  const [selectedId, setSelectedId] = useCanvasState<string>("cellular.selected.v2", "controller");
  const [pulseStep, setPulseStep] = useCanvasState<number>("cellular.pulse.v2", 0);
  const [cellExpanded, setCellExpanded] = useCanvasState<boolean>("cellular.cell-expanded.v2", true);
  const route = deriveRoute(recipe, workload, storage, start);
  const pulseEdgeId = route.orderedEdges[pulseStep % route.orderedEdges.length] ?? "";

  const selectRecipe = (next: RecipeId) => {
    setRecipe(next);
    if (next === "t0") {
      setStorage("retain");
      setStart("synchronized");
    } else if (next === "t1") {
      setStorage("sketch");
      setStart("synchronized");
    } else if (next === "t2") {
      setStorage("sketch");
      setStart("synchronized");
    } else {
      setStorage("sketch");
      setStart("synchronized");
      setTruth("full");
    }
    setPulseStep(0);
  };

  const selectNode = (id: string) => {
    setSelectedId(id);
    if (id === "cell") setCellExpanded((value) => !value);
  };

  const setRoadmap = (show: boolean) => {
    setTruth(show ? "full" : "built-only");
    if (!show && recipe === "t3") {
      setRecipe("t1");
      setStorage("exact-fold");
      setStart("synchronized");
    }
  };

  return (
    <div style={{ minHeight: "100%", padding: 22, background: t.bg.editor }}>
      <style>{CSS}</style>
      <Stack gap={18}>
        <Row align="start" gap={16} wrap>
          <Stack gap={4} style={{ maxWidth: 850 }}>
            <Text size="small" tone="tertiary" weight="semibold">AIPERF · CELLULAR RUNTIME · IMPLEMENTATION TRUTH</Text>
            <H1>One benchmark. Many autonomous cells. One measurement contract.</H1>
            <Text tone="secondary">
              Trace the authored run through deterministic ownership, nested worker shards, fold or retain,
              flat controller merge, hierarchy refusal, and the final report. Select any node for source-grounded evidence.
            </Text>
          </Stack>
          <Spacer />
          <Stack gap={5} style={{ minWidth: 230 }}>
            <Row gap={6} justify="end">
              <Pill size="sm" tone="success">Built route</Pill>
              <Pill size="sm" tone="warning">Partial proof / wiring</Pill>
              <Pill size="sm" tone="neutral">Planned</Pill>
              <Pill size="sm" tone="neutral">Rejected</Pill>
            </Row>
            <Text size="small" tone="quaternary" style={{ textAlign: "right" }}>
              Source: live rust/runtime + rust/cli filesystem · audited 2026-07-16
            </Text>
          </Stack>
        </Row>

        <Divider />
        <RecipeStrip
          recipe={recipe}
          workload={workload}
          storage={storage}
          start={start}
          truth={truth}
          onRecipe={selectRecipe}
          onWorkload={(value) => setWorkload(value as WorkloadKind)}
          onStorage={(value) => setStorage(value as StorageMode)}
          onStart={(value) => setStart(value as StartMode)}
          onTruth={setRoadmap}
          onPulse={() => {
            const next = (pulseStep + 1) % route.orderedEdges.length;
            setPulseStep(next);
            setSelectedId(route.orderedEdges[next]);
          }}
        />

        <FidelityStrip route={route} />
        {route.warning ? (
          <div style={{ borderLeft: `3px solid ${t.category.yellow}`, padding: "8px 12px", background: t.fill.quaternary }}>
            <Text size="small">{route.warning}</Text>
          </div>
        ) : null}

        <SystemAtlas
          route={route}
          truth={truth}
          selectedId={selectedId}
          pulseEdgeId={pulseEdgeId}
          onSelect={selectNode}
        />

        {cellExpanded ? <CellCrossSection storage={storage} workload={workload} /> : null}

        <div className="cellular-lower" style={{ display: "grid", gap: 16, alignItems: "start" }}>
          <EngineerInspector selectedId={selectedId} />
          <AbilityMap truth={truth} />
        </div>

        <Divider />
        <Row gap={12} align="center" wrap>
          <Code>{workload === "graph" ? "PartitionedGraphTraceSource" : "CellularAutonomousIssuer"}</Code>
          <Text size="small" tone="secondary">
            {workload === "graph"
              ? "Whole traces are cell-local; retain merge concatenates and renumbers."
              : "Global ordinal = phase_base + within_phase_local × cell_count + cell_id."}
          </Text>
          <Spacer />
          <Text size="small" tone="quaternary">Click Cell k / N to toggle its body plan.</Text>
        </Row>
      </Stack>
    </div>
  );
}

const STORY_CHAPTERS = [
  { label: "Launch", pages: [1, 2, 3, 4, 5] },
  { label: "Distribute", pages: [6, 7, 8, 9, 10] },
  { label: "Execute", pages: [11, 12, 13, 14, 15] },
  { label: "Reduce", pages: [16, 17, 18] },
  { label: "Scale", pages: [19, 20] },
] as const;

function normalizeStoryPage(value: number) {
  return Number.isInteger(value) && value >= 1 && value <= 20 ? value : 1;
}

function StoryRail({ page, onPage }: { page: number; onPage: (page: number) => void }) {
  return (
    <div className="cellular-story-rail">
      {STORY_CHAPTERS.map((chapter) => (
        <div key={chapter.label} style={{ display: "contents" }}>
          <Stack gap={6}>
            <Text size="small" tone="tertiary" weight="semibold">{chapter.label.toUpperCase()}</Text>
            <Row gap={5} wrap>
              {chapter.pages.map((number) => {
                const step = STORY_STEPS[number - 1];
                return (
                  <span key={number} style={{ display: "contents" }}>
                    <Pill
                      active={page === number}
                      tone={number < page ? "success" : "neutral"}
                      onClick={() => onPage(number)}
                      title={`Page ${number}: ${step.title}`}
                    >
                      {number}
                    </Pill>
                  </span>
                );
              })}
            </Row>
          </Stack>
        </div>
      ))}
    </div>
  );
}

function ProgressiveScene({
  step,
  selectedId,
  onSelect,
}: {
  step: StoryStep;
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const t = useHostTheme();
  const visibility = storyVisibility(step.page);
  const newNodes = new Set(step.addedNodeIds);
  const newEdges = new Set(step.addedEdgeIds);
  const visibleNodes = NODES.filter((node) => visibility.nodeIds.has(node.id));
  const visibleEdges = EDGES.filter(
    (edge) =>
      visibility.edgeIds.has(edge.id) &&
      visibility.nodeIds.has(edge.from) &&
      visibility.nodeIds.has(edge.to),
  );
  const visibleLanes = new Set(visibleNodes.map((node) => node.lane));
  const laneColor = (lane: Lane) =>
    lane === "control"
      ? t.category.yellow
      : lane === "data"
        ? t.category.blue
        : lane === "execution"
          ? t.category.green
          : t.category.purple;

  return (
    <div className="cellular-story-scroll" style={{ border: `1px solid ${t.stroke.secondary}`, borderRadius: 8, background: t.bg.editor }}>
      <svg className="cellular-story-svg" viewBox="0 0 1400 760" style={{ display: "block", width: "100%" }} role="img" aria-label={`Page ${step.page}: ${step.title}`}>
        <defs>
          {(["control", "data", "execution", "results"] as Lane[]).map((lane) => (
            <marker key={lane} id={`story-arrow-${lane}`} markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
              <path d="M0,0 L6,3 L0,6 Z" fill={laneColor(lane)} />
            </marker>
          ))}
        </defs>
        {(["control", "data", "execution", "results"] as Lane[])
          .filter((lane) => visibleLanes.has(lane))
          .map((lane) => (
            <g key={lane}>
              <rect x={10} y={LANE_Y[lane]} width={1380} height={LANE_H[lane]} rx={8} fill={t.fill.quaternary} />
              <text x={26} y={LANE_Y[lane] + 22} fill={laneColor(lane)} fontSize={10} fontWeight={700} letterSpacing="0.12em">
                {lane.toUpperCase()} PLANE
              </text>
            </g>
          ))}
        {visibleEdges.map((edge) => (
          <g key={edge.id} opacity={newEdges.has(edge.id) ? 1 : 0.2} onClick={() => onSelect(edge.id)} style={{ cursor: "pointer" }}>
            <path d={edge.d} fill="none" stroke="transparent" strokeWidth={12} />
            <path
              d={edge.d}
              fill="none"
              stroke={laneColor(edge.lane)}
              strokeWidth={newEdges.has(edge.id) ? 3 : 1.25}
              strokeDasharray={edge.status === "built" ? undefined : "4 5"}
              markerEnd={`url(#story-arrow-${edge.lane})`}
            />
          </g>
        ))}
        {visibleNodes.map((node) => {
          const selected = selectedId === node.id;
          const introduced = newNodes.has(node.id);
          const color = laneColor(node.lane);
          return (
            <g
              key={node.id}
              className="cellular-node"
              role="button"
              tabIndex={0}
              aria-label={`${node.label}. ${statusLabel(node.status)}. ${node.detail}`}
              onClick={() => onSelect(node.id)}
              onKeyDown={(event: { key: string; preventDefault: () => void }) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelect(node.id);
                }
              }}
              opacity={introduced || selected ? 1 : 0.28}
            >
              <rect
                x={node.x}
                y={node.y}
                width={node.w}
                height={node.h}
                rx={node.id === "cell" ? 22 : 7}
                fill={selected ? t.fill.secondary : t.bg.elevated}
                stroke={selected || introduced ? color : t.stroke.secondary}
                strokeWidth={selected ? 2.6 : introduced ? 1.8 : 1}
                strokeDasharray={node.status === "built" ? undefined : "4 3"}
              />
              {node.id === "cell" ? (
                <>
                  <circle cx={node.x + 20} cy={node.y + node.h / 2} r={8} fill={t.fill.tertiary} stroke={color} />
                  <circle cx={node.x + node.w - 20} cy={node.y + node.h / 2} r={8} fill={t.fill.tertiary} stroke={color} />
                </>
              ) : null}
              <text x={node.x + node.w / 2} y={node.y + 23} textAnchor="middle" fill={t.text.primary} fontSize={11.5} fontWeight={650}>
                {node.label}
              </text>
              <text x={node.x + node.w / 2} y={node.y + 40} textAnchor="middle" fill={t.text.tertiary} fontSize={9.2}>
                {node.detail}
              </text>
              {node.status !== "built" ? (
                <text x={node.x + node.w - 6} y={node.y - 6} textAnchor="end" fill={t.text.secondary} fontSize={8.2} fontWeight={700}>
                  {statusLabel(node.status)}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function EvidenceMargin({ step, selectedId }: { step: StoryStep; selectedId: string }) {
  const t = useHostTheme();
  const selectedNode = NODES.find((node) => node.id === selectedId);
  const selectedEdge = EDGES.find((edge) => edge.id === selectedId);
  return (
    <Stack gap={12} style={{ borderLeft: `1px solid ${t.stroke.secondary}`, paddingLeft: 18 }}>
      <Text size="small" tone="tertiary" weight="semibold">INVARIANT</Text>
      <Text>{step.invariant}</Text>
      <Divider />
      <Text size="small" tone="tertiary" weight="semibold">SOURCE EVIDENCE</Text>
      <Code>{step.symbol}</Code>
      <Text size="small" tone="secondary">{step.path}</Text>
      <Text size="small" tone="quaternary">{step.proof}</Text>
      <Divider />
      <Text size="small" weight="semibold">Introduced on this page</Text>
      <Text size="small" tone="secondary">{step.change}</Text>
      {selectedNode ? (
        <>
          <Divider />
          <Pill tone={statusTone(selectedNode.status)}>{statusLabel(selectedNode.status)}</Pill>
          <Text weight="semibold">{selectedNode.label}</Text>
          <Text size="small" tone="secondary">{selectedNode.detail}</Text>
        </>
      ) : selectedEdge ? (
        <>
          <Divider />
          <Text weight="semibold">{selectedEdge.from} → {selectedEdge.to}</Text>
          <Text size="small" tone="secondary">{selectedEdge.payload}</Text>
        </>
      ) : null}
    </Stack>
  );
}

const REDUCTION_COPY = {
  retain: ["CapturedRecord × N", "retain rows", "Partition", "O(records) · exact percentiles"],
  "exact-fold": ["CapturedRecord × N", "RunCapture::fold_streaming", "StorePartition · exact", "bounded rows · exact percentiles"],
  sketch: ["finite metric values", "t-digest + moments", "StorePartition · sketch", "bounded · approximate percentiles"],
} as const;

function ReductionSimulation({ mode }: { mode: ReductionMode }) {
  const t = useHostTheme();
  const labels = ["INPUT", "OPERATION", "WIRE OUTPUT", "FIDELITY"];
  return (
    <div className="cellular-story-reduction" style={{ borderTop: `1px solid ${t.stroke.secondary}`, borderBottom: `1px solid ${t.stroke.secondary}` }}>
      {REDUCTION_COPY[mode].map((value, index) => (
        <div key={labels[index]} style={{ padding: "10px 12px", borderLeft: index === 0 ? "none" : `1px solid ${t.stroke.tertiary}`, background: index === 2 ? t.fill.quaternary : "transparent" }}>
          <Text size="small" tone="quaternary" weight="semibold">{labels[index]}</Text>
          <Text size="small" weight="semibold" style={{ marginTop: 4 }}>{value}</Text>
        </div>
      ))}
    </div>
  );
}

function ScaleBoundaryStrip() {
  const t = useHostTheme();
  const tiers = [
    ["T0 Exact", "Retain · flat merge", "Built"],
    ["T1 Bounded", "Sketch · flat merge", "Built"],
    ["T2 Hierarchical", "Unavailable · requests are refused before controller startup", "Rejected"],
    ["T3 External sink", "No-central-merge streaming is planned; barrier-free is a separate built START mode", "Planned"],
  ] as const;
  return (
    <Stack gap={10}>
      <Row gap={10} align="center" wrap>
        <Text size="small" tone="tertiary" weight="semibold">CELLULAR FIDELITY LADDER</Text>
        <Pill tone="warning">Scheduled duration/unbounded rejected · graph duration built</Pill>
      </Row>
      <div className="cellular-story-reduction" style={{ borderTop: `1px solid ${t.stroke.secondary}`, borderBottom: `1px solid ${t.stroke.secondary}` }}>
        {tiers.map(([title, detail, status], index) => (
          <div key={title} style={{ padding: "10px 12px", borderLeft: index === 0 ? "none" : `1px solid ${t.stroke.tertiary}` }}>
            <Text size="small" weight="semibold">{title}</Text>
            <Text size="small" tone="secondary" style={{ marginTop: 4 }}>{detail}</Text>
            <Text size="small" weight="semibold" style={{ marginTop: 4, color: status === "Built" ? t.category.green : t.category.yellow }}>{status}</Text>
          </div>
        ))}
      </div>
    </Stack>
  );
}

function StoryPage({
  step,
  selectedId,
  onSelect,
  onPage,
}: {
  step: StoryStep;
  selectedId: string;
  onSelect: (id: string) => void;
  onPage: (page: number) => void;
}) {
  const t = useHostTheme();
  return (
    <div style={{ minHeight: "100%", padding: 22, background: t.bg.editor }}>
      <style>{`${CSS}\n${STORY_CSS}`}</style>
      <Stack gap={18}>
        <Row align="start" gap={16} wrap>
          <Stack gap={4} style={{ maxWidth: 850 }}>
            <Text size="small" tone="tertiary" weight="semibold">CELLULAR STORY · {step.chapter.toUpperCase()} · PAGE {step.page} OF 20</Text>
            <H1>{step.title}</H1>
            <Text tone="secondary">{step.thesis}</Text>
          </Stack>
          <Spacer />
          <Button variant="secondary" onClick={() => onPage(20)}>Jump to full atlas</Button>
        </Row>
        <Divider />
        <StoryRail page={step.page} onPage={onPage} />
        <div className="cellular-story-stage">
          <ProgressiveScene step={step} selectedId={selectedId} onSelect={onSelect} />
          <EvidenceMargin step={step} selectedId={selectedId} />
        </div>
        {step.simulation ? <ReductionSimulation mode={step.simulation} /> : null}
        {step.page === 19 ? <ScaleBoundaryStrip /> : null}
        <Divider />
        <Row gap={10} align="center">
          <Button variant="secondary" disabled={step.page === 1} onClick={() => onPage(step.page - 1)}>Back</Button>
          <Text size="small" tone="secondary">{step.page} / 20</Text>
          <Button onClick={() => onPage(step.page + 1)}>Next</Button>
          <Spacer />
          <Text size="small" tone="quaternary">Use ← and → outside controls to step.</Text>
        </Row>
      </Stack>
    </div>
  );
}

function shouldIgnoreStoryKey(target: EventTarget | null) {
  const element = target as HTMLElement | null;
  return Boolean(
    element?.isContentEditable ||
      element?.matches("input, textarea, select, button, [role='button']"),
  );
}

export default function CellularArchitectureAtlas() {
  const t = useHostTheme();
  const [storedPage, setStoredPage] = useCanvasState<number>("cellular.story-page.v2", 1);
  const [selectedId, setSelectedId] = useCanvasState<string>("cellular.story-selected.v1", "");
  const page = normalizeStoryPage(storedPage);
  const step: StoryStep = STORY_STEPS[page - 1];
  const goTo = (next: number) => setStoredPage(Math.max(1, Math.min(20, next)));
  const onStoryKeyDown = (event: { key: string; target: EventTarget | null; preventDefault: () => void }) => {
    if (shouldIgnoreStoryKey(event.target)) return;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") event.preventDefault();
    if (event.key === "ArrowLeft") goTo(page - 1);
    if (event.key === "ArrowRight") goTo(page + 1);
  };

  return (
    <div className="cellular-story-shell" tabIndex={0} onKeyDown={onStoryKeyDown} style={{ background: t.bg.editor }}>
      {step.fullAtlas ? (
        <>
          <div style={{ padding: "14px 22px", borderBottom: `1px solid ${t.stroke.secondary}` }}>
            <style>{STORY_CSS}</style>
            <Stack gap={12}>
              <Row gap={10} align="center" wrap>
                <Button variant="secondary" onClick={() => goTo(19)}>Back</Button>
                <Text size="small" tone="secondary">Page 20 / 20 · Full atlas</Text>
                <Spacer />
                <Button onClick={() => goTo(1)}>Restart story</Button>
              </Row>
              <StoryRail page={20} onPage={goTo} />
            </Stack>
          </div>
          <FullAtlasPage />
        </>
      ) : (
        <StoryPage step={step} selectedId={selectedId} onSelect={setSelectedId} onPage={goTo} />
      )}
    </div>
  );
}
