import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  CollapsibleSection,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Select,
  Spacer,
  Stack,
  Text,
  TextInput,
  Toggle,
  useCanvasAction,
  useCanvasState,
  useHostTheme,
} from "cursor/canvas";

type Mode = "workbook" | "compose" | "decisions";
type Status = "built" | "partial" | "feature-gated" | "approximate" | "rejected";
type Actor = "entry-point" | "controller" | "wire" | "cell" | "worker" | "aggregator";
type ChapterId =
  | "eligibility"
  | "ownership"
  | "control"
  | "distribution"
  | "execution"
  | "capture"
  | "merge"
  | "artifacts";
type ChapterFilter = ChapterId | "all";

type SourceRef = {
  path: string;
  startLine: number;
  endLine: number;
  symbol: string;
};

type EvidenceRef = {
  path: string;
  symbol: string;
  kind: "unit" | "integration" | "e2e" | "boundary";
};

type PseudocodeLine = { id: string; text: string };

type TraceFrame = {
  id: string;
  label: string;
  activeLineId: string;
  activeActors: readonly Actor[];
  activeLinks: readonly string[];
  before: readonly string[];
  after: readonly string[];
  emitted?: string;
  invariantChecks: readonly string[];
};

type AlgorithmDefinition = {
  id: string;
  chapter: ChapterId;
  title: string;
  status: Status;
  summary: string;
  source: SourceRef;
  evidence: readonly EvidenceRef[];
  inputs: readonly string[];
  outputs: readonly string[];
  state: readonly string[];
  invariants: readonly string[];
  complexity: { time: string; memory: string };
  gates: readonly string[];
  failures: readonly string[];
  pseudocode: readonly PseudocodeLine[];
  frames: readonly TraceFrame[];
  predecessors: readonly string[];
  successors: readonly string[];
  routeTags: readonly string[];
};

type SelectorState = {
  workload: "scheduled" | "graph";
  transport: "http" | "grpc" | "offline";
  dataset: "synthetic" | "file" | "public" | "dag-jsonl" | "weka" | "dynamo";
  fanout: "off" | "verify";
  turns: "single" | "multi";
  sampler: "sequential" | "shuffle" | "random";
  budget: "requests" | "sessions" | "duration" | "adaptive";
  storage: "retain" | "exact-fold" | "sketch";
  topology: "flat" | "local-tree" | "external-tree";
  start: "synchronized" | "phaser" | "barrier-free";
  deployment: "same-host" | "cross-host";
  artifacts: "summary" | "records" | "raw" | "csv" | "parquet" | "otlp";
  build: "lean" | "velo" | "full";
};
type SelectorKey = keyof SelectorState;
type EffectiveArtifact =
  | SelectorState["artifacts"]
  | "omitted"
  | "aggregate-otlp";
type EffectiveSettings = {
  workload: SelectorState["workload"];
  topology: SelectorState["topology"];
  artifacts: EffectiveArtifact;
  storage: SelectorState["storage"];
};
type GateStage =
  | "pre-controller"
  | "controller-prelaunch"
  | "cell-side"
  | "aggregator-receive";

type ValidationGate = {
  id: string;
  algorithmId: string;
  order: number;
  stage: GateStage;
  rejects: (selection: SelectorState) => boolean;
  reason: string;
};

type RouteResult =
  | {
      valid: true;
      algorithmIds: readonly string[];
      memory: string;
      fidelity: string;
      artifacts: string;
      effective: EffectiveSettings;
      environment: readonly string[];
      compileFeatures: readonly string[];
      limitations: readonly string[];
      evidence: readonly EvidenceRef[];
    }
  | {
      valid: false;
      algorithmIds: readonly string[];
      effective: EffectiveSettings;
      limitations: readonly string[];
      gateStage: GateStage;
      rejectedBy: string;
      reason: string;
    };

type RouteRecipe = {
  id: string;
  title: string;
  selection: SelectorState;
  kind: "canonical" | "rejected";
};

type DecisionCell = {
  id: string;
  title: string;
  leftLabel: string;
  left: SelectorState;
  rightLabel: string;
  right: SelectorState;
  invariant: string;
};

const ELIGIBILITY_IDS = [
  "execution-mode-dispatch",
  "execution-child-selection",
  "controller-promotion",
  "velo-feature-admission",
  "cell-count-resolution",
  "run-kind-classification",
  "cellular-run-shape-validation",
  "scheduled-budget-validation",
  "graph-budget-validation",
  "storage-compatibility-prediction",
  "sketch-artifact-validation",
  "execution-merge-backstops",
] as const;

const OWNERSHIP_IDS = [
  "modulo-cell-ownership",
  "owned-positions-tiling",
  "shared-seed-resolution",
  "cell-envelope-construction",
  "scheduled-session-slicing",
  "capacity-rate-ramp-slicing",
  "phase-ordinal-bases",
  "direct-issuance-authority",
  "cellular-issuance-authority",
  "multi-turn-detection",
  "conversation-ownership",
] as const;

const CONTROL_IDS = [
  "broadcast-attach-replay",
  "broadcast-add-fanout",
  "broadcast-finalize",
  "phaser-generation-advance",
  "phaser-await-generation",
  "phaser-late-attach",
  "velo-controller-bind",
  "velo-peer-connect",
  "handler-registration",
  "synchronized-start",
  "phaser-start",
  "barrier-free-launch",
  "local-cell-launch",
  "external-cell-launch",
  "controller-child-arbitration",
] as const;

const DISTRIBUTION_IDS = [
  "canonical-dataset-regeneration",
  "dataset-chunk-publish",
  "dataset-finalize",
  "owned-index-build",
  "dataset-velo-subscribe",
  "dataset-velo-replay-live",
  "controller-fanout-generation",
  "dispatch-on-issue",
  "dispatch-on-complete",
  "distribution-miss",
  "fanout-verification-overlay",
  "dataset-serve-plan",
  "dataset-manifest-validation",
  "dataset-safe-path-mapping",
  "dataset-http-zstd-reconstruct",
  "recorded-graph-file-enumeration",
] as const;

const EXECUTION_IDS = [
  "partitioned-scheduled-sampler",
  "partitioned-graph-source",
  "graph-global-instance-ordinal",
  "two-level-partition",
  "thread-phase-slicing",
  "scheduled-shard-runtime",
  "cell-envelope-fetch",
  "issuance-dispatch-injection",
  "scheduled-graph-runtime-branch",
] as const;

const CAPTURE_IDS = [
  "terminal-record-finalization",
  "retain-record-capture",
  "streaming-exact-fold",
  "sketch-scratch-harvest",
  "tdigest-insert-compress",
  "welford-aggregate-state",
  "tagged-sketch-merge",
  "column-store-append",
  "ingested-count-preservation",
  "partition-messagepack-encode",
  "partition-messagepack-decode",
] as const;

const MERGE_IDS = [
  "scheduled-global-ordinal-merge",
  "ordinal-duplicate-detection",
  "ordinal-missing-detection",
  "ordinal-range-detection",
  "graph-concatenation-renumber",
  "exact-fold-store-merge",
  "sketch-tdigest-merge",
  "controller-partition-collection",
  "hierarchical-tier-sizing",
  "heartbeat-aggregation",
  "final-report-assembly",
  "merged-report-fidelity-boundary",
] as const;

const ARTIFACT_IDS = [
  "artifact-authority-allowlist",
  "shard-local-concatenation",
  "cell-local-concatenation",
  "controller-global-concatenation",
  "artifact-http-zstd-upload",
  "partial-file-atomic-replace",
  "artifact-completion-barrier",
  "telemetry-drop-warning",
  "child-exit-arbitration",
  "controller-timeout",
  "cancellation-propagation",
  "terminal-failure-envelope",
] as const;

// The three distinct cellular wire planes, kept explicit so route tags never
// conflate them. The partition/register/heartbeat control plane and the phaser
// control plane are raw MessagePack over velo (tiny events, no application zstd);
// the dataset fan-out plane is MessagePack + zstd level 3 over velo (redundant
// request bodies compress well); Stage E (artifact upload) and Stage G (dataset
// serve) are HTTP streaming + zstd (Content-Encoding: zstd, bounded per-chunk).
const WIRE_FACTS = {
  partition: "wire: MessagePack over velo raw payload; no application zstd",
  phaserVelo: "wire: MessagePack over velo raw payload; no application zstd",
  datasetVelo: "wire: MessagePack + zstd level 3 over velo",
  stageE: "wire: HTTP streaming + zstd (Content-Encoding: zstd)",
  stageG: "wire: HTTP streaming + zstd (Content-Encoding: zstd)",
} as const;

function pseudocode(...text: readonly string[]): readonly PseudocodeLine[] {
  return text.map((line, index) => ({ id: `step-${index + 1}`, text: line }));
}

function admissionFrames(
  activeLineId: string,
  actors: readonly Actor[],
  admitted: { before: readonly string[]; after: readonly string[]; invariant: string },
  rejected: {
    before: readonly string[];
    after: readonly string[];
    invariant: string;
    emitted?: string;
    activeLineId?: string;
  },
): readonly TraceFrame[] {
  return [
    {
      id: "admission",
      label: "Admission path",
      activeLineId,
      activeActors: actors,
      activeLinks: [],
      before: admitted.before,
      after: admitted.after,
      emitted: "admitted",
      invariantChecks: [admitted.invariant],
    },
    {
      id: "rejection",
      label: "Rejection or boundary path",
      activeLineId: rejected.activeLineId ?? activeLineId,
      activeActors: actors,
      activeLinks: [],
      before: rejected.before,
      after: rejected.after,
      emitted: rejected.emitted ?? "rejected",
      invariantChecks: [rejected.invariant],
    },
  ];
}

const ELIGIBILITY_ALGORITHMS: readonly AlgorithmDefinition[] = [
  {
    id: "execution-mode-dispatch",
    chapter: "eligibility",
    title: "Dispatch the private execution protocol",
    status: "built",
    summary:
      "The unified binary recognizes an exact one-argument private mode, then dispatch routes cells and controllers while the aggregator role returns an explicit hierarchy-unavailable refusal.",
    source: { path: "rust/cli/src/execute_mode.rs", startLine: 60, endLine: 120, symbol: "is_execution_mode / dispatch" },
    evidence: [
      { path: "rust/cli/src/execute_mode.rs", symbol: "is_execution_mode exact-slice match at lines 60-65", kind: "boundary" },
      { path: "rust/cli/src/execute_mode.rs", symbol: "dispatch role routing at lines 83-120", kind: "boundary" },
    ],
    inputs: ["argv without argv[0]"],
    outputs: ["execute, cell, controller, or hierarchy-refusal result"],
    state: ["process role", "stdin/stdout protocol ownership"],
    invariants: ["Private roles are intercepted before clap.", "Unknown arguments never enter the private protocol."],
    complexity: { time: "O(1): exact comparison against a one-element slice and three constants", memory: "O(1)" },
    gates: ["argv contains an exact private flag"],
    failures: ["Unknown, misspelled, or extra-argument private invocations are not admitted; stdin and role-handler errors exit nonzero."],
    pseudocode: pseudocode(
      "admit only args == [--execute], [--cell], or [--aggregator]",
      "read stdin to EOF unless role is cell",
      "route cell immediately; reject aggregator as unavailable hierarchy",
      "promote a qualifying execute envelope to controller; otherwise run protocol v2",
    ),
    frames: admissionFrames(
      "step-1",
      ["entry-point"],
      { before: ["argv = [--cell]"], after: ["role = cell"], invariant: "Private flag bypasses clap." },
      { before: ["argv = [--cell, extra]"], after: ["role remains public CLI"], invariant: "Private mode requires one exact argument." },
    ),
    predecessors: [],
    successors: ["execution-child-selection", "controller-promotion"],
    routeTags: ["entry", "protocol", "all-runs"],
  },
  {
    id: "execution-child-selection",
    chapter: "eligibility",
    title: "Select the unified child executable",
    status: "built",
    summary:
      "Re-exec returns any present AIPERF_EXEC_BIN value, otherwise current_exe, and finally a bare platform-specific aiperf name.",
    source: { path: "rust/cli/src/exec_bin.rs", startLine: 12, endLine: 32, symbol: "resolve" },
    evidence: [
      { path: "rust/cli/src/exec_bin.rs", symbol: "AIPERF_EXEC_BIN override at lines 20-23", kind: "boundary" },
      { path: "rust/cli/src/exec_bin.rs", symbol: "current_exe and bare-name fallbacks at lines 24-31", kind: "boundary" },
    ],
    inputs: ["AIPERF_EXEC_BIN", "std::env::current_exe"],
    outputs: ["child executable PathBuf"],
    state: ["process environment"],
    invariants: ["Any present override, including an empty string, wins.", "The last fallback is a platform-specific bare executable name."],
    complexity: { time: "O(path length)", memory: "O(path length)" },
    gates: ["override is present; else current_exe succeeds; else bare-name fallback"],
    failures: ["An empty override resolves to an empty path and can fail later at spawn; bare-name fallback can fail OS path lookup."],
    pseudocode: pseudocode(
      "if AIPERF_EXEC_BIN is present, return PathBuf(value), even when value is empty",
      "else if current_exe() succeeds, return it",
      "else return aiperf.exe on Windows or aiperf elsewhere",
    ),
    frames: admissionFrames(
      "step-1",
      ["entry-point"],
      { before: ["AIPERF_EXEC_BIN=/tmp/aiperf"], after: ["child=/tmp/aiperf"], invariant: "Any present override wins." },
      { before: ["override absent; current_exe fails"], after: ["child=aiperf or aiperf.exe"], invariant: "Resolution remains infallible; spawn surfaces lookup failure.", emitted: "bare-name fallback" },
    ),
    predecessors: ["execution-mode-dispatch"],
    successors: ["controller-promotion", "velo-feature-admission"],
    routeTags: ["entry", "re-exec"],
  },
  {
    id: "controller-promotion",
    chapter: "eligibility",
    title: "Promote multi-cell execute to controller",
    status: "built",
    summary:
      "Dispatch promotes only a non-cell process carrying valid JSON, operation=execute, and a resolved cell count greater than one.",
    source: { path: "rust/cli/src/execute_mode.rs", startLine: 83, endLine: 120, symbol: "dispatch" },
    evidence: [
      { path: "rust/cli/src/execute_mode.rs", symbol: "four promotion predicates at lines 105-113", kind: "boundary" },
      { path: "rust/e2e-tests/tests/test_cellular.rs", symbol: "test_cellular_run_from_python_frontend", kind: "e2e" },
    ],
    inputs: ["stdin bytes", "AIPERF_CELL_ID presence", "operation", "resolved runtime.cells"],
    outputs: ["controller orchestration or ordinary execute request"],
    state: ["decoded request", "process role"],
    invariants: ["A cell child never recursively promotes.", "Only a valid execute envelope with cells>1 promotes."],
    complexity: { time: "O(request bytes)", memory: "O(request bytes)" },
    gates: ["AIPERF_CELL_ID absent", "stdin parses as JSON", "operation == execute", "cell_count_from_envelope > 1"],
    failures: ["A failed promotion predicate continues to ordinary v2 handling; controller execution failures emit a terminal failure."],
    pseudocode: pseudocode(
      "read stdin for a non-cell role",
      "if CELL_ID is absent and JSON parses and operation == execute and cells > 1, run controller",
      "otherwise configure defaults and run protocol v2",
    ),
    frames: admissionFrames(
      "step-2",
      ["entry-point", "controller"],
      { before: ["runtime.cells=3", "role=execute"], after: ["role=controller"], invariant: "Multi-cell request is promoted exactly once." },
      { before: ["runtime.cells=3", "operation=validate"], after: ["ordinary v2 validation"], invariant: "Non-execute operations do not promote.", emitted: "not promoted" },
    ),
    predecessors: ["execution-mode-dispatch", "execution-child-selection", "cell-count-resolution"],
    successors: ["velo-feature-admission", "cellular-run-shape-validation"],
    routeTags: ["entry", "controller", "multi-cell"],
  },
  {
    id: "velo-feature-admission",
    chapter: "eligibility",
    title: "Require Velo for cellular roles",
    status: "feature-gated",
    summary:
      "Controller and cell handlers fail closed when the binary lacks the Velo transport feature; aggregator invocation is an explicit hierarchy refusal.",
    source: { path: "rust/cli/src/execute_mode.rs", startLine: 221, endLine: 258, symbol: "run_aggregator / run_cell / run_controller" },
    evidence: [
      { path: "rust/cli/src/execute_mode.rs", symbol: "non-velo aggregator handling at lines 221-228", kind: "boundary" },
      { path: "rust/cli/src/execute_mode.rs", symbol: "non-velo cell/controller handling at lines 230-253", kind: "boundary" },
    ],
    inputs: ["compiled Cargo features", "requested cellular role"],
    outputs: ["role execution or unsupported-build error"],
    state: ["compile-time cfg(feature = velo)"],
    invariants: ["Unsupported binaries never silently collapse a cellular run into direct execution."],
    complexity: { time: "O(1)", memory: "O(1)" },
    gates: ["feature=velo for every cellular role"],
    failures: ["A multi-cell request on a non-Velo build returns an explicit Velo-required error."],
    pseudocode: pseudocode(
      "if compiled with velo, execute the requested cellular role",
      "otherwise return a role-specific unsupported-build error",
    ),
    frames: admissionFrames(
      "step-1",
      ["entry-point", "controller"],
      { before: ["build=velo", "cells=3"], after: ["controller admitted"], invariant: "Feature-bearing role is available." },
      { before: ["build=lean", "cells=3"], after: ["explicit Velo error"], invariant: "No degraded execution occurs." },
    ),
    predecessors: ["execution-child-selection", "controller-promotion"],
    successors: ["cellular-run-shape-validation"],
    routeTags: ["entry", "feature-gate", "velo"],
  },
  {
    id: "cell-count-resolution",
    chapter: "eligibility",
    title: "Resolve the requested cell count",
    status: "built",
    summary:
      "Promotion reads run.cfg.runtime.cells, converts to u32 after clamping the unsigned value into 1..=1024, and defaults missing or non-u64 values to one.",
    source: { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", startLine: 217, endLine: 223, symbol: "cell_count_from_envelope" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "cell_count_from_envelope clamp at lines 217-223", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "cell_count_reads_runtime_cells", kind: "unit" },
    ],
    inputs: ["BenchmarkRun config.runtime.cells"],
    outputs: ["u32 cell count in 1..=1024"],
    state: [],
    invariants: ["Result is always in 1..=1024.", "Absent or non-u64 cells means one."],
    complexity: { time: "O(1)", memory: "O(1)" },
    gates: ["runtime.cells optionally decodes as u64"],
    failures: ["Zero is clamped to one and values above 1024 are clamped to 1024 rather than rejected here."],
    pseudocode: pseudocode(
      "read envelope.run.cfg.runtime.cells as u64",
      "if present, return clamp(value, 1, 1024) as u32",
      "otherwise return 1u32",
    ),
    frames: admissionFrames(
      "step-1",
      ["entry-point"],
      { before: ["runtime.cells=4"], after: ["cell_count=4"], invariant: "Explicit topology is preserved." },
      { before: ["runtime.cells=0"], after: ["cell_count=1"], invariant: "Resolved count remains within the bounded domain.", emitted: "clamped" },
    ),
    predecessors: ["execution-mode-dispatch"],
    successors: ["controller-promotion"],
    routeTags: ["entry", "configuration"],
  },
  {
    id: "run-kind-classification",
    chapter: "eligibility",
    title: "Classify scheduled versus graph work",
    status: "built",
    summary:
      "CellularRunKind::detect examines every dataset; any dag_jsonl, weka_trace, or dynamo_trace format selects Graph, otherwise Scheduled.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_kind.rs", startLine: 51, endLine: 68, symbol: "CellularRunKind::detect" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_kind.rs", symbol: "detects_graph_and_scheduled_kinds", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/cellular_kind.rs", symbol: "all-dataset any match at lines 29-40", kind: "boundary" },
    ],
    inputs: ["all envelope run.cfg.datasets"],
    outputs: ["CellularRunKind::Scheduled or CellularRunKind::Graph"],
    state: [],
    invariants: ["Recorded graph formats never enter scheduled execution.", "Classification has no ambiguous third kind."],
    complexity: { time: "O(dataset count)", memory: "O(1)" },
    gates: ["any dataset carries a recognized graph format"],
    failures: ["Absent/non-array datasets or no recognized graph format classify Scheduled; this detector does not validate dataset shape."],
    pseudocode: pseudocode(
      "scan every run.cfg.datasets entry",
      "if any format is dag_jsonl, weka_trace, or dynamo_trace, return Graph",
      "otherwise return Scheduled",
    ),
    frames: admissionFrames(
      "step-1",
      ["controller"],
      { before: ["format=dag_jsonl"], after: ["kind=Graph"], invariant: "Recorded graph selects graph execution." },
      { before: ["datasets absent or all synthetic"], after: ["kind=Scheduled"], invariant: "No invented factory gate participates in classification.", emitted: "classified Scheduled" },
    ),
    predecessors: ["cellular-run-shape-validation"],
    successors: ["scheduled-budget-validation", "graph-budget-validation"],
    routeTags: ["eligibility", "classification"],
  },
  {
    id: "cellular-run-shape-validation",
    chapter: "eligibility",
    title: "Validate the common cellular shape",
    status: "built",
    summary:
      "Before launch, the controller admits HTTP/gRPC and graph datasets, while linear datasets must be supported synthetic/file/public shapes with sound single- or multi-turn ownership.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1358, endLine: 1469, symbol: "validate_cellular_run_shape" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "validate_cellular_run_shape at lines 1358-1469", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "warn_dropped_sidecar_telemetry at lines 1805-1834", kind: "boundary" },
    ],
    inputs: ["cellular envelope transport and all datasets", "predicted exact-fold compatibility"],
    outputs: ["admission or contextual shape error"],
    state: ["cellular_will_use_exact_fold result"],
    invariants: ["Graph formats bypass linear dataset restrictions.", "Multi-turn random replacement is never admitted."],
    complexity: { time: "O(dataset count)", memory: "O(1)" },
    gates: ["transport absent or HTTP/gRPC", "supported linear dataset kind/format", "multi-turn exact-fold", "multi-turn sequential/shuffle"],
    failures: ["Rejects non-HTTP/gRPC transports, unsupported linear dataset types/formats, multi-turn retain, and multi-turn random sampling."],
    pseudocode: pseudocode(
      "if transport is present, require http or grpc",
      "for each dataset: graph formats continue without linear checks",
      "require each linear dataset is supported synthetic, file, or public",
      "for multi-turn require exact-fold, a known format when file/public, and non-random sampling",
    ),
    frames: admissionFrames(
      "step-1",
      ["controller"],
      { before: ["transport=http", "synthetic single-turn"], after: ["shape admitted"], invariant: "Supported linear shape passes without artifact rejection." },
      { before: ["cells=3", "transport=dynosim_offline"], after: ["error before launch"], invariant: "Unsupported transport fails closed." },
    ),
    predecessors: ["controller-promotion", "velo-feature-admission"],
    successors: ["run-kind-classification", "storage-compatibility-prediction"],
    routeTags: ["eligibility", "validation", "fail-closed"],
  },
  {
    id: "scheduled-budget-validation",
    chapter: "eligibility",
    title: "Validate scheduled phase budgets",
    status: "built",
    summary:
      "Scheduled cellular accepts only request-bounded phase types with request or exact-fold session budgets, rejects duration/adaptive bounds, and requires every budget and cap to cover all cells.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1570, endLine: 1680, symbol: "CELLULAR_REQUEST_BOUNDED_PHASE_TYPES / validate_cellular_phase_budgets" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "phase type/budget/cap gates at lines 1570-1680", kind: "boundary" },
      { path: "rust/e2e-tests/tests/test_cellular_multiturn.rs", symbol: "test_cellular_multi_turn_retain_is_rejected", kind: "e2e" },
    ],
    inputs: ["scheduled phases", "cell_count", "predicted exact-fold compatibility"],
    outputs: ["scheduled admission or exact rejecting reason"],
    state: ["multi-turn classification"],
    invariants: ["Every phase budget and configured concurrency cap is at least cell_count.", "Only four request-bounded phase types are admitted."],
    complexity: { time: "O(phases)", memory: "O(1)" },
    gates: ["type in concurrency/poisson/gamma/constant", "requests or sessions", "budget>=cells", "sessions use exact-fold", "caps>=cells"],
    failures: ["Rejects unsupported/trace-driven phase types, missing or undersized budgets, sessions on retain, duration/adaptive bounds, and concurrency/prefill caps below cell_count."],
    pseudocode: pseudocode(
      "require phase type in {concurrency, poisson, gamma, constant}",
      "require requests or sessions; each present budget must be >= cell_count",
      "require exact-fold when sessions is present; reject duration or adaptive_scale",
      "require each concurrency/prefill_concurrency cap >= cell_count; ramps remain allowed",
    ),
    frames: admissionFrames(
      "step-1",
      ["controller"],
      { before: ["requests=60", "cells=3"], after: ["scheduled budget admitted"], invariant: "No cell receives an empty request slice." },
      { before: ["type=fixed_schedule", "requests=60", "cells=3"], after: ["unsupported phase-type error"], invariant: "Trace-driven phases cannot replay once per cell." },
    ),
    predecessors: ["run-kind-classification"],
    successors: ["storage-compatibility-prediction", "scheduled-session-slicing"],
    routeTags: ["eligibility", "scheduled", "budget"],
  },
  {
    id: "graph-budget-validation",
    chapter: "eligibility",
    title: "Validate graph trace budgets",
    status: "built",
    summary:
      "Graph cellular rejects static request budgets, allows sessions or duration to drive trace partitioning, and requires configured concurrency caps to cover all cells.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1682, endLine: 1729, symbol: "validate_graph_cellular_phases" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "graph requests/cap gates at lines 1682-1729", kind: "boundary" },
      { path: "rust/e2e-tests/tests/test_graph_cellular.rs", symbol: "test_graph_cellular_from_python_frontend", kind: "e2e" },
    ],
    inputs: ["graph phase bounds and caps", "cell_count"],
    outputs: ["graph admission or exact rejecting reason"],
    state: [],
    invariants: ["Static requests never select the unpartitioned cycling source.", "Configured caps do not floor into aggregate over-subscription."],
    complexity: { time: "O(phases)", memory: "O(1)" },
    gates: ["requests absent", "each concurrency/prefill_concurrency cap >= cell_count"],
    failures: ["Rejects any static requests budget and any configured concurrency/prefill cap below cell_count; it does not require exact-fold."],
    pseudocode: pseudocode(
      "for each graph phase, require requests is absent",
      "allow sessions or duration to drive PartitionedGraphTraceSource",
      "for each configured concurrency/prefill_concurrency cap, require cap >= cell_count",
      "allow retained graph records; merge concatenates them by cell",
    ),
    frames: admissionFrames(
      "step-2",
      ["controller"],
      { before: ["sessions=6", "requests absent", "cap=3", "cells=3"], after: ["graph phase admitted"], invariant: "Trace instances use the partitioned source." },
      { before: ["requests=6", "cells=3"], after: ["static-request error"], invariant: "Every cell is prevented from replaying the full cycle." },
    ),
    predecessors: ["run-kind-classification"],
    successors: ["storage-compatibility-prediction", "modulo-cell-ownership"],
    routeTags: ["eligibility", "graph", "budget"],
  },
  {
    id: "storage-compatibility-prediction",
    chapter: "eligibility",
    title: "Predict controller-visible exact-fold compatibility",
    status: "built",
    summary:
      "The controller predicts whether every cell will use exact-fold from signals visible without compiling the dataset; execution applies a broader live-consumer gate.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1249, endLine: 1308, symbol: "cellular_will_use_exact_fold" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "cellular_will_use_exact_fold at lines 1276-1308", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "exact_fold_eligible and ExactFoldInputs at lines 892-985", kind: "boundary" },
    ],
    inputs: ["&serde_json::Value envelope", "AIPERF_RUNTIME_EXACT_FOLD and heartbeat process environment", "metrics, phases, and records_parquet_path JSON pointers"],
    outputs: ["boolean controller prediction: exact-fold or not exact-fold"],
    state: [],
    invariants: ["The controller predictor intentionally does not call or fully mirror exact_fold_eligible.", "A false positive is caught by the merge backstop."],
    complexity: { time: "O(phase count) for adaptive_scale scanning; other checks are fixed JSON pointers/env reads", memory: "O(1)" },
    gates: ["exact fold env enabled", "heartbeat disabled", "not sketch", "no adaptive phase", "Parquet feature when requested"],
    failures: ["Returns false for forced retain, heartbeat, sketch, adaptive, or unsupported Parquet streaming; live inputs/accuracy/sinks/artifacts can still disqualify execution later."],
    pseudocode: pseudocode(
      "return false if exact-fold env is disabled or heartbeat lane is enabled",
      "return false for sketch storage or any adaptive_scale phase",
      "on a non-Parquet build, return false when records_parquet_path is requested",
      "otherwise return true; execution separately rejects accuracy/live/artifact consumers",
    ),
    frames: admissionFrames(
      "step-3",
      ["controller"],
      { before: ["env enabled", "no heartbeat/sketch/adaptive"], after: ["controller predicts exact-fold"], invariant: "Only controller-visible disqualifiers participate." },
      { before: ["sketch metrics enabled"], after: ["controller prediction=false"], invariant: "Sketch is distinct from exact-fold.", emitted: "not exact-fold" },
    ),
    predecessors: ["cellular-run-shape-validation"],
    successors: ["sketch-artifact-validation", "execution-merge-backstops"],
    routeTags: ["eligibility", "storage", "merge"],
  },
  {
    id: "sketch-artifact-validation",
    chapter: "eligibility",
    title: "Reject row-dependent sketch artifacts",
    status: "built",
    summary:
      "After the cell has crossed START and completed applicable distribution setup, cell-side run_v2 validate_plan rejects the four directly coded sketch conflicts: records, raw, outputs, and native per-record OTLP.",
    source: { path: "rust/runtime/src/runner_protocol/execute.rs", startLine: 620, endLine: 656, symbol: "validate_plan" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "sketch artifact ensures at lines 640-655", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "validate_plan plan-wide validation at lines 620-656", kind: "integration" },
    ],
    inputs: ["sketch selection", "artifact and exporter configuration"],
    outputs: ["cell run_v2 admission or one explicit incompatibility error"],
    state: ["cell envelope released after START", "applicable Stage G/fan-out distribution setup completed"],
    invariants: ["Sketch mode never promises data it discards."],
    complexity: { time: "O(1): two direct ensure! checks over fixed fields", memory: "O(1)" },
    gates: ["records_path, raw_path, outputs_path absent", "native_otel_enabled=false"],
    failures: ["Rejects records_path, raw_path, outputs_path, or native per-record OTLP while sketch is enabled; this cited block does not directly reject trace or timeslice."],
    pseudocode: pseudocode(
      "if storage is not sketch, return success",
      "require records_path, raw_path, and outputs_path are absent",
      "require native_otel_enabled is false",
    ),
    frames: admissionFrames(
      "step-2",
      ["cell"],
      { before: ["cell crossed START", "sketch", "summary only"], after: ["cell plan admitted"], invariant: "All outputs are summary-compatible." },
      { before: ["cell crossed START", "sketch", "raw JSONL requested"], after: ["cell-side artifact conflict error"], invariant: "Discarded rows are not advertised." },
    ),
    predecessors: ["storage-compatibility-prediction"],
    successors: ["execution-merge-backstops"],
    routeTags: ["eligibility", "sketch", "artifacts"],
  },
  {
    id: "execution-merge-backstops",
    chapter: "eligibility",
    title: "Revalidate actual cell payloads at merge",
    status: "built",
    summary:
      "The controller rejects scheduled multi-turn retain and mixed store/record payloads; graph retain is valid and concatenates cell partitions.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 719, endLine: 751, symbol: "run_cellular merge-mode backstops" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "mixed-mode and retain backstops at lines 719-751", kind: "boundary" },
      { path: "rust/e2e-tests/tests/test_cellular_multiturn.rs", symbol: "test_cellular_multi_turn_retain_is_rejected", kind: "e2e" },
    ],
    inputs: ["actual CellMessage payload variants", "run kind", "multi-turn classification"],
    outputs: ["selected store/record merge path or explicit error"],
    state: ["payloads received from every cell"],
    invariants: ["One run merges one payload mode.", "The multi-turn retain backstop applies only to Scheduled."],
    complexity: { time: "O(cell_count)", memory: "O(cell_count)" },
    gates: ["uniform payload mode", "scheduled multi-turn must ship stores"],
    failures: ["Rejects mixed StorePartition/Partition payloads and scheduled multi-turn raw records; graph raw records concatenate by cell and are admitted."],
    pseudocode: pseudocode(
      "classify every cell payload as folded store or retained records",
      "if kind is Scheduled and run is multi-turn and raw partitions exist, reject",
      "if stores exist, require raw partitions are empty and merge stores",
      "otherwise Scheduled merges global order and Graph concatenates by cell",
    ),
    frames: admissionFrames(
      "step-2",
      ["controller", "aggregator"],
      { before: ["all payloads=StorePartition"], after: ["store merge selected"], invariant: "Merge algebra is uniform." },
      { before: ["scheduled multi-turn payloads=Partition"], after: ["exact-fold limitation error"], invariant: "Graph retain remains exempt; scheduled corruption is blocked." },
    ),
    predecessors: ["storage-compatibility-prediction", "sketch-artifact-validation"],
    successors: [],
    routeTags: ["eligibility", "merge", "backstop"],
  },
];

const OWNERSHIP_ALGORITHMS: readonly AlgorithmDefinition[] = [
  {
    id: "modulo-cell-ownership",
    chapter: "ownership",
    title: "Own positions by modulo",
    status: "built",
    summary:
      "Cell k owns exactly the global positions whose remainder modulo cell_count is k.",
    source: { path: "rust/runtime/src/cellular/partition.rs", startLine: 31, endLine: 143, symbol: "CellPartition / ModuloCellPartition" },
    evidence: [
      { path: "rust/runtime/src/cellular/partition.rs", symbol: "CellPartition::owns and implementation at lines 38-40 and 133-135", kind: "boundary" },
      { path: "rust/runtime/src/cellular/partition.rs", symbol: "ownership_is_disjoint_and_complete_across_cells", kind: "unit" },
    ],
    inputs: ["cell_id", "cell_count", "global instance position", "optional AIPERF_CELL_ID/AIPERF_CELL_COUNT environment"],
    outputs: ["ownership boolean"],
    state: ["validated immutable partition"],
    invariants: ["cell_count >= 1 and cell_id < cell_count.", "Exactly one cell owns every global position."],
    complexity: { time: "O(1)", memory: "O(1)" },
    gates: ["ModuloCellPartition::new validates coordinates", "from_env returns Some only when both values parse and validate"],
    failures: ["new returns ZeroCells or IdOutOfRange; from_env collapses missing, malformed, zero, or out-of-range environment values to None, and issuance then selects Direct."],
    pseudocode: pseudocode(
      "require cell_count >= 1 and cell_id < cell_count",
      "owns(instance) = instance % cell_count == cell_id",
      "from_env: parse both vars and call new; any absence/parse/validation failure returns None",
      "issuance_authority_from_env: None selects DirectIssuanceAuthority",
    ),
    frames: admissionFrames(
      "step-2",
      ["cell"],
      { before: ["id=1", "count=3", "instance=4"], after: ["4 % 3 = 1; owned"], invariant: "Exactly cell 1 admits position 4." },
      { before: ["AIPERF_CELL_COUNT=0"], after: ["from_env=None", "issuance selects Direct"], invariant: "Malformed environment is not surfaced as a partition rejection.", emitted: "Direct fallback" },
    ),
    predecessors: ["graph-budget-validation"],
    successors: ["owned-positions-tiling", "cellular-issuance-authority", "conversation-ownership"],
    routeTags: ["ownership", "modulo", "all-workloads"],
  },
  {
    id: "owned-positions-tiling",
    chapter: "ownership",
    title: "Count each uneven owned slice",
    status: "built",
    summary:
      "A closed form counts positions k, k+N, k+2N below a finite total, including uneven remainders.",
    source: { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", startLine: 205, endLine: 212, symbol: "owned_positions" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "owned_positions_sum_to_total_and_tile", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "owned_positions closed form at lines 195-212", kind: "boundary" },
    ],
    inputs: ["total", "cell_id", "cell_count"],
    outputs: ["owned position count"],
    state: [],
    invariants: ["Cell counts sum exactly to total.", "Counts differ by at most one."],
    complexity: { time: "O(1)", memory: "O(1)" },
    gates: ["validated nonzero cell_count"],
    failures: ["cell_count=0 would make division invalid and is excluded by partition construction."],
    pseudocode: pseudocode(
      "if cell_id >= total, return 0",
      "owned = div_ceil(total - cell_id, cell_count)",
      "return owned",
    ),
    frames: admissionFrames(
      "step-2",
      ["controller"],
      { before: ["total=10", "id=0", "count=3"], after: ["owned=div_ceil(10,3)=4"], invariant: "10 tiles as 4+3+3." },
      { before: ["total=2", "id=2", "count=3"], after: ["owned=0"], invariant: "Empty slices are represented exactly.", emitted: "zero share" },
    ),
    predecessors: ["modulo-cell-ownership"],
    successors: ["scheduled-session-slicing", "capacity-rate-ramp-slicing"],
    routeTags: ["ownership", "tiling", "arithmetic"],
  },
  {
    id: "shared-seed-resolution",
    chapter: "ownership",
    title: "Resolve one shared seed",
    status: "built",
    summary:
      "The controller returns no injected seed when one is authored; otherwise DefaultHasher hashes benchmark_id, or a fixed fallback identity, once for every cell.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1925, endLine: 1946, symbol: "resolve_cellular_seed" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "DefaultHasher benchmark_id derivation at lines 1930-1945", kind: "boundary" },
      { path: "rust/e2e-tests/tests/test_cellular.rs", symbol: "test_cellular_autoderives_seed_when_absent", kind: "e2e" },
    ],
    inputs: ["optional run.random_seed", "run.benchmark_id"],
    outputs: ["None for an authored seed; otherwise Some(DefaultHasher result)"],
    state: ["controller-owned request before envelope cloning"],
    invariants: ["Authored seed is inherited verbatim because no injection occurs.", "Derived seed depends only on benchmark_id or the fixed fallback."],
    complexity: { time: "O(benchmark_id length)", memory: "O(1)" },
    gates: ["derive only when random_seed is absent"],
    failures: ["Missing/non-string benchmark_id hashes the literal aiperf-cellular; the hasher result is not zero-remapped."],
    pseudocode: pseudocode(
      "if run.random_seed is a u64, return None",
      "identity = run.benchmark_id string or aiperf-cellular",
      "seed = DefaultHasher(identity).finish()",
      "return Some(seed) for injection into each cell envelope",
    ),
    frames: admissionFrames(
      "step-3",
      ["controller"],
      { before: ["random_seed absent", "benchmark_id=run-7"], after: ["Some(hash(run-7)) injected"], invariant: "Every cell receives the same hash result." },
      { before: ["random_seed=42"], after: ["None; envelope clone retains 42"], invariant: "Authored determinism is not rewritten.", emitted: "inherit authored seed" },
    ),
    predecessors: ["cellular-run-shape-validation"],
    successors: ["cell-envelope-construction", "conversation-ownership"],
    routeTags: ["ownership", "seed", "determinism"],
  },
  {
    id: "cell-envelope-construction",
    chapter: "ownership",
    title: "Construct a cell-local envelope",
    status: "built",
    summary:
      "The controller clones the run, rewrites the cell artifact directory, optionally injects the shared seed, forces cells to one, uniformly divides workers, and slices phase controls.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1013, endLine: 1122, symbol: "build_cell_envelope" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "build_cell_envelope", kind: "integration" },
      { path: "rust/e2e-tests/tests/test_cellular.rs", symbol: "test_cellular_run_from_python_frontend", kind: "e2e" },
    ],
    inputs: ["shared envelope", "cell_id", "cell_count", "cell artifact directory", "optional injected seed"],
    outputs: ["one isolated cell BenchmarkRun"],
    state: ["cloned JSON config"],
    invariants: ["runtime.cells=1 prevents recursive promotion.", "Uniform worker division preserves the two-level partition assumption."],
    complexity: { time: "O(request size)", memory: "O(request size) per cell" },
    gates: ["valid cell identity and admitted shared request"],
    failures: ["Missing run object or phases array returns contextual construction errors; non-object phase entries are skipped."],
    pseudocode: pseudocode(
      "clone envelope; rewrite run.artifact_dir and optionally run.random_seed",
      "set runtime.cells=1 and workers=max(1, workers/cell_count)",
      "slice scheduled requests/sessions and all phase capacity/rate controls",
      "return the rewritten envelope; identity arrives separately through launcher environment",
      "return the cell-local request",
    ),
    frames: admissionFrames(
      "step-3",
      ["controller", "wire", "cell"],
      { before: ["workers=8", "requests=10", "id=1", "count=3"], after: ["workers=2", "requests=3", "runtime.cells=1"], invariant: "Artifact and execution controls become cell-local." },
      { before: ["phase entry is not an object"], after: ["entry is skipped unchanged", "construction continues"], invariant: "Only a missing run object or phases array errors.", emitted: "phase skipped" },
    ),
    predecessors: ["shared-seed-resolution"],
    successors: ["scheduled-session-slicing", "capacity-rate-ramp-slicing", "phase-ordinal-bases"],
    routeTags: ["ownership", "envelope", "projection"],
  },
  {
    id: "scheduled-session-slicing",
    chapter: "ownership",
    title: "Slice request or conversation budgets",
    status: "built",
    summary:
      "Scheduled phases replace global request counts or multi-turn session counts with the exact owned-position count for the target cell.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1061, endLine: 1097, symbol: "build_cell_envelope phase slicing" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "session slicing and debug tiling assertion at lines 1074-1097", kind: "boundary" },
      { path: "rust/e2e-tests/tests/test_cellular_multiturn.rs", symbol: "test_cellular_multi_turn_exact_fold_matches_single_cell", kind: "e2e" },
    ],
    inputs: ["phase requests or sessions", "cell_id", "cell_count", "run kind"],
    outputs: ["cell-local finite budget"],
    state: ["cell envelope phase object"],
    invariants: ["Local budgets sum to the global budget.", "Multi-turn work is partitioned per conversation, never per turn."],
    complexity: { time: "O(phases)", memory: "O(1) beyond envelope" },
    gates: ["every present requests field is sliced", "scheduled sessions are sliced", "graph sessions are left whole"],
    failures: ["This helper does not reject requests-only multi-turn phases; upstream budget validation accepts requests or sessions and rejects only when neither valid bound is present."],
    pseudocode: pseudocode(
      "for requests: local = id >= total ? 0 : div_ceil(total - id, count)",
      "for scheduled sessions: assert sum_k owned_positions(total,k,count) == total",
      "replace global sessions with owned_positions(total,id,count)",
      "leave graph session budgets global because graph partitions traces internally",
    ),
    frames: admissionFrames(
      "step-3",
      ["controller"],
      { before: ["sessions=10", "id=0", "count=3"], after: ["local sessions=4"], invariant: "Conversation slices tile exactly." },
      { before: ["multi-turn phase has requests=10 and no sessions", "id=1", "count=3"], after: ["local requests=3", "phase accepted"], invariant: "Requests-only is a valid fixed turn budget even for a multi-turn dataset.", emitted: "request slice", activeLineId: "step-1" },
    ),
    predecessors: ["cell-envelope-construction", "owned-positions-tiling", "scheduled-budget-validation"],
    successors: ["conversation-ownership"],
    routeTags: ["ownership", "sessions", "budget"],
  },
  {
    id: "capacity-rate-ramp-slicing",
    chapter: "ownership",
    title: "Slice capacity, rate, and ramp targets",
    status: "approximate",
    summary:
      "Concurrency caps use owned-position arithmetic with a minimum of one; rate divides evenly, and each unchanged-duration ramp targets those sliced values.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1098, endLine: 1119, symbol: "build_cell_envelope capacity and rate slicing" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "ramp admission rationale at lines 1660-1665", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "warn_cellular_approximations ramp warning at lines 1968-1983", kind: "integration" },
    ],
    inputs: ["concurrency", "prefill_concurrency", "rate", "ramp specs", "cell_id", "cell_count"],
    outputs: ["cell-local caps and offered rate"],
    state: ["cell envelope phase controls"],
    invariants: ["Rate shares sum to global rate up to FP rounding.", "Every cell retains capacity to progress."],
    complexity: { time: "O(phases)", memory: "O(1)" },
    gates: ["scheduled validation has already required each configured concurrency/prefill cap >= cell_count"],
    failures: ["Caps below cell_count are rejected before slicing; admitted ramps are warned because aggregate start is near cell_count rather than one."],
    pseudocode: pseudocode(
      "before construction, require each configured concurrency/prefill cap >= cell_count",
      "local_cap = max(1, id >= total ? 0 : div_ceil(total - id, count))",
      "local_rate = global_rate / cell_count",
      "keep ramp duration and strategy unchanged",
      "ramp each cell toward its sliced cap or rate target",
    ),
    frames: admissionFrames(
      "step-2",
      ["controller", "cell"],
      { before: ["concurrency=8", "id=1", "count=3"], after: ["local cap=3"], invariant: "Caps tile to eight before minimum-one edge cases." },
      { before: ["concurrency=2", "count=3"], after: ["rejected by scheduled phase validation"], invariant: "The minimum-one over-subscription case never reaches envelope slicing." },
    ),
    predecessors: ["cell-envelope-construction", "owned-positions-tiling"],
    successors: [],
    routeTags: ["ownership", "capacity", "rate", "ramp", "approximation"],
  },
  {
    id: "phase-ordinal-bases",
    chapter: "ownership",
    title: "Allocate disjoint phase ordinal bases",
    status: "built",
    summary:
      "Phases execute in array order; each base is the running sum of every prior phase's requests, while a session-only phase contributes zero.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 2006, endLine: 2037, symbol: "phase_ordinal_bases" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "phase_ordinal_bases", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "LocalLauncher::cell_command injects CELL_PHASE_ORDINAL_BASES_ENV at lines 127-128", kind: "integration" },
    ],
    inputs: ["ordered phases with names and optional requests"],
    outputs: ["phase-to-u64 base map"],
    state: [],
    invariants: ["Each base equals the sum of all prior request counts.", "Session-only exact-fold phases do not advance the retain-path base."],
    complexity: { time: "O(phases)", memory: "O(active phases)" },
    gates: ["finite validated phase budgets"],
    failures: ["A phase without a name returns an error; session-only phases intentionally contribute zero rather than failing."],
    pseudocode: pseudocode(
      "base=0",
      "for each phase in array order: require name and store bases[name]=base",
      "base += phase.requests as u64, or +=0 when requests is absent",
      "return all named bases",
    ),
    frames: admissionFrames(
      "step-2",
      ["controller", "cell"],
      { before: ["warmup requests=12", "profiling follows"], after: ["bases={warmup:0, profiling:12}"], invariant: "All prior request counts contribute." },
      { before: ["warmup sessions=4", "profiling requests=9"], after: ["bases={warmup:0, profiling:0}"], invariant: "Session-only exact-fold work contributes zero to retain ordinals.", emitted: "zero request contribution" },
    ),
    predecessors: ["cell-envelope-construction"],
    successors: ["direct-issuance-authority", "cellular-issuance-authority"],
    routeTags: ["ownership", "ordinal", "phase"],
  },
  {
    id: "direct-issuance-authority",
    chapter: "ownership",
    title: "Issue direct dense ordinals",
    status: "built",
    summary:
      "The stateless direct authority ignores phase-local inputs and returns the caller's cumulative flat_local ordinal unchanged.",
    source: { path: "rust/runtime/src/cellular/issuance.rs", startLine: 60, endLine: 91, symbol: "DirectIssuanceAuthority" },
    evidence: [
      { path: "rust/runtime/src/cellular/issuance.rs", symbol: "direct_issuer_is_the_cumulative_slot_over_the_cell_of_one", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/cellular_cell.rs", symbol: "issuance_authority_from_env direct branch at lines 188-198", kind: "integration" },
    ],
    inputs: ["flat_local", "ignored phase_ordinal_base", "ignored within_phase_local"],
    outputs: ["dense global ordinal"],
    state: ["immutable identity ModuloCellPartition(0,1)"],
    invariants: ["global_ordinal = flat_local.", "The authority owns no issuance counter."],
    complexity: { time: "O(1) per turn", memory: "O(1)" },
    gates: ["issuance_authority_from_env selects Direct when ModuloCellPartition::from_env returns None", "issuance_authority_for always selects CellularAutonomousIssuer, including for an explicit identity partition"],
    failures: ["No arithmetic failure occurs in the authority; the caller is responsible for supplying a valid cumulative flat_local."],
    pseudocode: pseudocode(
      "ignore phase_ordinal_base and within_phase_local",
      "return flat_local",
    ),
    frames: admissionFrames(
      "step-1",
      ["worker"],
      { before: ["flat_local=12", "base=99", "within=7"], after: ["global=12"], invariant: "Only cumulative flat_local controls identity issuance." },
      { before: ["flat_local sequence repeats 12"], after: ["global repeats 12"], invariant: "Stateless authority does not repair an invalid caller sequence.", emitted: "identity mapping" },
    ),
    predecessors: ["phase-ordinal-bases"],
    successors: [],
    routeTags: ["ownership", "issuance", "direct"],
  },
  {
    id: "cellular-issuance-authority",
    chapter: "ownership",
    title: "Issue interleaved global ordinals",
    status: "built",
    summary:
      "A stateless cellular authority maps the caller's within_phase_local through its already-validated partition: base + local*count + id.",
    source: { path: "rust/runtime/src/cellular/issuance.rs", startLine: 93, endLine: 132, symbol: "CellularAutonomousIssuer" },
    evidence: [
      { path: "rust/runtime/src/cellular/issuance.rs", symbol: "cellular_issuers_tile_the_dense_ordinal_space_across_phase_bases", kind: "unit" },
      { path: "rust/runtime/src/cellular/issuance.rs", symbol: "cellular_ordinal_is_the_phase_base_plus_instance_index_for_round_robin", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/cellular_cell.rs", symbol: "issuance_authority_from_env / issuance_authority_for at lines 188-211", kind: "integration" },
    ],
    inputs: ["ignored flat_local", "phase_ordinal_base", "caller within_phase_local", "validated partition"],
    outputs: ["globally unique dispatch ordinal"],
    state: ["immutable validated ModuloCellPartition"],
    invariants: ["global = phase_base + within_phase_local * cell_count + cell_id.", "The authority owns no counter."],
    complexity: { time: "O(1) per turn", memory: "O(1)" },
    gates: ["ModuloCellPartition was validated before issuer construction"],
    failures: ["The issuer constructor cannot reject because it receives a validated partition; invalid caller-local sequencing can duplicate/gap ordinals, and unchecked usize multiply/add can panic with overflow checks or wrap without them."],
    pseudocode: pseudocode(
      "read cell_id and cell_count from the already-validated partition",
      "global = phase_base + within_phase_local * cell_count + cell_id",
      "return global",
    ),
    frames: admissionFrames(
      "step-2",
      ["cell", "worker"],
      { before: ["base=12", "id=1", "count=3", "within=1"], after: ["global=16"], invariant: "Caller-local index maps to the owned interleave." },
      { before: ["phase_base=usize::MAX", "within=1", "count=2"], after: ["usize arithmetic overflows: checked panic or unchecked wrap"], invariant: "The formula has no checked overflow result.", emitted: "overflow boundary" },
    ),
    predecessors: ["phase-ordinal-bases", "modulo-cell-ownership"],
    successors: [],
    routeTags: ["ownership", "issuance", "cellular"],
  },
  {
    id: "multi-turn-detection",
    chapter: "ownership",
    title: "Detect conversation-owned scheduled work",
    status: "built",
    summary:
      "Any non-graph dataset not proven single-turn, or any phase carrying sessions, is classified as multi-turn cellular work.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1174, endLine: 1233, symbol: "dataset_is_single_turn / cellular_run_is_multi_turn" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "single/multi-turn format allowlists at lines 1149-1172", kind: "boundary" },
      { path: "rust/e2e-tests/tests/test_cellular_multiturn.rs", symbol: "module contract at lines 4-22", kind: "e2e" },
    ],
    inputs: ["dataset type/format/turns", "phase sessions fields"],
    outputs: ["multi-turn boolean"],
    state: [],
    invariants: ["Unknown linear formats fail toward multi-turn conservatism.", "Graph datasets use whole-trace ownership instead."],
    complexity: { time: "O(datasets + phases)", memory: "O(1)" },
    gates: ["scheduled dataset has multi-turn evidence or sessions budget"],
    failures: ["Unknown/unwired file formats are not assumed single-turn; incompatible retain mode is rejected later."],
    pseudocode: pseudocode(
      "dataset_multi = any non-graph dataset not proven strictly single-turn",
      "session_bound = any active phase contains sessions",
      "return dataset_multi or session_bound",
    ),
    frames: admissionFrames(
      "step-3",
      ["controller"],
      { before: ["inputs_json with authored payload arrays"], after: ["multi_turn=true"], invariant: "Conversation is the ownership unit." },
      { before: ["synthetic turns fixed at 1", "no sessions"], after: ["multi_turn=false"], invariant: "Single-turn retains request ownership." },
    ),
    predecessors: ["scheduled-budget-validation"],
    successors: ["scheduled-session-slicing", "conversation-ownership", "execution-merge-backstops"],
    routeTags: ["ownership", "multi-turn", "classification"],
  },
  {
    id: "conversation-ownership",
    chapter: "ownership",
    title: "Yield only owned conversation draws",
    status: "built",
    summary:
      "PartitionedSampler advances through the shared deterministic draw sequence and returns only positions owned by its modulo partition.",
    source: { path: "rust/runtime/src/dataset/sampler.rs", startLine: 269, endLine: 333, symbol: "PartitionedSampler" },
    evidence: [
      { path: "rust/runtime/src/dataset/sampler.rs", symbol: "partitioned_sampler_yields_disjoint_owned_positions", kind: "unit" },
      { path: "rust/e2e-tests/tests/test_cellular_multiturn.rs", symbol: "test_cellular_multi_turn_exact_fold_matches_single_cell", kind: "e2e" },
    ],
    inputs: ["deterministic inner Sampler", "ModuloCellPartition"],
    outputs: ["next owned SessionId"],
    state: ["inner sampler", "replicated global draw position"],
    invariants: ["Every inner draw advances position.", "Sequential/shuffle cells partition conversation positions without collision."],
    complexity: { time: "Amortized O(cell_count) draws per yield", memory: "O(inner sampler state)" },
    gates: ["cell_count>1 applies wrapper; multi-turn random replacement is rejected"],
    failures: ["Empty concrete samplers are rejected; random replacement cannot guarantee supported multi-turn ownership parity."],
    pseudocode: pseudocode(
      "loop: id = inner.next()",
      "owned = position % cell_count == cell_id",
      "position = position + 1",
      "if owned, return id; otherwise continue",
    ),
    frames: admissionFrames(
      "step-2",
      ["cell", "worker"],
      { before: ["id=1", "count=3", "position=4"], after: ["conversation yielded", "position=5"], invariant: "Only the modulo owner returns the draw." },
      { before: ["id=1", "count=3", "position=5"], after: ["draw skipped", "loop continues"], invariant: "Skipping still advances the shared position replica." },
    ),
    predecessors: ["modulo-cell-ownership", "shared-seed-resolution", "multi-turn-detection", "scheduled-session-slicing"],
    successors: [],
    routeTags: ["ownership", "conversation", "sampler"],
  },
];

const CONTROL_ALGORITHMS: readonly AlgorithmDefinition[] = [
  {
    id: "broadcast-attach-replay",
    chapter: "control",
    title: "Attach a consumer at the replay+live seam",
    status: "built",
    summary:
      "Under one Mutex, attach snapshots the current history as replay events (plus a trailing Finalized if already sealed) and registers a fresh live sender atomically, so no item slips between the snapshot and the live registration.",
    source: { path: "rust/runtime/src/cellular/broadcast.rs", startLine: 132, endLine: 152, symbol: "Broadcast::attach" },
    evidence: [
      { path: "rust/runtime/src/cellular/broadcast.rs", symbol: "single-lock snapshot + finalized short-circuit at lines 133-151", kind: "boundary" },
      { path: "rust/runtime/src/cellular/broadcast.rs", symbol: "replay_plus_live_reconstructs_full_order_for_every_attach_time", kind: "unit" },
      { path: "rust/runtime/src/cellular/broadcast.rs", symbol: "no_gap_or_duplicate_at_the_attach_seam", kind: "unit" },
    ],
    inputs: ["producer-owned Broadcast<T>"],
    outputs: ["Subscription<T> { replay events, live receiver }"],
    state: ["Mutex<Inner { history, finalized, senders }>"],
    invariants: [
      "The snapshot and live registration happen under one lock, so replay concat live reconstructs the full add order.",
      "An already-finalized broadcast returns replay + Finalized and a dead live channel (no live sender registered).",
    ],
    complexity: { time: "O(history): clones the history vector under the lock", memory: "O(history): the replay snapshot" },
    gates: ["lock acquired (a poisoned lock is recovered via into_inner, never surfaced)"],
    failures: ["No fallible path; a consumer that never drains its live channel blocks only itself, never the producer or peers."],
    pseudocode: pseudocode(
      "lock inner",
      "replay = history.iter().cloned().map(Item)",
      "if finalized: push Finalized, return dead-live Subscription",
      "else register a fresh sender and return { replay, live }",
    ),
    frames: [
      {
        id: "live-attach",
        label: "Attach before finalize",
        activeLineId: "step-4",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["history = [0, 1]", "finalized = false"],
        after: ["replay = [0, 1]", "live = open sender registered"],
        emitted: "subscription",
        invariantChecks: ["Snapshot + registration under one lock: no item lands between them."],
      },
      {
        id: "post-finalize-attach",
        label: "Attach after finalize",
        activeLineId: "step-3",
        activeActors: ["controller", "cell"],
        activeLinks: [],
        before: ["history = [0, 1, 2]", "finalized = true"],
        after: ["replay = [0, 1, 2, Finalized]", "live = dead channel"],
        emitted: "replay-only",
        invariantChecks: ["A late attach still reconstructs the full order plus the terminal."],
      },
    ],
    predecessors: [],
    successors: ["broadcast-add-fanout", "broadcast-finalize", "phaser-generation-advance", "dataset-velo-replay-live"],
    routeTags: ["control", "broadcast", "replay-on-attach"],
  },
  {
    id: "broadcast-add-fanout",
    chapter: "control",
    title: "Append and fan out one item",
    status: "built",
    summary:
      "Under the same lock as attach, add appends the item to history and fans it out to every live sender, retaining only senders whose send succeeds; it is a no-op returning false once finalized.",
    source: { path: "rust/runtime/src/cellular/broadcast.rs", startLine: 158, endLine: 168, symbol: "Broadcast::add" },
    evidence: [
      { path: "rust/runtime/src/cellular/broadcast.rs", symbol: "finalized guard + retain-on-send-ok at lines 159-167", kind: "boundary" },
      { path: "rust/runtime/src/cellular/broadcast.rs", symbol: "a_dropped_consumer_does_not_block_the_producer_or_others", kind: "unit" },
      { path: "rust/runtime/src/cellular/broadcast.rs", symbol: "add_after_finalize_is_rejected_and_finalize_is_idempotent", kind: "unit" },
    ],
    inputs: ["item: T (Clone)"],
    outputs: ["bool accepted"],
    state: ["Mutex<Inner { history, finalized, senders }>"],
    invariants: [
      "History grows only while not finalized; every accepted item is appended before fan-out.",
      "A closed receiver is pruned by retain, so one dropped consumer never blocks the producer or peers.",
    ],
    complexity: { time: "O(senders): one clone + send per live consumer", memory: "O(1) amortized beyond the history append" },
    gates: ["not finalized"],
    failures: ["Adding after finalize returns false (safe no-op, not a panic); a send failure silently prunes that receiver."],
    pseudocode: pseudocode(
      "lock inner",
      "if finalized: return false",
      "history.push(item.clone())",
      "senders.retain(|tx| tx.send(Item(item.clone())).is_ok())",
    ),
    frames: [
      {
        id: "normal-add",
        label: "Fan out to live consumers",
        activeLineId: "step-4",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["senders = [a, b]", "history = [0]"],
        after: ["history = [0, 1]", "a and b received Item(1)"],
        emitted: "accepted",
        invariantChecks: ["Every live consumer sees the item after it is appended."],
      },
      {
        id: "add-after-finalize",
        label: "Add on a sealed broadcast",
        activeLineId: "step-2",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["finalized = true"],
        after: ["history unchanged"],
        emitted: "false (rejected)",
        invariantChecks: ["A sealed producer never appends."],
      },
    ],
    predecessors: ["broadcast-attach-replay"],
    successors: ["broadcast-finalize", "phaser-generation-advance", "dataset-chunk-publish"],
    routeTags: ["control", "broadcast", "fan-out"],
  },
  {
    id: "broadcast-finalize",
    chapter: "control",
    title: "Seal the broadcast idempotently",
    status: "built",
    summary:
      "finalize sets the sealed flag and drains every live sender, sending the terminal Finalized to each; after this add is a no-op and a late attach replays history plus Finalized. Idempotent.",
    source: { path: "rust/runtime/src/cellular/broadcast.rs", startLine: 173, endLine: 182, symbol: "Broadcast::finalize" },
    evidence: [
      { path: "rust/runtime/src/cellular/broadcast.rs", symbol: "idempotent guard + drain-and-send-Finalized at lines 174-181", kind: "boundary" },
      { path: "rust/runtime/src/cellular/broadcast.rs", symbol: "add_after_finalize_is_rejected_and_finalize_is_idempotent", kind: "unit" },
      { path: "rust/runtime/src/cellular/broadcast.rs", symbol: "replay_plus_live_reconstructs_full_order_for_every_attach_time", kind: "unit" },
    ],
    inputs: ["existing live senders"],
    outputs: ["terminal Finalized fanned out to every live consumer"],
    state: ["Mutex<Inner { finalized, senders }>"],
    invariants: [
      "Finalize is idempotent: a second call short-circuits and re-sends nothing.",
      "Every live consumer's stream terminates so its next/await completes.",
    ],
    complexity: { time: "O(senders): one terminal send per live consumer", memory: "O(1)" },
    gates: ["not already finalized"],
    failures: ["A closed receiver's terminal send is ignored; no error is surfaced."],
    pseudocode: pseudocode(
      "lock inner",
      "if finalized: return (idempotent)",
      "finalized = true",
      "for tx in senders.drain(..): tx.send(Finalized)",
    ),
    frames: [
      {
        id: "first-finalize",
        label: "First finalize drains senders",
        activeLineId: "step-4",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["finalized = false", "senders = [a, b]"],
        after: ["finalized = true", "a and b received Finalized", "senders drained"],
        emitted: "terminal",
        invariantChecks: ["Every consumer stream terminates."],
      },
      {
        id: "second-finalize",
        label: "Second finalize is a no-op",
        activeLineId: "step-2",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["finalized = true"],
        after: ["no re-send"],
        emitted: "no-op",
        invariantChecks: ["Finalize is idempotent."],
      },
    ],
    predecessors: ["broadcast-add-fanout"],
    successors: ["phaser-generation-advance", "dataset-finalize"],
    routeTags: ["control", "broadcast", "terminal"],
  },
  {
    id: "phaser-generation-advance",
    chapter: "control",
    title: "Advance the monotonic generation",
    status: "built",
    summary:
      "advance does fetch_add(1, SeqCst) on the AtomicU64 then broadcasts a PhaseEvent carrying the new generation and its transition, returning the generation. The counter is monotonic and never resets, so consumers gate on >=.",
    source: { path: "rust/runtime/src/cellular/phaser.rs", startLine: 83, endLine: 90, symbol: "Phaser::advance" },
    evidence: [
      { path: "rust/runtime/src/cellular/phaser.rs", symbol: "fetch_add(1, SeqCst) + broadcast.add(PhaseEvent) at lines 84-89", kind: "boundary" },
      { path: "rust/runtime/src/cellular/phaser.rs", symbol: "generations_are_monotonic_and_carry_transitions", kind: "unit" },
    ],
    inputs: ["transition: PhaseTransition"],
    outputs: ["u64 new generation", "broadcast PhaseEvent { generation, transition }"],
    state: ["Arc<AtomicU64> generation", "Broadcast<PhaseEvent>"],
    invariants: [
      "Generation strictly increases by exactly one per advance and never resets.",
      "Each broadcast event carries its own generation and transition.",
    ],
    complexity: { time: "O(subscribers): the underlying broadcast fan-out", memory: "O(1) beyond the broadcast history append" },
    gates: ["broadcast not finalized (else the add is a silent no-op)"],
    failures: ["advance after finalize still increments the counter but the broadcast.add is a no-op, so a post-seal generation is unobservable; finalize is the intended last transition (advance(Done) then seal)."],
    pseudocode: pseudocode(
      "generation = generation.fetch_add(1, SeqCst) + 1",
      "broadcast.add(PhaseEvent { generation, transition })",
      "return generation",
    ),
    frames: [
      {
        id: "started",
        label: "Advance to Started",
        activeLineId: "step-1",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["generation = 0"],
        after: ["generation = 1", "broadcast Item(PhaseEvent { 1, Started })"],
        emitted: "generation 1",
        invariantChecks: ["Monotonic +1 increment."],
      },
      {
        id: "sealed-advance",
        label: "Advance on a sealed broadcast",
        activeLineId: "step-2",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["broadcast finalized", "generation = 3"],
        after: ["generation = 4", "broadcast.add is a no-op"],
        emitted: "counter++ but unobserved",
        invariantChecks: ["A post-seal generation is not delivered to any subscriber."],
      },
    ],
    predecessors: ["broadcast-add-fanout", "synchronized-start"],
    successors: ["phaser-await-generation", "phaser-start", "controller-fanout-generation"],
    routeTags: ["control", "phaser", "generation"],
  },
  {
    id: "phaser-await-generation",
    chapter: "control",
    title: "Await a target generation with a >= gate",
    status: "built",
    summary:
      "await_generation returns Ok as soon as the highest observed generation is >= target (replay fast-path for an already-passed target), pulling replay-then-live events; it returns Err(PhaserClosed) if the stream finalizes before the target is reached.",
    source: { path: "rust/runtime/src/cellular/phaser.rs", startLine: 187, endLine: 197, symbol: "PhaserSubscription::await_generation" },
    evidence: [
      { path: "rust/runtime/src/cellular/phaser.rs", symbol: "seen_generation >= target fast-path + >= loop gate at lines 188-196", kind: "boundary" },
      { path: "rust/runtime/src/cellular/phaser.rs", symbol: "await_generation_returns_from_replay_for_already_passed_targets", kind: "unit" },
      { path: "rust/runtime/src/cellular/phaser.rs", symbol: "await_generation_blocks_then_wakes_on_live_advance", kind: "unit" },
      { path: "rust/runtime/src/cellular/phaser.rs", symbol: "await_after_finalize_errors_if_target_never_reached", kind: "unit" },
    ],
    inputs: ["target: u64", "replay snapshot + live receiver (via next)"],
    outputs: ["Ok(()) on reaching target", "Err(PhaserClosed) on early finalize"],
    state: ["cursor into replay", "seen_generation (highest observed)", "finalized flag"],
    invariants: [
      "Reached means observing any event with generation >= target (monotonic, cyclic-safe: never equality).",
      "An already-passed target returns immediately from the replay-tracked seen_generation.",
    ],
    complexity: { time: "O(events until target or finalize)", memory: "O(1) cursor state" },
    gates: ["seen_generation >= target, or a pulled event's generation >= target"],
    failures: ["A stream finalized before target yields Err(PhaserClosed) (an aborted run); a target never reached on a live-only stream blocks until finalize, then errors — it never hangs past seal."],
    pseudocode: pseudocode(
      "if seen_generation >= target: return Ok(())",
      "while let Some(event) = next().await:",
      "  if event.generation >= target: return Ok(())",
      "return Err(PhaserClosed)  // stream finalized first",
    ),
    frames: [
      {
        id: "replay-passed",
        label: "Target already passed (replay)",
        activeLineId: "step-1",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["seen_generation = 3", "target = 2"],
        after: ["Ok(()) without pulling live"],
        emitted: "immediate Ok",
        invariantChecks: ["generation >= target satisfied from replay, no equality."],
      },
      {
        id: "finalize-before-target",
        label: "Abort before reaching target",
        activeLineId: "step-4",
        activeActors: ["cell", "controller"],
        activeLinks: [],
        before: ["seen_generation = 1", "target = 5", "stream finalizes"],
        after: ["next() drains to None"],
        emitted: "Err(PhaserClosed)",
        invariantChecks: ["Finalization surfaces as an error, never an indefinite hang."],
      },
    ],
    predecessors: ["phaser-generation-advance"],
    successors: ["phaser-late-attach", "dispatch-on-issue"],
    routeTags: ["control", "phaser", "await"],
  },
  {
    id: "phaser-late-attach",
    chapter: "control",
    title: "Subscribe to the phaser late over velo",
    status: "feature-gated",
    summary:
      "The velo phaser distribution: a cell registers the event push handler BEFORE subscribing, the controller attaches under the broadcast lock, ships the replay snapshot in the unary reply, and spawns a pump forwarding the live tail — so a generation advanced concurrently with subscribe lands in exactly one of {reply snapshot, pushed live}.",
    source: { path: "rust/runtime/src/cellular/transport/phaser_velo.rs", startLine: 133, endLine: 181, symbol: "PhaserClient::subscribe" },
    evidence: [
      { path: "rust/runtime/src/cellular/transport/phaser_velo.rs", symbol: "PhaserServer::bind attach_raw + spawn pump at lines 61-119", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/phaser_velo.rs", symbol: "register event handler before register_peer + unary subscribe at lines 137-171", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/phaser_velo.rs", symbol: "cell_observes_replay_then_live_generations_over_velo", kind: "integration" },
    ],
    inputs: ["cell velo Arc<Velo>", "controller PeerInfo"],
    outputs: ["PhaserSubscription { replay = reply snapshot, live = push channel }"],
    state: ["controller-side broadcast attach seam", "per-cell pump task"],
    invariants: [
      "The event handler is registered before the subscribe request, so no pushed live event is dropped.",
      "A late attach still reaches every already-advanced generation via the replay snapshot.",
    ],
    complexity: { time: "O(replay): the snapshot shipped in the reply + live tail", memory: "O(replay) reply body (raw MessagePack)" },
    gates: ["velo feature compiled", "controller phaser bound"],
    failures: ["The pump task (not the run) ends when the broadcast finalizes or am_send errors (the cell went away); a decode failure fails the subscribe."],
    pseudocode: pseudocode(
      "cell: register HANDLER_PHASER_EVENT -> local channel",
      "cell: register_peer(controller); unary HANDLER_PHASER_SUBSCRIBE(cell_peer)",
      "controller: register_peer(cell); { replay, live } = phaser.attach_raw()",
      "controller: spawn pump forwarding live -> am_send(HANDLER_PHASER_EVENT); reply { replay }",
      "cell: reconstruct PhaserSubscription { replay, live }",
    ),
    frames: [
      {
        id: "replay-reaches",
        label: "Late subscribe replays passed generations",
        activeLineId: "step-4",
        activeActors: ["controller", "wire", "cell"],
        activeLinks: [],
        before: ["controller advanced to generation 2 before subscribe"],
        after: ["reply.replay carries generations up to 2"],
        emitted: "replay snapshot",
        invariantChecks: ["A late attach still observes every passed generation."],
      },
      {
        id: "live-pushed",
        label: "Later generation pushed live",
        activeLineId: "step-4",
        activeActors: ["controller", "wire", "cell"],
        activeLinks: [],
        before: ["cell subscribed at generation 2"],
        after: ["controller advances to 3", "pump pushes PhaseEvent(3) to the cell channel"],
        emitted: "live event",
        invariantChecks: ["A concurrent generation lands in exactly one of reply or live, never both or neither."],
      },
    ],
    predecessors: ["phaser-await-generation", "velo-peer-connect", "handler-registration"],
    successors: ["synchronized-start", "controller-fanout-generation"],
    routeTags: ["control", "phaser", "velo", WIRE_FACTS.phaserVelo],
  },
  {
    id: "velo-controller-bind",
    chapter: "control",
    title: "Bind the controller transport and handlers",
    status: "feature-gated",
    summary:
      "bind_controller registers the register/heartbeat/partition/store_partition handlers over raw-MessagePack velo payloads and exposes a merged recv stream plus an all-registered barrier; the register handler counts the Nth cell and returns its sliced spec plus the START handle.",
    source: { path: "rust/runtime/src/cellular/transport/velo_transport.rs", startLine: 81, endLine: 216, symbol: "VeloControllerTransport::bind_controller" },
    evidence: [
      { path: "rust/runtime/src/cellular/transport/velo_transport.rs", symbol: "HANDLER_REGISTER count-to-barrier + spec/START reply at lines 94-129", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/velo_transport.rs", symbol: "HANDLER_PARTITION / HANDLER_STORE_PARTITION register-shipper + ack at lines 158-209", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/velo_transport.rs", symbol: "cell_registers_ships_heartbeat_and_partition", kind: "integration" },
    ],
    inputs: ["velo Arc<Velo>", "spec_for(cell_id)", "cell_count", "start_event: EventHandle"],
    outputs: ["VeloControllerTransport { recv stream, all_registered barrier }"],
    state: ["mpsc receiver of decoded CellMessage", "AtomicU32 registered count", "Notify barrier"],
    invariants: [
      "Every handler body decodes and re-encodes raw MessagePack; no application zstd is used on the control plane.",
      "The Nth registration releases the all-registered barrier exactly once.",
    ],
    complexity: { time: "O(1) per handler invocation", memory: "O(channel capacity) = 1024 buffered messages" },
    gates: ["velo feature compiled", "each handler registers on velo without conflict"],
    failures: ["A missing launch spec for a registering cell_id errors the register reply; a decode failure pushes a CellTransportError::Decode into the stream."],
    pseudocode: pseudocode(
      "create mpsc(1024); registered = AtomicU32(0)",
      "register HANDLER_REGISTER: register_peer(cell); if ++registered == cell_count notify barrier; reply { spec, START }",
      "register HANDLER_HEARTBEAT: push decoded CellMessage::Heartbeat",
      "register HANDLER_PARTITION / HANDLER_STORE_PARTITION: register_peer(shipper); push; reply ack",
      "return transport { receiver, all_registered }",
    ),
    frames: [
      {
        id: "barrier-release",
        label: "Nth register releases the barrier",
        activeLineId: "step-2",
        activeActors: ["cell", "wire", "controller"],
        activeLinks: [],
        before: ["registered = cell_count - 1"],
        after: ["registered = cell_count", "all_registered notified", "reply carries spec + START"],
        emitted: "barrier released",
        invariantChecks: ["The barrier fires exactly once at the Nth registration."],
      },
      {
        id: "partition-ack",
        label: "Partition ship acked to a fresh instance",
        activeLineId: "step-4",
        activeActors: ["cell", "wire", "controller"],
        activeLinks: [],
        before: ["cell ships from a fresh velo instance"],
        after: ["controller register_peer(shipper)", "partition pushed", "ack returned"],
        emitted: "ack",
        invariantChecks: ["A ship from an unseen instance is re-registered so its ack routes back."],
      },
    ],
    predecessors: ["controller-promotion", "velo-feature-admission"],
    successors: ["handler-registration", "synchronized-start", "velo-peer-connect"],
    routeTags: ["control", "velo", "controller", WIRE_FACTS.partition],
  },
  {
    id: "velo-peer-connect",
    chapter: "control",
    title: "Dial the controller by endpoint coordinate",
    status: "feature-gated",
    summary:
      "connect_controller parses a tcp://HOST:PORT (getaddrinfo-resolved, DNS-capable) or uds://PATH coordinate and retries velo.connect every 200ms up to a 60s timeout, returning the controller's real PeerInfo — discovery-free bootstrap by coordinate alone.",
    source: { path: "rust/runtime/src/cellular/transport/connect.rs", startLine: 124, endLine: 138, symbol: "connect_controller" },
    evidence: [
      { path: "rust/runtime/src/cellular/transport/connect.rs", symbol: "retry loop with CONNECT_TIMEOUT/CONNECT_RETRY_INTERVAL at lines 126-137", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/connect.rs", symbol: "parse_endpoint tcp getaddrinfo + uds branches at lines 92-120", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/connect.rs", symbol: "connect_controller_bootstraps_by_endpoint", kind: "integration" },
      { path: "rust/runtime/src/cellular/transport/connect.rs", symbol: "parse_recognizes_tcp_and_uds_endpoints", kind: "unit" },
    ],
    inputs: ["velo instance", "coordinate string (tcp:// or uds://)"],
    outputs: ["controller PeerInfo"],
    state: ["retry deadline (now + 60s)"],
    invariants: [
      "Connection needs only the endpoint coordinate — no registry or discovery service.",
      "A DNS name resolves through to_socket_addrs, so a k8s headless-service FQDN works.",
    ],
    complexity: { time: "O(retries) bounded by 60s / 200ms", memory: "O(1)" },
    gates: ["coordinate parses to a known scheme", "velo.connect succeeds before the deadline"],
    failures: ["An unrecognized or unresolvable coordinate errors immediately; an unreachable controller retries until CONNECT_TIMEOUT then errors (timed out)."],
    pseudocode: pseudocode(
      "endpoint = parse_endpoint(coordinate)  // tcp getaddrinfo or uds",
      "deadline = now + 60s",
      "loop: match velo.connect(endpoint): Ok(peer) return peer",
      "  Err(e): if now >= deadline return Err(e); sleep 200ms",
    ),
    frames: [
      {
        id: "connect-retry",
        label: "Connect after the controller binds",
        activeLineId: "step-3",
        activeActors: ["cell", "wire"],
        activeLinks: [],
        before: ["controller listener not yet up", "3 retries elapsed"],
        after: ["velo.connect Ok", "returns controller PeerInfo"],
        emitted: "connected",
        invariantChecks: ["A pre-bind race is absorbed by bounded retry."],
      },
      {
        id: "connect-timeout",
        label: "Controller never reachable",
        activeLineId: "step-4",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["now >= deadline", "velo.connect still Err"],
        after: ["Err(...).context(timed out)"],
        emitted: "timed-out error",
        invariantChecks: ["Bootstrap fails loudly at the deadline rather than hanging."],
      },
    ],
    predecessors: ["velo-controller-bind"],
    successors: ["handler-registration", "phaser-late-attach", "dataset-velo-subscribe"],
    routeTags: ["control", "velo", "bootstrap"],
  },
  {
    id: "handler-registration",
    chapter: "control",
    title: "Register the cell and fetch its spec",
    status: "feature-gated",
    summary:
      "VeloCellClient::connect register_peers the controller, then register sends a unary CellRegister (the cell's PeerInfo + cell_id) and decodes the RegisterReply carrying its sliced execute envelope and the START event handle; every ship carries the sender's PeerInfo so the controller register_peers a fresh instance to route the ack back.",
    source: { path: "rust/runtime/src/cellular/transport/velo_transport.rs", startLine: 247, endLine: 279, symbol: "VeloCellClient::connect / register / await_start" },
    evidence: [
      { path: "rust/runtime/src/cellular/transport/velo_transport.rs", symbol: "register_peer(controller) + unary HANDLER_REGISTER at lines 247-267", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/velo_transport.rs", symbol: "ship_from_a_fresh_instance_is_acked", kind: "integration" },
      { path: "rust/runtime/src/cellular/transport/velo_transport.rs", symbol: "cell_registers_ships_heartbeat_and_partition", kind: "integration" },
    ],
    inputs: ["cell velo Arc<Velo>", "controller PeerInfo", "cell_id"],
    outputs: ["RegisterReply { sliced envelope, start_event }"],
    state: ["controller peer registered on the cell velo"],
    invariants: [
      "A reply or push routes only to a peer the responder has registered.",
      "A ship from a fresh velo instance re-registers that instance so the ack still reaches it.",
    ],
    complexity: { time: "O(1) unary round-trip", memory: "O(spec bytes) reply" },
    gates: ["velo feature compiled", "controller peer registered before the unary send"],
    failures: ["An IO error on register surfaces as CellTransportError::Io; a decode failure as ::Decode; a poisoned START event surfaces as an error to await_start."],
    pseudocode: pseudocode(
      "connect: register_peer(controller)",
      "register: cell_peer = encode(velo.peer_info()); body = encode(CellRegister { cell_id, cell_peer })",
      "  unary HANDLER_REGISTER(body) -> instance(controller)",
      "  decode RegisterReply { envelope, start_event }",
    ),
    frames: [
      {
        id: "register-reply",
        label: "Register returns spec + START",
        activeLineId: "step-3",
        activeActors: ["cell", "wire", "controller"],
        activeLinks: [],
        before: ["controller peer registered on the cell velo"],
        after: ["RegisterReply { sliced envelope, start_event }"],
        emitted: "spec + START",
        invariantChecks: ["The cell learns exactly its sliced envelope."],
      },
      {
        id: "fresh-instance-ack",
        label: "Ship from a fresh instance is acked",
        activeLineId: "step-2",
        activeActors: ["cell", "wire", "controller"],
        activeLinks: [],
        before: ["cell ships its partition from a new velo instance"],
        after: ["controller register_peers the sender", "ack routes back"],
        emitted: "ack",
        invariantChecks: ["Every ship carries its PeerInfo so the ack is routable."],
      },
    ],
    predecessors: ["velo-peer-connect", "velo-controller-bind"],
    successors: ["synchronized-start", "cell-envelope-construction"],
    routeTags: ["control", "velo", "register", WIRE_FACTS.partition],
  },
  {
    id: "synchronized-start",
    chapter: "control",
    title: "Release all cells together (default start)",
    status: "feature-gated",
    summary:
      "The default start: the controller waits (biased select against a failure and a registration timeout) for the all-registered barrier, then triggers the run-wide START event so every cell begins dispatching together; a cell awaits START via await_start.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 640, endLine: 658, symbol: "cellular start policy (default branch)" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "select! await_all_registered | failure | register_timeout then start_event.trigger() at lines 640-652", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/velo_transport.rs", symbol: "await_all_registered / await_start at lines 220-279", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/velo_transport.rs", symbol: "synchronized_start_releases_all_cells_together", kind: "integration" },
    ],
    inputs: ["all-registered barrier", "failure_rx", "register timeout", "start_event"],
    outputs: ["triggered START event (all cells resume)"],
    state: ["registered count", "start_event handle"],
    invariants: [
      "START triggers only after every cell has registered (an O(N) fan-in rendezvous).",
      "The select is biased so registration completion is checked before the failure/timeout arms.",
    ],
    complexity: { time: "O(N) fan-in over cell_count registrations", memory: "O(N) failure channel" },
    gates: ["all cell_count cells registered within register_timeout, and no cell failed first"],
    failures: ["A cell failing before registration is caught by failure_rx and bails; exceeding register_timeout bails; a poisoned START event surfaces to awaiting cells as an error."],
    pseudocode: pseudocode(
      "select biased: await_all_registered() => proceed",
      "  | failure_rx.recv() => bail",
      "  | sleep(register_timeout) => bail",
      "start_event.trigger()",
    ),
    frames: [
      {
        id: "all-registered",
        label: "All cells registered, trigger START",
        activeLineId: "step-1",
        activeActors: ["controller", "wire", "cell"],
        activeLinks: [],
        before: ["cell_count registrations complete"],
        after: ["start_event triggered", "cells wake from await_start"],
        emitted: "START",
        invariantChecks: ["Every cell begins together."],
      },
      {
        id: "register-timeout",
        label: "Registration times out",
        activeLineId: "step-3",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["fewer than cell_count registered", "register_timeout elapsed"],
        after: ["bail: cells did not all register"],
        emitted: "abort",
        invariantChecks: ["A stuck rendezvous aborts loudly, never hangs."],
      },
    ],
    predecessors: ["handler-registration", "velo-controller-bind"],
    successors: ["phaser-start", "barrier-free-launch", "phaser-generation-advance"],
    routeTags: ["control", "start", "synchronized"],
  },
  {
    id: "phaser-start",
    chapter: "control",
    title: "Drive START through the phaser (opt-in)",
    status: "feature-gated",
    summary:
      "Opt-in (AIPERF_CELL_PHASER_START): the controller constructs a Phaser and binds a PhaserServer on its velo before moving it into the transport, then drives START as generation 1 via advance(Started) through the monotonic phaser instead of (alongside) the single-shot START event.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 396, endLine: 406, symbol: "phaser bind (phaser_start.then + PhaserServer::bind)" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "phaser = phaser_start.then(Phaser::new); PhaserServer::bind at lines 396-406", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "advance(PhaseTransition::Started) at lines 656-658", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "phaser_start env flag parse at lines 205-211", kind: "boundary" },
    ],
    inputs: ["AIPERF_CELL_PHASER_START flag", "controller velo"],
    outputs: ["bound PhaserServer", "advance(Started) driving generation 1 after the rendezvous"],
    state: ["Option<Phaser>", "Option<PhaserServer> (held for the run)"],
    invariants: [
      "When enabled, the phaser is bound before the velo moves into the transport, so cells can subscribe.",
      "START is generation 1 via advance(Started); a cell registering after it sees the passed generation via replay.",
    ],
    complexity: { time: "O(1) bind + O(subscribers) advance", memory: "O(1) plus the phaser broadcast history" },
    gates: ["AIPERF_CELL_PHASER_START truthy", "velo feature compiled"],
    failures: ["Binding the phaser control plane can fail (context: binding phaser control plane); default off leaves phaser = None and only the event START fires."],
    pseudocode: pseudocode(
      "phaser = phaser_start.then(Phaser::new)",
      "if Some(phaser): PhaserServer::bind(velo.clone(), phaser.clone())",
      "... after the register rendezvous and start_event.trigger():",
      "if Some(phaser): phaser.advance(Started)  // generation 1",
    ),
    frames: [
      {
        id: "phaser-on",
        label: "Phaser-START selected",
        activeLineId: "step-2",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["AIPERF_CELL_PHASER_START=1"],
        after: ["PhaserServer bound", "advance(Started) broadcasts generation 1"],
        emitted: "generation 1",
        invariantChecks: ["Cells subscribe and wake at generation 1."],
      },
      {
        id: "phaser-off",
        label: "Default off",
        activeLineId: "step-1",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["flag unset"],
        after: ["phaser = None", "only the event START fires"],
        emitted: "no phaser",
        invariantChecks: ["Default start is unchanged when the opt-in is off."],
      },
    ],
    predecessors: ["synchronized-start", "phaser-generation-advance"],
    successors: ["controller-fanout-generation"],
    routeTags: ["control", "phaser", "start", "opt-in"],
  },
  {
    id: "barrier-free-launch",
    chapter: "control",
    title: "Trigger START without the rendezvous (opt-in)",
    status: "approximate",
    summary:
      "Opt-in tier-T3 (AIPERF_CELL_BARRIER_FREE): the controller triggers START immediately without the O(N) register rendezvous; a cell registering after the trigger sees the completed event via velo's completed-event cache. Start correlation is aggregate-equivalent (arrival-epoch jitter), not byte-identical.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 635, endLine: 652, symbol: "cellular start policy (barrier-free branch)" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "barrier_free branch logs then skips await_all_registered, else select at lines 635-649", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "barrier_free env flag parse at lines 197-203", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "CELL_BARRIER_FREE_ENV const at line 45", kind: "boundary" },
    ],
    inputs: ["AIPERF_CELL_BARRIER_FREE flag", "start_event"],
    outputs: ["immediately triggered START (no fan-in)"],
    state: ["start policy branch"],
    invariants: [
      "When enabled, START triggers without gathering all N registrations, so each cell starts on its own registration.",
      "A failed cell is still caught by the collect loop's failure watch.",
    ],
    complexity: { time: "O(1): no O(N) fan-in", memory: "O(1)" },
    gates: ["AIPERF_CELL_BARRIER_FREE truthy"],
    failures: ["Looser cross-cell start correlation (arrival-epoch jitter) is accepted as aggregate-equivalent, not an error; a hard cell failure still aborts via the failure watch."],
    pseudocode: pseudocode(
      "if barrier_free: log; skip await_all_registered",
      "else: select await_all_registered | failure | timeout",
      "start_event.trigger()  // both paths converge here",
    ),
    frames: [
      {
        id: "immediate",
        label: "Barrier-free immediate trigger",
        activeLineId: "step-1",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["AIPERF_CELL_BARRIER_FREE=1"],
        after: ["START triggered before all cells register"],
        emitted: "immediate START",
        invariantChecks: ["No O(N) rendezvous; late cells read the completed-event cache."],
      },
      {
        id: "default-rendezvous",
        label: "Default fan-in when off",
        activeLineId: "step-2",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["flag unset"],
        after: ["await_all_registered before trigger"],
        emitted: "synchronized",
        invariantChecks: ["Default is the tight synchronized start."],
      },
    ],
    predecessors: ["synchronized-start"],
    successors: ["local-cell-launch", "external-cell-launch"],
    routeTags: ["control", "start", "barrier-free", "opt-in"],
  },
  {
    id: "local-cell-launch",
    chapter: "control",
    title: "Spawn cell subprocesses locally",
    status: "built",
    summary:
      "The default LocalLauncher builds one Command per cell (current_exe --cell) injecting cell_id/cell_count/controller coordinate/phase-ordinal bases/artifact authority via env, with kill_on_drop so a controller abort SIGKILLs every cell.",
    source: { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", startLine: 105, endLine: 166, symbol: "LocalLauncher::cell_command / launch" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "env injection + kill_on_drop at lines 113-147", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "spawn loop 0..cell_count at lines 152-164", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "local_launcher_sets_cell_env", kind: "unit" },
    ],
    inputs: ["CellLaunchContext { cell_count, controller_coordinate, phase_ordinal_bases, artifact_authority }"],
    outputs: ["Vec<CellHandle> wrapping spawned children"],
    state: ["per-cell tokio::process::Command"],
    invariants: [
      "Each cell is injected its cell_id/cell_count/controller coordinate and the ordinal bases.",
      "kill_on_drop(true) means a dropped watcher SIGKILLs its cell, so a failed run leaves no load-generating orphan.",
    ],
    complexity: { time: "O(N) spawns", memory: "O(N) child handles" },
    gates: ["current_exe resolvable (else a stale fallback name is used, surfaced at spawn)"],
    failures: ["A spawn failure is contexted (spawning cell {id}) and aborts the launch; an artifact authority is injected only when present (Stage E)."],
    pseudocode: pseudocode(
      "for cell_id in 0..cell_count:",
      "  cmd = Command(current_exe).arg(--cell).env(cell_id, cell_count, controller, bases).kill_on_drop(true)",
      "  if artifact_authority: cmd.env(CELL_ARTIFACT_ADDR)",
      "  handles.push(CellHandle { child: spawn(cmd) })",
    ),
    frames: [
      {
        id: "spawn-with-env",
        label: "Spawn cell with injected env",
        activeLineId: "step-2",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["cell_count = 4"],
        after: ["4 --cell children spawned with cell_id/cell_count/coordinate/bases"],
        emitted: "children",
        invariantChecks: ["Each cell knows its identity and the controller coordinate."],
      },
      {
        id: "artifact-authority",
        label: "Stage E authority injected",
        activeLineId: "step-3",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["artifact_authority = Some(host:port)"],
        after: ["CELL_ARTIFACT_ADDR env set for HTTP artifact upload"],
        emitted: "authority",
        invariantChecks: ["The authority is injected only when HTTP artifact shipping is on."],
      },
    ],
    predecessors: ["barrier-free-launch", "phase-ordinal-bases"],
    successors: ["controller-child-arbitration", "handler-registration"],
    routeTags: ["control", "launch", "local", "same-host"],
  },
  {
    id: "external-cell-launch",
    chapter: "control",
    title: "Expect operator-created cell pods",
    status: "built",
    summary:
      "The K8sLauncher spawns nothing: the operator/JobSet already created the cell pods, so it logs the expected cell_count and returns childless CellHandles; the pods discover the controller from operator-injected env.",
    source: { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", startLine: 172, endLine: 184, symbol: "K8sLauncher::launch" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "log expected + childless CellHandle map at lines 173-183", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "select_launcher (local default, k8s) at lines 188-192", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "k8s_launcher_spawns_nothing_but_expects_all_cells", kind: "unit" },
    ],
    inputs: ["CellLaunchContext { cell_count }", "AIPERF_CELL_LAUNCHER=k8s"],
    outputs: ["cell_count childless CellHandles"],
    state: ["no owned children"],
    invariants: [
      "No local process is spawned; the operator owns pod lifecycle.",
      "A childless handle's wait_failure never resolves, so pod liveness is not this launcher's concern.",
    ],
    complexity: { time: "O(N) handle construction", memory: "O(N) childless handles" },
    gates: ["AIPERF_CELL_LAUNCHER == k8s"],
    failures: ["A pod that never registers is not caught by a child exit (there is none) but by the controller's registration/collect timeout backstop."],
    pseudocode: pseudocode(
      "log: expecting cell_count pods to register (no local spawn)",
      "return (0..cell_count).map(|id| CellHandle { child: None, id })",
    ),
    frames: [
      {
        id: "expect-pods",
        label: "Expect pods, spawn nothing",
        activeLineId: "step-2",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["AIPERF_CELL_LAUNCHER=k8s", "cell_count = 8"],
        after: ["8 childless handles returned"],
        emitted: "expected",
        invariantChecks: ["The controller waits for pod registrations, not for spawns."],
      },
      {
        id: "no-registration-backstop",
        label: "A pod never registers",
        activeLineId: "step-1",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["one pod fails to come up"],
        after: ["no child exit to observe"],
        emitted: "collect-timeout backstop",
        invariantChecks: ["The registration/collect deadline is the k8s liveness backstop."],
      },
    ],
    predecessors: ["barrier-free-launch"],
    successors: ["controller-child-arbitration"],
    routeTags: ["control", "launch", "k8s", "cross-host"],
  },
  {
    id: "controller-child-arbitration",
    chapter: "control",
    title: "Watch cells for hard failure",
    status: "feature-gated",
    summary:
      "The Velo-gated controller loop spawns one watcher per cell handle: a local child exiting non-zero forwards a failure that bails the run; a clean exit parks (pending) so the collect keeps serving; a k8s handle never resolves (the collect deadline is its backstop). The collect select is biased so a ready cell message wins a ship-then-exit race. Its CellHandle::wait_failure primitive is ungated.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 592, endLine: 620, symbol: "per-handle failure watch" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "spawn wait_failure -> failure_tx per handle at lines 596-604", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "CellHandle::wait_failure park-on-clean-exit / never-resolve-for-k8s at lines 76-93", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "biased collect select: transport message before failure/deadline at lines 688-709", kind: "boundary" },
    ],
    inputs: ["cell CellHandles", "failure_rx"],
    outputs: ["a hard-failure string that bails the run, or nothing (clean exit parks)"],
    state: ["one spawned watcher task per handle", "mpsc failure channel"],
    invariants: [
      "A non-zero exit forwards a failure; a clean exit parks so the transport keeps being served (no false failure).",
      "The collect select is biased so a cell that ships then exits has its message taken before its exit is seen.",
    ],
    complexity: { time: "O(N) watcher tasks", memory: "O(N) failure channel" },
    gates: ["velo feature compiled for the controller loop", "a handle resolves with a non-success status (local child only); CellHandle::wait_failure itself is ungated"],
    failures: ["Any cell hard failure aborts the run via failure_rx; a k8s pod that hangs is bounded by the collect timeout, not this watch."],
    pseudocode: pseudocode(
      "for handle in handles: spawn { report = handle.wait_failure().await; failure_tx.send(report) }",
      "wait_failure: clean exit -> pending() (park); non-zero -> return diagnostic; k8s (no child) -> pending()",
      "collect: select biased { message | failure => bail | deadline => bail }",
    ),
    frames: [
      {
        id: "nonzero-exit",
        label: "Cell exits non-zero",
        activeLineId: "step-2",
        activeActors: ["controller", "cell"],
        activeLinks: [],
        before: ["a cell process exits with status 1"],
        after: ["wait_failure returns a diagnostic", "failure_rx bails the run"],
        emitted: "abort",
        invariantChecks: ["A hard cell failure aborts rather than hanging the collect."],
      },
      {
        id: "clean-exit-parks",
        label: "Clean exit parks the watcher",
        activeLineId: "step-2",
        activeActors: ["controller", "cell"],
        activeLinks: [],
        before: ["a cell exits 0 after shipping"],
        after: ["wait_failure parks on pending()", "collect keeps serving the transport"],
        emitted: "no false failure",
        invariantChecks: ["A clean exit is not a failure; the ship-then-exit race resolves in the cell's favour."],
      },
    ],
    predecessors: ["local-cell-launch", "external-cell-launch"],
    successors: ["execution-merge-backstops"],
    routeTags: ["control", "failure", "watch"],
  },
];

const DISTRIBUTION_ALGORITHMS: readonly AlgorithmDefinition[] = [
  {
    id: "canonical-dataset-regeneration",
    chapter: "distribution",
    title: "Regenerate the dataset from a shared seed",
    status: "built",
    summary:
      "The canonical measured distribution: a synthetic/inline/public cell (or any same-host cell whose controller-local path is readable) ships no dataset — download_cell_dataset_if_needed returns the envelope unchanged and the cell regenerates the identical dataset space from the controller-injected shared seed via its ordinary execute path. Fan-out is NOT this path.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_cell.rs", startLine: 308, endLine: 327, symbol: "download_cell_dataset_if_needed (no-ship branches)" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_cell.rs", symbol: "from_env None + no dataset path + no authority return-unchanged at lines 311-327", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "default = per-cell seed regeneration / Stage-G serve comment at lines 212-215", kind: "boundary" },
    ],
    inputs: ["cell envelope bytes", "ModuloCellPartition::from_env", "cellular_file_dataset_path", "cell_artifact_authority"],
    outputs: ["the envelope unchanged (compile + regenerate locally)"],
    state: ["injected shared seed in the envelope"],
    invariants: [
      "Every cell composes the identical dataset space from one shared seed; ownership tiling selects each cell's slice.",
      "No dataset bytes cross the wire on this path.",
    ],
    complexity: { time: "O(1) decision (regeneration cost is the ordinary execute path)", memory: "O(1) here" },
    gates: ["is a cell (from_env Some)", "and either: no file dataset path, or no artifact authority"],
    failures: ["Not a cell (from_env None) returns unchanged (single-process path); a file/graph dataset WITH a resolvable authority instead takes the Stage-G reconstruct path (a different entry)."],
    pseudocode: pseudocode(
      "if ModuloCellPartition::from_env() is None: return envelope (single-process)",
      "if no cellular_file_dataset_path(envelope): return envelope (synthetic/inline/public -> regenerate)",
      "if cell_artifact_authority() is None: return envelope (same-host/shared-FS -> read local path)",
      "else: Stage-G fetch + reconstruct (see dataset-http-zstd-reconstruct)",
    ),
    frames: [
      {
        id: "synthetic-regenerate",
        label: "Synthetic cell regenerates",
        activeLineId: "step-2",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["dataset = synthetic", "shared seed injected"],
        after: ["envelope unchanged", "cell regenerates its owned slice from the seed"],
        emitted: "no ship",
        invariantChecks: ["Identical dataset space from one seed; no bytes on the wire."],
      },
      {
        id: "same-host-local",
        label: "Same-host file cell reads local path",
        activeLineId: "step-3",
        activeActors: ["cell", "controller"],
        activeLinks: [],
        before: ["file dataset", "no artifact authority (same host)"],
        after: ["envelope unchanged", "cell reads the controller-local path directly"],
        emitted: "no ship",
        invariantChecks: ["A readable shared path needs no HTTP transfer."],
      },
    ],
    predecessors: ["shared-seed-resolution", "cell-envelope-construction"],
    successors: ["conversation-ownership"],
    routeTags: ["distribution", "synthetic", "shared-seed", "canonical"],
  },
  {
    id: "dataset-chunk-publish",
    chapter: "distribution",
    title: "Publish a dataset chunk (fan-out)",
    status: "built",
    summary:
      "The fan-out publisher assigns a monotonic chunk_id via fetch_add and broadcasts a DatasetChunk of DatasetRequest { request_id, payload } add-only over the broadcast primitive; requests are keyed by stable request_id, never arrival order.",
    source: { path: "rust/runtime/src/cellular/dataset_session.rs", startLine: 84, endLine: 90, symbol: "DatasetPublisher::add" },
    evidence: [
      { path: "rust/runtime/src/cellular/dataset_session.rs", symbol: "next_chunk.fetch_add + broadcast.add(DatasetChunk) at lines 85-89", kind: "boundary" },
      { path: "rust/runtime/src/cellular/dataset_session.rs", symbol: "each_cell_indexes_only_its_owned_shard_keyed_by_request_id", kind: "unit" },
      { path: "rust/runtime/src/cellular/dataset_session.rs", symbol: "a_late_cell_still_indexes_the_full_owned_shard_via_replay", kind: "unit" },
    ],
    inputs: ["requests: Vec<DatasetRequest<R>>"],
    outputs: ["u64 chunk_id", "broadcast DatasetChunk { chunk_id, requests }"],
    state: ["Arc<AtomicU64> next_chunk", "Broadcast<DatasetChunk<R>>"],
    invariants: [
      "chunk_id is monotonic; a streaming controller passes chunk_id + 1 to phaser.advance(ShardsAvailable).",
      "Requests are addressed by stable request_id, so a late cell reconstructs its shard identically.",
    ],
    complexity: { time: "O(subscribers) fan-out per chunk", memory: "O(chunk) history growth" },
    gates: ["broadcast not finalized"],
    failures: ["Adding after finalize is the broadcast no-op (inherited from broadcast-add-fanout)."],
    pseudocode: pseudocode(
      "chunk_id = next_chunk.fetch_add(1, SeqCst)",
      "broadcast.add(DatasetChunk { chunk_id, requests })",
      "return chunk_id",
    ),
    frames: [
      {
        id: "publish-chunk-0",
        label: "Publish the first chunk",
        activeLineId: "step-2",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["next_chunk = 0", "16 requests (ids 0..16)"],
        after: ["chunk_id = 0 broadcast", "next_chunk = 1"],
        emitted: "chunk 0",
        invariantChecks: ["Keyed by request_id, not arrival order."],
      },
      {
        id: "late-replay",
        label: "Late cell still receives the chunk",
        activeLineId: "step-2",
        activeActors: ["controller", "cell"],
        activeLinks: [],
        before: ["a cell attaches after chunk 0 was published"],
        after: ["chunk 0 delivered via replay-on-attach"],
        emitted: "replayed",
        invariantChecks: ["Add-only broadcast means no published chunk is missed."],
      },
    ],
    predecessors: ["controller-fanout-generation", "broadcast-add-fanout"],
    successors: ["dataset-finalize", "owned-index-build", "dataset-velo-replay-live"],
    routeTags: ["distribution", "fan-out", "opt-in", "verification"],
  },
  {
    id: "dataset-finalize",
    chapter: "distribution",
    title: "Seal the dataset broadcast",
    status: "built",
    summary:
      "finalize seals the dataset broadcast so a cell attaching after it still replays every chunk plus the terminal; sealing is required before build_owned can drain (else the live tail blocks forever).",
    source: { path: "rust/runtime/src/cellular/dataset_session.rs", startLine: 94, endLine: 96, symbol: "DatasetPublisher::finalize" },
    evidence: [
      { path: "rust/runtime/src/cellular/dataset_session.rs", symbol: "broadcast.finalize() delegation at lines 94-96", kind: "boundary" },
      { path: "rust/runtime/src/cellular/dataset_session.rs", symbol: "a_late_cell_still_indexes_the_full_owned_shard_via_replay", kind: "unit" },
    ],
    inputs: ["the dataset broadcast"],
    outputs: ["sealed broadcast (terminal fanned out)"],
    state: ["Broadcast<DatasetChunk<R>>"],
    invariants: [
      "After finalize a late attach replays every chunk plus Finalized, so its index is complete.",
      "build_owned only terminates once the broadcast is sealed.",
    ],
    complexity: { time: "O(subscribers): one terminal send each", memory: "O(1)" },
    gates: ["not already finalized (idempotent via the broadcast)"],
    failures: ["Failing to finalize would block every cell's build_owned indefinitely; the bounded run always distributes fully then finalizes."],
    pseudocode: pseudocode(
      "broadcast.finalize()  // seal + fan out Finalized",
      "downstream build_owned drains to completion (unblocked)",
    ),
    frames: [
      {
        id: "seal",
        label: "Seal after full distribution",
        activeLineId: "step-1",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["all chunks published"],
        after: ["broadcast sealed", "late cells replay full shard + Finalized"],
        emitted: "terminal",
        invariantChecks: ["A late index is still complete."],
      },
      {
        id: "unblock-drain",
        label: "build_owned can drain",
        activeLineId: "step-2",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["cell draining collect_until_finalized"],
        after: ["drain terminates at Finalized"],
        emitted: "index built",
        invariantChecks: ["Draining requires the seal."],
      },
    ],
    predecessors: ["dataset-chunk-publish", "broadcast-finalize"],
    successors: ["owned-index-build", "dataset-velo-subscribe"],
    routeTags: ["distribution", "fan-out", "terminal"],
  },
  {
    id: "owned-index-build",
    chapter: "distribution",
    title: "Index only this cell's owned shard",
    status: "built",
    summary:
      "build_owned drains the subscription to Finalized, inserting only requests where owns(request_id) into a request_id-keyed HashMap; every non-owned request is observed then dropped, so peak RAM is O(owned) — the cell's ~1/N shard — even though every chunk is observed.",
    source: { path: "rust/runtime/src/cellular/dataset_session.rs", startLine: 122, endLine: 136, symbol: "DatasetIndex::build_owned" },
    evidence: [
      { path: "rust/runtime/src/cellular/dataset_session.rs", symbol: "collect_until_finalized + owns() filter insert at lines 126-135", kind: "boundary" },
      { path: "rust/runtime/src/cellular/dataset_session.rs", symbol: "each_cell_indexes_only_its_owned_shard_keyed_by_request_id", kind: "unit" },
      { path: "rust/runtime/src/cellular/dataset_session.rs", symbol: "arrival_order_does_not_affect_the_index", kind: "unit" },
    ],
    inputs: ["Subscription<DatasetChunk<R>>", "owns: Fn(u64) -> bool"],
    outputs: ["DatasetIndex<R> { owned: HashMap<u64, R> }"],
    state: ["owned HashMap keyed by request_id"],
    invariants: [
      "The index is keyed by request_id, so arrival order does not affect it.",
      "Non-owned requests are observed then dropped, bounding peak RAM to O(owned).",
    ],
    complexity: { time: "O(dataset): every request is observed once", memory: "O(owned) = ~O(dataset / N)" },
    gates: ["the producer finalized (else the live tail blocks)"],
    failures: ["A cell that never finalizes upstream blocks in collect_until_finalized; owned shards across cells tile the dataset disjointly by construction."],
    pseudocode: pseudocode(
      "chunks = sub.collect_until_finalized().await",
      "for chunk in chunks: for request in chunk.requests:",
      "  if owns(request.request_id): owned.insert(request_id, payload)",
    ),
    frames: [
      {
        id: "keep-owned",
        label: "Cell 1 of 3 keeps id % 3 == 1",
        activeLineId: "step-3",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["ids 0..9 observed", "owns(id) = id % 3 == 1"],
        after: ["owned = {1, 4, 7}"],
        emitted: "indexed",
        invariantChecks: ["Only owned request_ids are retained."],
      },
      {
        id: "drop-nonowned",
        label: "Non-owned requests dropped",
        activeLineId: "step-3",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["id 2 observed", "owns(2) = false"],
        after: ["id 2 dropped, not inserted"],
        emitted: "O(1/N) RAM",
        invariantChecks: ["Peak memory is the owned shard, not the whole dataset."],
      },
    ],
    predecessors: ["dataset-finalize", "modulo-cell-ownership"],
    successors: ["dispatch-on-issue"],
    routeTags: ["distribution", "fan-out", "index"],
  },
  {
    id: "dataset-velo-subscribe",
    chapter: "distribution",
    title: "Build the owned index over velo (zstd)",
    status: "feature-gated",
    summary:
      "Over velo the cell registers the chunk push handler (zunpack), register_peers the controller, sends a unary subscribe, zstd-decompresses (level 3) the replay reply and each pushed chunk, and builds its owned index — MessagePack + zstd level 3, distinct from the raw-rmp partition/phaser control plane.",
    source: { path: "rust/runtime/src/cellular/transport/dataset_velo.rs", startLine: 144, endLine: 186, symbol: "DatasetClient::build_owned_index" },
    evidence: [
      { path: "rust/runtime/src/cellular/transport/dataset_velo.rs", symbol: "register HANDLER_DATASET_CHUNK zunpack + unary subscribe + zunpack reply at lines 150-186", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/dataset_velo.rs", symbol: "zpack/zunpack rmp + zstd level 3 at lines 40-53", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/dataset_velo.rs", symbol: "cells_build_disjoint_owned_indexes_over_velo", kind: "integration" },
    ],
    inputs: ["cell velo Arc<Velo>", "controller PeerInfo", "owns: Fn(u64) -> bool"],
    outputs: ["DatasetIndex<WirePayload>"],
    state: ["chunk push channel", "cursor into replay"],
    invariants: [
      "Replay-on-attach delivers the cell's full owned shard even on a late subscribe.",
      "The dataset wire is MessagePack + zstd level 3, not the raw MessagePack of the control plane.",
    ],
    complexity: { time: "O(dataset): every chunk decompressed + observed", memory: "O(owned) index + O(chunk) decode" },
    gates: ["velo feature compiled", "subscribe reply decompresses + decodes"],
    failures: ["A zstd/decode failure fails the subscribe; a missing finalize would block build_owned."],
    pseudocode: pseudocode(
      "register HANDLER_DATASET_CHUNK -> channel (zunpack each pushed chunk)",
      "register_peer(controller); unary HANDLER_DATASET_SUBSCRIBE(cell_peer)",
      "reply = zunpack(reply_bytes)  // rmp + zstd L3",
      "DatasetIndex::build_owned(replay + live, owns)",
    ),
    frames: [
      {
        id: "subscribe-replay",
        label: "Subscribe replays the owned shard",
        activeLineId: "step-3",
        activeActors: ["cell", "wire", "controller"],
        activeLinks: [],
        before: ["controller published all chunks + finalized"],
        after: ["reply.replay zstd-decoded", "owned index built"],
        emitted: "index",
        invariantChecks: ["A late subscriber still gets its full shard via replay."],
      },
      {
        id: "live-chunk-indexed",
        label: "Live chunk pushed and indexed",
        activeLineId: "step-1",
        activeActors: ["controller", "wire", "cell"],
        activeLinks: [],
        before: ["cell subscribed mid-stream"],
        after: ["pushed chunk zunpacked into the live channel", "owned entries inserted"],
        emitted: "live chunk",
        invariantChecks: ["Dataset velo wire is zstd-compressed MessagePack."],
      },
    ],
    predecessors: ["dataset-finalize", "velo-peer-connect", "handler-registration"],
    successors: ["owned-index-build", "dispatch-on-issue"],
    routeTags: ["distribution", "velo", "fan-out", "opt-in", WIRE_FACTS.datasetVelo],
  },
  {
    id: "dataset-velo-replay-live",
    chapter: "distribution",
    title: "Serve dataset replay + live pump (zstd)",
    status: "feature-gated",
    summary:
      "The controller's dataset service attaches a broadcast consumer under the seam lock, zstd-packs (level 3) the replay snapshot into the subscribe reply, and spawns a pump that zstd-packs and pushes each live chunk to the cell until finalize or the cell disconnects.",
    source: { path: "rust/runtime/src/cellular/transport/dataset_velo.rs", startLine: 79, endLine: 134, symbol: "DatasetServer::bind" },
    evidence: [
      { path: "rust/runtime/src/cellular/transport/dataset_velo.rs", symbol: "attach_raw under lock + spawn pump zpack live + zpack(reply) at lines 100-127", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/dataset_velo.rs", symbol: "zpack contrasted with uncompressed rmp phaser plane at lines 36-45", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/dataset_velo.rs", symbol: "cells_build_disjoint_owned_indexes_over_velo", kind: "integration" },
    ],
    inputs: ["velo Arc<Velo>", "DatasetPublisher<WirePayload>"],
    outputs: ["subscribe reply (zstd replay)", "per-cell live pump task"],
    state: ["broadcast attach seam", "per-cell pump task"],
    invariants: [
      "The replay snapshot ships in the reply and the live tail is pumped, split at the atomic attach seam.",
      "The pump ends (task, not run) on finalize (terminal event) or an am_send error (cell gone).",
    ],
    complexity: { time: "O(replay) reply + O(live) pump", memory: "O(replay) compressed reply body" },
    gates: ["velo feature compiled", "publisher bound to the service"],
    failures: ["A zpack failure breaks the pump; a decode failure fails the subscribe handler."],
    pseudocode: pseudocode(
      "on subscribe: register_peer(cell); { replay, live } = publisher.attach_raw()",
      "spawn pump: while live.recv(): zpack(event) -> am_send(HANDLER_DATASET_CHUNK); stop on terminal/err",
      "reply = zpack(DatasetSubscribeReply { replay })",
    ),
    frames: [
      {
        id: "replay-in-reply",
        label: "Replay snapshot in the reply",
        activeLineId: "step-3",
        activeActors: ["controller", "wire"],
        activeLinks: [],
        before: ["cell subscribes after chunks 0..2"],
        after: ["reply carries zstd-packed replay of 0..2"],
        emitted: "reply",
        invariantChecks: ["Snapshot taken atomically at the attach seam."],
      },
      {
        id: "live-pump",
        label: "Live chunk pumped then terminal",
        activeLineId: "step-2",
        activeActors: ["controller", "wire", "cell"],
        activeLinks: [],
        before: ["cell subscribed at chunk 2"],
        after: ["chunk 3 zpacked + pushed", "Finalized ends the pump task"],
        emitted: "live + terminal",
        invariantChecks: ["The pump task, not the run, ends at the terminal."],
      },
    ],
    predecessors: ["dataset-chunk-publish", "broadcast-attach-replay"],
    successors: ["dataset-velo-subscribe"],
    routeTags: ["distribution", "velo", "fan-out", WIRE_FACTS.datasetVelo],
  },
  {
    id: "controller-fanout-generation",
    chapter: "distribution",
    title: "Generate and broadcast the fan-out dataset",
    status: "feature-gated",
    summary:
      "Opt-in (AIPERF_CELL_DATASET_FANOUT): the controller binds the dataset service, builds each endpoint-ready chat body once, publishes them in 16-request chunks via DatasetPublisher::add, advances the phaser ShardsAvailable(chunk_id + 1) per chunk when the phaser is active, then finalizes — the fan-out is the real dispatch source for the verification overlay, not the canonical measured path.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 414, endLine: 475, symbol: "dataset fan-out block" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "DatasetServer::bind + total = profiling_request_budget at lines 414-422", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "16-chunk publish + phaser.advance(ShardsAvailable(chunk_id+1)) + finalize at lines 438-468", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "profiling_request_budget (requests|sessions) at lines 2045-2066", kind: "boundary" },
    ],
    inputs: ["AIPERF_CELL_DATASET_FANOUT flag", "profiling request budget", "endpoint url + model"],
    outputs: ["a sealed dataset broadcast of endpoint-ready WireRequest bodies"],
    state: ["DatasetPublisher", "DatasetServer (held for the run)", "Option<Phaser>"],
    invariants: [
      "Each request body is built once on the controller, so a cell POSTs exactly what was published.",
      "When the phaser is active, ShardsAvailable(chunk_id + 1) marks shards [0, chunk_id] available per chunk.",
    ],
    complexity: { time: "O(total requests) to build + publish", memory: "O(total) broadcast history (retained for replay)" },
    gates: ["AIPERF_CELL_DATASET_FANOUT truthy", "velo feature compiled", "run cfg has an endpoint url"],
    failures: ["A missing endpoint url errors the fan-out; a bounded run distributes fully up front (profiling_request_budget), so an unbounded (duration) profiling phase yields 0 total here."],
    pseudocode: pseudocode(
      "if dataset_fanout: bind DatasetServer(publisher)",
      "total = profiling_request_budget(envelope)",
      "for [start, start+16) in 0..total: build WireRequest bodies; chunk_id = publisher.add(requests)",
      "  if phaser: phaser.advance(ShardsAvailable(chunk_id + 1))",
      "publisher.finalize()",
    ),
    frames: [
      {
        id: "publish-and-mark",
        label: "Publish chunk, mark shards available",
        activeLineId: "step-3",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["total = 32", "phaser active"],
        after: ["chunk 0 (ids 0..16) published", "phaser advance ShardsAvailable(1)"],
        emitted: "shard 0 available",
        invariantChecks: ["The availability interlock advances per chunk."],
      },
      {
        id: "finalize-distribution",
        label: "Finalize after full distribution",
        activeLineId: "step-5",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["all 32 requests published in 2 chunks"],
        after: ["publisher.finalize()", "cells can complete build_owned"],
        emitted: "sealed",
        invariantChecks: ["A bounded run distributes fully up front."],
      },
    ],
    predecessors: ["phaser-generation-advance", "dataset-velo-replay-live"],
    successors: ["dataset-chunk-publish", "fanout-verification-overlay"],
    routeTags: ["distribution", "fan-out", "opt-in", "verification"],
  },
  {
    id: "dispatch-on-issue",
    chapter: "distribution",
    title: "Issue a request exactly once",
    status: "built",
    summary:
      "The per-request dispatch state machine: InFlight/Done -> Duplicate (exactly-once-issue dedup, no state change); Unknown and indexed -> Issue(payload) transitioning InFlight and counting issued; Unknown and not indexed -> Miss (counted).",
    source: { path: "rust/runtime/src/cellular/dispatch_state.rs", startLine: 89, endLine: 108, symbol: "DispatchTracker::on_issue" },
    evidence: [
      { path: "rust/runtime/src/cellular/dispatch_state.rs", symbol: "match on state then index.get -> Issue/Duplicate/Miss at lines 94-107", kind: "boundary" },
      { path: "rust/runtime/src/cellular/dispatch_state.rs", symbol: "indexed_request_issues_once_then_dedups", kind: "unit" },
      { path: "rust/runtime/src/cellular/dispatch_state.rs", symbol: "in_flight_accounting_tracks_issued_minus_completed", kind: "unit" },
    ],
    inputs: ["request_id: u64", "index: &DatasetIndex<R>"],
    outputs: ["DispatchDecision<R>: Issue(payload) | Duplicate | Miss"],
    state: ["HashMap<u64, RequestState> (Unknown = absence)", "issued counter"],
    invariants: [
      "A request already InFlight or Done re-issues as Duplicate with no state change (exactly-once issue).",
      "An indexed, not-yet-issued request transitions to InFlight and increments issued.",
    ],
    complexity: { time: "O(1) map lookup + insert", memory: "O(issued) recorded states" },
    gates: ["Unknown state and index.get(request_id) is Some"],
    failures: ["An unindexed request returns Miss and increments the miss counter (see distribution-miss); state stays Unknown."],
    pseudocode: pseudocode(
      "match states.get(id): InFlight|Done -> Duplicate",
      "None -> match index.get(id):",
      "  Some(payload) -> insert InFlight; issued += 1; Issue(payload)",
      "  None -> misses += 1; Miss",
    ),
    frames: [
      {
        id: "issue-once",
        label: "Indexed id issues once",
        activeLineId: "step-3",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["id 4 Unknown", "index owns 4"],
        after: ["id 4 InFlight", "issued = 1", "Issue(payload)"],
        emitted: "Issue",
        invariantChecks: ["A first issue transitions to InFlight."],
      },
      {
        id: "dedup",
        label: "Re-issue is a duplicate",
        activeLineId: "step-1",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["id 4 InFlight"],
        after: ["no state change"],
        emitted: "Duplicate",
        invariantChecks: ["Exactly-once issue: a re-issue is a no-op."],
      },
    ],
    predecessors: ["owned-index-build", "phaser-await-generation"],
    successors: ["dispatch-on-complete", "distribution-miss"],
    routeTags: ["distribution", "dispatch", "exactly-once"],
  },
  {
    id: "dispatch-on-complete",
    chapter: "distribution",
    title: "Complete a request idempotently",
    status: "built",
    summary:
      "on_complete transitions InFlight -> Done, incrementing completed only when the request was actually in-flight; it is idempotent, and a completion for a never-issued request records a defensive Done without double-counting.",
    source: { path: "rust/runtime/src/cellular/dispatch_state.rs", startLine: 112, endLine: 118, symbol: "DispatchTracker::on_complete" },
    evidence: [
      { path: "rust/runtime/src/cellular/dispatch_state.rs", symbol: "was_inflight guard + insert Done + completed++ at lines 113-117", kind: "boundary" },
      { path: "rust/runtime/src/cellular/dispatch_state.rs", symbol: "in_flight_accounting_tracks_issued_minus_completed", kind: "unit" },
    ],
    inputs: ["request_id: u64"],
    outputs: ["state -> Done", "completed counter increment (only if was InFlight)"],
    state: ["HashMap<u64, RequestState>", "completed counter"],
    invariants: [
      "completed increments only on an InFlight -> Done transition, so in-flight = issued - completed stays correct.",
      "on_complete is idempotent: a second completion does not double-count.",
    ],
    complexity: { time: "O(1) map insert", memory: "O(1)" },
    gates: ["was_inflight to increment completed"],
    failures: ["A completion for a never-issued request is recorded as Done (a defensive terminal) but does not increment completed."],
    pseudocode: pseudocode(
      "was_inflight = states.get(id) == InFlight",
      "states.insert(id, Done)",
      "if was_inflight: completed += 1",
    ),
    frames: [
      {
        id: "complete-inflight",
        label: "Complete an in-flight request",
        activeLineId: "step-3",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["id 4 InFlight", "completed = 0"],
        after: ["id 4 Done", "completed = 1"],
        emitted: "completed",
        invariantChecks: ["in-flight = issued - completed."],
      },
      {
        id: "recomplete",
        label: "Re-complete is idempotent",
        activeLineId: "step-1",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["id 4 Done"],
        after: ["still Done", "completed unchanged"],
        emitted: "no double count",
        invariantChecks: ["Idempotent completion."],
      },
    ],
    predecessors: ["dispatch-on-issue"],
    successors: [],
    routeTags: ["distribution", "dispatch", "idempotent"],
  },
  {
    id: "distribution-miss",
    chapter: "distribution",
    title: "Count a distribution miss, never skip",
    status: "built",
    summary:
      "An issue for a request this cell has not indexed is a counted, surfaced DispatchDecision::Miss — never a silent skip; the tracker counts the miss and leaves state Unknown, so a later index can still issue it, and distribution_misses is a distinct error class.",
    source: { path: "rust/runtime/src/cellular/dispatch_state.rs", startLine: 102, endLine: 129, symbol: "on_issue Miss arm + distribution_misses" },
    evidence: [
      { path: "rust/runtime/src/cellular/dispatch_state.rs", symbol: "misses += 1; Miss (state unchanged) at lines 102-105", kind: "boundary" },
      { path: "rust/runtime/src/cellular/dispatch_state.rs", symbol: "distribution_misses accessor at lines 127-129", kind: "boundary" },
      { path: "rust/runtime/src/cellular/dispatch_state.rs", symbol: "unknown_request_is_a_counted_miss_not_a_silent_skip", kind: "unit" },
    ],
    inputs: ["request_id not in the index"],
    outputs: ["DispatchDecision::Miss", "incremented miss counter"],
    state: ["misses counter", "state map (unchanged for a miss)"],
    invariants: [
      "A miss is counted and surfaced, never silently skipped.",
      "A miss does not mark state, so a subsequently-indexed request can still issue.",
    ],
    complexity: { time: "O(1)", memory: "O(1)" },
    gates: ["Unknown state and index.get(request_id) is None"],
    failures: ["A nonzero miss count is a distinct error class (an incomplete fan-out), surfaced separately from server errors; the caller should bounded-await the index before accepting a miss."],
    pseudocode: pseudocode(
      "on Unknown and not indexed: misses += 1; return Miss",
      "state stays Unknown (a later index can issue it)",
      "distribution_misses() reports the accepted total distinctly",
    ),
    frames: [
      {
        id: "counted-miss",
        label: "Unindexed id is a counted miss",
        activeLineId: "step-1",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["id 99 Unknown", "index does not own 99"],
        after: ["misses = 1", "Miss returned"],
        emitted: "Miss (counted)",
        invariantChecks: ["Never a silent skip."],
      },
      {
        id: "later-issuable",
        label: "State unchanged, later issuable",
        activeLineId: "step-2",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["miss recorded for id 99"],
        after: ["id 99 still Unknown"],
        emitted: "issuable after index",
        invariantChecks: ["A miss does not poison future issuance."],
      },
    ],
    predecessors: ["dispatch-on-issue"],
    successors: ["fanout-verification-overlay"],
    routeTags: ["distribution", "dispatch", "miss", "fail-closed"],
  },
  {
    id: "fanout-verification-overlay",
    chapter: "distribution",
    title: "Verify fan-out by dispatching the owned shard",
    status: "feature-gated",
    summary:
      "Opt-in overlay (AIPERF_CELL_DATASET_FANOUT), a no-op when unset: the cell builds its owned index over velo, runs the dispatch state machine over its owned slice actually POSTing each owned request, and fails closed if any distribution miss occurred — proving the fan-out delivered each cell its owned shard. It runs alongside, not instead of, the canonical measured dispatch.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_cell.rs", startLine: 481, endLine: 555, symbol: "verify_dataset_fanout" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_cell.rs", symbol: "flag no-op guard + build_owned_index + dispatch loop at lines 487-538", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_cell.rs", symbol: "ensure distribution_misses == 0 fail-closed at lines 549-553", kind: "boundary" },
    ],
    inputs: ["AIPERF_CELL_DATASET_FANOUT flag", "controller coordinate", "cell partition"],
    outputs: ["Ok(()) when misses == 0", "Err on any distribution miss"],
    state: ["DatasetIndex over velo", "DispatchTracker"],
    invariants: [
      "A no-op when the flag is unset (returns Ok immediately).",
      "The fan-out is the real dispatch source here: each Issue actually POSTs the endpoint-ready body.",
    ],
    complexity: { time: "O(owned) POSTs", memory: "O(owned) index" },
    gates: ["AIPERF_CELL_DATASET_FANOUT truthy", "velo feature compiled", "distribution_misses == 0"],
    failures: ["Any distribution miss fails closed (ensure! with a per-cell miss count); a non-2xx or send error is logged (not fatal) — only misses gate the overlay."],
    pseudocode: pseudocode(
      "if flag unset: return Ok(())",
      "index = DatasetClient::build_owned_index(velo, controller, owns)",
      "for id in index.owned_ids(): on_issue -> POST body; on_complete",
      "ensure!(tracker.distribution_misses() == 0)",
    ),
    frames: [
      {
        id: "overlay-off",
        label: "Overlay off is a no-op",
        activeLineId: "step-1",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["AIPERF_CELL_DATASET_FANOUT unset"],
        after: ["returns Ok immediately"],
        emitted: "skipped",
        invariantChecks: ["The canonical measured dispatch is unaffected."],
      },
      {
        id: "overlay-on",
        label: "Overlay dispatches owned shard",
        activeLineId: "step-4",
        activeActors: ["cell", "wire"],
        activeLinks: [],
        before: ["flag on", "owned shard indexed over velo"],
        after: ["each owned request POSTed", "misses == 0 asserted"],
        emitted: "verified",
        invariantChecks: ["Fail-closed on any miss (incomplete fan-out)."],
      },
    ],
    predecessors: ["distribution-miss", "dataset-velo-subscribe", "controller-fanout-generation"],
    successors: [],
    routeTags: ["distribution", "fan-out", "opt-in", "verification", "fail-closed"],
  },
  {
    id: "dataset-serve-plan",
    chapter: "distribution",
    title: "Plan the Stage-G dataset serve set",
    status: "feature-gated",
    summary:
      "build_dataset_serve_plan builds the controller's serve set: for a graph trace it enumerates the loader's own read set into a flat name->path map + DatasetManifest { kind, base_name, files }; a scheduled file/path dataset ships as one file. Fails closed on a directory-shaped scheduled dataset or duplicate shard names.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1490, endLine: 1568, symbol: "build_dataset_serve_plan" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "graph enumerate -> kind/name map (dup rejection) + manifest at lines 1512-1545", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "scheduled single-file require + one name->path at lines 1546-1566", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "build_dataset_serve_plan_ships_single_dir_and_prefix", kind: "unit" },
    ],
    inputs: ["format: Option<&str>", "source: &Path"],
    outputs: ["HashMap<String, PathBuf> serve map", "DatasetManifest"],
    state: ["name -> path map", "ordered names"],
    invariants: [
      "For a graph trace the serve set is the loader's own enumeration (byte-for-byte the 1-cell read set).",
      "Shard names must be unique (a duplicate name fails closed, since flat names reconstruct the tree).",
    ],
    complexity: { time: "O(files) enumeration + map build", memory: "O(files)" },
    gates: ["velo feature compiled", "graph format enumerates successfully, or the scheduled source is a single readable file"],
    failures: ["A scheduled file/path dataset that is a directory or missing fails closed; two shards with the same file name fail closed."],
    pseudocode: pseudocode(
      "if graph format: (kind, base_name, files) = enumerate_recorded_trace_files(format, source)",
      "  for path in files: name = file_name(path); ensure unique; map.insert(name, path)",
      "  manifest = { kind_str, base_name, names }",
      "else: ensure source.is_file(); map = { name -> source }; manifest = { file, name }",
    ),
    frames: [
      {
        id: "weka-dir",
        label: "WEKA directory -> shard manifest",
        activeLineId: "step-1",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["format = weka_trace", "source = a directory of .json shards"],
        after: ["kind = dir", "files = enumerated .json read set"],
        emitted: "dir manifest",
        invariantChecks: ["Serve set equals the loader's own enumeration."],
      },
      {
        id: "scheduled-dir-reject",
        label: "Scheduled directory rejected",
        activeLineId: "step-4",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["format = single_turn file dataset", "source is a directory"],
        after: ["ensure source.is_file() fails"],
        emitted: "rejected",
        invariantChecks: ["A scheduled dataset ships as a single file only."],
      },
    ],
    predecessors: ["cellular-run-shape-validation", "recorded-graph-file-enumeration"],
    successors: ["dataset-manifest-validation", "dataset-http-zstd-reconstruct"],
    routeTags: ["distribution", "stage-g", "file", WIRE_FACTS.stageG],
  },
  {
    id: "dataset-manifest-validation",
    chapter: "distribution",
    title: "Serve and consume the dataset manifest",
    status: "feature-gated",
    summary:
      "The controller serves the DatasetManifest as JSON at GET /dataset-manifest (404 when none registered); the cell fetches it whole (names, not bytes), and reconstruct maps kind dir -> dest_dir, file/prefix -> dest_dir/base_name, else fails closed on an unknown kind.",
    source: { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", startLine: 641, endLine: 650, symbol: "serve_dataset_manifest (+ fetch_dataset_manifest / reconstruct kind match)" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "serve_dataset_manifest 404-when-none at lines 641-650", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "fetch_dataset_manifest collect-whole + decode at lines 846-882", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "reconstruct kind dir/file/prefix/unknown at lines 927-936", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "manifest_directory_reconstructs_identical_tree", kind: "integration" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "manifest_prefix_points_path_at_stem_beside_shards", kind: "integration" },
    ],
    inputs: ["registered DatasetManifest (server)", "GET /dataset-manifest (cell)"],
    outputs: ["DatasetManifest JSON", "the local path datasets/0.path should point at"],
    state: ["UploadState.manifest"],
    invariants: [
      "A run with no manifest returns 404 (synthetic / same-host / no-dataset).",
      "The manifest carries file NAMES, so it is collected whole; only the shard bytes stream.",
    ],
    complexity: { time: "O(1) serve + O(manifest) decode", memory: "O(manifest) names" },
    gates: ["velo feature compiled", "a manifest is registered (else 404)", "kind is dir/file/prefix"],
    failures: ["An unknown manifest kind fails closed (bail); a non-2xx manifest fetch errors."],
    pseudocode: pseudocode(
      "server: manifest.clone().map(Json).ok_or(404)",
      "cell: GET /dataset-manifest; ensure success; decode DatasetManifest",
      "reconstruct: kind dir -> dest_dir; file|prefix -> dest_dir/base_name; other -> bail",
    ),
    frames: [
      {
        id: "dir-manifest",
        label: "Directory manifest maps to dest_dir",
        activeLineId: "step-3",
        activeActors: ["controller", "wire", "cell"],
        activeLinks: [],
        before: ["kind = dir", "files = [a.json, b.json]"],
        after: ["datasets/0.path = dest_dir"],
        emitted: "dir path",
        invariantChecks: ["The loader scans the reconstructed directory."],
      },
      {
        id: "unknown-kind",
        label: "Unknown kind rejected",
        activeLineId: "step-3",
        activeActors: ["cell"],
        activeLinks: [],
        before: ["kind = something-else"],
        after: ["bail: unknown dataset manifest kind"],
        emitted: "rejected",
        invariantChecks: ["Fail closed on an unrecognized layout."],
      },
    ],
    predecessors: ["dataset-serve-plan"],
    successors: ["dataset-safe-path-mapping", "dataset-http-zstd-reconstruct"],
    routeTags: ["distribution", "stage-g", "manifest", WIRE_FACTS.stageG],
  },
  {
    id: "dataset-safe-path-mapping",
    chapter: "distribution",
    title: "Map shipped names to safe local paths",
    status: "feature-gated",
    summary:
      "Two defense-in-depth validators map every shipped name to a safe local path: a dataset file name must be a single Normal flat component (no /, .., root); an uploaded artifact relpath must be present verbatim in the run's shippable_relatives allowlist with only Normal components — so a cell can never traverse out of its cell-{id} dir.",
    source: { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", startLine: 889, endLine: 899, symbol: "validate_dataset_relname (+ validate_artifact_relpath / shippable_relatives)" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "validate_dataset_relname single-Normal-component gate at lines 889-899", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "validate_artifact_relpath absolute/non-normal/allowlist gates at lines 276-294", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "shippable_relatives allowlist derivation at lines 946-964", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "dataset_relname_validation_rejects_traversal", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "server_rejects_unallowed_upload", kind: "integration" },
    ],
    inputs: ["a manifest-supplied dataset name", "a client-supplied artifact relpath", "the run's ArtifactSpec"],
    outputs: ["a validated safe PathBuf, or a fail-closed error"],
    state: ["allowlist HashSet<String> from shippable_relatives"],
    invariants: [
      "A dataset name must be exactly one Normal path component (no traversal).",
      "An artifact relpath must be verbatim in the allowlist and all-Normal, so bytes land only at known per-record locations inside cell-{id}.",
    ],
    complexity: { time: "O(components) per name", memory: "O(allowlist)" },
    gates: ["velo feature compiled", "dataset name is a single Normal component", "artifact relpath is relative, all-Normal, and in the allowlist"],
    failures: ["An absolute path, any non-Normal component (.., ., root), or an out-of-allowlist path fails closed."],
    pseudocode: pseudocode(
      "dataset name: single Normal component else bail",
      "artifact relpath: not absolute; every component Normal; allowed.contains(rel) else bail",
      "allowlist = shippable_relatives(cfg.artifacts)",
    ),
    frames: [
      {
        id: "flat-name-ok",
        label: "Flat name accepted",
        activeLineId: "step-1",
        activeActors: ["cell", "controller"],
        activeLinks: [],
        before: ["name = shard.000001.jsonl.gz"],
        after: ["accepted as one Normal component"],
        emitted: "ok",
        invariantChecks: ["Flat names reconstruct the tree safely."],
      },
      {
        id: "traversal-reject",
        label: "Traversal rejected",
        activeLineId: "step-2",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["relpath = ../../etc/passwd"],
        after: ["non-Normal component -> bail"],
        emitted: "rejected",
        invariantChecks: ["A cell can never traverse out of its cell dir."],
      },
    ],
    predecessors: ["dataset-manifest-validation"],
    successors: ["dataset-http-zstd-reconstruct"],
    routeTags: ["distribution", "stage-g", "safety", "fail-closed"],
  },
  {
    id: "dataset-http-zstd-reconstruct",
    chapter: "distribution",
    title: "Stream and reconstruct the shipped dataset",
    status: "feature-gated",
    summary:
      "Stage G bulk transfer: the controller streams each source file zstd-compressed (Content-Encoding: zstd, one CHUNK_SIZE=65536 chunk at a time); the cell GETs each named file, streaming-decompresses to a .part then atomically renames — the whole file is never resident on either end, and a mid-stream error truncates the body so the cell fails rather than landing a partial file.",
    source: { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", startLine: 912, endLine: 937, symbol: "reconstruct_shipped_dataset (+ fetch_dataset_to_file / serve_dataset / decode_channel_to_file)" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "per-file validate + fetch + kind path at lines 917-936", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "serve_dataset chunked zstd + mid-stream truncate at lines 585-635", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "decode_channel_to_file zstd-to-.part-then-rename at lines 243-261", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "serve_then_download_round_trips_dataset_bytes", kind: "integration" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "compiled_dataset_matches_between_original_and_shipped_file", kind: "integration" },
    ],
    inputs: ["authority", "DatasetManifest", "dest_dir"],
    outputs: ["reconstructed local tree", "the datasets/0.path stem per kind"],
    state: ["bounded mpsc channel (4)", "DecompressToFile .part sink"],
    invariants: [
      "Peak memory is O(CHUNK_SIZE) per transfer, independent of file size or shard count.",
      "A mid-stream read error truncates the body so the cell's decoder fails rather than landing a partial file.",
    ],
    complexity: { time: "O(total bytes)", memory: "O(chunk) = 65536 bytes per transfer" },
    gates: ["velo feature compiled", "each name validates", "each GET returns 2xx"],
    failures: ["A non-2xx fetch errors; a writer/decode failure is contexted; a truncated body fails the decode (no partial landed)."],
    pseudocode: pseudocode(
      "for name in manifest.files: validate_dataset_relname(name)",
      "  fetch_dataset_to_file(authority, name, dest_dir/name)  // GET, stream, zstd-decode to .part, rename",
      "return kind dir -> dest_dir; file|prefix -> dest_dir/base_name",
    ),
    frames: [
      {
        id: "stream-decode-rename",
        label: "Stream, decompress, atomic rename",
        activeLineId: "step-2",
        activeActors: ["controller", "wire", "cell"],
        activeLinks: [],
        before: ["GET /dataset/{name}", "Content-Encoding: zstd"],
        after: ["chunks decoded to .part", "atomic rename to final name"],
        emitted: "landed",
        invariantChecks: ["Whole file never resident; bounded per-chunk memory."],
      },
      {
        id: "midstream-error",
        label: "Mid-stream error, no partial",
        activeLineId: "step-2",
        activeActors: ["controller", "cell"],
        activeLinks: [],
        before: ["server read error mid-stream"],
        after: ["body truncated", "cell decode fails", ".part discarded"],
        emitted: "fail not partial",
        invariantChecks: ["A truncated transfer never lands a partial final file."],
      },
    ],
    predecessors: ["dataset-safe-path-mapping", "dataset-manifest-validation"],
    successors: [],
    routeTags: ["distribution", "stage-g", "http", "zstd", WIRE_FACTS.stageG],
  },
  {
    id: "recorded-graph-file-enumeration",
    chapter: "distribution",
    title: "Enumerate a recorded trace's read set",
    status: "built",
    summary:
      "enumerate_recorded_trace_files is the shipping-side mirror of the graph loaders: for weka/aiperf/dynamo/dag_jsonl it returns (kind, base_name, ordered file set) reusing the loaders' own enumeration (json_documents_in_dir / discover_dynamo_segments), so the shipped set is byte-for-byte the 1-cell read set; it fails closed on missing/empty-dir/unmatched-prefix/dag-directory/unsupported-format.",
    source: { path: "rust/runtime/src/graph/recorded/source.rs", startLine: 326, endLine: 394, symbol: "enumerate_recorded_trace_files" },
    evidence: [
      { path: "rust/runtime/src/graph/recorded/source.rs", symbol: "weka/aiperf dir|file, dynamo File|Directory|SegmentedPrefix, dag_jsonl file-only at lines 337-393", kind: "boundary" },
      { path: "rust/runtime/src/graph/recorded/source.rs", symbol: "enumerate_weka_directory_matches_loader_json_read_set", kind: "unit" },
      { path: "rust/runtime/src/graph/recorded/source.rs", symbol: "enumerate_dynamo_directory_and_prefix_match_discovery_order", kind: "unit" },
      { path: "rust/runtime/src/graph/recorded/source.rs", symbol: "enumerate_rejects_dag_jsonl_directory_and_missing_paths", kind: "unit" },
    ],
    inputs: ["format: &str", "path: &Path"],
    outputs: ["(RecordedTracePathKind, base_name, Vec<PathBuf> in read order)"],
    state: ["none (pure enumeration)"],
    invariants: [
      "The enumerated set reuses the loaders' own discovery, so it equals the 1-cell read set (no over/under-ship).",
      "dag_jsonl reads a single file only; a directory/prefix is rejected here.",
    ],
    complexity: { time: "O(files) directory scan", memory: "O(files)" },
    gates: ["format is a recorded graph format", "path resolves to a supported layout"],
    failures: ["A missing path, empty directory, unmatched prefix, dag_jsonl directory, or unsupported format fails closed (the same errors the loader would raise, before cells launch)."],
    pseudocode: pseudocode(
      "weka_trace|aiperf_trace: dir -> Directory + json_documents_in_dir; file -> File; else error",
      "dynamo_trace: File|Directory|SegmentedPrefix via discover_dynamo_segments",
      "dag_jsonl: file -> File; else reject",
      "other: reject (not a recorded graph trace format)",
    ),
    frames: [
      {
        id: "weka-directory",
        label: "WEKA directory -> ordered shard list",
        activeLineId: "step-1",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["format = weka_trace", "path = a dir of .json"],
        after: ["Directory kind", "files = json_documents_in_dir read set"],
        emitted: "enumerated",
        invariantChecks: ["Read set equals the loader's own enumeration."],
      },
      {
        id: "dag-dir-reject",
        label: "dag_jsonl directory rejected",
        activeLineId: "step-3",
        activeActors: ["controller"],
        activeLinks: [],
        before: ["format = dag_jsonl", "path is a directory"],
        after: ["error: dag_jsonl reads a single file"],
        emitted: "rejected",
        invariantChecks: ["Fail closed before launching cells."],
      },
    ],
    predecessors: ["run-kind-classification"],
    successors: ["dataset-serve-plan"],
    routeTags: ["distribution", "stage-g", "graph", "enumeration"],
  },
];

const EXECUTION_ALGORITHMS: readonly AlgorithmDefinition[] = [
  {
    id: "partitioned-scheduled-sampler",
    chapter: "execution",
    title: "Draw only the cell-owned scheduled positions",
    status: "built",
    summary:
      "PartitionedSampler advances the shared deterministic inner draw stream one position at a time and returns a draw only when ModuloCellPartition owns that position. Sequential/shuffle unions reproduce the one-cell draw set; random-with-replacement remains well formed but is explicitly outside one-cell-identical determinism.",
    source: { path: "rust/runtime/src/dataset/sampler.rs", startLine: 269, endLine: 332, symbol: "PartitionedSampler / Sampler::next" },
    evidence: [
      { path: "rust/runtime/src/dataset/sampler.rs", symbol: "partitioned_sampler_yields_disjoint_owned_positions", kind: "unit" },
      { path: "rust/runtime/src/dataset/sampler.rs", symbol: "PartitionedSampler::for_partition", kind: "boundary" },
    ],
    inputs: ["inner sampler", "ModuloCellPartition { cell_id, cell_count }"],
    outputs: ["inner sampler ID from the next owned draw position"],
    state: ["global draw position", "inner sampler state"],
    invariants: ["A value returns only when partition.owns(position) is true.", "Every skipped position still advances the same inner stream."],
    complexity: { time: "up to/amortized O(cell_count * inner next) per emitted draw", memory: "O(inner sampler)" },
    gates: ["ModuloCellPartition already validated", "multi-cell partitions wrap; identity/None passes the inner sampler through"],
    failures: ["The wrapper itself is infallible after partition construction; random sampling is accepted with a weaker determinism contract."],
    pseudocode: pseudocode(
      "loop: id = inner.next(); owned = partition.owns(position)",
      "position += 1",
      "if owned: return id",
    ),
    frames: admissionFrames("step-3", ["cell", "worker"], {
      before: ["cell 1/3", "position 7"], after: ["inner draw returned", "position 7 owned and emitted"], invariant: "7 mod 3 = 1",
    }, {
      before: ["cell 1/3", "position 8"], after: ["inner draw consumed but skipped", "position advances to 9"], invariant: "A non-owned draw is never emitted.", activeLineId: "step-2",
    }),
    predecessors: ["modulo-cell-ownership", "owned-positions-tiling"],
    successors: ["conversation-ownership", "scheduled-shard-runtime"],
    routeTags: ["execution", "scheduled", "sampler", "partition"],
  },
  {
    id: "partitioned-graph-source",
    chapter: "execution",
    title: "Partition graph templates before replay",
    status: "built",
    summary:
      "PartitionedGraphTraceSource computes global_ordinal = next_local*cell_count + cell_id, applies the strategy-aware draw at that ordinal, and emits the cloned template with a globally stamped instance ID.",
    source: { path: "rust/runtime/src/graph/workload.rs", startLine: 242, endLine: 350, symbol: "PartitionedGraphTraceSource / GraphTraceSource::next_trace" },
    evidence: [
      { path: "rust/runtime/src/graph/workload.rs", symbol: "partitioned_source_interleaves_and_covers_the_single_cell_set", kind: "unit" },
      { path: "rust/runtime/src/graph/workload.rs", symbol: "partitioned_source_rejects_bad_partitions", kind: "unit" },
    ],
    inputs: ["trace templates", "optional global session limit", "cell_id", "cell_count", "PermutationDraw"],
    outputs: ["GraphTracePlan for the next owned global ordinal"],
    state: ["next_local", "immutable templates", "strategy-aware draw"],
    invariants: ["global_ordinal = next_local*cell_count + cell_id.", "The union of cell ordinals tiles the single-cell ordinal sequence."],
    complexity: { time: "O(selected plan size + draw work)", memory: "O(templates + draw state)" },
    gates: ["at least one template", "cell_count >= 1", "cell_id < cell_count", "positive configured session limit"],
    failures: ["Invalid coordinates/budget or checked ordinal overflow errors; reaching the global session limit returns None."],
    pseudocode: pseudocode(
      "global = next_local * cell_count + cell_id; stop if global >= session_limit",
      "template = templates[draw.index(global, templates.len)]",
      "stamp trace.id with instance-global; next_local += 1; return template",
    ),
    frames: admissionFrames("step-2", ["cell", "worker"], {
      before: ["cell 1/3", "global ordinal 4"], after: ["draw selects one template", "selected GraphTracePlan cloned"], invariant: "The strategy-aware draw is evaluated at global ordinal 4.",
    }, {
      before: ["cloned template selected", "global ordinal 4"], after: ["instance ID stamped", "next_local advanced", "plan returned"], invariant: "The returned clone carries the global ownership ordinal.", activeLineId: "step-3",
    }),
    predecessors: ["owned-positions-tiling", "recorded-graph-file-enumeration"],
    successors: ["graph-global-instance-ordinal", "scheduled-graph-runtime-branch"],
    routeTags: ["execution", "graph", "partition"],
  },
  {
    id: "graph-global-instance-ordinal",
    chapter: "execution",
    title: "Assign graph identities from global position",
    status: "built",
    summary:
      "PartitionedGraphTraceSource stamps each cloned trace ID with its computed global ordinal, while the non-partitioned CyclingGraphTraceSource uses a run-scoped GraphTraceInstanceSequence so independently prepared phases remain collision-free.",
    source: { path: "rust/runtime/src/graph/workload.rs", startLine: 94, endLine: 111, symbol: "GraphTraceInstanceSequence::take" },
    evidence: [
      { path: "rust/runtime/src/graph/workload.rs", symbol: "independently_budgeted_phase_sources_share_run_unique_instance_ids", kind: "unit" },
      { path: "rust/runtime/src/graph/workload.rs", symbol: "PartitionedGraphTraceSource::next_trace lines 331-350", kind: "boundary" },
    ],
    inputs: ["run-scoped sequence or partition global ordinal", "trace template ID"],
    outputs: ["trace.id suffixed with ::instance-{ordinal}"],
    state: ["run-scoped next ordinal for non-partitioned sources"],
    invariants: ["Every admitted root trace receives one run-unique ordinal.", "Partitioned IDs equal ownership global ordinals and therefore merge across cells."],
    complexity: {
      time: "O(1) ordinal take; O(template-ID length) formatting/stamping",
      memory: "O(1) retained sequence state; O(template-ID length) output ID allocation",
    },
    gates: ["trace source construction already validated templates"],
    failures: ["Checked u64 identity exhaustion returns GraphWorkloadError."],
    pseudocode: pseudocode(
      "ordinal = shared sequence.take() or partition global_ordinal",
      "sequence.take checks ordinal+1 for u64 overflow",
      "trace.id = template_id + '::instance-' + ordinal",
    ),
    frames: admissionFrames("step-3", ["cell", "worker"], {
      before: ["warmup emitted instances 0,1", "profiling shares sequence"], after: ["profiling receives instance 2"], invariant: "Phase-local budgets cannot reset run identity.",
    }, {
      before: ["next ordinal is u64::MAX"], after: ["identity exhaustion error"], invariant: "Identity wrap cannot create a collision.", activeLineId: "step-2",
    }),
    predecessors: ["partitioned-graph-source"],
    successors: [],
    routeTags: ["execution", "graph", "identity"],
  },
  {
    id: "two-level-partition",
    chapter: "execution",
    title: "Compose cell and thread ownership",
    status: "built",
    summary:
      "two_level_partition flattens the nested grid in cell-major stride order: nested_id = cell_id + cell_count * thread_id and nested_count = cell_count * worker_count. The tempting cell_id * worker_count + thread_id formula is explicitly wrong because it breaks equivalence with a flat modulo partition.",
    source: { path: "rust/runtime/src/runner_protocol/sharded_scheduled.rs", startLine: 105, endLine: 125, symbol: "two_level_partition" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/sharded_scheduled.rs", symbol: "two_level_partition_nests_and_tiles", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/sharded_scheduled.rs", symbol: "per_thread_slice_counts_match_global_two_level", kind: "unit" },
    ],
    inputs: ["cell_id", "cell_count", "thread_id", "worker_count"],
    outputs: ["ModuloCellPartition(nested_id, nested_count)"],
    state: ["none"],
    invariants: [
      "nested_id = cell_id + cell_count * thread_id.",
      "nested_count = cell_count * worker_count.",
      "Reject cell_id * worker_count + thread_id: it does not preserve flat modulo ownership.",
    ],
    complexity: { time: "O(1)", memory: "O(1)" },
    gates: ["cell_count > 0", "worker_count > 0", "coordinates in range", "checked arithmetic fits u32"],
    failures: ["Zero counts, out-of-range coordinates, or multiplication/addition overflow return an error."],
    pseudocode: pseudocode(
      "nested_id = cell_id + cell_count * thread_id",
      "nested_count = cell_count * worker_count",
      "return ModuloCellPartition(nested_id, nested_count)  // never cell_id*worker_count + thread_id",
    ),
    frames: admissionFrames("step-2", ["cell", "worker"], {
      before: ["cell 1/3", "thread 2/4"], after: ["nested_id = 7", "nested_count = 12"], invariant: "7 = 1 + 3*2; owned slots are 7 mod 12",
    }, {
      before: ["wrong flat formula gives 6"], after: ["formula rejected by contract"], invariant: "6 would belong to cell 0 under modulo-3, not cell 1.", activeLineId: "step-3",
    }),
    predecessors: ["modulo-cell-ownership", "owned-positions-tiling"],
    successors: ["thread-phase-slicing", "issuance-dispatch-injection"],
    routeTags: ["execution", "nested-sharding", "identity"],
  },
  {
    id: "thread-phase-slicing",
    chapter: "execution",
    title: "Slice request budgets and rates per thread",
    status: "built",
    summary:
      "slice_phase_for_thread clones a PhaseSpec, slices common.requests and common.prefill_concurrency through owned_positions, slices concurrency caps the same way with a floor of one, and divides Poisson/Constant/Gamma rate by W. UserCentric and FixedSchedule are returned unchanged.",
    source: { path: "rust/runtime/src/runner_protocol/sharded_scheduled.rs", startLine: 138, endLine: 192, symbol: "slice_phase_for_thread" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/sharded_scheduled.rs", symbol: "concurrency_phase_slices_requests_and_concurrency_by_w", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/sharded_scheduled.rs", symbol: "rate_phase_splits_rate_by_w_and_floors_caps", kind: "unit" },
    ],
    inputs: ["PhaseSpec", "thread_id", "worker_count"],
    outputs: ["cloned PhaseSpec with requests, rate, concurrency, and prefill_concurrency sliced where applicable"],
    state: ["none"],
    invariants: ["Thread request budgets tile the authored requests budget.", "The function does not slice sessions, ramps, UserCentric, or FixedSchedule fields."],
    complexity: { time: "O(phase fields)", memory: "O(cloned PhaseSpec size)" },
    gates: ["Upstream sharded execution admits supported request-bounded phases and valid worker coordinates."],
    failures: ["No local error result: the function clones and returns a PhaseSpec; unsupported trace-driven variants are left unchanged for upstream rejection."],
    pseudocode: pseudocode(
      "clone phase; slice common.requests and common.prefill_concurrency with owned_positions",
      "slice concurrency caps with owned_positions(...).max(1); divide Poisson/Constant/Gamma rate by workers",
      "leave UserCentric and FixedSchedule unchanged; return cloned PhaseSpec",
    ),
    frames: admissionFrames("step-1", ["cell", "worker"], {
      before: ["10 requests", "thread 1/3"], after: ["owns positions 1,4,7", "local requests 3"], invariant: "Thread totals tile all 10 requests.",
    }, {
      before: ["UserCentric or FixedSchedule phase"], after: ["cloned phase returned unchanged"], invariant: "Trace-driven fields are not partially or incorrectly sliced.", activeLineId: "step-3",
    }),
    predecessors: ["two-level-partition", "scheduled-budget-validation"],
    successors: ["scheduled-shard-runtime"],
    routeTags: ["execution", "scheduled", "budget", "rate"],
  },
  {
    id: "scheduled-shard-runtime",
    chapter: "execution",
    title: "Run one scheduler and transport per core",
    status: "built",
    summary:
      "run_sharded_scheduled spawns one OS thread per worker, builds a current-thread Tokio runtime and LocalSet in each, executes the unchanged scheduled phase engine, then absorbs outcomes on the coordinator.",
    source: { path: "rust/runtime/src/runner_protocol/sharded_scheduled.rs", startLine: 249, endLine: 392, symbol: "run_sharded_scheduled / merge_shards" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/sharded_scheduled.rs", symbol: "merge_shards lines 363-392", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "execute_scheduled_shard lines 2195-2424", kind: "boundary" },
    ],
    inputs: ["Arc<ShardedShared>", "profiling sidecars", "coordinator Clock"],
    outputs: ["mode-aware merged ScheduledShardOutcome"],
    state: ["W thread handles", "per-thread current_thread runtime + LocalSet", "once-per-cell sidecars"],
    invariants: ["No Arc/Mutex is placed on the per-request hot path.", "Each thread owns its scheduler, transport, capture, and nested partition."],
    complexity: { time: "O(work + W joins)", memory: "O(W*local runtime + selected capture mode)" },
    gates: ["workers > 1", "request-bounded phase variants", "no static accuracy evaluator"],
    failures: ["A worker build/run/join error aborts the sharded run with thread context."],
    pseudocode: pseudocode(
      "for thread_id in 0..W: spawn OS thread with current_thread runtime + LocalSet",
      "inside thread: execute_scheduled_shard(shared, thread_id)",
      "join all; absorb mode-matching shard outcomes; sort retained records by global request_index",
    ),
    frames: admissionFrames("step-1", ["cell", "worker"], {
      before: ["workers = 4", "request-bounded phase"], after: ["4 isolated local runtimes", "4 nested partitions"], invariant: "Scheduler and transport remain co-located per core.",
    }, {
      before: ["one shard returns Folded", "another returns Retained"], after: ["merge error"], invariant: "All shards in one run use the same storage mode.", activeLineId: "step-3",
    }),
    predecessors: ["thread-phase-slicing", "issuance-dispatch-injection"],
    successors: ["retain-record-capture", "streaming-exact-fold", "sketch-scratch-harvest"],
    routeTags: ["execution", "scheduled", "thread-per-core"],
  },
  {
    id: "cell-envelope-fetch",
    chapter: "execution",
    title: "Fetch opaque cell bytes after START",
    status: "feature-gated",
    summary:
      "fetch_cell_envelope reads the controller coordinate and validates the cell partition coordinates through ModuloCellPartition::from_env, connects over velo, registers the cell, waits on the selected phaser or event START barrier, captures the shared timing origin, and returns reply.envelope as opaque Vec<u8>.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_cell.rs", startLine: 405, endLine: 468, symbol: "fetch_cell_envelope" },
    evidence: [
      { path: "rust/runtime/src/cellular/transport/velo_transport.rs", symbol: "VeloCellClient::register lines 252-267", kind: "boundary" },
      { path: "rust/runtime/src/cellular/transport/velo_transport.rs", symbol: "VeloCellClient::await_start lines 269-278", kind: "boundary" },
    ],
    inputs: ["AIPERF_CELL_CONTROLLER_ADDR", "AIPERF_CELL_ID/_COUNT", "optional phaser-start flag"],
    outputs: ["opaque reply.envelope Vec<u8> after START"],
    state: ["velo instance and controller peer", "RegisterReply envelope + start event", "optional phaser subscription"],
    invariants: ["Environment partition coordinates are validated before registration.", "The function does not decode or semantically validate reply.envelope; bytes are released only after the selected START gate."],
    complexity: { time: "O(envelope bytes)", memory: "O(envelope bytes)" },
    gates: ["velo feature compiled", "controller address present", "cell partition env valid", "registration succeeds"],
    failures: ["Missing env, controller connect/register, MessagePack reply decode, poisoned event, or finalized phaser errors before returning bytes."],
    pseudocode: pseudocode(
      "read controller coordinate and cell_id from ModuloCellPartition::from_env; connect velo",
      "reply = VeloCellClient.register(cell_id); await phaser generation 1 or reply.start_event",
      "capture shared timing origin; return reply.envelope",
    ),
    frames: admissionFrames("step-3", ["controller", "wire", "cell"], {
      before: ["cell 2/4 connects", "registration reply carries envelope + start event"], after: ["START releases", "envelope bytes returned"], invariant: "Envelope release follows the run-wide gate.",
    }, {
      before: ["START event poisoned or phaser finalized"], after: ["fetch returns error", "envelope not released"], invariant: "Controller abort cannot become an unsynchronized run.", activeLineId: "step-2",
    }),
    predecessors: ["cell-envelope-construction", "local-cell-launch", "synchronized-start"],
    successors: ["scheduled-graph-runtime-branch", "issuance-dispatch-injection"],
    routeTags: ["execution", "cell", "messagepack", WIRE_FACTS.partition],
  },
  {
    id: "issuance-dispatch-injection",
    chapter: "execution",
    title: "Inject one partition into sampler and issuer",
    status: "built",
    summary:
      "execute_scheduled_shard computes one two-level ModuloCellPartition and injects it both into issuance_authority_for and PreparedNativeConversationSourceFactory, preserving global ordinal and sampler ownership alignment.",
    source: { path: "rust/runtime/src/runner_protocol/execute.rs", startLine: 2203, endLine: 2308, symbol: "execute_scheduled_shard partition injection" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_cell.rs", symbol: "issuance_authority_for", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "run_capture_finish_stamps_global_index_and_joins_worker_records", kind: "unit" },
    ],
    inputs: ["two-level ModuloCellPartition", "phase ordinal bases", "conversation source factory"],
    outputs: ["partitioned sampler source", "CellularAutonomousIssuer using the same coordinates"],
    state: ["per-thread RunCapture", "per-thread source factory"],
    invariants: ["Sampler and issuer receive the exact same partition value.", "within*(cell_count*worker_count)+nested_id identifies the owned global slot."],
    complexity: { time: "O(1) injection", memory: "O(1)" },
    gates: ["valid two-level partition", "phase bases available"],
    failures: ["Partition construction, endpoint preparation, or source creation errors abort the shard."],
    pseudocode: pseudocode(
      "partition = two_level_partition(cell_id, cells, thread_id, workers)",
      "capture.issuance = issuance_authority_for(partition)",
      "source_factory.cell_partition = Some(partition); run sliced phases",
    ),
    frames: admissionFrames("step-2", ["cell", "worker"], {
      before: ["nested partition 7/12", "capture constructed"], after: ["issuance authority receives 7/12"], invariant: "Global request_index uses the nested partition.",
    }, {
      before: ["issuance authority uses 7/12"], after: ["source factory receives 7/12", "sliced phases run"], invariant: "Sampler ownership and request_index stride use the same partition.", activeLineId: "step-3",
    }),
    predecessors: ["two-level-partition", "cellular-issuance-authority"],
    successors: ["scheduled-shard-runtime", "terminal-record-finalization"],
    routeTags: ["execution", "scheduled", "issuance", "sampler"],
  },
  {
    id: "scheduled-graph-runtime-branch",
    chapter: "execution",
    title: "Choose graph or scheduled runtime once",
    status: "built",
    summary:
      "execute_native performs a single dataset-plan branch: Graph plans require no static-accuracy state and enter execute_graph_native; all non-graph plans enter execute_scheduled_native.",
    source: { path: "rust/runtime/src/runner_protocol/execute.rs", startLine: 1303, endLine: 1319, symbol: "execute_native" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "execute_graph_native lines 1396-1851", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "execute_native_inner lines 2427-3297", kind: "boundary" },
    ],
    inputs: ["NativeRunSpec.dataset", "prepared accuracy state", "transport/graph factories"],
    outputs: ["NativeReport from exactly one runtime family"],
    state: ["run-owned sidecars"],
    invariants: ["A Graph plan is never reinterpreted as a linear dataset.", "Static accuracy state cannot enter graph execution."],
    complexity: { time: "O(1) dispatch", memory: "O(1)" },
    gates: ["NativeRunSpec already validated"],
    failures: ["Graph plus prepared static-accuracy state fails before graph execution."],
    pseudocode: pseudocode(
      "if dataset is Graph: require accuracy is None; execute_graph_native",
      "else: execute_scheduled_native",
    ),
    frames: admissionFrames("step-1", ["cell", "worker"], {
      before: ["dataset = Graph", "accuracy = None"], after: ["graph runtime selected"], invariant: "Graph bundle remains canonical.",
    }, {
      before: ["dataset = Graph", "accuracy prepared"], after: ["fail closed"], invariant: "Static accuracy never leaks into graph execution.", activeLineId: "step-1",
    }),
    predecessors: ["run-kind-classification", "cell-envelope-fetch"],
    successors: ["partitioned-graph-source", "partitioned-scheduled-sampler"],
    routeTags: ["execution", "branch", "scheduled", "graph"],
  },
];

const CAPTURE_ALGORITHMS: readonly AlgorithmDefinition[] = [
  {
    id: "terminal-record-finalization",
    chapter: "capture",
    title: "Finalize each terminal record once",
    status: "built",
    summary:
      "The scheduled CapturePhaseProcessor selects exactly one terminal path: fold-and-drop consumes or synthesizes a worker record and folds it immediately; retain mode labels it for the scheduled finish-time UUID join and optionally snapshots it for live consumers.",
    source: { path: "rust/runtime/src/runner_protocol/execute.rs", startLine: 5722, endLine: 5778, symbol: "CapturePhaseProcessor::process" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "run_capture_finish_synthesizes_fallback_for_pre_worker_failures", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "exact_fold_flags_and_dense_dispatch_ordinals", kind: "unit" },
    ],
    inputs: ["IssuedCredit", "TurnDispatchOutcome", "RunCapture mode"],
    outputs: ["one retained label/snapshot or one folded record"],
    state: ["staged live record", "fold ordinal map", "streaming accumulator"],
    invariants: ["Every completed turn follows exactly one capture path.", "A pre-worker failure is synthesized rather than omitted."],
    complexity: { time: "O(record metrics)", memory: "mode-dependent" },
    gates: ["terminal outcome available"],
    failures: ["Only a fold-and-drop record-lane write can return an error through fold_streaming; retain-mode labeling, live-sink emission, and heartbeat observation are infallible here."],
    pseudocode: pseudocode(
      "if folds_records: take staged record or synthesize fallback",
      "  take exact-fold ordinal when enabled; fold_streaming; return",
      "label retained record; optionally snapshot once for live sink and heartbeat",
    ),
    frames: admissionFrames("step-2", ["worker", "cell"], {
      before: ["successful turn staged a worker record", "exact-fold enabled"], after: ["record folded once", "clean heavy record dropped"], invariant: "No finish-time join also processes this turn.",
    }, {
      before: ["turn failed before worker staging"], after: ["fallback record synthesized and folded"], invariant: "Error and completed counts still include the turn.", activeLineId: "step-1",
    }),
    predecessors: ["scheduled-shard-runtime", "issuance-dispatch-injection"],
    successors: ["retain-record-capture", "streaming-exact-fold", "sketch-scratch-harvest"],
    routeTags: ["capture", "terminal", "record"],
  },
  {
    id: "retain-record-capture",
    chapter: "capture",
    title: "Join retained worker records in dispatch order",
    status: "built",
    summary:
      "RunCapture::finish resolves drained records by UUID, synthesizes missing worker records, emits in identity dispatch order, and stamps each row with the issuance authority's global request_index.",
    source: { path: "rust/runtime/src/runner_protocol/execute.rs", startLine: 5308, endLine: 5367, symbol: "RunCapture::finish" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "run_capture_finish_stamps_global_index_and_joins_worker_records", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "run_capture_finish_worker_split_matches_single_worker_byte_for_byte", kind: "unit" },
    ],
    inputs: ["dispatch-ordered identities", "coordinator labels", "issued times", "worker-drained records"],
    outputs: ["Vec<CapturedRecord> in dispatch order with global request_index"],
    state: ["identity list", "UUID maps for labels, outputs, raw exchanges, drained records"],
    invariants: ["Worker-local drain order never determines report order.", "Each dispatched identity resolves to exactly one record and one unique global index."],
    complexity: { time: "O(records)", memory: "O(records)" },
    gates: ["folds_records == false"],
    failures: ["Duplicate drained UUIDs, unresolved identities, count mismatch, or missing issuer timestamps fail finalization."],
    pseudocode: pseudocode(
      "records_by_uuid = resolve drained records; synthesize missing identities",
      "for identity in dispatch order: remove UUID record; patch phase/session/admit/global index",
      "emit CapturedRecord with output and raw exchange",
    ),
    frames: admissionFrames("step-2", ["worker", "cell"], {
      before: ["drain order A,C,B", "identity order A,B,C"], after: ["A, B, C removed in identity order", "coordinator fields patched"], invariant: "Worker completion order cannot choose row placement.",
    }, {
      before: ["A, B, C records patched"], after: ["CapturedRecord values emitted A,B,C with output/raw fields"], invariant: "Final output order is dispatch identity order.", activeLineId: "step-3",
    }),
    predecessors: ["terminal-record-finalization"],
    successors: ["column-store-append", "partition-messagepack-encode"],
    routeTags: ["capture", "retain", "exact", "O(records)"],
  },
  {
    id: "streaming-exact-fold",
    chapter: "capture",
    title: "Fold exact rows at completion",
    status: "built",
    summary:
      "RunCapture::fold_record stamps coordinator fields and a dense request_index, processes the record into an Exact accumulator, streams any lane/OTLP artifact, retains only errors, and drops clean record payloads.",
    source: { path: "rust/runtime/src/runner_protocol/execute.rs", startLine: 5369, endLine: 5480, symbol: "RunCapture::fold_streaming / fold_record" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "exact_fold_matches_legacy_retain_byte_for_byte", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "fold_record_folds_otel_matching_post_run_loop", kind: "unit" },
    ],
    inputs: ["RecordIngest", "phase/session/admit facts", "dense fold request_index"],
    outputs: ["Exact accumulator row", "optional streamed artifact/OTLP observation", "errored subset"],
    state: ["MetricsAccumulator in Exact mode", "record lane", "errored records"],
    invariants: ["Exact-fold and sketch are mutually exclusive.", "Dense absolute or shard-local slots preserve exact columns; only errored/canceled records remain resident."],
    complexity: { time: "O(metric catalog per record)", memory: "O(exact scalar columns), not O(token-arrival payloads)" },
    gates: ["exact_fold_eligible", "MetricsStorageMode::Exact"],
    failures: ["MetricsAccumulator::process_record and OTLP observation are infallible; only record_lane.write can propagate an error. Eligibility keeps incompatible retained-record consumers off this path."],
    pseudocode: pseudocode(
      "stamp phase, session_num, admit_ns, and request_index",
      "accumulator.process_record(record)",
      "stream requested lane/OTLP; retain only error; drop clean record",
    ),
    frames: admissionFrames("step-2", ["worker", "cell"], {
      before: ["record for dispatch slot 4 completes before slot 2"], after: ["insert exact row at slot 4", "clean payload dropped"], invariant: "Completion order does not move the exact row.",
    }, {
      before: ["errored record completes"], after: ["metrics folded", "CapturedRecord retained for error grouping"], invariant: "Dropping clean records does not erase error detail.", activeLineId: "step-3",
    }),
    predecessors: ["terminal-record-finalization"],
    successors: ["column-store-append", "ingested-count-preservation", "partition-messagepack-encode"],
    routeTags: ["capture", "exact-fold", "fold-and-drop"],
  },
  {
    id: "sketch-scratch-harvest",
    chapter: "capture",
    title: "Harvest one scratch row into sketches",
    status: "approximate",
    summary:
      "MetricsAccumulator processes a record into a transient row; in Sketch mode ColumnStore harvests every finite record metric into per-(phase,tag) TagSketch values, then clears exact rows while preserving ingested_total.",
    source: { path: "rust/runtime/src/metrics_core/accumulator.rs", startLine: 491, endLine: 523, symbol: "MetricsAccumulator::process_record_with_token_arrivals" },
    evidence: [
      { path: "rust/runtime/src/metrics_core/store.rs", symbol: "harvest_row_to_sketch lines 811-839", kind: "boundary" },
      { path: "rust/runtime/src/metrics_core/accumulator.rs", symbol: "sketch_mode_keeps_counts_sums_extrema_exact_and_percentiles_close", kind: "unit" },
    ],
    inputs: ["RecordIngest", "Sketch storage config"],
    outputs: ["updated SketchColumns", "empty exact row storage", "incremented ingested_total"],
    state: ["single transient exact row", "FxHashMap<(Phase, u16), TagSketch>"],
    invariants: ["Only finite values enter sketches.", "Rows are cleared after harvest; ingested_total is not cleared."],
    complexity: { time: "O(record metric columns)", memory: "O(tags * digest compression)" },
    gates: ["MetricsStorageMode::Sketch"],
    failures: ["Unsupported per-record artifacts are rejected before execution; non-finite metric values are skipped."],
    pseudocode: pseudocode(
      "insert/process one record into scratch row",
      "for finite metric value: sketch[(phase, tag)].add(value)",
      "increment ingested_total; clear exact rows",
    ),
    frames: admissionFrames("step-2", ["worker", "cell"], {
      before: ["scratch row contains TTFT=7 and NaN metric"], after: ["TTFT digest/count updated", "NaN ignored"], invariant: "Sketches contain finite observations only.",
    }, {
      before: ["harvest complete", "record_count = 1"], after: ["record_count = 0", "ingested_count increased"], invariant: "Clearing row storage cannot clear the processed-record total.", activeLineId: "step-3",
    }),
    predecessors: ["terminal-record-finalization"],
    successors: ["tdigest-insert-compress", "welford-aggregate-state", "ingested-count-preservation"],
    routeTags: ["capture", "sketch", "bounded-memory", "approximate"],
  },
  {
    id: "tdigest-insert-compress",
    chapter: "capture",
    title: "Bound streaming quantile state",
    status: "approximate",
    summary:
      "TDigest::add ignores non-finite values, appends a unit centroid, updates exact count/min/max, and eagerly compresses above a deterministic threshold; compression sorts and greedily clusters adjacent centroids under the K1 scale.",
    source: { path: "rust/runtime/src/cellular/sketch.rs", startLine: 40, endLine: 99, symbol: "TDigest / TDigest::add" },
    evidence: [
      { path: "rust/runtime/src/cellular/sketch.rs", symbol: "TDigest::compress lines 198-232", kind: "boundary" },
      { path: "rust/runtime/src/cellular/sketch.rs", symbol: "centroid_count_stays_bounded_by_compression", kind: "unit" },
    ],
    inputs: ["finite f64 value", "compression delta"],
    outputs: ["updated centroid set", "exact count/min/max", "approximate interior quantiles"],
    state: ["centroids", "total_weight", "min", "max", "compress_threshold"],
    invariants: ["NaN and infinities never enter the digest.", "q=0 and q=1 use exact extrema; interior quantiles are approximate."],
    complexity: { time: "amortized append plus periodic O(C log C) compression", memory: "O(compression)" },
    gates: ["compression clamped to at least 1"],
    failures: ["No error is raised for non-finite input; it is intentionally ignored."],
    pseudocode: pseudocode(
      "if value is not finite: return",
      "append centroid(value,1); update total_weight/min/max",
      "if centroid count > threshold: sort and greedily compress under K1 scale",
    ),
    frames: admissionFrames("step-2", ["worker"], {
      before: ["digest count 8", "add finite 12.5"], after: ["unit centroid appended", "count 9", "extrema updated"], invariant: "Count equals finite values ingested.",
    }, {
      before: ["centroids exceed threshold"], after: ["sorted neighboring centroids clustered"], invariant: "Compression bounds state while retaining exact min/max.", activeLineId: "step-3",
    }),
    predecessors: ["sketch-scratch-harvest"],
    successors: ["tagged-sketch-merge"],
    routeTags: ["capture", "sketch", "tdigest", "quantiles"],
  },
  {
    id: "welford-aggregate-state",
    chapter: "capture",
    title: "Track exact counts and streaming variance",
    status: "approximate",
    summary:
      "TagSketch::add updates integer count, floating sum, exact min/max, Welford mean/M2, and its t-digest; merge uses Chan's parallel Welford update and merges the digest.",
    source: { path: "rust/runtime/src/metrics_core/store.rs", startLine: 512, endLine: 621, symbol: "TagSketch::add / merge / std" },
    evidence: [
      { path: "rust/runtime/src/metrics_core/accumulator.rs", symbol: "sketch_partitions_merge_associatively", kind: "unit" },
      { path: "rust/runtime/src/metrics_core/store.rs", symbol: "TagSketch::merge lines 567-589", kind: "boundary" },
    ],
    inputs: ["finite value stream or another TagSketch"],
    outputs: ["count, sum, min, max, mean, M2, digest"],
    state: ["u64 count", "f64 sum/mean/M2", "exact extrema", "TDigest"],
    invariants: ["Count and extrema remain exact.", "Mean/M2 and floating sum may vary by a few ULPs with merge/completion order."],
    complexity: { time: "O(1) add; O(digest merge) merge", memory: "O(digest compression)" },
    gates: ["finite value"],
    failures: ["Empty aggregates contribute nothing; non-finite values are excluded before add."],
    pseudocode: pseudocode(
      "add: count++; sum += x; update min/max; Welford mean/M2; digest.add(x)",
      "merge: combine count/sum/extrema; apply Chan mean/M2; digest.merge(other)",
      "std = sqrt(M2 / count) for population ddof=0",
    ),
    frames: admissionFrames("step-1", ["worker"], {
      before: ["count=2", "mean=4", "M2=2", "x=7"], after: ["count=3", "mean/M2 advanced", "digest advanced"], invariant: "One finite value changes every aggregate exactly once.",
    }, {
      before: ["two shard aggregates"], after: ["Chan-combined mean/M2", "digests merged"], invariant: "Merged count is the exact sum of shard counts.", activeLineId: "step-2",
    }),
    predecessors: ["sketch-scratch-harvest"],
    successors: ["tagged-sketch-merge"],
    routeTags: ["capture", "sketch", "welford", "variance"],
  },
  {
    id: "tagged-sketch-merge",
    chapter: "capture",
    title: "Merge per-phase metric sketches",
    status: "approximate",
    summary:
      "SketchColumns::merge walks the other store's (Phase, u16) entries, creates missing destinations with the same compression, and merges each TagSketch.",
    source: { path: "rust/runtime/src/metrics_core/store.rs", startLine: 633, endLine: 693, symbol: "SketchColumns::merge" },
    evidence: [
      { path: "rust/runtime/src/metrics_core/accumulator.rs", symbol: "sketch_partitions_merge_associatively", kind: "unit" },
      { path: "rust/runtime/src/cellular/sketch.rs", symbol: "merge_is_deterministic_regardless_of_shard_order", kind: "unit" },
    ],
    inputs: ["destination SketchColumns", "source SketchColumns"],
    outputs: ["union keyed by (Phase, metric-tag index as u16)"],
    state: ["FxHashMap<(Phase, u16), TagSketch>"],
    invariants: ["Warmup and profiling values never mix.", "Different metric tags never share aggregate state."],
    complexity: { time: "O(tags * digest merge)", memory: "O(union of tags * compression)" },
    gates: ["both ColumnStore values expose SketchColumns when invoked through append_store"],
    failures: ["No local error result; non-finite observations were already excluded by TagSketch::add."],
    pseudocode: pseudocode(
      "for ((phase, tag), other_aggregate) in other",
      "  destination = entry(phase, tag).or_insert(empty with compression)",
      "  destination.merge(other_aggregate)",
    ),
    frames: admissionFrames("step-3", ["controller"], {
      before: ["cell 0 has profiling/TTFT", "cell 1 has profiling/TTFT"], after: ["one profiling/TTFT aggregate"], invariant: "Counts add and digests merge under the same key.",
    }, {
      before: ["warmup/TTFT and profiling/TTFT"], after: ["two distinct entries remain after their per-key merges"], invariant: "Phase boundaries survive merge.", activeLineId: "step-3",
    }),
    predecessors: ["tdigest-insert-compress", "welford-aggregate-state"],
    successors: ["column-store-append"],
    routeTags: ["capture", "sketch", "phase-tag", "merge"],
  },
  {
    id: "column-store-append",
    chapter: "capture",
    title: "Append exact rows or merge sketches",
    status: "built",
    summary:
      "ColumnStore::append_store always adds ingested_total, merges SketchColumns only when both stores have sketches, then appends and remaps any source rows. The method performs no exact/sketch mode comparison. MetricsAccumulator::merge separately checks MetricsConfig equality before calling it; merge_store_partitions calls append_store directly.",
    source: { path: "rust/runtime/src/metrics_core/store.rs", startLine: 935, endLine: 1047, symbol: "ColumnStore::append_store" },
    evidence: [
      { path: "rust/runtime/src/metrics_core/store.rs", symbol: "worker_stores_merge_with_numeric_categorical_and_ragged_alignment", kind: "unit" },
      { path: "rust/runtime/src/metrics_core/accumulator.rs", symbol: "MetricsAccumulator::merge lines 540-569", kind: "boundary" },
    ],
    inputs: ["destination ColumnStore", "source ColumnStore"],
    outputs: ["destination with summed ingested_total, both-present sketches merged, and source rows appended"],
    state: ["row columns/ragged metadata", "optional SketchColumns", "ingested_total"],
    invariants: ["append_store itself does not reject a storage-mode mismatch.", "Sketches merge only when both Option<SketchColumns> values are Some; source rows append independently."],
    complexity: { time: "O(rows+columns) Exact; O(tags*digest merge) Sketch", memory: "Exact O(rows); Sketch O(tags*compression)" },
    gates: ["source stores with rows must be dense before append", "MetricsAccumulator::merge callers separately require identical MetricsConfig"],
    failures: ["The only local rejection is the assertion that every source row is occupied; append_store returns no mode/config error."],
    pseudocode: pseudocode(
      "self.ingested_total += other.ingested_total",
      "if self.sketch and other.sketch are both Some: merge sketches",
      "if other has rows: assert dense, then append/remap row, numeric, categorical, and ragged columns",
    ),
    frames: admissionFrames("step-3", ["controller"], {
      before: ["two exact shard stores with dense local rows"], after: ["rows concatenated into one store"], invariant: "No row is coerced into a sketch.",
    }, {
      before: ["only one store has SketchColumns", "source has no rows"], after: ["no mode error", "ingested_total still added", "no sketch merge"], invariant: "append_store has no cross-mode validation; caller configuration is a separate concern.", activeLineId: "step-2",
    }),
    predecessors: ["retain-record-capture", "streaming-exact-fold", "tagged-sketch-merge"],
    successors: ["ingested-count-preservation", "partition-messagepack-encode"],
    routeTags: ["capture", "column-store", "append", "mode-aware"],
  },
  {
    id: "ingested-count-preservation",
    chapter: "capture",
    title: "Preserve record totals after row clearing",
    status: "built",
    summary:
      "ColumnStore keeps ingested_total separately from row_count. Sketch harvest increments ingested_total and clear_rows removes row vectors, so record_count() can be 0 while ingested_count() remains the true nonzero processed total; append adds these totals.",
    source: { path: "rust/runtime/src/metrics_core/store.rs", startLine: 1090, endLine: 1102, symbol: "ColumnStore::record_count / ingested_count" },
    evidence: [
      { path: "rust/runtime/src/metrics_core/accumulator.rs", symbol: "MetricsAccumulator::ingested_count lines 476-478", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/execute.rs", symbol: "cell store shipping uses ingested_count for sketch lines 3083-3114", kind: "boundary" },
    ],
    inputs: ["processed record events", "row harvest/clear operations", "appended stores"],
    outputs: ["stable u64 ingested_count independent of retained row count"],
    state: ["ingested_total", "row vectors"],
    invariants: ["ingested_count counts processed records even when no rows remain.", "Sketch record_count()==0 does not mean an empty run."],
    complexity: { time: "O(1) query/update", memory: "O(1)" },
    gates: ["none"],
    failures: ["Using record_count as issued count in sketch mode undercounts to zero; shipping code therefore reads ingested_count."],
    pseudocode: pseudocode(
      "on record ingest: ingested_total += 1",
      "on sketch harvest: clear row vectors but keep ingested_total",
      "report issued = ingested_count; record_count may be 0",
    ),
    frames: admissionFrames("step-2", ["worker", "cell"], {
      before: ["ingested_total=9", "scratch row for record 10"], after: ["ingested_total=10", "row harvested then cleared"], invariant: "The processed total survives row eviction.",
    }, {
      before: ["sketch store record_count=0", "ingested_count=10"], after: ["cell ships issued=10"], invariant: "Zero retained rows is not interpreted as zero work.", activeLineId: "step-3",
    }),
    predecessors: ["sketch-scratch-harvest", "column-store-append"],
    successors: ["partition-messagepack-encode"],
    routeTags: ["capture", "count", "sketch", "shipping"],
  },
  {
    id: "partition-messagepack-encode",
    chapter: "capture",
    title: "Encode terminal partitions for the wire",
    status: "built",
    summary:
      "RecordsShardPartition::to_bytes and ColumnStorePartition::to_bytes directly serialize their serde structs with rmp_serde::to_vec. Their wire fields are only {cell_id, records} and {cell_id, store}; heartbeat counters are separate and there is no codec-layer semantic validation.",
    source: { path: "rust/runtime/src/cellular/shard.rs", startLine: 80, endLine: 83, symbol: "RecordsShardPartition::to_bytes" },
    evidence: [
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "ColumnStorePartition::to_bytes lines 341-344", kind: "boundary" },
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "records_partition_wire_round_trip_is_lossless_and_stable", kind: "unit" },
    ],
    inputs: ["RecordsShardPartition { cell_id, records } or ColumnStorePartition { cell_id, store }"],
    outputs: ["MessagePack bytes from rmp_serde::to_vec"],
    state: ["none"],
    invariants: ["The codec adds no cell_count, heartbeat counters, or epoch_ns fields.", "A store partition carries ingested_total only as part of its serialized ColumnStore."],
    complexity: { time: "O(partition payload)", memory: "O(encoded payload)" },
    gates: ["partition value already constructed"],
    failures: ["Only rmp_serde serialization failure is mapped to PartitionCodecError::Encode."],
    pseudocode: pseudocode(
      "take RecordsShardPartition {cell_id,records} or ColumnStorePartition {cell_id,store}",
      "bytes = rmp_serde::to_vec(self)",
      "map serializer error to PartitionCodecError::Encode; return bytes",
    ),
    frames: admissionFrames("step-2", ["cell", "wire"], {
      before: ["ColumnStorePartition {cell_id: 1, store}"], after: ["rmp_serde::to_vec bytes"], invariant: "Only cell_id and the serialized store are partition fields.",
    }, {
      before: ["rmp_serde::to_vec completed"], after: ["MessagePack bytes returned"], invariant: "Return adds no metadata or semantic validation.", activeLineId: "step-3",
    }),
    predecessors: ["retain-record-capture", "column-store-append", "ingested-count-preservation"],
    successors: ["partition-messagepack-decode"],
    routeTags: ["capture", "wire", "messagepack", WIRE_FACTS.partition],
  },
  {
    id: "partition-messagepack-decode",
    chapter: "capture",
    title: "Decode terminal partition bytes",
    status: "built",
    summary:
      "RecordsShardPartition::from_bytes and ColumnStorePartition::from_bytes directly deserialize MessagePack with rmp_serde::from_slice. They perform no follow-up semantic validation; the only codec failure is deserialization into the target struct.",
    source: { path: "rust/runtime/src/cellular/shard.rs", startLine: 85, endLine: 88, symbol: "RecordsShardPartition::from_bytes" },
    evidence: [
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "ColumnStorePartition::from_bytes lines 346-349", kind: "boundary" },
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "store_partition_wire_round_trip_preserves_summary", kind: "unit" },
    ],
    inputs: ["MessagePack bytes"],
    outputs: ["deserialized RecordsShardPartition or ColumnStorePartition"],
    state: ["none"],
    invariants: ["The decoded target has only its declared serde fields.", "from_bytes does not validate cell ownership or metric semantics."],
    complexity: { time: "O(partition payload)", memory: "O(decoded payload)" },
    gates: ["bytes deserialize as the requested partition type"],
    failures: ["Malformed or type-incompatible bytes produce PartitionCodecError::Decode; there is no Validation variant."],
    pseudocode: pseudocode(
      "partition = rmp_serde::from_slice(bytes)",
      "map deserializer error to PartitionCodecError::Decode",
      "return partition without semantic validation",
    ),
    frames: admissionFrames("step-3", ["wire", "controller"], {
      before: ["valid MessagePack for {cell_id,records}"], after: ["RecordsShardPartition returned"], invariant: "Decode reconstructs the declared fields only.",
    }, {
      before: ["truncated or wrong-shape bytes"], after: ["PartitionCodecError::Decode"], invariant: "Decode failure is the codec's only rejection path.", activeLineId: "step-2",
    }),
    predecessors: ["partition-messagepack-encode"],
    successors: [],
    routeTags: ["capture", "wire", "messagepack", WIRE_FACTS.partition],
  },
];

const MERGE_ALGORITHMS: readonly AlgorithmDefinition[] = [
  {
    id: "scheduled-global-ordinal-merge", chapter: "merge", title: "Re-ingest scheduled records in global ordinal order", status: "built",
    summary: "The retain path flattens every cell partition, validates each request_index against a dense 0..total permutation, stably sorts by that ordinal, and processes records in exactly the single-cell slot and append order.",
    source: { path: "rust/runtime/src/cellular/shard.rs", startLine: 108, endLine: 141, symbol: "merge_records_in_global_order" },
    evidence: [
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "merged_cell_records_are_byte_identical_to_a_single_cell_run lines 502-536", kind: "unit" },
      { path: "rust/e2e-tests/tests/test_cellular.rs", symbol: "test_cellular_matches_single_cell lines 98-157", kind: "e2e" },
    ],
    inputs: ["MetricsConfig", "RecordsShardPartition values whose union carries scheduled global request_index ordinals"],
    outputs: ["MetricsAccumulator populated in ascending global dispatch order"],
    state: ["flattened records", "seen[0..total)", "order-sensitive accumulator append order"],
    invariants: ["The ordinal union is a permutation of 0..total before any record is inserted.", "Warmup and profiling absolute slots match the single-cell phase blocks."],
    complexity: { time: "O(R log R)", memory: "O(R)" },
    gates: ["scheduled retain partition path", "every record has a dense unique in-range global ordinal"],
    failures: ["Missing, duplicate, or out-of-range ordinals return RecordsMergeError before accumulation."],
    pseudocode: pseudocode("records = flatten(partitions)", "validate every request_index against seen[0..records.len)", "stable-sort records by request_index", "process each record into a fresh accumulator"),
    frames: admissionFrames("step-2", ["controller"], {
      before: ["cell 0 ordinals {0,2}", "cell 1 ordinals {1,3}"], after: ["seen={0,1,2,3}", "records sorted 0,1,2,3"], invariant: "Arrival order cannot change re-ingest order.",
    }, {
      before: ["all ordinals validated and sorted"], after: ["record slots and ragged append order match one-cell"], invariant: "Each record contributes exactly once.", activeLineId: "step-4",
    }),
    predecessors: ["partition-messagepack-decode", "controller-partition-collection"], successors: ["final-report-assembly"],
    routeTags: ["merge", "scheduled", "retain", "byte-exact"],
  },
  {
    id: "ordinal-duplicate-detection", chapter: "merge", title: "Reject duplicate global ordinals", status: "built",
    summary: "Validation marks seen[ordinal] with mem::replace; a second claim for the same in-range slot fails immediately, before sorting or accumulator mutation.",
    source: { path: "rust/runtime/src/cellular/shard.rs", startLine: 108, endLine: 141, symbol: "merge_records_in_global_order" },
    evidence: [{ path: "rust/runtime/src/cellular/shard.rs", symbol: "merge_rejects_ordinals_that_are_not_a_permutation lines 608-638", kind: "unit" }],
    inputs: ["record.request_index", "seen bitmap sized to record count"], outputs: ["unique seen mark or DuplicateOrdinal(ordinal)"],
    state: ["seen[ordinal] boolean"], invariants: ["No two records may occupy one absolute metric slot."],
    complexity: { time: "O(1) per record", memory: "O(R) shared bitmap" }, gates: ["ordinal is present and ordinal < total"],
    failures: ["A second mark returns RecordsMergeError::DuplicateOrdinal without partial accumulation."],
    pseudocode: pseudocode("if ordinal >= total: defer to range rejection", "already_seen = replace(seen[ordinal], true)", "if already_seen: return DuplicateOrdinal(ordinal)"),
    frames: admissionFrames("step-2", ["controller"], {
      before: ["seen[4]=false", "record ordinal=4"], after: ["seen[4]=true"], invariant: "First ownership claim is accepted.",
    }, {
      before: ["seen[4]=true", "second record ordinal=4"], after: ["DuplicateOrdinal(4)"], invariant: "Overlapping cell claims never overwrite.", activeLineId: "step-3",
    }),
    predecessors: ["controller-partition-collection"], successors: ["scheduled-global-ordinal-merge"],
    routeTags: ["merge", "ordinal", "fail-closed"],
  },
  {
    id: "ordinal-missing-detection", chapter: "merge", title: "Reject records without global ordinals", status: "built",
    summary: "A scheduled retain record with request_index=None is rejected as MissingOrdinal. Combined with uniqueness and range checks over exactly total records, accepted ordinals necessarily cover every dense slot.",
    source: { path: "rust/runtime/src/cellular/shard.rs", startLine: 108, endLine: 141, symbol: "merge_records_in_global_order" },
    evidence: [{ path: "rust/runtime/src/cellular/shard.rs", symbol: "merge_rejects_ordinals_that_are_not_a_permutation lines 631-638", kind: "unit" }],
    inputs: ["record.request_index: Option<usize>"], outputs: ["present ordinal or MissingOrdinal"],
    state: ["validation scan"], invariants: ["Every scheduled retain record carries its absolute dispatch slot.", "Unique in-range ordinals across R records imply no hole in 0..R."],
    complexity: { time: "O(1) per record", memory: "O(1) incremental" }, gates: ["scheduled global-order merge"],
    failures: ["None returns RecordsMergeError::MissingOrdinal before any record placement."],
    pseudocode: pseudocode("read record.request_index", "if None: return MissingOrdinal", "otherwise continue duplicate/range validation"),
    frames: admissionFrames("step-1", ["controller"], {
      before: ["record request_index=Some(7)"], after: ["ordinal 7 enters validation"], invariant: "A concrete slot accompanies the record.",
    }, {
      before: ["record request_index=None"], after: ["MissingOrdinal"], invariant: "An indexless scheduled record is never append-positioned implicitly.", activeLineId: "step-2",
    }),
    predecessors: ["controller-partition-collection"], successors: ["scheduled-global-ordinal-merge"],
    routeTags: ["merge", "ordinal", "completeness"],
  },
  {
    id: "ordinal-range-detection", chapter: "merge", title: "Reject out-of-range global ordinals", status: "built",
    summary: "The validation scan accepts ordinal only when ordinal < total records; larger values return a structured OrdinalOutOfRange and cannot force sparse allocation or insert_record_at panic.",
    source: { path: "rust/runtime/src/cellular/shard.rs", startLine: 108, endLine: 141, symbol: "merge_records_in_global_order" },
    evidence: [{ path: "rust/runtime/src/cellular/shard.rs", symbol: "merge_rejects_ordinals_that_are_not_a_permutation lines 619-629", kind: "unit" }],
    inputs: ["ordinal", "total flattened record count"], outputs: ["in-range ordinal or OrdinalOutOfRange {ordinal,total}"],
    state: ["validation scan"], invariants: ["Accepted ordinals index only the dense total-sized domain."],
    complexity: { time: "O(1) per record", memory: "O(1) incremental" }, gates: ["request_index is Some"],
    failures: ["ordinal >= total returns RecordsMergeError::OrdinalOutOfRange before allocation by max ordinal."],
    pseudocode: pseudocode("ordinal = record.request_index", "if ordinal >= total: return OrdinalOutOfRange", "otherwise mark seen[ordinal]"),
    frames: admissionFrames("step-2", ["controller"], {
      before: ["total=8", "ordinal=7"], after: ["ordinal accepted"], invariant: "Largest valid slot is total-1.",
    }, {
      before: ["total=8", "ordinal=80"], after: ["OrdinalOutOfRange {80,8}"], invariant: "Malformed wire data cannot expand storage sparsely.", activeLineId: "step-2",
    }),
    predecessors: ["controller-partition-collection"], successors: ["scheduled-global-ordinal-merge"],
    routeTags: ["merge", "ordinal", "bounds"],
  },
  {
    id: "graph-concatenation-renumber", chapter: "merge", title: "Concatenate graph cells and renumber densely", status: "approximate",
    summary: "Graph cells carry colliding local request_index values. The merge sorts partitions by cell_id, sorts each cell by local index, assigns fresh dense global slots, and preserves each record's authoritative phase field.",
    source: { path: "rust/runtime/src/cellular/shard.rs", startLine: 171, endLine: 191, symbol: "merge_records_by_concatenation" },
    evidence: [
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "concatenation_merges_all_cells_and_renumbers_densely lines 943-971", kind: "unit" },
      { path: "rust/e2e-tests/tests/test_graph_cellular.rs", symbol: "test_graph_cellular_from_python_frontend lines 30-105", kind: "e2e" },
    ],
    inputs: ["graph RecordsShardPartition values with per-cell local ordinals"], outputs: ["dense accumulator ordered by cell_id then local request_index"],
    state: ["sorted partitions", "monotonic global slot counter"], invariants: ["Every graph record receives one unique dense slot.", "Phase membership remains record.phase, never inferred from the new slot."],
    complexity: { time: "O(C log C + Σ Rc log Rc)", memory: "O(max Rc) plus accumulator" },
    gates: ["graph raw-record partition path"], failures: ["No permutation precondition; any partition set can be densely renumbered.", "Result is deterministic per topology, not byte-identical to one-cell summation order."],
    pseudocode: pseudocode("sort partitions by cell_id", "for each partition: sort records by local request_index", "assign request_index=next dense slot", "process record and increment slot"),
    frames: admissionFrames("step-2", ["controller", "cell"], {
      before: ["cell 0 local {0,1}", "cell 1 local {0,1,2}"], after: ["partitions ordered 0 then 1", "local order preserved"], invariant: "Cross-cell local collisions are expected.",
    }, {
      before: ["five locally ordered records"], after: ["global slots {0,1,2,3,4}"], invariant: "No graph record is overwritten or dropped.", activeLineId: "step-3",
    }),
    predecessors: ["partition-messagepack-decode", "controller-partition-collection"], successors: ["final-report-assembly"],
    routeTags: ["merge", "graph", "renumber", "topology-deterministic"],
  },
  {
    id: "exact-fold-store-merge", chapter: "merge", title: "Append exact folded stores by producer id", status: "built",
    summary: "Folded ColumnStorePartition values are sorted by cell_id and reduced through ColumnStore::append_store; empty input creates an empty accumulator and nonempty input preserves the merged store.",
    source: { path: "rust/runtime/src/cellular/shard.rs", startLine: 357, endLine: 373, symbol: "merge_store_partitions" },
    evidence: [
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "n_store_partitions_merge_within_tolerance_of_the_union lines 856-932", kind: "unit" },
      { path: "rust/e2e-tests/tests/test_cellular.rs", symbol: "test_cellular_exact_fold_matches_retain lines 303-365", kind: "e2e" },
    ],
    inputs: ["MetricsConfig", "exact ColumnStorePartition values"], outputs: ["MetricsAccumulator reconstructed from appended exact store"],
    state: ["cell_id-sorted accumulated partition"], invariants: ["Each partition is appended once.", "Fixed producer-id order makes arrival order irrelevant."],
    complexity: { time: "O(C log C + stored columns)", memory: "O(merged exact store)" },
    gates: ["uniform StorePartition run", "exact-fold storage"], failures: ["No explicit mode validation inside append_store; mixed terminal partition kinds are rejected by the controller."],
    pseudocode: pseudocode("sort store partitions by cell_id", "append each store into the accumulated store", "return accumulator from merged store or empty accumulator"),
    frames: admissionFrames("step-1", ["controller"], {
      before: ["stores arrive ids 2,0,1"], after: ["reduction order 0,1,2"], invariant: "Network arrival does not choose floating reduction order.",
    }, {
      before: ["three dense exact stores"], after: ["one appended exact store"], invariant: "Counts/extrema/percentiles match; reordered float reductions may differ by last ULP.", activeLineId: "step-2",
    }),
    predecessors: ["column-store-append", "controller-partition-collection"], successors: ["final-report-assembly"],
    routeTags: ["merge", "exact-fold", "store", "within-tolerance"],
  },
  {
    id: "sketch-tdigest-merge", chapter: "merge", title: "Merge bounded sketch stores associatively", status: "approximate",
    summary: "Sketch stores use the same sorted store-partition reduction; append_store combines per-tag t-digests and Welford state while ingested_count carries the true record total after rows were cleared.",
    source: { path: "rust/runtime/src/cellular/shard.rs", startLine: 357, endLine: 373, symbol: "merge_store_partitions" },
    evidence: [
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "sketch_store_partitions_merge_matches_single_sketch_and_carry_the_count lines 723-803", kind: "unit" },
      { path: "rust/runtime/src/cellular/sketch.rs", symbol: "TDigest::merge lines 128-143", kind: "boundary" },
      { path: "rust/runtime/src/cellular/sketch.rs", symbol: "quantiles_converge_to_the_exact_report_percentiles lines 308-327", kind: "unit" },
      { path: "rust/runtime/src/cellular/sketch.rs", symbol: "merge_matches_a_single_digest_of_the_whole lines 330-358", kind: "unit" },
      { path: "rust/e2e-tests/tests/test_cellular.rs", symbol: "test_cellular_sketch_matches_single_cell lines 384-455", kind: "e2e" },
    ],
    inputs: ["sketch ColumnStorePartition values with t-digests, Welford aggregates, and ingested_total"], outputs: ["bounded merged sketch accumulator"],
    state: ["per-(phase,tag) t-digests", "Welford aggregate state", "ingested_total"],
    invariants: ["Counts and anchored extrema survive merge exactly.", "Interior percentiles remain approximate and topology-dependent.", "Record total comes from ingested_count, not retained row_count."],
    complexity: { time: "O(C log C + C·tags·compression log compression)", memory: "O(tags·compression)" },
    gates: ["uniform StorePartition run", "MetricsStorageMode::Sketch"], failures: ["No raw rows remain for per-record artifacts.", "Floating sums/means/std can drift by merge order; quantiles are t-digest estimates."],
    pseudocode: pseudocode("sort sketch stores by producer id", "append ingested_total and exact aggregates", "merge each matching t-digest by centroid concatenate+compress", "report merged.ingested_count as run total"),
    frames: admissionFrames("step-3", ["controller"], {
      before: ["three sketches each count=10", "row_count=0"], after: ["merged digest count=30", "ingested_count=30"], invariant: "Fold-and-clear does not erase work totals.",
    }, {
      before: ["three centroid sets"], after: ["one compressed centroid set", "exact min/max retained"], invariant: "Approximation is explicit and bounded by compression.", activeLineId: "step-3",
    }),
    predecessors: ["tagged-sketch-merge", "ingested-count-preservation", "controller-partition-collection"], successors: ["final-report-assembly"],
    routeTags: ["merge", "sketch", "tdigest", "bounded-memory"],
  },
  {
    id: "controller-partition-collection", chapter: "merge", title: "Collect the expected terminal partition count", status: "feature-gated",
    summary: "The Velo controller stops when raw plus store partition vector length reaches the configured cell count. It does not validate terminal partition producer IDs for uniqueness or range, so a duplicate producer can satisfy the count while another producer is absent.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 660, endLine: 710, symbol: "run_cellular terminal collection block" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "collected length closure and unvalidated partition pushes lines 683-704", kind: "boundary" },
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "RecordsShardPartition::cell_id and ColumnStorePartition::cell_id producer metadata", kind: "boundary" },
    ],
    inputs: ["ControllerTransport messages", "expected_partitions count", "child failure channel", "collect deadline"], outputs: ["count-complete raw or store partition vectors plus heartbeat map"],
    state: ["unkeyed raw partition vector", "unkeyed store partition vector", "heartbeats keyed by supplied cell_id", "deadline"],
    invariants: ["Heartbeats do not increment the terminal partition count.", "The implemented completion predicate is vector length only; it does not prove one partition per expected producer."],
    complexity: { time: "O(messages)", memory: "O(terminal payloads + cells·heartbeat)" },
    gates: ["velo feature", "controller mode"], failures: ["Transport close, cell failure, or deadline expiry aborts collection.", "Mixed raw/store kinds are rejected after collection.", "Duplicate or out-of-range terminal producer IDs are not rejected here and can mask a missing expected producer."],
    pseudocode: pseudocode("expected = cell_count", "while raw.len + stores.len < expected: receive biased select", "append every terminal partition without producer-ID validation; heartbeat replaces map[cell_id]", "on close/failure/timeout: abort"),
    frames: admissionFrames("step-3", ["controller", "wire", "cell"], {
      before: ["expected=3", "two stores collected", "heartbeat arrives"], after: ["two stores still", "heartbeat[cell] replaced"], invariant: "Progress telemetry cannot satisfy completion.",
    }, {
      before: ["expected=3", "stores from producer IDs {0,0,1} arrive"], after: ["vector length reaches 3", "collection exits although producer 2 is missing"], invariant: "This is the documented count-only limitation, not a uniqueness guarantee.", activeLineId: "step-3",
    }),
    predecessors: ["velo-controller-bind", "controller-child-arbitration"], successors: ["scheduled-global-ordinal-merge", "graph-concatenation-renumber", "exact-fold-store-merge", "sketch-tdigest-merge"],
    routeTags: ["merge", "controller", "velo", "barrier"],
  },
  {
    id: "hierarchical-tier-sizing", chapter: "merge", title: "Refuse hierarchical aggregation", status: "rejected",
    summary: "A requested fanout is rejected before imported acquisition, scratch creation, artifact binding, Velo binding, or launcher execution. Supported cellular runs are flat controller-to-cell stars.",
    source: { path: "rust/runtime/src/engine/cellular_aggregator.rs", startLine: 9, endLine: 24, symbol: "is_hierarchy_requested" },
    evidence: [
      { path: "rust/runtime/src/engine/cellular_controller.rs", symbol: "hierarchy_refuses_before_any_startup_side_effect", kind: "unit" },
      { path: "rust/e2e-tests/tests/test_cellular.rs", symbol: "test_cellular_hierarchy_is_refused", kind: "e2e" },
    ],
    inputs: ["cell_count", "AIPERF_CELL_AGG_FANOUT"], outputs: ["flat topology or refusal"],
    state: ["parsed fanout"], invariants: ["A hierarchy request has no startup side effects.", "Flat topology has one terminal partition per cell."],
    complexity: { time: "O(1)", memory: "O(1)" }, gates: ["fanout requests hierarchy"],
    failures: ["A requested hierarchy returns an unavailable error before startup."],
    pseudocode: pseudocode("parse fanout from env", "if fanout requests hierarchy: return refusal", "otherwise use the flat star"),
    frames: admissionFrames("step-3", ["controller"], {
      before: ["cells=7", "fanout=3"], after: ["unavailable hierarchy error"], invariant: "No startup side effect occurs.",
    }, {
      before: ["cells=7", "fanout=7"], after: ["flat topology"], invariant: "A pointless tier is not created.", activeLineId: "step-2",
    }),
    predecessors: ["controller-promotion"], successors: [],
    routeTags: ["merge", "hierarchical", "refusal"],
  },
  {
    id: "heartbeat-aggregation", chapter: "merge", title: "Merge terminal-adjacent heartbeat messages", status: "approximate",
    summary: "MetricsHeartbeat::merge can combine arbitrary snapshots by summing counters/saturation and t-digest-merging latency distributions. Current production cells send exactly one heartbeat immediately before their terminal partition, so controller replacement-by-cell supports repeated updates but is not currently a periodic live stream.",
    source: { path: "rust/runtime/src/cellular/heartbeat.rs", startLine: 65, endLine: 75, symbol: "MetricsHeartbeat::merge" },
    evidence: [
      { path: "rust/runtime/src/cellular/heartbeat.rs", symbol: "heartbeat_merge_sums_counters_and_merges_sketches lines 236-288", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/cellular_cell.rs", symbol: "CellRecordsShipper::ship sends one heartbeat then terminal lines 743-790", kind: "boundary" },
    ],
    inputs: ["one production MetricsHeartbeat per cell; merge type also supports repeated snapshots"], outputs: ["summed counters/saturation and merged latency sketches"],
    state: ["controller map keyed by supplied cell_id", "max observed_at_ns", "counter sums", "three t-digests"], invariants: ["Current cell shipping orders one heartbeat immediately before one terminal partition.", "Heartbeat percentiles are approximate diagnostics; final report data comes from terminal partitions."],
    complexity: { time: "O(C·compression log compression)", memory: "O(C latest heartbeats + compression)" },
    gates: ["at least one terminal-adjacent heartbeat for sidecar output"], failures: ["Missing heartbeat yields no sidecar contribution.", "The current production path provides no periodic in-run heartbeat cadence despite the merge type supporting repeated snapshots."],
    pseudocode: pseudocode("cell sends one heartbeat immediately before its terminal partition", "controller stores heartbeat by supplied cell_id", "after collection, sum counters and merge ttft/itl/latency t-digests", "project finite quantiles to cellular-heartbeat.json"),
    frames: admissionFrames("step-1", ["controller", "cell"], {
      before: ["cell completes local execution"], after: ["one heartbeat sent", "terminal partition sent next"], invariant: "Production emission is terminal-adjacent, not periodic.",
    }, {
      before: ["terminal collection finishes with heartbeat messages"], after: ["counters summed", "digests merged", "approximate sidecar written"], invariant: "Heartbeat merge does not replace terminal metric merge.", activeLineId: "step-3",
    }),
    predecessors: ["controller-partition-collection", "tdigest-insert-compress", "final-report-assembly"], successors: ["controller-global-concatenation"],
    routeTags: ["merge", "heartbeat", "terminal-adjacent", "tdigest"],
  },
  {
    id: "final-report-assembly", chapter: "merge", title: "Assemble and persist the merged report", status: "partial",
    summary: "The controller exports profiling and optional warmup metrics, writes native-v2.json, and invokes best-effort exporters before it waits for Stage E artifacts or concatenates per-cell files; those earlier outputs are not rolled back by a later artifact failure.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 740, endLine: 815, symbol: "run_cellular report publication block" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "minimal RunOutcome construction and pre-artifact exporter invocation at lines 758-815", kind: "boundary" },
      { path: "rust/e2e-tests/tests/test_cellular.rs", symbol: "test_cellular_run_from_python_frontend lines 37-64", kind: "e2e" },
      { path: "rust/e2e-tests/tests/test_graph_cellular.rs", symbol: "test_graph_cellular_from_python_frontend lines 30-105", kind: "e2e" },
    ],
    inputs: ["uniform merged accumulator", "authored metrics/export config", "model and endpoint config", "report path"], outputs: ["native-v2.json", "native exporter outputs", "CellularRunOutcome record count"],
    state: ["profiling summary", "optional warmup", "minimal RunOutcome", "NativeReport", "published report/export files before artifact completion"],
    invariants: ["Sketch record totals use ingested_count.", "Warmup is included only when its result map is nonempty.", "native-v2.json and best-effort exporter outputs are published before the Stage E barrier and per-cell artifact concat."],
    complexity: { time: "O(metrics report + exporter costs)", memory: "O(report serialization)" },
    gates: ["all expected terminal partitions collected", "no mixed raw/store partition kinds"],
    failures: ["Merge, report serialization, directory creation, or native-v2 write fails before artifact handling.", "Individual exporters log failures but do not fail the run.", "A later artifact barrier/concat failure returns run failure after native-v2 and exporter files may already exist."],
    pseudocode: pseudocode("select store merge or run-kind record merge", "record_count = merged.ingested_count", "export profiling and optional warmup", "build minimal RunOutcome and NativeReport", "write native-v2.json; run exporters best-effort before artifact barrier/concat"),
    frames: admissionFrames("step-1", ["controller"], {
      before: ["three uniform StorePartitions"], after: ["one merged accumulator", "record_count from ingested_count"], invariant: "Sketch zero-row stores still report work.",
    }, {
      before: ["NativeReport serialized"], after: ["native-v2.json written", "exporters invoked", "artifact barrier still pending"], invariant: "Report publication precedes artifact completion and can survive a later artifact failure.", activeLineId: "step-5",
    }),
    predecessors: ["scheduled-global-ordinal-merge", "graph-concatenation-renumber", "exact-fold-store-merge", "sketch-tdigest-merge"], successors: ["merged-report-fidelity-boundary", "heartbeat-aggregation", "controller-global-concatenation"],
    routeTags: ["merge", "report", "native-v2", "export"],
  },
  {
    id: "merged-report-fidelity-boundary", chapter: "merge", title: "Declare merged-report fidelity by storage and run kind", status: "partial",
    summary: "Fidelity depends on the merge path: only scheduled retain re-ingests in byte-exact global ordinal order; graph retain is deterministic per topology, exact-fold store append is tolerance-level for order-sensitive floats, and sketch percentiles are approximate. All paths omit coordinator finalize provenance, grouped per-error detail, per-record OTLP accumulators, and server/GPU/network side channels.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 758, endLine: 790, symbol: "run_cellular merged-report construction boundary" },
    evidence: [
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "merge_records_in_global_order lines 108-141", kind: "boundary" },
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "merged_cell_records_are_byte_identical_to_a_single_cell_run lines 501-536", kind: "unit" },
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "merge_records_by_concatenation lines 171-191", kind: "boundary" },
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "concatenation_is_deterministic_regardless_of_partition_order lines 973-1000", kind: "unit" },
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "merge_store_partitions lines 357-373", kind: "boundary" },
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "n_store_partitions_merge_within_tolerance_of_the_union lines 856-932", kind: "unit" },
      { path: "rust/runtime/src/cellular/shard.rs", symbol: "sketch_store_partitions_merge_matches_single_sketch_and_carry_the_count lines 723-803", kind: "unit" },
      { path: "rust/runtime/src/cellular/sketch.rs", symbol: "TDigest::merge lines 128-143", kind: "boundary" },
      { path: "rust/runtime/src/cellular/sketch.rs", symbol: "quantiles_converge_to_the_exact_report_percentiles lines 308-327", kind: "unit" },
      { path: "rust/runtime/src/cellular/sketch.rs", symbol: "merge_matches_a_single_digest_of_the_whole lines 330-358", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "warn_dropped_sidecar_telemetry lines 1805-1835", kind: "boundary" },
      { path: "rust/runtime/src/metrics_core/report.rs", symbol: "NativeReport::from_outcome initializes otel_per_record=None at lines 1014-1018", kind: "boundary" },
      { path: "rust/cli/src/execute_mode.rs", symbol: "run_controller terminal provenance lines 288-323", kind: "boundary" },
    ],
    inputs: ["run kind", "retain/exact-fold/sketch storage path", "cell-shipped records or stores"], outputs: ["minimal merged NativeReport with path-specific fidelity"],
    state: ["selected merge contract", "omitted provenance/error groups/per-record OTLP/sidecars"], invariants: ["Scheduled retain alone claims byte-exact global-order re-ingestion.", "Graph retain, exact-fold, and sketch retain their separately stated deterministic/tolerance/approximation contracts.", "Coordinator provenance, grouped errors, per-record OTLP accumulators, and server/GPU/network side channels are omitted."],
    complexity: { time: "O(1) policy boundary", memory: "O(1)" },
    gates: ["cellular merged-report path"], failures: ["Treating every record-derived aggregate as byte-identical to one-cell overstates graph-retain, exact-fold, and sketch fidelity.", "Treating the report as field-for-field one-cell parity ignores omitted provenance, grouped errors, per-record OTLP accumulators, and side channels."],
    pseudocode: pseudocode("select scheduled-retain, graph-retain, exact-fold, or sketch fidelity contract", "assemble the minimal report under that contract", "omit coordinator provenance, grouped error details, and per-record OTLP", "omit server/GPU/network side-channel data; warn when configured"),
    frames: admissionFrames("step-3", ["controller"], {
      before: ["scheduled retain supplies a valid global ordinal permutation"], after: ["records re-ingested in byte-exact single-cell order"], invariant: "This byte-exact ordering claim does not transfer to other merge paths.",
    }, {
      before: ["graph retain, exact-fold, or sketch merge selected", "sidecars configured"], after: ["path-specific non-byte-exact contract applies", "side-channel series omitted"], invariant: "Approximation/tolerance and omission boundaries remain explicit.", activeLineId: "step-4",
    }),
    predecessors: ["final-report-assembly"], successors: ["telemetry-drop-warning", "terminal-failure-envelope"],
    routeTags: ["merge", "fidelity", "omission", "boundary"],
  },
];

const ARTIFACT_ALGORITHMS: readonly AlgorithmDefinition[] = [
  {
    id: "artifact-authority-allowlist", chapter: "artifacts", title: "Constrain upload authority and artifact paths", status: "feature-gated",
    summary: "The controller derives the exact shippable path set from ArtifactSpec. Each client path must be relative, all-Normal, and allowlisted before joining beneath the route-selected temp_root/cell-id; this validates the file path but does not authenticate or range-check cell_id.",
    source: { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", startLine: 265, endLine: 294, symbol: "validate_artifact_relpath" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "path_validation_rejects_traversal_and_unknown lines 1052-1074", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "server_rejects_unallowed_upload lines 1434-1461", kind: "integration" },
    ],
    inputs: ["client-supplied relative path", "unvalidated route cell_id", "controller-derived HashSet of ArtifactSpec paths"], outputs: ["validated relative PathBuf joined under the route-selected cell directory"],
    state: ["run-scoped file allowlist", "unvalidated cell_id route parameter"], invariants: ["Absolute, dot, parent, root, and prefix path components never reach filesystem join.", "Only configured records/raw/CSV/parquet/outputs/inputs file paths may land.", "Path validation does not establish that cell_id belongs to the run."],
    complexity: { time: "O(path components + allowlist lookup)", memory: "O(configured artifact names)" },
    gates: ["Stage E HTTP artifact server active"], failures: ["Empty, absolute, traversal-bearing, or unknown file path returns HTTP 400.", "An arbitrary u32 cell_id is accepted as a landing-directory name."],
    pseudocode: pseudocode("reject empty or absolute path", "require every component is Normal", "require exact membership in run allowlist", "join only after validation beneath cell-id root"),
    frames: admissionFrames("step-3", ["controller", "cell"], {
      before: ["allowed contains profile_export.jsonl"], after: ["same relative path accepted"], invariant: "Authority is run-scoped.",
    }, {
      before: ["route cell_id=99", "file path ../native-v2.json"], after: ["400 traversal rejected", "no filesystem write"], invariant: "The file cannot traverse out of the route-selected directory; cell_id validity is a separate unchecked boundary.", activeLineId: "step-2",
    }),
    predecessors: ["controller-promotion"], successors: ["artifact-http-zstd-upload", "partial-file-atomic-replace"],
    routeTags: ["artifacts", "security", "allowlist", "stage-e"],
  },
  {
    id: "shard-local-concatenation", chapter: "artifacts", title: "Fuse per-shard artifacts by format", status: "built",
    summary: "When worker sharding is active, the coordinator fuses shard JSONL by byte append, CSV with one header, and outputs.json by data-array merge; Parquet row-group concat exists only under #[cfg(feature=\"parquet\")]. Successful checked cleanup can fail the operation; error-path Drop cleanup is best-effort and ignores removal errors.",
    source: { path: "rust/runtime/src/runner_protocol/shard_artifacts.rs", startLine: 84, endLine: 156, symbol: "concatenate_shard_artifacts" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/shard_artifacts.rs", symbol: "per_shard_concat_matches_batch_over_union lines 463-597", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/shard_artifacts.rs", symbol: "concatenate_shard_artifacts parquet cfg branch lines 139-145", kind: "boundary" },
    ],
    inputs: ["ArtifactSpec", "worker count", "per-shard files"], outputs: ["one final JSONL/CSV/outputs file per requested artifact", "Parquet final only in parquet-feature builds", "checked scratch cleanup only on the successful concat path"],
    state: ["ordered shard paths", "best-effort all-exit cleanup guard", "checked success-path cleanup loop"], invariants: ["JSONL is byte-appended in shard order.", "CSV writes one header.", "inputs.json is excluded because coordinator writes it once.", "Drop cleanup attempts removal but provides no success guarantee and cannot replace the original error."],
    complexity: { time: "O(total artifact bytes)", memory: "JSONL/CSV use fs::read per source: O(largest source file); outputs.json may retain merged data; feature-enabled Parquet is library-dependent" },
    gates: ["worker sharding exists and requested per-record artifacts", "Parquet concatenation additionally requires Cargo feature parquet"], failures: ["Any compiled format concat error aborts; subsequent Drop cleanup ignores remove_dir_all errors and may leave scratch behind.", "After all concats succeed, checked cleanup failure is contextualized and fails the operation.", "On a lite build, the Parquet branch is absent: a requested Parquet path creates no final Parquet here and does not itself make concat fail."],
    pseudocode: pseudocode("install all-exit shard cleanup guard", "for each configured JSONL/CSV/outputs format, collect shard paths", "if Cargo feature parquet: concatenate configured Parquet row groups", "remove shard dirs loudly on success"),
    frames: admissionFrames("step-2", ["worker", "controller"], {
      before: ["three JSONL shard files"], after: ["one final JSONL in shard order"], invariant: "No row is reparsed for JSONL.",
    }, {
      before: ["concat fails before checked cleanup"], after: ["original concat error returned", "Drop merely attempts scratch removal"], invariant: "Best-effort error cleanup may leave directories and never overrides the primary error.", activeLineId: "step-1",
    }),
    predecessors: [], successors: [],
    routeTags: ["artifacts", "shard", "concat", "format-aware"],
  },
  {
    id: "cell-local-concatenation", chapter: "artifacts", title: "Fuse per-cell artifacts on the controller host", status: "built",
    summary: "The controller treats each cell directory as one shard and fuses JSONL, CSV, and outputs.json. Parquet row-group concat exists only under #[cfg(feature=\"parquet\")]. inputs.json is not concatenated: one byte-identical full-dataset copy is selected separately.",
    source: { path: "rust/runtime/src/runner_protocol/shard_artifacts.rs", startLine: 158, endLine: 203, symbol: "concatenate_cell_artifacts" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/shard_artifacts.rs", symbol: "per_cell_concat_matches_batch_over_union lines 636-757", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/shard_artifacts.rs", symbol: "concatenate_cell_artifacts parquet cfg branch lines 195-201", kind: "boundary" },
      { path: "rust/e2e-tests/tests/test_cellular.rs", symbol: "test_cellular_emits_per_record_artifacts_matching_single_cell lines 195-287", kind: "e2e" },
    ],
    inputs: ["cell directories ordered by cell_id", "ArtifactSpec", "final artifact directory"], outputs: ["merged JSONL/CSV/outputs artifacts", "merged Parquet only in parquet-feature builds"],
    state: ["source path list per configured artifact"], invariants: ["Cell source cleanup belongs to ScratchTreeGuard, not this function.", "Per-record rows are concatenated; full-dataset inputs.json is copied once."],
    complexity: { time: "O(total artifact bytes)", memory: "JSONL/CSV use fs::read per source: O(largest cell file); outputs.json may retain merged data; feature-enabled Parquet is library-dependent" },
    gates: ["per-record artifact requested", "same-host: terminal partition collection already orders after local writes", "Stage E: upload barrier must first release", "Parquet concatenation additionally requires Cargo feature parquet"],
    failures: ["Missing optional source files are handled by compiled format helpers; malformed CSV/outputs or feature-enabled Parquet fails concat.", "On a lite build, the Parquet branch is absent: a requested Parquet path creates no merged Parquet here and does not itself make concat fail."],
    pseudocode: pseudocode("map each configured JSONL/CSV/outputs path across cell dirs", "concat those formats with their format-specific routine", "if Cargo feature parquet: concatenate configured Parquet row groups", "leave cell dirs for ScratchTreeGuard"),
    frames: admissionFrames("step-2", ["controller", "cell"], {
      before: ["cell dirs contain records/raw/CSV"], after: ["source lists ordered by cell id"], invariant: "Topology fixes artifact concatenation order.",
    }, {
      before: ["all cell source lists"], after: ["one file per compiled artifact format"], invariant: "Lite builds do not synthesize Parquet; inputs.json is not duplicated.", activeLineId: "step-2",
    }),
    predecessors: [], successors: ["controller-global-concatenation"],
    routeTags: ["artifacts", "cell", "concat", "stage-d"],
  },
  {
    id: "controller-global-concatenation", chapter: "artifacts", title: "Publish controller-global artifact outputs", status: "feature-gated",
    summary: "After native-v2 and exporter publication, the controller conditionally waits for Stage E /done cardinality, then concatenates requested JSONL/CSV/outputs artifacts, conditionally concatenates Parquet only under #[cfg(feature=\"parquet\")], and copies one inputs.json. A late barrier or concat failure can leave earlier report/export files on disk.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 829, endLine: 895, symbol: "run_cellular artifact barrier and concat block" },
    evidence: [
      { path: "rust/e2e-tests/tests/test_cellular_http_shipping.rs", symbol: "test_cellular_http_shipping_matches_single_cell lines 354-570", kind: "e2e" },
      { path: "rust/runtime/src/runner_protocol/shard_artifacts.rs", symbol: "concatenate_cell_artifacts parquet cfg branch lines 195-201", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/shard_artifacts.rs", symbol: "copy_cell_inputs_json lines 205-245", kind: "boundary" },
    ],
    inputs: ["controller-local cell directories", "ArtifactSpec", "artifact directory"], outputs: ["global records/raw/CSV/outputs artifacts", "global Parquet only in parquet-feature builds", "one inputs.json"],
    state: ["already-published report/export files", "HTTP landing-root override when force seam is active", "cell_dirs in id order"], invariants: ["Stage E concat follows the count-based /done barrier; same-host concat relies on cells writing before terminal partition shipment.", "inputs.json remains the single full-dataset document, not N concatenated copies."],
    complexity: { time: "O(total artifact bytes)", memory: "JSONL/CSV use fs::read per source: O(largest cell file); outputs.json may retain merged data; feature-enabled Parquet is library-dependent" },
    gates: ["requested artifact paths", "Stage E uses the /done cardinality barrier; same-host does not", "Parquet publication additionally requires Cargo feature parquet"], failures: ["Barrier or compiled-format concat/copy error fails the cellular run after native-v2/export files may already have been published.", "The /done barrier itself does not validate expected cell IDs.", "On a lite build, requested Parquet has no compiled concat branch, so no global Parquet is created here and that absence alone does not fail concatenation."],
    pseudocode: pseudocode("native-v2 and exporters have already run", "if Stage E: wait for /done set cardinality", "select landed or same-host cell dirs in cell-id order", "concat JSONL/CSV/outputs; if Cargo feature parquet, concat Parquet; copy first inputs.json"),
    frames: admissionFrames("step-1", ["controller", "cell"], {
      before: ["native-v2 and exporter files already exist", "Stage E uploads still possible"], after: ["controller enters /done cardinality barrier"], invariant: "Report publication occurs before artifact completion.",
    }, {
      before: ["artifact barrier or concat fails late"], after: ["run returns failure", "previously published native-v2/export files can remain"], invariant: "Failure does not roll back earlier publication.", activeLineId: "step-4",
    }),
    predecessors: ["cell-local-concatenation", "final-report-assembly", "heartbeat-aggregation"], successors: ["terminal-failure-envelope"],
    routeTags: ["artifacts", "controller", "global", "publication"],
  },
  {
    id: "artifact-http-zstd-upload", chapter: "artifacts", title: "Stream Stage E artifacts over HTTP and zstd", status: "feature-gated",
    summary: "Each cross-host cell streams every existing allowlisted file through a level-3 zstd reader in at-most-64-KiB chunks over a bounded request channel, then posts /done. The server forwards body frames through a four-slot channel to blocking streaming decode.",
    source: { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", startLine: 654, endLine: 680, symbol: "ship_cell_artifacts" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "streaming_zstd_round_trips_byte_for_byte lines 1003-1034", kind: "unit" },
      { path: "rust/e2e-tests/tests/test_cellular_http_shipping.rs", symbol: "test_cellular_http_shipping_matches_single_cell lines 354-570", kind: "e2e" },
    ],
    inputs: ["controller authority", "cell_id", "cell artifact directory", "shippable relative paths"], outputs: ["controller-local byte-identical files", "per-cell done marker"],
    state: ["FileCompressor CHUNK_SIZE=65536", "zstd level 3", "bounded body/decode channels"], invariants: ["Neither sender nor receiver buffers a whole file.", "Only existing requested files are POSTed.", "/done is posted after all file responses succeed."],
    complexity: { time: "O(total bytes)", memory: "O(CHUNK_SIZE)" },
    gates: ["velo feature", "cross-host or force seam", "HTTP artifact shipping enabled", "nonempty shippable paths"],
    failures: ["Open/compress/connect/HTTP/decode/write failures fail cell execution; /done is not posted after a failed file."],
    pseudocode: pseudocode("for each existing relative artifact", "stream zstd chunks into HTTP POST with backpressure", "server streams frames into blocking decoder", "after every upload succeeds, POST /cell/id/done"),
    frames: admissionFrames("step-2", ["cell", "wire", "controller"], {
      before: ["large JSONL on cell disk"], after: ["bounded compressed frames in flight"], invariant: "Peak transfer memory is file-size independent.",
    }, {
      before: ["all artifact POSTs returned success"], after: ["/done accepted"], invariant: "Completion marker is causally after file completion.", activeLineId: "step-4",
    }),
    predecessors: ["artifact-authority-allowlist"], successors: ["partial-file-atomic-replace", "artifact-completion-barrier"],
    routeTags: ["artifacts", "http", "zstd", "bounded-memory"],
  },
  {
    id: "partial-file-atomic-replace", chapter: "artifacts", title: "Commit compressed and plain uploads by atomic rename", status: "feature-gated",
    summary: "Both DecompressToFile and PlainToFile create final-path.part, stream bytes into it, flush and fsync the file, then rename it onto the final path. A failed or truncated transfer does not intentionally publish a partial final artifact.",
    source: { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", startLine: 142, endLine: 237, symbol: "part_path_for, DecompressToFile, and PlainToFile" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "streaming_zstd_round_trips_byte_for_byte lines 1003-1034", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "plain_uncompressed_sink_round_trips lines 1077-1089", kind: "unit" },
    ],
    inputs: ["validated final path", "compressed or plain chunk stream"], outputs: ["atomically published final file"],
    state: [".part staging path", "DecompressToFile for zstd", "PlainToFile for identity encoding"], invariants: ["Neither sink intentionally creates the final path before finish.", "Both finish paths flush, unwrap, fsync, then rename.", "Failed transfer leaves at most a .part file from this attempt."],
    complexity: { time: "O(file bytes)", memory: "O(CHUNK_SIZE + codec buffers)" },
    gates: ["upload or dataset download receiver"], failures: ["Filesystem/decoder/fsync/rename failure propagates; final file is not intentionally published early."],
    pseudocode: pseudocode("choose DecompressToFile for zstd or PlainToFile for identity", "create parent dirs and final.part", "stream decoded or plain chunks into part", "flush and fsync part file", "rename part atomically onto final"),
    frames: admissionFrames("step-3", ["controller", "wire"], {
      before: ["first chunks received"], after: ["dest.jsonl.part grows", "dest.jsonl absent"], invariant: "Readers cannot observe truncation.",
    }, {
      before: ["complete compressed or plain stream in part"], after: ["part fsynced and renamed", "final visible"], invariant: "Both encoding branches use the same staged publication discipline.", activeLineId: "step-5",
    }),
    predecessors: ["artifact-http-zstd-upload"], successors: ["artifact-completion-barrier"],
    routeTags: ["artifacts", "atomic", "staged-publication", "atomic-visibility"],
  },
  {
    id: "artifact-completion-barrier", chapter: "artifacts", title: "Wait for all distinct artifact completions", status: "feature-gated",
    summary: "A watch channel publishes the raw u32 IDs posted to /done, and wait_for_cells releases when set cardinality reaches cell_count. The handler performs no expected-range validation, so an invalid ID can increase cardinality and mask a missing expected cell.",
    source: { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", startLine: 426, endLine: 465, symbol: "ArtifactUploadServer::wait_for_cells" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "wait_for_cells_returns_when_completed_before_waiter_registers lines 1091-1135", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "cell_done inserts unvalidated route cell_id lines 563-577", kind: "boundary" },
    ],
    inputs: ["watch<HashSet<cell_id>>", "cell_count", "artifact upload timeout"], outputs: ["barrier release or missing-cell timeout error"],
    state: ["versioned HashSet of unvalidated supplied u32 IDs"], invariants: ["A duplicate of the same ID cannot increase cardinality.", "A completion sent before waiter registration is still observed.", "Cardinality completion does not prove that every expected ID in 0..cell_count posted /done."],
    complexity: { time: "O(done updates + cell_count on timeout)", memory: "O(cell_count)" },
    gates: ["Stage E file uploads active; dataset-serve-only mode skips it"], failures: ["An out-of-range supplied ID is accepted and can release the barrier while an expected cell is missing.", "Only a timeout enumerates missing expected IDs.", "Server shutdown ends wait and lets downstream concat behavior decide."],
    pseudocode: pseudocode("cell_done inserts supplied u32 without range validation", "subscribe to versioned done set", "return when set cardinality >= cell_count", "otherwise await changed; on timeout enumerate missing expected IDs"),
    frames: admissionFrames("step-2", ["controller", "cell"], {
      before: ["all /done markers arrived before wait starts"], after: ["first borrow observes complete set"], invariant: "No notification permit is required.",
    }, {
      before: ["cell_count=3", "done={0,1,99}", "expected cell 2 missing"], after: ["cardinality is 3", "barrier releases without reporting missing 2"], invariant: "This is the documented unvalidated-ID limitation.", activeLineId: "step-3",
    }),
    predecessors: ["artifact-http-zstd-upload", "partial-file-atomic-replace"], successors: ["controller-global-concatenation"],
    routeTags: ["artifacts", "barrier", "watch", "timeout"],
  },
  {
    id: "telemetry-drop-warning", chapter: "artifacts", title: "Warn on unsupported side-channel aggregation", status: "partial",
    summary: "At controller startup, active server_metrics, gpu_telemetry, and network_latency config is detected and warned as omitted. Defaults make rejection impractical; cross-cell side-channel aggregation remains future wiring.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 1805, endLine: 1835, symbol: "warn_dropped_sidecar_telemetry" },
    evidence: [{ path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "run_cellular warning invocation lines 281-287", kind: "boundary" }],
    inputs: ["run envelope side-channel config"], outputs: ["one startup warning listing active omitted sidecars"],
    state: ["three known sidecar names"], invariants: ["Warning is explicit; omitted telemetry is never silently represented as merged.", "Record-derived metrics continue."],
    complexity: { time: "O(1)", memory: "O(1)" }, gates: ["at least one sidecar is active"],
    failures: ["No hard failure; cellular run proceeds without merged server/GPU/network side-channel data."],
    pseudocode: pseudocode("inspect server_metrics, gpu_telemetry, network_latency", "collect configured active sidecars", "if nonempty, warn they are dropped in merged report"),
    frames: admissionFrames("step-1", ["controller"], {
      before: ["gpu telemetry enabled by config"], after: ["startup warning names gpu_telemetry"], invariant: "Omission is operator-visible.",
    }, {
      before: ["record partitions merge successfully"], after: ["final report has record metrics but no GPU series"], invariant: "Absence is not converted to zeros.", activeLineId: "step-3",
    }),
    predecessors: ["merged-report-fidelity-boundary"], successors: ["terminal-failure-envelope"],
    routeTags: ["artifacts", "telemetry", "warning", "omission"],
  },
  {
    id: "child-exit-arbitration", chapter: "artifacts", title: "Arbitrate terminal messages against child exits", status: "feature-gated",
    summary: "Local cell watchers send only hard failures. Registration and collection use biased select, so a ready registration or terminal partition wins over a simultaneous nonzero exit; a clean cell exit parks forever and is not misclassified.",
    source: { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", startLine: 76, endLine: 93, symbol: "CellHandle::wait_failure" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "run_cellular failure watchers and biased selects lines 600-710", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "CellHandle::wait_failure lines 73-93", kind: "boundary" },
    ],
    inputs: ["local cell exit status", "transport event readiness"], outputs: ["hard-failure diagnostic or accepted progress/terminal message"],
    state: ["failure channel", "biased registration/collection select"], invariants: ["Successful child exit is not failure.", "Ship-then-exit race accepts an already-ready terminal message first.", "Nonzero exit before required progress aborts."],
    complexity: { time: "O(children) watcher tasks", memory: "O(children)" },
    gates: ["local launcher has child handles; k8s relies on deadlines"], failures: ["Nonzero status or wait error becomes a contextual controller failure."],
    pseudocode: pseudocode("watch each local child; park forever on success", "send nonzero/wait failure into failure channel", "in biased select, consume ready transport progress first", "otherwise abort on failure diagnostic"),
    frames: admissionFrames("step-3", ["controller", "cell"], {
      before: ["partition and nonzero exit become ready together"], after: ["partition consumed first"], invariant: "Completed shipment is not discarded by scheduler race.",
    }, {
      before: ["cell exits 1 before partition"], after: ["controller aborts with cell id and status"], invariant: "A missing producer cannot become a hang.", activeLineId: "step-4",
    }),
    predecessors: ["controller-promotion"], successors: ["controller-timeout", "cancellation-propagation", "terminal-failure-envelope"],
    routeTags: ["failure", "child-exit", "arbitration", "biased-select"],
  },
  {
    id: "controller-timeout", chapter: "artifacts", title: "Bound registration, collection, and upload waits", status: "feature-gated",
    summary: "Three independent deadlines bound startup registration, terminal partition collection, and Stage E upload completion. Defaults are five minutes, two hours, and five minutes respectively and each can be overridden independently.",
    source: { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", startLine: 956, endLine: 997, symbol: "collect_timeout, artifact_upload_timeout, register_timeout" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/artifact_shipping.rs", symbol: "wait_for_cells_times_out_and_names_missing_cells lines 1138-1172", kind: "unit" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "run_cellular registration and collect timeout branches lines 641-708", kind: "boundary" },
    ],
    inputs: ["registration state", "terminal partition count", "artifact done set", "environment timeout overrides"], outputs: ["bounded progress or typed contextual error"],
    state: ["separate startup/run/upload deadlines"], invariants: ["Each wait has a finite deadline.", "Upload timeout starts only after metric partitions and report/export publication are complete.", "Timeouts bound waiting but do not validate producer or /done identities."],
    complexity: { time: "O(1) timers", memory: "O(1)" }, gates: ["synchronized start for registration timeout", "all runs for collect timeout", "Stage E for upload timeout"],
    failures: ["Registration names startup timeout; collection includes received/expected counts; upload lists missing expected IDs only when it actually times out.", "Duplicate terminal producers or invalid /done IDs can satisfy count predicates before their deadlines."],
    pseudocode: pseudocode("race all-registered against child failure and register deadline", "race partition receive against child failure and collect deadline", "if Stage E, wait done-set with upload deadline", "propagate first terminal error"),
    frames: admissionFrames("step-2", ["controller", "cell"], {
      before: ["k8s cell never registers"], after: ["registration deadline aborts"], invariant: "Startup wait is finite.",
    }, {
      before: ["metrics shipped", "cell dies mid-artifact upload"], after: ["upload deadline reports missing cell"], invariant: "A completed metrics plane does not mask incomplete artifacts.", activeLineId: "step-3",
    }),
    predecessors: ["child-exit-arbitration", "artifact-completion-barrier"], successors: ["cancellation-propagation", "terminal-failure-envelope"],
    routeTags: ["failure", "timeout", "registration", "collection"],
  },
  {
    id: "cancellation-propagation", chapter: "artifacts", title: "Stop local load when the controller aborts", status: "partial",
    summary: "Local cell subprocesses are configured kill_on_drop. Any controller error unwinds run_cellular, drops watcher tasks and their owned child handles, and SIGKILLs remaining local cells so they cannot continue generating load. Kubernetes pod cancellation remains operator-owned.",
    source: { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", startLine: 108, endLine: 148, symbol: "LocalLauncher::cell_command" },
    evidence: [
      { path: "rust/runtime/src/runner_protocol/cell_launcher.rs", symbol: "LocalLauncher::cell_command kill_on_drop lines 108-148", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "controller-owned runtime block_on scope lines 296-304", kind: "boundary" },
      { path: "rust/runtime/src/runner_protocol/cellular_controller.rs", symbol: "cell failure watcher task ownership lines 592-603", kind: "boundary" },
    ],
    inputs: ["controller task lifetime", "local Child handles"], outputs: ["local child process termination on controller abort"],
    state: ["kill_on_drop(true) child ownership"], invariants: ["A failed local controller does not orphan load-generating cells.", "This is process-lifetime cancellation, not a cross-cell was_cancelled report flag."],
    complexity: { time: "O(local children)", memory: "O(1) per child handle" },
    gates: ["LocalLauncher subprocess mode"], failures: ["Termination is forceful SIGKILL rather than graceful phase cancellation.", "K8sLauncher has no Child handle; operator/pod lifecycle must stop remote cells."],
    pseudocode: pseudocode("spawn local cell with kill_on_drop=true", "watch child inside controller-owned task", "on any controller error, unwind and drop tasks/handles", "OS kills still-running local children"),
    frames: admissionFrames("step-1", ["controller", "cell"], {
      before: ["four local cells running", "one hard failure arrives"], after: ["controller returns error"], invariant: "First terminal failure ends orchestration.",
    }, {
      before: ["three sibling Child handles dropped"], after: ["siblings SIGKILLed"], invariant: "Failed run leaves no local benchmark traffic.", activeLineId: "step-4",
    }),
    predecessors: ["child-exit-arbitration", "controller-timeout"], successors: ["terminal-failure-envelope"],
    routeTags: ["failure", "cancellation", "local", "kill-on-drop"],
  },
  {
    id: "terminal-failure-envelope", chapter: "artifacts", title: "Emit failures caught after application composition", status: "feature-gated",
    summary: "After compose_stock_application succeeds, the CLI wraps run_cellular in catch_unwind: returned errors emit cellular_run_failed and in-scope panics emit internal_panic. Composition happens before catch_unwind and exits directly on failure, so a terminal envelope is not unconditional.",
    source: { path: "rust/cli/src/execute_mode.rs", startLine: 258, endLine: 377, symbol: "run_controller, emit_cellular_failure, and compose_stock_application" },
    evidence: [
      { path: "rust/cli/src/execute_mode.rs", symbol: "emit_cellular_failure lines 338-366", kind: "boundary" },
      { path: "rust/cli/src/execute_mode.rs", symbol: "run_controller catch_unwind arbitration lines 275-335", kind: "boundary" },
      { path: "rust/cli/src/execute_mode.rs", symbol: "compose_stock_application direct exit boundary lines 359-375", kind: "boundary" },
    ],
    inputs: ["successful RunnerApplication composition", "run_cellular Result or panic payload", "benchmark_id"], outputs: ["conditional RunTerminalV2 JSON line; direct process exit remains possible before catch"],
    state: ["pre-catch composition", "caught success or failure terminal branch", "typed stage and error detail"], invariants: ["Only panics inside the run_cellular catch boundary are converted to internal_panic.", "Caught failure carries report_path=None and success=false.", "Successful terminal provenance labels scheduled versus graph correctly."],
    complexity: { time: "O(error formatting)", memory: "O(error detail)" },
    gates: ["velo-enabled cellular controller mode", "compose_stock_application must succeed to enter catch_unwind"], failures: ["compose_stock_application exits with code 2 before catch_unwind and emits no run_terminal envelope.", "If terminal stdout writing itself fails, process termination is the remaining signal."],
    pseudocode: pseudocode("compose stock application outside catch; failure exits directly", "catch_unwind around run_cellular", "on Ok outcome emit successful run_terminal with provenance", "on returned error emit cellular_run_failed", "on caught panic emit internal_panic"),
    frames: admissionFrames("step-4", ["controller"], {
      before: ["composition succeeded", "cell collection returns contextual error"], after: ["run_terminal success=false", "code=cellular_run_failed"], invariant: "Post-composition operational errors remain typed.",
    }, {
      before: ["compose_stock_application fails before catch"], after: ["process exits 2", "no terminal envelope guaranteed"], invariant: "The catch boundary does not cover composition.", activeLineId: "step-1",
    }),
    predecessors: ["final-report-assembly", "telemetry-drop-warning", "child-exit-arbitration", "controller-timeout", "cancellation-propagation", "controller-global-concatenation"], successors: [],
    routeTags: ["failure", "terminal", "protocol-v2", "panic"],
  },
];

const STORAGE_INVARIANTS = {
  retain: "one finalized record remains available for global-order merge",
  exactFold: "one finalized record contributes once, then is dropped",
  sketch: "every finite record metric contributes once; row storage remains bounded",
} as const;

const STORAGE_MODE_DETAILS = {
  retain: {
    rows: "All CapturedRecord values remain available through finish and merge.",
    memory: "O(records), including per-record timing/output/raw state requested by the run.",
    fidelity: "Scheduled retain supports byte-exact global-ordinal re-ingest; graph retain uses deterministic cell/local concatenation and is not byte-identical to one-cell float reduction order.",
    count: "record_count and ingested_count both reflect retained processed rows.",
  },
  exactFold: {
    rows: "Clean heavy records are dropped after exact scalar insertion; only error/detail consumers survive.",
    memory: "Exact NaN-sparse scalar columns grow with records, but token-arrival and heavy record payload retention does not.",
    fidelity: "Dense request_index placement preserves exact percentiles/timeslices; appended shards are within floating summation tolerance.",
    count: "Exact stores retain rows, so record_count is usable; ingested_count is also preserved.",
  },
  sketch: {
    rows: "Each transient row is harvested into per-(phase,tag) t-digest + Welford state and then cleared.",
    memory: "O(tags * compression) per shard/cell, independent of request count.",
    fidelity: "Counts/extrema and integer totals are exact; interior quantiles are approximate and float sum/mean/std can drift by a few ULPs.",
    count: "record_count() may be 0 after clear_rows while ingested_count remains the nonzero true processed total.",
  },
} as const;

const ALGORITHMS: readonly AlgorithmDefinition[] = [
  ...ELIGIBILITY_ALGORITHMS,
  ...OWNERSHIP_ALGORITHMS,
  ...CONTROL_ALGORITHMS,
  ...DISTRIBUTION_ALGORITHMS,
  ...EXECUTION_ALGORITHMS,
  ...CAPTURE_ALGORITHMS,
  ...MERGE_ALGORITHMS,
  ...ARTIFACT_ALGORITHMS,
];
const ROUTE_ORDER = [
  "eligibility",
  "ownership",
  "control",
  "distribution",
  "execution",
  "capture",
  "merge",
  "artifacts",
] as const;
const ARTIFACT_TERMINAL_IDS = [
  "controller-global-concatenation",
  "artifact-completion-barrier",
] as const;

const FANOUT_GRAPH_ZERO_REQUESTS_LIMITATION =
  "Graph duration/adaptive fanout has no request/session budget: the controller publishes zero verification requests, so the overlay proves nothing.";
const FANOUT_GRPC_HTTP_PROBE_LIMITATION =
  "gRPC fanout verification still constructs HTTP chat probes; HTTP send failures only warn, so zero distribution misses does not prove gRPC dispatch.";
const CELLULAR_PER_RECORD_OTLP_LIMITATION =
  "The controller rebuilds NativeReport from merged records/stores with otel_per_record=None; aggregate OTLP can export, but cell-local per-record histogram accumulators are not shipped.";
const WORKLOAD_CLASSIFICATION_LIMITATION =
  "The authored workload selector is overridden by dataset shape: dag_jsonl, weka_trace, and dynamo_trace execute as graph; all other dataset shapes execute as scheduled.";
const CROSS_HOST_HIERARCHY_REFUSAL =
  "Cross-host hierarchy requests are refused before controller startup.";
const SAME_HOST_HIERARCHY_REFUSAL =
  "Same-host hierarchy requests are refused before controller startup.";
const PARQUET_SKIPPED_LIMITATION =
  "Per-record Parquet is accepted on a build without the parquet feature, then warned and skipped; no Parquet concat stage runs.";
const PARQUET_RETAIN_LIMITATION =
  "Requested exact-fold resolves to effective retain on a build without the parquet feature because the Parquet streaming lane is unavailable, even though the final Parquet output is warned and skipped.";
const ADAPTIVE_RETAIN_LIMITATION =
  "Requested exact-fold resolves to effective retain for scheduled adaptive scale because control windows consume retained per-turn records; cellular scheduled adaptive execution then fails its prelaunch phase gate.";
const SKETCH_COLUMNAR_OMITTED_LIMITATION =
  "Sketch mode accepts CSV/Parquet path fields but retains no rows for those writers, so the per-record output is omitted.";
const GATE_STAGE_LABELS: Readonly<Record<GateStage, string>> = {
  "pre-controller": "Pre-controller role admission",
  "controller-prelaunch": "Controller prelaunch validation",
  "cell-side": "Cell-side execution validation after startup",
  "aggregator-receive": "Reserved hierarchy refusal stage",
};

function effectiveWorkload(selection: SelectorState): SelectorState["workload"] {
  return ["dag-jsonl", "weka", "dynamo"].includes(selection.dataset)
    ? "graph"
    : "scheduled";
}

function effectiveTopology(selection: SelectorState): SelectorState["topology"] {
  return selection.topology;
}

function effectiveArtifact(selection: SelectorState): EffectiveArtifact {
  if (
    selection.storage === "sketch" &&
    ["csv", "parquet"].includes(selection.artifacts)
  ) {
    return "omitted";
  }
  if (selection.artifacts === "parquet" && selection.build !== "full") {
    return "omitted";
  }
  if (selection.artifacts === "otlp") return "aggregate-otlp";
  return selection.artifacts;
}

function effectiveStorage(selection: SelectorState): SelectorState["storage"] {
  if (selection.storage !== "exact-fold") return selection.storage;
  if (selection.artifacts === "parquet" && selection.build !== "full") {
    return "retain";
  }
  if (
    effectiveWorkload(selection) === "scheduled" &&
    selection.budget === "adaptive"
  ) {
    return "retain";
  }
  return "exact-fold";
}

function effectiveSettings(selection: SelectorState): EffectiveSettings {
  return {
    workload: effectiveWorkload(selection),
    topology: effectiveTopology(selection),
    artifacts: effectiveArtifact(selection),
    storage: effectiveStorage(selection),
  };
}

const VALIDATION_GATES: readonly ValidationGate[] = [
  {
    id: "velo-build-required",
    algorithmId: "velo-feature-admission",
    order: 10,
    stage: "pre-controller",
    rejects: (selection) => selection.build === "lean",
    reason: "Multi-cell controller and cell roles require a Velo-bearing build; the aggregator role is an explicit refusal.",
  },
  {
    id: "offline-transport-rejected",
    algorithmId: "cellular-run-shape-validation",
    order: 20,
    stage: "controller-prelaunch",
    rejects: (selection) => selection.transport === "offline",
    reason: "Cellular execution is wired only for HTTP and gRPC; DynoSim drivers do not ship cell partitions.",
  },
  {
    id: "multiturn-retain-rejected",
    algorithmId: "cellular-run-shape-validation",
    order: 40,
    stage: "controller-prelaunch",
    rejects: (selection) => effectiveWorkload(selection) === "scheduled" && selection.turns === "multi" && effectiveStorage(selection) !== "exact-fold",
    reason: "cellular-run-shape validation rejects multi-turn conversations when effective storage is RETAIN; only exact-fold's order-independent store concatenation is sound.",
  },
  {
    id: "multiturn-random-rejected",
    algorithmId: "cellular-run-shape-validation",
    order: 41,
    stage: "controller-prelaunch",
    rejects: (selection) => effectiveWorkload(selection) === "scheduled" && selection.turns === "multi" && selection.sampler === "random",
    reason: "cellular-run-shape validation rejects multi-turn random-with-replacement because it has no stable per-cell conversation partition; use sequential or shuffle.",
  },
  {
    id: "scheduled-duration-rejected",
    algorithmId: "scheduled-budget-validation",
    order: 50,
    stage: "controller-prelaunch",
    rejects: (selection) => effectiveWorkload(selection) === "scheduled" && selection.budget === "duration",
    reason: "Scheduled cellular execution requires a finite requests or exact-fold sessions budget.",
  },
  {
    id: "scheduled-adaptive-rejected",
    algorithmId: "scheduled-budget-validation",
    order: 51,
    stage: "controller-prelaunch",
    rejects: (selection) => effectiveWorkload(selection) === "scheduled" && selection.budget === "adaptive",
    reason: "Scheduled adaptive cellular execution lacks cross-cell scaling consensus and fails closed.",
  },
  {
    id: "scheduled-sessions-retain-rejected",
    algorithmId: "scheduled-budget-validation",
    order: 52,
    stage: "controller-prelaunch",
    rejects: (selection) =>
      effectiveWorkload(selection) === "scheduled" &&
      selection.budget === "sessions" &&
      effectiveStorage(selection) !== "exact-fold",
    reason: "A scheduled sessions budget is admitted only on the exact-fold merge path.",
  },
  {
    id: "graph-requests-rejected",
    algorithmId: "graph-budget-validation",
    order: 53,
    stage: "controller-prelaunch",
    rejects: (selection) => effectiveWorkload(selection) === "graph" && selection.budget === "requests",
    reason: "Graph cellular execution partitions sessions or duration, not static-node request budgets.",
  },
  {
    id: "sketch-record-artifacts-rejected",
    algorithmId: "sketch-artifact-validation",
    order: 54,
    stage: "cell-side",
    rejects: (selection) =>
      selection.storage === "sketch" &&
      ["records", "raw"].includes(selection.artifacts),
    reason: "validate_plan rejects records_path and raw_path when sketch storage is enabled.",
  },
  {
    id: "sketch-otlp-rejected",
    algorithmId: "sketch-artifact-validation",
    order: 55,
    stage: "cell-side",
    rejects: (selection) =>
      selection.storage === "sketch" &&
      selection.artifacts === "otlp",
    reason: "validate_plan rejects native per-record OTLP when sketch storage is enabled.",
  },
  {
    id: "hierarchy-request-rejected",
    algorithmId: "hierarchical-tier-sizing",
    order: 1,
    stage: "pre-controller",
    rejects: (selection) => selection.topology !== "flat",
    reason: "Hierarchical cellular aggregation is unavailable and is refused before controller startup.",
  },
];

const CHAPTERS: readonly { id: ChapterId; label: string; thesis: string }[] = [
  { id: "eligibility", label: "Entry & eligibility", thesis: "Fail closed before cells launch." },
  { id: "ownership", label: "Ownership & budgets", thesis: "Tile work without gaps or collisions." },
  { id: "control", label: "Control plane", thesis: "Coordinate generations and process state." },
  { id: "distribution", label: "Data distribution", thesis: "Deliver or regenerate only owned work." },
  { id: "execution", label: "Execution", thesis: "Nest cell and worker shards." },
  { id: "capture", label: "Capture & storage", thesis: "Retain, fold, or sketch terminal records." },
  { id: "merge", label: "Merge & reporting", thesis: "Reduce partitions under explicit fidelity rules." },
  { id: "artifacts", label: "Artifacts & failure", thesis: "Move bytes safely and select one terminal outcome." },
];

const ACTORS: readonly Actor[] = [
  "entry-point",
  "controller",
  "wire",
  "cell",
  "worker",
  "aggregator",
];

const ACTOR_LABELS: Readonly<Record<Actor, string>> = {
  "entry-point": "entry point",
  controller: "controller",
  wire: "wire",
  cell: "cell",
  worker: "worker",
  aggregator: "hierarchy refusal",
};

const WORKBOOK_CSS = `
.caw-shell { min-height: 100%; }
.caw-header {
  position: sticky;
  top: 0;
  z-index: 4;
}
.caw-sheet {
  display: grid;
  grid-template-columns: minmax(560px, 2fr) minmax(280px, .8fr);
  gap: 18px;
}
.caw-main {
  display: grid;
  grid-template-rows: minmax(360px, 1fr) auto;
  gap: 14px;
  min-width: 0;
}
.caw-composer {
  display: grid;
  grid-template-columns: minmax(260px, .65fr) minmax(640px, 1.6fr);
  gap: 18px;
}
.caw-route-scroll,
.caw-trace-scroll { overflow-x: auto; }
.caw-trace-svg { min-width: 840px; }
.caw-active-link {
  stroke-dasharray: 8 6;
  animation: caw-flow 1.6s linear infinite;
}
@keyframes caw-flow { to { stroke-dashoffset: -20; } }
.caw-status {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  padding: 2px 7px;
  border: 1px solid;
  font-size: 11px;
  line-height: 1.4;
}
.caw-status-partial,
.caw-status-approximate { border-style: dashed; }
.caw-status-feature-gated { border-style: dotted; }
.caw-contract-rail summary { cursor: pointer; }
.caw-frame-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 10px;
  align-items: stretch;
}
.caw-route-facets {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.caw-shell :is(button, a, input, select, textarea, summary, [tabindex]):focus-visible {
  outline: 2px solid currentColor;
  outline-offset: 3px;
}
.workbook-index-scroll { max-height: min(68vh, 760px); overflow-y: auto; scrollbar-gutter: stable; }
.workbook-index-row,
.trace-stage,
.trace-actor,
.trace-link,
.trace-frame { transition: opacity 180ms ease, color 180ms ease, background 180ms ease, stroke 180ms ease; }
.workbook-play-reduced-note { display: none; }
.composed-route {
  display: flex;
  gap: 10px;
  padding: 4px 2px 12px;
  scrollbar-gutter: stable;
  scroll-snap-type: x proximity;
}
.composed-route-stop {
  flex: 0 0 min(300px, 78vw);
  scroll-snap-align: start;
}
.composed-route-stop summary { cursor: pointer; }
.recipe-strip {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding: 2px 2px 8px;
  scrollbar-gutter: stable;
}
.decision-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 0;
}
.decision-sides {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 18px;
}
.workbook-live-region {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
@media (max-width: 980px) {
  .caw-sheet,
  .caw-composer { grid-template-columns: 1fr !important; }
  .caw-main { grid-template-rows: auto; }
  .workbook-layout { grid-template-columns: 1fr !important; }
  .workbook-index-scroll { max-height: 360px; }
  .algorithm-workbench { grid-template-columns: 1fr !important; }
  .composer-layout { grid-template-columns: 1fr !important; }
  .caw-contract-rail {
    padding-left: 0 !important;
    padding-top: 12px;
    border-left: 0 !important;
  }
  .decision-grid,
  .decision-sides { grid-template-columns: 1fr; }
}
.caw-reduced .workbook-play { display: none !important; }
.caw-reduced .workbook-play-reduced-note { display: inline; }
.caw-reduced .caw-active-link { animation: none; }
.caw-reduced .workbook-index-row,
.caw-reduced .trace-stage,
.caw-reduced .trace-actor,
.caw-reduced .trace-link,
.caw-reduced .trace-frame { transition: none !important; }
.caw-reduced .composed-route { scroll-behavior: auto; scroll-snap-type: none; }
@media (max-width: 620px) {
  .caw-frame-grid,
  .caw-route-facets { grid-template-columns: 1fr; }
  .caw-frame-emission { text-align: left !important; }
}
@media (prefers-reduced-motion: reduce) {
  .caw-active-link { animation: none; }
  .workbook-play { display: none !important; }
  .workbook-play-reduced-note { display: inline; }
  .workbook-index-row,
  .trace-stage,
  .trace-actor,
  .trace-link,
  .trace-frame { transition: none !important; }
  .composed-route { scroll-behavior: auto; scroll-snap-type: none; }
}
`;

function searchableText(algorithm: AlgorithmDefinition): string {
  return [
    algorithm.id,
    algorithm.title,
    algorithm.summary,
    algorithm.source.symbol,
    algorithm.source.path,
    ...algorithm.invariants,
    ...algorithm.failures,
    ...algorithm.gates,
    ...algorithm.evidence.flatMap((item) => [item.path, item.symbol]),
  ]
    .join(" ")
    .toLowerCase();
}

const SEARCH_INDEX = new Map(
  ALGORITHMS.map((algorithm) => [algorithm.id, searchableText(algorithm)]),
);

function filterAlgorithms(
  algorithms: readonly AlgorithmDefinition[],
  query: string,
  chapter?: ChapterId,
): readonly AlgorithmDefinition[] {
  const needle = query.trim().toLowerCase();
  return algorithms.filter(
    (algorithm) =>
      (!chapter || algorithm.chapter === chapter) &&
      (!needle || SEARCH_INDEX.get(algorithm.id)?.includes(needle)),
  );
}

const WORKBOOK_ROOT_ID = "cellular-algorithm-workbook-root";
const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)";
let playbackGeneration = 0;
let playbackTimers: readonly number[] = [];
let reducedMotionListenerInstalled = false;
let keyboardListenerInstalled = false;
let manualReducedMotion = false;

type KeyboardNavigation = {
  previousAlgorithm: () => void;
  nextAlgorithm: () => void;
  previousFrame?: () => void;
  nextFrame?: () => void;
};

let keyboardNavigation: KeyboardNavigation | undefined;

function cancelWorkbookPlayback(): void {
  playbackGeneration += 1;
  for (const timer of playbackTimers) clearTimeout(timer);
  playbackTimers = [];
}

function reducedMotionIsActive(): boolean {
  return (
    manualReducedMotion ||
    typeof window !== "undefined" && window.matchMedia(REDUCED_MOTION_QUERY).matches
  );
}

function playbackCanMutate(generation: number): boolean {
  return (
    generation === playbackGeneration &&
    !reducedMotionIsActive() &&
    typeof document !== "undefined" &&
    document.getElementById(WORKBOOK_ROOT_ID)?.isConnected === true
  );
}

function startWorkbookPlayback(
  frameIndexes: readonly number[],
  onFrame: (frameIndex: number) => void,
): void {
  cancelWorkbookPlayback();
  if (frameIndexes.length === 0 || reducedMotionIsActive()) return;
  const generation = playbackGeneration;
  playbackTimers = frameIndexes.map((frameIndex, offset) =>
    setTimeout(() => {
      if (!playbackCanMutate(generation)) {
        if (generation === playbackGeneration) cancelWorkbookPlayback();
        return;
      }
      onFrame(frameIndex);
      if (offset === frameIndexes.length - 1) cancelWorkbookPlayback();
    }, (offset + 1) * 900),
  );
}

function installReducedMotionCancellation(): void {
  if (reducedMotionListenerInstalled || typeof window === "undefined") return;
  const media = window.matchMedia(REDUCED_MOTION_QUERY);
  media.addEventListener("change", cancelWorkbookPlayback);
  reducedMotionListenerInstalled = true;
}

function isShortcutTarget(target: EventTarget | null): boolean {
  return (
    target instanceof Element &&
    target.closest(
      "input, select, textarea, button, a, [contenteditable]:not([contenteditable='false'])",
    ) !== null
  );
}

function installKeyboardNavigation(): void {
  if (keyboardListenerInstalled || typeof window === "undefined") return;
  window.addEventListener("keydown", (event) => {
    if (
      isShortcutTarget(event.target) ||
      event.altKey ||
      event.ctrlKey ||
      event.metaKey ||
      !keyboardNavigation
    ) {
      return;
    }
    const action =
      event.key === "ArrowLeft"
        ? keyboardNavigation.previousFrame
        : event.key === "ArrowRight"
          ? keyboardNavigation.nextFrame
          : event.key === "["
            ? keyboardNavigation.previousAlgorithm
            : event.key === "]"
              ? keyboardNavigation.nextAlgorithm
              : undefined;
    if (!action) return;
    event.preventDefault();
    cancelWorkbookPlayback();
    action();
  });
  keyboardListenerInstalled = true;
}

const DEFAULT_SELECTION: SelectorState = {
  workload: "scheduled",
  transport: "http",
  dataset: "synthetic",
  fanout: "off",
  turns: "single",
  sampler: "sequential",
  budget: "requests",
  storage: "exact-fold",
  topology: "flat",
  start: "synchronized",
  deployment: "same-host",
  artifacts: "summary",
  build: "velo",
};

const SELECTOR_OPTIONS: Readonly<{ [K in SelectorKey]: readonly SelectorState[K][] }> = {
  workload: ["scheduled", "graph"],
  transport: ["http", "grpc", "offline"],
  dataset: ["synthetic", "file", "public", "dag-jsonl", "weka", "dynamo"],
  fanout: ["off", "verify"],
  turns: ["single", "multi"],
  sampler: ["sequential", "shuffle", "random"],
  budget: ["requests", "sessions", "duration", "adaptive"],
  storage: ["retain", "exact-fold", "sketch"],
  topology: ["flat", "local-tree", "external-tree"],
  start: ["synchronized", "phaser", "barrier-free"],
  deployment: ["same-host", "cross-host"],
  artifacts: ["summary", "records", "raw", "csv", "parquet", "otlp"],
  build: ["lean", "velo", "full"],
};

const SELECTOR_LABELS: Readonly<Record<SelectorKey, string>> = {
  workload: "Workload",
  transport: "Transport",
  dataset: "Dataset",
  fanout: "Dataset fanout",
  turns: "Turns",
  sampler: "Sampler",
  budget: "Budget",
  storage: "Storage",
  topology: "Topology",
  start: "Start",
  deployment: "Deployment",
  artifacts: "Artifacts",
  build: "Build",
};

function optionLabel(value: string): string {
  return value
    .split("-")
    .map((part) => part === "grpc" ? "gRPC" : part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function legalSelectorValue<K extends SelectorKey>(
  key: K,
  value: unknown,
): value is SelectorState[K] {
  return typeof value === "string" && (SELECTOR_OPTIONS[key] as readonly string[]).includes(value);
}

function normalizeSelection(value: unknown): SelectorState {
  const candidate = value && typeof value === "object"
    ? value as Partial<Record<SelectorKey, unknown>> & { artifacts?: unknown }
    : {};
  const normalized = { ...DEFAULT_SELECTION };
  for (const key of Object.keys(SELECTOR_OPTIONS) as SelectorKey[]) {
    const raw = key === "artifacts" && candidate.artifacts === "columnar"
      ? "parquet"
      : candidate[key];
    if (legalSelectorValue(key, raw)) {
      (normalized as Record<SelectorKey, SelectorState[SelectorKey]>)[key] = raw;
    }
  }
  return normalized;
}

function validateWorkbook(
  algorithms: readonly AlgorithmDefinition[],
  gates: readonly ValidationGate[],
): readonly string[] {
  const errors: string[] = [];
  const ids = new Set<string>();
  const chapterCounts = new Map<ChapterId, number>();

  for (const algorithm of algorithms) {
    if (ids.has(algorithm.id)) errors.push(`duplicate algorithm: ${algorithm.id}`);
    ids.add(algorithm.id);
    chapterCounts.set(algorithm.chapter, (chapterCounts.get(algorithm.chapter) ?? 0) + 1);
    if (!algorithm.source.path || algorithm.source.startLine > algorithm.source.endLine) {
      errors.push(`invalid source: ${algorithm.id}`);
    }
    if (algorithm.evidence.length === 0) errors.push(`missing evidence: ${algorithm.id}`);
    if (algorithm.invariants.length === 0) errors.push(`missing invariant: ${algorithm.id}`);
    if (algorithm.failures.length === 0) errors.push(`missing failure contract: ${algorithm.id}`);
    const lineIds = new Set<string>();
    for (const line of algorithm.pseudocode) {
      if (lineIds.has(line.id)) errors.push(`duplicate pseudocode line: ${algorithm.id}/${line.id}`);
      lineIds.add(line.id);
    }
    if (algorithm.frames.length < 2) errors.push(`fewer than two trace frames: ${algorithm.id}`);
    const frameIds = new Set<string>();
    for (const frame of algorithm.frames) {
      if (frameIds.has(frame.id)) errors.push(`duplicate trace frame: ${algorithm.id}/${frame.id}`);
      frameIds.add(frame.id);
      if (algorithm.pseudocode.filter((line) => line.id === frame.activeLineId).length !== 1) {
        errors.push(`frame does not resolve to exactly one pseudocode line: ${algorithm.id}/${frame.id}`);
      }
    }
  }

  for (const chapter of CHAPTERS) {
    if ((chapterCounts.get(chapter.id) ?? 0) === 0) {
      errors.push(`chapter has no algorithms: ${chapter.id}`);
    }
  }
  for (const algorithm of algorithms) {
    const indexed = SEARCH_INDEX.get(algorithm.id);
    if (
      !indexed ||
      !filterAlgorithms(algorithms, algorithm.id).some((item) => item.id === algorithm.id)
    ) {
      errors.push(`algorithm is unreachable from index search: ${algorithm.id}`);
    }
  }
  for (const algorithm of algorithms) {
    for (const linkedId of [...algorithm.predecessors, ...algorithm.successors]) {
      if (!ids.has(linkedId)) errors.push(`unknown algorithm link: ${algorithm.id}/${linkedId}`);
    }
  }
  for (const gate of gates) {
    if (!ids.has(gate.algorithmId)) errors.push(`unknown gate algorithm: ${gate.id}`);
  }
  const gateOrders = new Set<number>();
  const stageRank: Readonly<Record<GateStage, number>> = {
    "pre-controller": 0,
    "controller-prelaunch": 1,
    "cell-side": 2,
    "aggregator-receive": 3,
  };
  let previousStageRank = -1;
  for (const gate of gates) {
    if (gateOrders.has(gate.order)) errors.push(`duplicate gate order: ${gate.order}`);
    gateOrders.add(gate.order);
  }
  for (const gate of [...gates].sort((left, right) => left.order - right.order)) {
    if (stageRank[gate.stage] < previousStageRank) {
      errors.push(`gate stage regresses in runtime order: ${gate.id}`);
    }
    previousStageRank = stageRank[gate.stage];
  }
  for (const id of ELIGIBILITY_IDS) {
    if (!ids.has(id)) errors.push(`missing eligibility algorithm: ${id}`);
  }
  for (const id of OWNERSHIP_IDS) {
    if (!ids.has(id)) errors.push(`missing ownership algorithm: ${id}`);
  }
  for (const id of CONTROL_IDS) {
    if (!ids.has(id)) errors.push(`missing control algorithm: ${id}`);
  }
  for (const id of DISTRIBUTION_IDS) {
    if (!ids.has(id)) errors.push(`missing distribution algorithm: ${id}`);
  }
  for (const id of EXECUTION_IDS) {
    if (!ids.has(id)) errors.push(`missing execution algorithm: ${id}`);
  }
  for (const id of CAPTURE_IDS) {
    if (!ids.has(id)) errors.push(`missing capture algorithm: ${id}`);
  }
  for (const id of MERGE_IDS) {
    if (!ids.has(id)) errors.push(`missing merge algorithm: ${id}`);
  }
  for (const id of ARTIFACT_IDS) {
    if (!ids.has(id)) errors.push(`missing artifact algorithm: ${id}`);
  }
  errors.push(...validateComposerCoverage());
  return errors;
}

function firstRejection(selection: SelectorState): ValidationGate | undefined {
  return [...VALIDATION_GATES]
    .sort((left, right) => left.order - right.order)
    .find((gate) => gate.rejects(selection));
}

function unique<T>(values: readonly T[]): readonly T[] {
  return [...new Set(values)];
}

function routeFragments(selection: SelectorState): Readonly<Record<ChapterId, readonly string[]>> {
  const effective = effectiveSettings(selection);
  const isGraph = effective.workload === "graph";
  const isCrossHostFile = selection.deployment === "cross-host" &&
    ["file", "dag-jsonl", "weka", "dynamo"].includes(selection.dataset);
  const fileArtifact = ["records", "raw", "csv", "parquet"].includes(effective.artifacts);

  const eligibility = [
    "execution-mode-dispatch",
    "execution-child-selection",
    "cell-count-resolution",
    "controller-promotion",
    "velo-feature-admission",
    "cellular-run-shape-validation",
    "run-kind-classification",
    isGraph ? "graph-budget-validation" : "scheduled-budget-validation",
    "storage-compatibility-prediction",
    "sketch-artifact-validation",
    "execution-merge-backstops",
  ];
  const ownership = [
    "modulo-cell-ownership",
    "owned-positions-tiling",
    "shared-seed-resolution",
    "cell-envelope-construction",
    ...(!isGraph && selection.budget === "sessions" ? ["scheduled-session-slicing"] : []),
    "capacity-rate-ramp-slicing",
    ...(!isGraph
      ? [
          "phase-ordinal-bases",
          "direct-issuance-authority",
          "cellular-issuance-authority",
        ]
      : []),
    ...(!isGraph && selection.turns === "multi"
      ? ["multi-turn-detection", "conversation-ownership"]
      : []),
  ];
  const control = [
    "velo-controller-bind",
    "handler-registration",
    selection.deployment === "cross-host" ? "external-cell-launch" : "local-cell-launch",
    "velo-peer-connect",
    "controller-child-arbitration",
    selection.start === "phaser"
      ? "phaser-start"
      : selection.start === "barrier-free"
        ? "barrier-free-launch"
        : "synchronized-start",
  ];
  const datasetPlan = isCrossHostFile
    ? ["dataset-serve-plan", ...(isGraph ? ["recorded-graph-file-enumeration"] : [])]
    : [];
  const datasetMaterialization = isCrossHostFile
    ? [
        "dataset-manifest-validation",
        "dataset-safe-path-mapping",
        "dataset-http-zstd-reconstruct",
      ]
    : ["canonical-dataset-regeneration"];
  const fanoutVerification = selection.fanout === "verify"
    ? [
        "controller-fanout-generation",
        "dataset-velo-subscribe",
        "dataset-velo-replay-live",
        "owned-index-build",
        "dispatch-on-issue",
        "dispatch-on-complete",
        "distribution-miss",
        "fanout-verification-overlay",
      ]
    : [];
  const distribution = [...datasetPlan, ...fanoutVerification, ...datasetMaterialization];
  const execution = isGraph
    ? [
        "partitioned-graph-source",
        "graph-global-instance-ordinal",
        "scheduled-graph-runtime-branch",
      ]
    : [
        "partitioned-scheduled-sampler",
        "two-level-partition",
        "thread-phase-slicing",
        "scheduled-shard-runtime",
        "issuance-dispatch-injection",
        "scheduled-graph-runtime-branch",
      ];
  const capture = effective.storage === "retain"
    ? ["terminal-record-finalization", "retain-record-capture"]
    : effective.storage === "exact-fold"
      ? ["terminal-record-finalization", "streaming-exact-fold", "column-store-append", "ingested-count-preservation", "partition-messagepack-encode"]
      : ["terminal-record-finalization", "sketch-scratch-harvest", "tdigest-insert-compress", "welford-aggregate-state", "tagged-sketch-merge", "ingested-count-preservation", "partition-messagepack-encode"];
  const storageMerge = effective.storage === "retain"
    ? isGraph
      ? ["graph-concatenation-renumber"]
      : ["scheduled-global-ordinal-merge", "ordinal-duplicate-detection", "ordinal-missing-detection", "ordinal-range-detection"]
    : effective.storage === "exact-fold"
      ? ["exact-fold-store-merge"]
      : ["sketch-tdigest-merge"];
  const topologyMerge = ["controller-partition-collection"];
  const merge = [
    ...topologyMerge,
    ...storageMerge,
    "heartbeat-aggregation",
    "final-report-assembly",
    "merged-report-fidelity-boundary",
  ];
  const artifacts = [
    ...(fileArtifact
      ? [
          "shard-local-concatenation",
          "cell-local-concatenation",
          ...(selection.deployment === "cross-host"
            ? ["artifact-authority-allowlist", "artifact-http-zstd-upload", "partial-file-atomic-replace", "artifact-completion-barrier"]
            : []),
          "controller-global-concatenation",
        ]
      : []),
    "telemetry-drop-warning",
    "child-exit-arbitration",
    "controller-timeout",
    "cancellation-propagation",
    "terminal-failure-envelope",
  ];

  return { eligibility, ownership, control, distribution, execution, capture, merge, artifacts };
}

function routeAlgorithmIds(selection: SelectorState): readonly string[] {
  const fragments = routeFragments(selection);
  return unique(ROUTE_ORDER.flatMap((chapter) => fragments[chapter]));
}

function effectiveSettingLimitations(
  selection: SelectorState,
  effective: EffectiveSettings,
): readonly string[] {
  return [
    ...(selection.workload !== effective.workload
      ? [WORKLOAD_CLASSIFICATION_LIMITATION]
      : []),
    ...(selection.topology !== effective.topology &&
    selection.deployment === "cross-host"
      ? [CROSS_HOST_HIERARCHY_REFUSAL]
      : []),
    ...(selection.topology !== effective.topology &&
    selection.deployment === "same-host"
      ? [SAME_HOST_HIERARCHY_REFUSAL]
      : []),
    ...(selection.storage !== effective.storage
      ? [selection.artifacts === "parquet" && selection.build !== "full"
          ? PARQUET_RETAIN_LIMITATION
          : ADAPTIVE_RETAIN_LIMITATION]
      : []),
    ...(selection.artifacts === "parquet" && selection.build !== "full"
      ? [PARQUET_SKIPPED_LIMITATION]
      : []),
    ...(selection.storage === "sketch" &&
    ["csv", "parquet"].includes(selection.artifacts)
      ? [SKETCH_COLUMNAR_OMITTED_LIMITATION]
      : []),
  ];
}

function rejectedAlgorithmIds(
  selection: SelectorState,
  rejection: ValidationGate,
): readonly string[] {
  const fragments = routeFragments(selection);
  if (rejection.stage === "pre-controller") {
    return unique([
      "execution-mode-dispatch",
      "execution-child-selection",
      "cell-count-resolution",
      "controller-promotion",
      rejection.algorithmId,
    ]);
  }
  if (rejection.stage === "cell-side") {
    return unique([
      ...fragments.eligibility.filter(
        (id) =>
          id !== "sketch-artifact-validation" &&
          id !== "execution-merge-backstops",
      ),
      ...fragments.ownership,
      ...fragments.control,
      ...fragments.distribution,
      rejection.algorithmId,
    ]);
  }
  const algorithmIds = routeAlgorithmIds(selection);
  const gateIndex = algorithmIds.indexOf(rejection.algorithmId);
  return gateIndex >= 0
    ? algorithmIds.slice(0, gateIndex + 1)
    : [...algorithmIds, rejection.algorithmId];
}

function deriveRoute(selection: SelectorState): RouteResult {
  const effective = effectiveSettings(selection);
  const algorithmIds = routeAlgorithmIds(selection);
  const effectiveLimitations = effectiveSettingLimitations(selection, effective);
  const rejection = firstRejection(selection);
  if (rejection) {
    return {
      valid: false,
      algorithmIds: rejectedAlgorithmIds(selection, rejection),
      effective,
      limitations: effectiveLimitations,
      gateStage: rejection.stage,
      rejectedBy: rejection.id,
      reason: rejection.reason,
    };
  }

  const memory = effective.storage === "retain"
    ? "O(completed records): full captured records remain until merge."
    : effective.storage === "exact-fold"
      ? "O(metric values): captured records fold into an exact column store and are dropped."
      : "O(shards × digest + concurrency): bounded t-digests and transient records.";
  const fidelity = effective.storage === "sketch"
    ? "Counts, sums, minima, and maxima are exact; percentiles and standard deviation are approximate."
    : effective.storage === "exact-fold"
      ? "Counts, extrema, and retained-column percentiles are exact; floating sums and means may differ within merge-order tolerance."
      : effective.workload === "graph"
        ? "Raw graph records concatenate by cell and receive dense replacement indices."
        : "Scheduled records are restored in byte-exact global dispatch order.";
  const artifactSummary: Readonly<Record<SelectorState["artifacts"], string>> = {
    summary: "Merged native report and console summary; side-channel telemetry is warned and omitted.",
    records: "Per-record records stream per shard, concatenate per cell, then concatenate globally.",
    raw: "Raw request/response JSONL follows shard, cell, and controller concatenation.",
    csv: "Per-record CSV streams per shard and fuses through cellular artifact stages without an optional Cargo feature.",
    parquet: "Per-record Parquet streams and concatenates row groups on a build carrying the `parquet` feature.",
    otlp: "Aggregate OTLP exports from the merged report; cell-local per-record histogram accumulators are not shipped.",
  };
  const storageControls = selection.storage === "retain"
    ? ["AIPERF_RUNTIME_EXACT_FOLD=0", "AIPERF_METRICS_SKETCH=0"]
    : selection.storage === "exact-fold"
      ? ["AIPERF_RUNTIME_EXACT_FOLD=1", "AIPERF_METRICS_SKETCH=0"]
      : ["AIPERF_METRICS_SKETCH=1"];
  const fileArtifact = ["records", "raw", "csv", "parquet"].includes(effective.artifacts);
  const crossHostDatasetShipping = selection.deployment === "cross-host" &&
    ["file", "dag-jsonl", "weka", "dynamo"].includes(selection.dataset);
  const stageEArtifactShipping = selection.deployment === "cross-host" && fileArtifact;
  const environment = [
    `AIPERF_CELL_LAUNCHER=${selection.deployment === "cross-host" ? "k8s" : "local"}`,
    ...storageControls,
    ...(selection.start === "phaser" ? ["AIPERF_CELL_PHASER_START=1"] : []),
    ...(selection.start === "barrier-free" ? ["AIPERF_CELL_BARRIER_FREE=1"] : []),
    ...(selection.fanout === "verify" ? ["AIPERF_CELL_DATASET_FANOUT=1"] : []),
    ...(crossHostDatasetShipping || stageEArtifactShipping
      ? ["AIPERF_CELL_HTTP_ARTIFACT_SHIPPING=1"]
      : []),
  ];
  const compileFeatures = selection.build === "full"
    ? ["dynosim", "parquet", "velo"]
    : ["velo"];
  const limitations = [
    ...effectiveLimitations,
    ...(selection.fanout === "verify" &&
    effective.workload === "graph" &&
    ["duration", "adaptive"].includes(selection.budget)
      ? [FANOUT_GRAPH_ZERO_REQUESTS_LIMITATION]
      : []),
    ...(selection.fanout === "verify" && selection.transport === "grpc"
      ? [FANOUT_GRPC_HTTP_PROBE_LIMITATION]
      : []),
    ...(selection.artifacts === "otlp"
      ? [CELLULAR_PER_RECORD_OTLP_LIMITATION]
      : []),
  ];
  const evidence = unique(
    algorithmIds
      .map((id) => ALGORITHMS.find((algorithm) => algorithm.id === id))
      .filter((algorithm): algorithm is AlgorithmDefinition => algorithm !== undefined)
      .flatMap((algorithm) => algorithm.evidence),
  );
  return {
    valid: true,
    algorithmIds,
    memory,
    fidelity,
    artifacts: effective.artifacts === "omitted"
      ? `Requested ${selection.artifacts}; runtime emits no per-record file for this effective configuration.`
      : effective.artifacts === "aggregate-otlp"
        ? artifactSummary.otlp
        : artifactSummary[effective.artifacts],
    effective,
    environment,
    compileFeatures,
    limitations,
    evidence,
  };
}

const MAX_ROUTE_CACHE_ENTRIES = 512;
const ROUTE_CACHE = new Map<string, RouteResult>();

function cachedRoute(selection: unknown): RouteResult {
  const normalized = normalizeSelection(selection);
  const key = JSON.stringify(normalized);
  const existing = ROUTE_CACHE.get(key);
  if (existing) return existing;
  const derived = deriveRoute(normalized);
  if (ROUTE_CACHE.size >= MAX_ROUTE_CACHE_ENTRIES) {
    const oldest = ROUTE_CACHE.keys().next().value;
    if (oldest !== undefined) ROUTE_CACHE.delete(oldest);
  }
  ROUTE_CACHE.set(key, derived);
  return derived;
}

type RouteFixture = {
  name: string;
  selection: SelectorState;
  valid: boolean;
  includes?: readonly string[];
  excludes?: readonly string[];
  ordered?: readonly (readonly [string, string])[];
  rejectedBy?: string;
  limitations?: readonly string[];
  environmentIncludes?: readonly string[];
  environmentExcludes?: readonly string[];
  compileFeatures?: readonly string[];
  effective?: Partial<EffectiveSettings>;
  gateStage?: GateStage;
};

const GATE_FIXTURES: readonly RouteFixture[] = [
  { name: "lean fanout phaser build", selection: { ...DEFAULT_SELECTION, build: "lean", fanout: "verify", start: "phaser" }, valid: false, rejectedBy: "velo-build-required", gateStage: "pre-controller", includes: ["controller-promotion"], ordered: [["cell-count-resolution", "controller-promotion"], ["controller-promotion", "velo-feature-admission"]] },
  { name: "offline rejection", selection: { ...DEFAULT_SELECTION, transport: "offline" }, valid: false, rejectedBy: "offline-transport-rejected", gateStage: "controller-prelaunch" },
  { name: "multi-turn retain rejection", selection: { ...DEFAULT_SELECTION, turns: "multi", budget: "sessions", storage: "retain" }, valid: false, rejectedBy: "multiturn-retain-rejected", gateStage: "controller-prelaunch", includes: ["cellular-run-shape-validation"] },
  { name: "multi-turn random rejection", selection: { ...DEFAULT_SELECTION, turns: "multi", budget: "sessions", sampler: "random" }, valid: false, rejectedBy: "multiturn-random-rejected", gateStage: "controller-prelaunch", includes: ["cellular-run-shape-validation"], excludes: ["conversation-ownership"] },
  { name: "scheduled duration rejection", selection: { ...DEFAULT_SELECTION, budget: "duration" }, valid: false, rejectedBy: "scheduled-duration-rejected", gateStage: "controller-prelaunch" },
  { name: "scheduled adaptive rejection", selection: { ...DEFAULT_SELECTION, budget: "adaptive" }, valid: false, rejectedBy: "scheduled-adaptive-rejected", gateStage: "controller-prelaunch" },
  { name: "scheduled sessions retain rejection", selection: { ...DEFAULT_SELECTION, budget: "sessions", storage: "retain" }, valid: false, rejectedBy: "scheduled-sessions-retain-rejected", gateStage: "controller-prelaunch" },
  { name: "graph request rejection", selection: { ...DEFAULT_SELECTION, workload: "graph", dataset: "dag-jsonl" }, valid: false, rejectedBy: "graph-requests-rejected", excludes: ["partitioned-graph-source"], gateStage: "controller-prelaunch" },
  { name: "sketch records rejection", selection: { ...DEFAULT_SELECTION, storage: "sketch", artifacts: "records" }, valid: false, rejectedBy: "sketch-record-artifacts-rejected", gateStage: "cell-side", includes: ["local-cell-launch", "controller-child-arbitration", "synchronized-start", "canonical-dataset-regeneration", "sketch-artifact-validation"], excludes: ["terminal-record-finalization"], ordered: [["local-cell-launch", "controller-child-arbitration"], ["controller-child-arbitration", "synchronized-start"], ["synchronized-start", "canonical-dataset-regeneration"], ["canonical-dataset-regeneration", "sketch-artifact-validation"]] },
  { name: "sketch OTLP rejection", selection: { ...DEFAULT_SELECTION, storage: "sketch", artifacts: "otlp" }, valid: false, rejectedBy: "sketch-otlp-rejected", gateStage: "cell-side", includes: ["local-cell-launch", "controller-child-arbitration", "synchronized-start", "canonical-dataset-regeneration", "sketch-artifact-validation"], excludes: ["terminal-record-finalization"], ordered: [["local-cell-launch", "controller-child-arbitration"], ["controller-child-arbitration", "synchronized-start"], ["synchronized-start", "canonical-dataset-regeneration"], ["canonical-dataset-regeneration", "sketch-artifact-validation"]] },
  { name: "hierarchy request refusal", selection: { ...DEFAULT_SELECTION, topology: "local-tree", storage: "retain" }, valid: false, rejectedBy: "hierarchy-request-rejected", gateStage: "pre-controller", includes: ["hierarchical-tier-sizing"], excludes: ["retain-record-capture", "final-report-assembly"] },
];

const TARGETED_REJECTION_FIXTURES: readonly RouteFixture[] = [
  {
    name: "multi-turn exact-fold Parquet Velo becomes retain",
    selection: { ...DEFAULT_SELECTION, turns: "multi", budget: "sessions", artifacts: "parquet", build: "velo" },
    valid: false,
    rejectedBy: "multiturn-retain-rejected",
    gateStage: "controller-prelaunch",
    includes: ["cellular-run-shape-validation"],
    limitations: [PARQUET_RETAIN_LIMITATION, PARQUET_SKIPPED_LIMITATION],
    effective: { storage: "retain", artifacts: "omitted" },
  },
  {
    name: "tree exact-fold Parquet Velo is refused before startup",
    selection: { ...DEFAULT_SELECTION, topology: "local-tree", artifacts: "parquet", build: "velo" },
    valid: false,
    rejectedBy: "hierarchy-request-rejected",
    gateStage: "pre-controller",
    includes: ["hierarchical-tier-sizing"],
    excludes: ["retain-record-capture", "streaming-exact-fold", "scheduled-global-ordinal-merge", "exact-fold-store-merge", "final-report-assembly"],
    limitations: [PARQUET_RETAIN_LIMITATION, PARQUET_SKIPPED_LIMITATION],
    effective: { storage: "retain", artifacts: "omitted" },
  },
  {
    name: "cell-side sketch rejection follows fanout verification",
    selection: { ...DEFAULT_SELECTION, storage: "sketch", artifacts: "records", fanout: "verify" },
    valid: false,
    rejectedBy: "sketch-record-artifacts-rejected",
    gateStage: "cell-side",
    includes: ["synchronized-start", "controller-fanout-generation", "fanout-verification-overlay", "canonical-dataset-regeneration", "sketch-artifact-validation"],
    ordered: [["synchronized-start", "controller-fanout-generation"], ["controller-fanout-generation", "fanout-verification-overlay"], ["fanout-verification-overlay", "canonical-dataset-regeneration"], ["canonical-dataset-regeneration", "sketch-artifact-validation"]],
    effective: { storage: "sketch", artifacts: "records" },
  },
  {
    name: "cell-side sketch rejection follows Stage G reconstruction",
    selection: { ...DEFAULT_SELECTION, storage: "sketch", artifacts: "raw", dataset: "file", deployment: "cross-host" },
    valid: false,
    rejectedBy: "sketch-record-artifacts-rejected",
    gateStage: "cell-side",
    includes: ["synchronized-start", "dataset-serve-plan", "dataset-manifest-validation", "dataset-safe-path-mapping", "dataset-http-zstd-reconstruct", "sketch-artifact-validation"],
    ordered: [["synchronized-start", "dataset-serve-plan"], ["dataset-serve-plan", "dataset-manifest-validation"], ["dataset-manifest-validation", "dataset-safe-path-mapping"], ["dataset-safe-path-mapping", "dataset-http-zstd-reconstruct"], ["dataset-http-zstd-reconstruct", "sketch-artifact-validation"]],
    effective: { workload: "scheduled", storage: "sketch", artifacts: "raw" },
  },
];

const VALID_ROUTE_FIXTURES: readonly RouteFixture[] = [
  { name: "scheduled retain", selection: { ...DEFAULT_SELECTION, storage: "retain" }, valid: true, includes: ["retain-record-capture", "scheduled-global-ordinal-merge"], excludes: ["streaming-exact-fold", "sketch-tdigest-merge", "graph-concatenation-renumber"], environmentIncludes: ["AIPERF_RUNTIME_EXACT_FOLD=0"] },
  { name: "scheduled exact fold", selection: DEFAULT_SELECTION, valid: true, includes: ["streaming-exact-fold", "exact-fold-store-merge"], excludes: ["retain-record-capture", "sketch-tdigest-merge", "graph-concatenation-renumber"], environmentIncludes: ["AIPERF_RUNTIME_EXACT_FOLD=1"] },
  { name: "scheduled sketch", selection: { ...DEFAULT_SELECTION, storage: "sketch" }, valid: true, includes: ["sketch-scratch-harvest", "sketch-tdigest-merge"], excludes: ["retain-record-capture", "exact-fold-store-merge", "graph-concatenation-renumber"], environmentIncludes: ["AIPERF_METRICS_SKETCH=1"] },
  { name: "graph retain", selection: { ...DEFAULT_SELECTION, workload: "graph", dataset: "dag-jsonl", budget: "sessions", storage: "retain" }, valid: true, includes: ["partitioned-graph-source", "graph-concatenation-renumber"], excludes: ["partitioned-scheduled-sampler", "two-level-partition", "thread-phase-slicing", "issuance-dispatch-injection", "exact-fold-store-merge", "sketch-tdigest-merge"], environmentIncludes: ["AIPERF_RUNTIME_EXACT_FOLD=0"] },
  { name: "graph exact fold", selection: { ...DEFAULT_SELECTION, workload: "graph", dataset: "weka", budget: "duration" }, valid: true, includes: ["partitioned-graph-source", "exact-fold-store-merge"], excludes: ["partitioned-scheduled-sampler", "two-level-partition", "thread-phase-slicing", "issuance-dispatch-injection", "graph-concatenation-renumber", "sketch-tdigest-merge"] },
  { name: "graph sketch", selection: { ...DEFAULT_SELECTION, workload: "graph", dataset: "dynamo", budget: "adaptive", storage: "sketch" }, valid: true, includes: ["partitioned-graph-source", "sketch-tdigest-merge"], excludes: ["partitioned-scheduled-sampler", "two-level-partition", "thread-phase-slicing", "issuance-dispatch-injection", "graph-concatenation-renumber", "exact-fold-store-merge"] },
  { name: "dataset overrides scheduled selector to graph", selection: { ...DEFAULT_SELECTION, dataset: "weka", budget: "duration" }, valid: true, includes: ["partitioned-graph-source", "graph-budget-validation"], excludes: ["partitioned-scheduled-sampler", "scheduled-budget-validation"], limitations: [WORKLOAD_CLASSIFICATION_LIMITATION], effective: { workload: "graph" } },
  { name: "dataset overrides graph selector to scheduled", selection: { ...DEFAULT_SELECTION, workload: "graph" }, valid: true, includes: ["partitioned-scheduled-sampler", "scheduled-budget-validation"], excludes: ["partitioned-graph-source", "graph-budget-validation"], limitations: [WORKLOAD_CLASSIFICATION_LIMITATION], effective: { workload: "scheduled" } },
  { name: "graph OTLP admitted aggregate-only", selection: { ...DEFAULT_SELECTION, workload: "graph", dataset: "weka", budget: "sessions", artifacts: "otlp" }, valid: true, includes: ["partitioned-graph-source", "final-report-assembly"], excludes: ["artifact-http-zstd-upload"], limitations: [CELLULAR_PER_RECORD_OTLP_LIMITATION], effective: { workload: "graph", artifacts: "aggregate-otlp" } },
  { name: "scheduled CSV cross-host", selection: { ...DEFAULT_SELECTION, artifacts: "csv", deployment: "cross-host" }, valid: true, includes: ["shard-local-concatenation", "artifact-http-zstd-upload"], ordered: [["shard-local-concatenation", "artifact-http-zstd-upload"]], environmentIncludes: ["AIPERF_CELL_HTTP_ARTIFACT_SHIPPING=1"] },
  { name: "gRPC file records", selection: { ...DEFAULT_SELECTION, transport: "grpc", dataset: "file", artifacts: "records" }, valid: true, includes: ["canonical-dataset-regeneration", "controller-global-concatenation"] },
  { name: "public raw cross-host", selection: { ...DEFAULT_SELECTION, dataset: "public", artifacts: "raw", deployment: "cross-host" }, valid: true, includes: ["canonical-dataset-regeneration", "artifact-http-zstd-upload"] },
  { name: "scheduled Parquet full", selection: { ...DEFAULT_SELECTION, artifacts: "parquet", build: "full" }, valid: true, includes: ["shard-local-concatenation"], compileFeatures: ["dynosim", "parquet", "velo"] },
  { name: "scheduled Parquet skipped without feature", selection: { ...DEFAULT_SELECTION, artifacts: "parquet", build: "velo" }, valid: true, includes: ["retain-record-capture", "scheduled-global-ordinal-merge"], excludes: ["streaming-exact-fold", "exact-fold-store-merge", "shard-local-concatenation", "cell-local-concatenation", "controller-global-concatenation"], limitations: [PARQUET_RETAIN_LIMITATION, PARQUET_SKIPPED_LIMITATION], effective: { storage: "retain", artifacts: "omitted" } },
  { name: "sketch CSV accepted but omitted", selection: { ...DEFAULT_SELECTION, storage: "sketch", artifacts: "csv" }, valid: true, excludes: ["shard-local-concatenation", "cell-local-concatenation", "controller-global-concatenation"], limitations: [SKETCH_COLUMNAR_OMITTED_LIMITATION], effective: { artifacts: "omitted" } },
  { name: "sketch Parquet accepted but omitted", selection: { ...DEFAULT_SELECTION, storage: "sketch", artifacts: "parquet", build: "full" }, valid: true, excludes: ["shard-local-concatenation", "cell-local-concatenation", "controller-global-concatenation"], limitations: [SKETCH_COLUMNAR_OMITTED_LIMITATION], effective: { artifacts: "omitted" } },
  { name: "sketch Parquet skipped and rowless", selection: { ...DEFAULT_SELECTION, storage: "sketch", artifacts: "parquet", build: "velo" }, valid: true, excludes: ["shard-local-concatenation", "cell-local-concatenation", "controller-global-concatenation"], limitations: [PARQUET_SKIPPED_LIMITATION, SKETCH_COLUMNAR_OMITTED_LIMITATION], effective: { storage: "sketch", artifacts: "omitted" } },
  { name: "scheduled OTLP cross-host", selection: { ...DEFAULT_SELECTION, artifacts: "otlp", deployment: "cross-host" }, valid: true, includes: ["final-report-assembly"], excludes: ["artifact-http-zstd-upload"], limitations: [CELLULAR_PER_RECORD_OTLP_LIMITATION], environmentExcludes: ["AIPERF_CELL_HTTP_ARTIFACT_SHIPPING=1"] },
  { name: "local hierarchy request", selection: { ...DEFAULT_SELECTION, topology: "local-tree" }, valid: false, rejectedBy: "hierarchy-request-rejected", gateStage: "pre-controller", includes: ["hierarchical-tier-sizing"] },
  { name: "cross-host hierarchy request", selection: { ...DEFAULT_SELECTION, topology: "local-tree", deployment: "cross-host" }, valid: false, rejectedBy: "hierarchy-request-rejected", gateStage: "pre-controller", includes: ["hierarchical-tier-sizing"] },
  { name: "external hierarchy request", selection: { ...DEFAULT_SELECTION, topology: "external-tree" }, valid: false, rejectedBy: "hierarchy-request-rejected", gateStage: "pre-controller", includes: ["hierarchical-tier-sizing"] },
  { name: "external sketch hierarchy request", selection: { ...DEFAULT_SELECTION, topology: "external-tree", deployment: "cross-host", storage: "sketch", build: "full" }, valid: false, rejectedBy: "hierarchy-request-rejected", gateStage: "pre-controller", includes: ["hierarchical-tier-sizing"] },
  { name: "scheduled multi-turn shuffle", selection: { ...DEFAULT_SELECTION, turns: "multi", sampler: "shuffle", budget: "sessions" }, valid: true, includes: ["conversation-ownership"] },
  { name: "graph selector turns are not linear gate", selection: { ...DEFAULT_SELECTION, workload: "graph", dataset: "dag-jsonl", budget: "sessions", turns: "multi", sampler: "random", storage: "retain" }, valid: true, includes: ["partitioned-graph-source"] },
  { name: "phaser start", selection: { ...DEFAULT_SELECTION, start: "phaser" }, valid: true, includes: ["phaser-start"] },
  { name: "barrier-free start", selection: { ...DEFAULT_SELECTION, start: "barrier-free", build: "full" }, valid: true, includes: ["barrier-free-launch"] },
  { name: "Stage G summary-only file transfer", selection: { ...DEFAULT_SELECTION, dataset: "file", deployment: "cross-host" }, valid: true, includes: ["dataset-serve-plan", "dataset-http-zstd-reconstruct"], excludes: ["artifact-http-zstd-upload"], environmentIncludes: ["AIPERF_CELL_HTTP_ARTIFACT_SHIPPING=1"] },
  { name: "cross-host graph transfer", selection: { ...DEFAULT_SELECTION, workload: "graph", dataset: "dynamo", budget: "sessions", deployment: "cross-host" }, valid: true, includes: ["recorded-graph-file-enumeration", "dataset-http-zstd-reconstruct"], excludes: ["controller-fanout-generation"], ordered: [["dataset-serve-plan", "dataset-http-zstd-reconstruct"]], environmentIncludes: ["AIPERF_CELL_HTTP_ARTIFACT_SHIPPING=1"] },
  { name: "fanout off measured path", selection: DEFAULT_SELECTION, valid: true, includes: ["canonical-dataset-regeneration"], excludes: ["controller-fanout-generation", "dataset-velo-subscribe", "fanout-verification-overlay"], limitations: [] },
  { name: "fanout scheduled HTTP", selection: { ...DEFAULT_SELECTION, fanout: "verify" }, valid: true, includes: ["controller-fanout-generation", "dataset-velo-replay-live", "owned-index-build", "dispatch-on-issue", "dispatch-on-complete", "distribution-miss", "fanout-verification-overlay", "canonical-dataset-regeneration"], ordered: [["fanout-verification-overlay", "canonical-dataset-regeneration"]], limitations: [], environmentIncludes: ["AIPERF_CELL_DATASET_FANOUT=1"] },
  { name: "fanout graph sessions HTTP", selection: { ...DEFAULT_SELECTION, workload: "graph", dataset: "dag-jsonl", budget: "sessions", fanout: "verify" }, valid: true, includes: ["controller-fanout-generation", "fanout-verification-overlay", "partitioned-graph-source"], limitations: [] },
  { name: "fanout graph duration boundary", selection: { ...DEFAULT_SELECTION, workload: "graph", dataset: "weka", budget: "duration", fanout: "verify" }, valid: true, includes: ["controller-fanout-generation", "fanout-verification-overlay"], limitations: [FANOUT_GRAPH_ZERO_REQUESTS_LIMITATION] },
  { name: "fanout graph adaptive boundary", selection: { ...DEFAULT_SELECTION, workload: "graph", dataset: "dynamo", budget: "adaptive", fanout: "verify" }, valid: true, includes: ["controller-fanout-generation", "fanout-verification-overlay"], limitations: [FANOUT_GRAPH_ZERO_REQUESTS_LIMITATION] },
  { name: "fanout gRPC HTTP-probe boundary", selection: { ...DEFAULT_SELECTION, fanout: "verify", transport: "grpc" }, valid: true, includes: ["controller-fanout-generation", "fanout-verification-overlay"], limitations: [FANOUT_GRPC_HTTP_PROBE_LIMITATION] },
  { name: "fanout cross-host Stage G order", selection: { ...DEFAULT_SELECTION, fanout: "verify", dataset: "file", deployment: "cross-host" }, valid: true, includes: ["dataset-serve-plan", "controller-fanout-generation", "fanout-verification-overlay", "dataset-manifest-validation", "dataset-safe-path-mapping", "dataset-http-zstd-reconstruct"], ordered: [["dataset-serve-plan", "controller-fanout-generation"], ["controller-fanout-generation", "fanout-verification-overlay"], ["fanout-verification-overlay", "dataset-manifest-validation"], ["dataset-manifest-validation", "dataset-safe-path-mapping"], ["dataset-safe-path-mapping", "dataset-http-zstd-reconstruct"]], limitations: [], environmentIncludes: ["AIPERF_CELL_HTTP_ARTIFACT_SHIPPING=1"] },
  { name: "fanout verification full", selection: { ...DEFAULT_SELECTION, fanout: "verify", build: "full" }, valid: true, includes: ["owned-index-build", "distribution-miss"], compileFeatures: ["dynosim", "parquet", "velo"], limitations: [] },
];

const ROUTE_FIXTURES = [
  ...GATE_FIXTURES,
  ...TARGETED_REJECTION_FIXTURES,
  ...VALID_ROUTE_FIXTURES,
] as const;

const ROUTE_RECIPES: readonly RouteRecipe[] = [
  {
    id: "scheduled-retain-raw",
    title: "Scheduled retain · raw exact order",
    selection: { ...DEFAULT_SELECTION, storage: "retain", artifacts: "raw" },
    kind: "canonical",
  },
  {
    id: "scheduled-exact-fold",
    title: "Scheduled exact fold · bounded",
    selection: DEFAULT_SELECTION,
    kind: "canonical",
  },
  {
    id: "scheduled-sketch",
    title: "Scheduled sketch · bounded memory",
    selection: { ...DEFAULT_SELECTION, storage: "sketch" },
    kind: "canonical",
  },
  {
    id: "scheduled-multiturn",
    title: "Scheduled multi-turn · exact fold",
    selection: {
      ...DEFAULT_SELECTION,
      turns: "multi",
      sampler: "shuffle",
      budget: "sessions",
    },
    kind: "canonical",
  },
  {
    id: "graph-retain",
    title: "Graph retain · concatenate",
    selection: {
      ...DEFAULT_SELECTION,
      workload: "graph",
      dataset: "dag-jsonl",
      budget: "sessions",
      storage: "retain",
    },
    kind: "canonical",
  },
  {
    id: "graph-exact-fold",
    title: "Graph exact fold",
    selection: {
      ...DEFAULT_SELECTION,
      workload: "graph",
      dataset: "weka",
      budget: "duration",
    },
    kind: "canonical",
  },
  {
    id: "graph-sketch-dynamo",
    title: "Graph sketch · Dynamo trace",
    selection: {
      ...DEFAULT_SELECTION,
      workload: "graph",
      dataset: "dynamo",
      budget: "adaptive",
      storage: "sketch",
    },
    kind: "canonical",
  },
  {
    id: "public-dataset",
    title: "Scheduled public dataset",
    selection: { ...DEFAULT_SELECTION, dataset: "public" },
    kind: "canonical",
  },
  {
    id: "nested-thread-per-core",
    title: "Nested thread-per-core",
    selection: { ...DEFAULT_SELECTION, transport: "grpc" },
    kind: "canonical",
  },
  {
    id: "phaser-start",
    title: "Phaser START",
    selection: { ...DEFAULT_SELECTION, start: "phaser" },
    kind: "canonical",
  },
  {
    id: "barrier-free",
    title: "Barrier-free launch",
    selection: { ...DEFAULT_SELECTION, start: "barrier-free" },
    kind: "canonical",
  },
  {
    id: "stage-g-cross-host",
    title: "Stage G · cross-host dataset",
    selection: { ...DEFAULT_SELECTION, dataset: "file", deployment: "cross-host" },
    kind: "canonical",
  },
  {
    id: "fanout-verification",
    title: "Fan-out verification overlay",
    selection: { ...DEFAULT_SELECTION, fanout: "verify" },
    kind: "canonical",
  },
  {
    id: "local-tree",
    title: "Rejected hierarchy request",
    selection: { ...DEFAULT_SELECTION, topology: "local-tree" },
    kind: "rejected",
  },
  {
    id: "external-tree",
    title: "Rejected external hierarchy request",
    selection: {
      ...DEFAULT_SELECTION,
      topology: "external-tree",
      deployment: "cross-host",
    },
    kind: "rejected",
  },
  {
    id: "stage-e-artifact",
    title: "Stage E · CSV artifact upload",
    selection: {
      ...DEFAULT_SELECTION,
      dataset: "file",
      deployment: "cross-host",
      artifacts: "csv",
    },
    kind: "canonical",
  },
  {
    id: "dataset-classification-override",
    title: "Scheduled intent · graph dataset",
    selection: { ...DEFAULT_SELECTION, dataset: "weka", budget: "duration" },
    kind: "canonical",
  },
  {
    id: "graph-otlp-aggregate-only",
    title: "Graph OTLP · aggregate only",
    selection: {
      ...DEFAULT_SELECTION,
      workload: "graph",
      dataset: "weka",
      budget: "sessions",
      artifacts: "otlp",
    },
    kind: "canonical",
  },
  {
    id: "parquet-warn-skip",
    title: "Parquet · Velo build skip",
    selection: { ...DEFAULT_SELECTION, artifacts: "parquet", build: "velo" },
    kind: "canonical",
  },
  {
    id: "sketch-csv-omitted",
    title: "Sketch CSV · accepted, omitted",
    selection: { ...DEFAULT_SELECTION, storage: "sketch", artifacts: "csv" },
    kind: "canonical",
  },
  {
    id: "cross-host-hierarchy-refusal",
    title: "Cross-host hierarchy refusal",
    selection: { ...DEFAULT_SELECTION, topology: "local-tree", deployment: "cross-host" },
    kind: "rejected",
  },
  ...GATE_FIXTURES.map((fixture): RouteRecipe => ({
    id: `rejected-${fixture.rejectedBy}`,
    title: `Rejected · ${fixture.rejectedBy}`,
    selection: fixture.selection,
    kind: "rejected",
  })),
];

const DECISIONS: readonly DecisionCell[] = [
  {
    id: "budget-shape",
    title: "Dataset-classified run kind",
    leftLabel: "Graph intent · synthetic",
    left: { ...DEFAULT_SELECTION, workload: "graph" },
    rightLabel: "Scheduled intent · WEKA",
    right: {
      ...DEFAULT_SELECTION,
      dataset: "weka",
      budget: "duration",
    },
    invariant: "Dataset shape, not the authored workload selector, determines the effective run kind.",
  },
  {
    id: "storage-fidelity",
    title: "Exact fold or sketch",
    leftLabel: "Exact fold",
    left: DEFAULT_SELECTION,
    rightLabel: "Sketch",
    right: { ...DEFAULT_SELECTION, storage: "sketch" },
    invariant: "Both drop completed rows; only sketch trades interior distribution fidelity for request-count-independent memory.",
  },
  {
    id: "conversation-ownership",
    title: "Single-turn or multi-turn ownership",
    leftLabel: "Single-turn requests",
    left: DEFAULT_SELECTION,
    rightLabel: "Multi-turn sessions",
    right: {
      ...DEFAULT_SELECTION,
      turns: "multi",
      sampler: "shuffle",
      budget: "sessions",
    },
    invariant: "A conversation remains wholly owned by one cell; turns never split across cells.",
  },
  {
    id: "retain-merge",
    title: "Global order or graph concatenation",
    leftLabel: "Scheduled retain",
    left: { ...DEFAULT_SELECTION, storage: "retain", artifacts: "raw" },
    rightLabel: "Graph retain",
    right: {
      ...DEFAULT_SELECTION,
      workload: "graph",
      dataset: "dag-jsonl",
      budget: "sessions",
      storage: "retain",
      artifacts: "raw",
    },
    invariant: "Retained records preserve payload fidelity, while workload identity selects the controller merge law.",
  },
  {
    id: "aggregation-topology",
    title: "Hierarchy requests are refused",
    leftLabel: "Local hierarchy request",
    left: { ...DEFAULT_SELECTION, topology: "local-tree", deployment: "cross-host" },
    rightLabel: "External hierarchy request",
    right: { ...DEFAULT_SELECTION, topology: "external-tree", deployment: "cross-host" },
    invariant: "No hierarchy topology is executable: every hierarchy request is refused before controller startup.",
  },
  {
    id: "dataset-verification",
    title: "Canonical data or verification overlay",
    leftLabel: "Canonical regeneration",
    left: DEFAULT_SELECTION,
    rightLabel: "Fan-out verification",
    right: { ...DEFAULT_SELECTION, fanout: "verify" },
    invariant: "Fan-out verifies ownership and delivery but never replaces the measured execution dataset path.",
  },
  {
    id: "host-shipping",
    title: "Shared host or HTTP + zstd",
    leftLabel: "Same-host file",
    left: { ...DEFAULT_SELECTION, dataset: "file", artifacts: "csv" },
    rightLabel: "Cross-host file",
    right: {
      ...DEFAULT_SELECTION,
      dataset: "file",
      deployment: "cross-host",
      artifacts: "csv",
    },
    invariant: "Cross-host placement adds Stage G input reconstruction and Stage E artifact upload around the same execution route.",
  },
  {
    id: "feature-admission",
    title: "Parquet emitted or skipped",
    leftLabel: "Full build · emitted",
    left: { ...DEFAULT_SELECTION, artifacts: "parquet", build: "full" },
    rightLabel: "Velo build · warned/skipped",
    right: { ...DEFAULT_SELECTION, artifacts: "parquet", build: "velo" },
    invariant: "The wire request is accepted on both builds; only a parquet-bearing build emits and concatenates the file.",
  },
];

function pairwiseSelections(): readonly SelectorState[] {
  const keys = Object.keys(SELECTOR_OPTIONS) as SelectorKey[];
  const selections: SelectorState[] = [];
  for (let leftIndex = 0; leftIndex < keys.length; leftIndex += 1) {
    for (let rightIndex = leftIndex + 1; rightIndex < keys.length; rightIndex += 1) {
      const left = keys[leftIndex];
      const right = keys[rightIndex];
      for (const leftValue of SELECTOR_OPTIONS[left]) {
        for (const rightValue of SELECTOR_OPTIONS[right]) {
          selections.push({
            ...DEFAULT_SELECTION,
            [left]: leftValue,
            [right]: rightValue,
          } as SelectorState);
        }
      }
    }
  }
  return selections;
}

function finiteInteractionSelections(): readonly SelectorState[] {
  return [
    ...pairwiseSelections(),
    {
      ...DEFAULT_SELECTION,
      workload: "graph",
      dataset: "weka",
      budget: "duration",
      fanout: "verify",
    },
    {
      ...DEFAULT_SELECTION,
      transport: "grpc",
      fanout: "verify",
    },
  ];
}

function validateComposerCoverage(): readonly string[] {
  const errors: string[] = [];
  const hasTerminal = (result: Extract<RouteResult, { valid: true }>) =>
    result.algorithmIds.includes("final-report-assembly") ||
    ARTIFACT_TERMINAL_IDS.some((id) => result.algorithmIds.includes(id));
  const validateEffectiveRoute = (
    selection: SelectorState,
    result: RouteResult,
    label: string,
  ) => {
    const expected = effectiveSettings(selection);
    const assertOrderedStops = (ids: readonly string[], name: string) => {
      const indices = ids.map((id) => result.algorithmIds.indexOf(id));
      if (
        indices.some((index) => index < 0) ||
        indices.some((index, position) => position > 0 && index <= indices[position - 1])
      ) {
        errors.push(`${name}: ${label}`);
      }
    };
    const launchId = selection.deployment === "cross-host"
      ? "external-cell-launch"
      : "local-cell-launch";
    const startId = selection.start === "phaser"
      ? "phaser-start"
      : selection.start === "barrier-free"
        ? "barrier-free-launch"
        : "synchronized-start";
    for (const key of Object.keys(expected) as (keyof EffectiveSettings)[]) {
      if (result.effective[key] !== expected[key]) {
        errors.push(`effective ${key} mismatch: ${label}`);
      }
    }
    if (
      selection.storage !== expected.storage &&
      !result.limitations.some((item) =>
        item === PARQUET_RETAIN_LIMITATION || item === ADAPTIVE_RETAIN_LIMITATION)
    ) {
      errors.push(`storage override lacks limitation: ${label}`);
    }
    if (!result.valid) {
      const gate = VALIDATION_GATES.find((item) => item.id === result.rejectedBy);
      if (!gate || gate.stage !== result.gateStage) {
        errors.push(`rejected route gate stage mismatch: ${label}`);
      }
      if (gate && result.algorithmIds.at(-1) !== gate.algorithmId) {
        errors.push(`rejected route does not stop at gate algorithm: ${label}`);
      }
      if (result.gateStage === "pre-controller") {
        assertOrderedStops(
          ["cell-count-resolution", "controller-promotion", "velo-feature-admission"],
          "pre-controller route order mismatch",
        );
      }
      if (
        result.gateStage === "cell-side" ||
        result.gateStage === "aggregator-receive"
      ) {
        assertOrderedStops(
          [launchId, "controller-child-arbitration", startId],
          "control route order mismatch",
        );
      }
      if (result.gateStage === "cell-side" && gate) {
        assertOrderedStops(
          [startId, ...routeFragments(selection).distribution, gate.algorithmId],
          "cell-side distribution route order mismatch",
        );
      }
      return;
    }
    assertOrderedStops(
      [launchId, "controller-child-arbitration", startId],
      "control route order mismatch",
    );
    const scheduledOnly = [
      "partitioned-scheduled-sampler",
      "two-level-partition",
      "thread-phase-slicing",
      "issuance-dispatch-injection",
    ];
    if (
      expected.workload === "graph" &&
      scheduledOnly.some((id) => result.algorithmIds.includes(id))
    ) {
      errors.push(`effective graph route contains scheduled-only stop: ${label}`);
    }
    if (
      expected.workload === "scheduled" &&
      result.algorithmIds.includes("partitioned-graph-source")
    ) {
      errors.push(`effective scheduled route contains graph source: ${label}`);
    }
    if (
      expected.artifacts === "omitted" &&
      ["shard-local-concatenation", "cell-local-concatenation", "controller-global-concatenation"].some(
        (id) => result.algorithmIds.includes(id),
      )
    ) {
      errors.push(`omitted artifact route contains concat stop: ${label}`);
    }
    if (
      expected.storage === "retain" &&
      (!result.algorithmIds.includes("retain-record-capture") ||
        result.algorithmIds.includes("streaming-exact-fold"))
    ) {
      errors.push(`effective retain route has wrong capture path: ${label}`);
    }
    if (
      expected.storage === "exact-fold" &&
      (!result.algorithmIds.includes("streaming-exact-fold") ||
        result.algorithmIds.includes("retain-record-capture"))
    ) {
      errors.push(`effective exact-fold route has wrong capture path: ${label}`);
    }
    const storageMergeId = expected.storage === "retain"
      ? expected.workload === "graph"
        ? "graph-concatenation-renumber"
        : "scheduled-global-ordinal-merge"
      : expected.storage === "exact-fold"
        ? "exact-fold-store-merge"
        : "sketch-tdigest-merge";
    if (
      result.algorithmIds.indexOf("controller-partition-collection") >
      result.algorithmIds.indexOf(storageMergeId)
    ) {
      errors.push(`storage merge precedes partition collection: ${label}`);
    }
    if (
      selection.workload !== expected.workload &&
      !result.limitations.includes(WORKLOAD_CLASSIFICATION_LIMITATION)
    ) {
      errors.push(`workload override lacks limitation: ${label}`);
    }
    if (
      selection.topology !== expected.topology &&
      !result.limitations.some((item) =>
        item === CROSS_HOST_HIERARCHY_REFUSAL || item === SAME_HOST_HIERARCHY_REFUSAL)
    ) {
      errors.push(`topology override lacks limitation: ${label}`);
    }
    if (
      expected.artifacts === "omitted" &&
      !result.limitations.some((item) =>
        item === PARQUET_SKIPPED_LIMITATION || item === SKETCH_COLUMNAR_OMITTED_LIMITATION)
    ) {
      errors.push(`artifact omission lacks limitation: ${label}`);
    }
  };
  for (const fixture of ROUTE_FIXTURES) {
    const result = deriveRoute(fixture.selection);
    if (result.valid !== fixture.valid) errors.push(`route fixture validity: ${fixture.name}`);
    for (const algorithmId of fixture.includes ?? []) {
      if (!result.algorithmIds.includes(algorithmId)) {
        errors.push(`route fixture missing ${algorithmId}: ${fixture.name}`);
      }
    }
    for (const algorithmId of fixture.excludes ?? []) {
      if (result.algorithmIds.includes(algorithmId)) {
        errors.push(`route fixture unexpectedly includes ${algorithmId}: ${fixture.name}`);
      }
    }
    for (const [before, after] of fixture.ordered ?? []) {
      const beforeIndex = result.algorithmIds.indexOf(before);
      const afterIndex = result.algorithmIds.indexOf(after);
      if (beforeIndex < 0 || afterIndex < 0 || beforeIndex >= afterIndex) {
        errors.push(`route fixture order ${before} -> ${after}: ${fixture.name}`);
      }
    }
    if (fixture.rejectedBy && (result.valid || result.rejectedBy !== fixture.rejectedBy)) {
      errors.push(`route fixture rejection: ${fixture.name}`);
    }
    if (result.valid && !hasTerminal(result)) {
      errors.push(`valid route lacks report or artifact terminal: ${fixture.name}`);
    }
    validateEffectiveRoute(fixture.selection, result, `fixture ${fixture.name}`);
    if (!result.valid && !result.rejectedBy) {
      errors.push(`invalid route lacks rejectedBy: ${fixture.name}`);
    }
    if (
      fixture.gateStage !== undefined &&
      (result.valid || result.gateStage !== fixture.gateStage)
    ) {
      errors.push(`route fixture gate stage: ${fixture.name}`);
    }
    if (fixture.limitations !== undefined) {
      if (
        result.limitations.length !== fixture.limitations.length ||
        fixture.limitations.some((limitation) => !result.limitations.includes(limitation))
      ) {
        errors.push(`route fixture limitations: ${fixture.name}`);
      }
    }
    for (const [key, value] of Object.entries(fixture.effective ?? {}) as [
      keyof EffectiveSettings,
      EffectiveSettings[keyof EffectiveSettings],
    ][]) {
      if (result.effective[key] !== value) {
        errors.push(`route fixture effective ${key}: ${fixture.name}`);
      }
    }
    if (result.valid) {
      for (const value of fixture.environmentIncludes ?? []) {
        if (!result.environment.includes(value)) {
          errors.push(`route fixture missing environment ${value}: ${fixture.name}`);
        }
      }
      for (const value of fixture.environmentExcludes ?? []) {
        if (result.environment.includes(value)) {
          errors.push(`route fixture unexpectedly includes environment ${value}: ${fixture.name}`);
        }
      }
      if (
        fixture.compileFeatures !== undefined &&
        (result.compileFeatures.length !== fixture.compileFeatures.length ||
          fixture.compileFeatures.some((feature) => !result.compileFeatures.includes(feature)))
      ) {
        errors.push(`route fixture compile features: ${fixture.name}`);
      }
    }
  }
  for (const gate of VALIDATION_GATES) {
    if (!GATE_FIXTURES.some((fixture) => fixture.rejectedBy === gate.id)) {
      errors.push(`validation gate lacks an explicit fixture: ${gate.id}`);
    }
  }
  const recipeIds = new Set<string>();
  for (const recipe of ROUTE_RECIPES) {
    if (recipeIds.has(recipe.id)) errors.push(`duplicate route recipe: ${recipe.id}`);
    recipeIds.add(recipe.id);
    const normalized = normalizeSelection(recipe.selection);
    if (JSON.stringify(normalized) !== JSON.stringify(recipe.selection)) {
      errors.push(`route recipe has an invalid selector: ${recipe.id}`);
    }
    for (const key of Object.keys(SELECTOR_OPTIONS) as SelectorKey[]) {
      if (!legalSelectorValue(key, recipe.selection[key])) {
        errors.push(`route recipe has illegal selector value: ${recipe.id}/${key}`);
      }
    }
    const result = cachedRoute(recipe.selection);
    if (recipe.kind === "canonical" && !result.valid) {
      errors.push(`canonical route recipe rejects: ${recipe.id}`);
    }
    if (recipe.kind === "rejected" && result.valid) {
      errors.push(`rejected route recipe passes: ${recipe.id}`);
    }
    if (!result.valid && !result.rejectedBy) {
      errors.push(`recipe invalid route lacks rejectedBy: ${recipe.id}`);
    }
    validateEffectiveRoute(recipe.selection, result, `recipe ${recipe.id}`);
  }
  const stageGRecipe = ROUTE_RECIPES.find((recipe) => recipe.id === "stage-g-cross-host");
  const stageGRoute = stageGRecipe ? cachedRoute(stageGRecipe.selection) : undefined;
  if (
    !stageGRoute ||
    !stageGRoute.valid ||
    !stageGRoute.algorithmIds.includes("dataset-http-zstd-reconstruct") ||
    !stageGRoute.environment.includes("AIPERF_CELL_HTTP_ARTIFACT_SHIPPING=1")
  ) {
    errors.push("Stage G recipe lacks its derived HTTP shipping route");
  }
  for (const gate of VALIDATION_GATES) {
    if (
      !ROUTE_RECIPES.some((recipe) => {
        const result = cachedRoute(recipe.selection);
        return recipe.kind === "rejected" && !result.valid && result.rejectedBy === gate.id;
      })
    ) {
      errors.push(`validation gate lacks a rejected recipe: ${gate.id}`);
    }
  }
  const decisionIds = new Set<string>();
  for (const decision of DECISIONS) {
    if (decisionIds.has(decision.id)) errors.push(`duplicate decision: ${decision.id}`);
    decisionIds.add(decision.id);
    for (const [side, selection] of [["left", decision.left], ["right", decision.right]] as const) {
      const result = cachedRoute(selection);
      if (result.algorithmIds.length === 0) {
        errors.push(`decision route is empty: ${decision.id}/${side}`);
      }
      if (!result.valid && !VALIDATION_GATES.some((gate) => gate.id === result.rejectedBy)) {
        errors.push(`decision route has unknown rejection: ${decision.id}/${side}`);
      }
      if (!result.valid && !result.rejectedBy) {
        errors.push(`decision invalid route lacks rejectedBy: ${decision.id}/${side}`);
      }
      validateEffectiveRoute(selection, result, `decision ${decision.id}/${side}`);
    }
    const leftRoute = cachedRoute(decision.left);
    const rightRoute = cachedRoute(decision.right);
    const bands = collapseSharedRoute(leftRoute.algorithmIds, rightRoute.algorithmIds);
    if (bands.leftDelta.some((id) => bands.rightDelta.includes(id))) {
      errors.push(`decision distinct algorithms overlap: ${decision.id}`);
    }
    if (
      decision.id === "storage-fidelity" &&
      ["ingested-count-preservation", "partition-messagepack-encode"].some(
        (id) => bands.leftDelta.includes(id) || bands.rightDelta.includes(id),
      )
    ) {
      errors.push("storage decision labels shared wire algorithms as distinct");
    }
  }
  const coveredValues = new Set<string>();
  const explicitSelections = [
    ...ROUTE_RECIPES.map((recipe) => recipe.selection),
    ...DECISIONS.flatMap((decision) => [decision.left, decision.right]),
  ];
  for (const selection of explicitSelections) {
    for (const key of Object.keys(SELECTOR_OPTIONS) as SelectorKey[]) {
      coveredValues.add(`${key}=${selection[key]}`);
    }
  }
  for (const key of Object.keys(SELECTOR_OPTIONS) as SelectorKey[]) {
    for (const value of SELECTOR_OPTIONS[key]) {
      if (!coveredValues.has(`${key}=${String(value)}`)) {
        errors.push(`selector value lacks a recipe or decision: ${key}=${String(value)}`);
      }
    }
  }
  const interactions = finiteInteractionSelections();
  if (interactions.length !== 805) {
    errors.push(`finite interaction count is ${interactions.length}, expected 805`);
  }
  for (const [index, selection] of interactions.entries()) {
    const result = deriveRoute(selection);
    if (result.algorithmIds.length === 0) errors.push(`interaction route is empty: ${index}`);
    if (!result.valid && !VALIDATION_GATES.some((gate) => gate.id === result.rejectedBy)) {
      errors.push(`interaction route has unknown rejection: ${index}`);
    }
    if (!result.valid && !result.rejectedBy) {
      errors.push(`interaction invalid route lacks rejectedBy: ${index}`);
    }
    if (result.valid && !hasTerminal(result)) {
      errors.push(`interaction valid route lacks report or artifact terminal: ${index}`);
    }
    validateEffectiveRoute(selection, result, `interaction ${index}`);
  }
  const normalizedLegacy = normalizeSelection({ artifacts: "columnar", workload: "bogus" });
  if (normalizedLegacy.artifacts !== "parquet" || normalizedLegacy.workload !== DEFAULT_SELECTION.workload) {
    errors.push("persisted selector normalization failed");
  }
  return errors;
}

const MODE_OPTIONS: readonly { id: Mode; label: string }[] = [
  { id: "workbook", label: "Workbook" },
  { id: "compose", label: "Compose route" },
  { id: "decisions", label: "Decision log" },
];

function normalizeMode(value: unknown): Mode {
  return typeof value === "string" && MODE_OPTIONS.some((item) => item.id === value)
    ? value as Mode
    : "workbook";
}

function normalizeChapter(value: unknown): ChapterFilter {
  return value === "all" ||
    typeof value === "string" && CHAPTERS.some((chapter) => chapter.id === value)
    ? value as ChapterFilter
    : "all";
}

function normalizeAlgorithmId(value: unknown): string {
  return typeof value === "string" && ALGORITHMS.some((algorithm) => algorithm.id === value)
    ? value
    : ALGORITHMS[0].id;
}

function normalizeQuery(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function normalizeReducedMotion(value: unknown): boolean {
  return typeof value === "boolean" ? value : false;
}

function ModeTabs({
  mode,
  onChange,
}: {
  mode: Mode;
  onChange: (mode: Mode) => void;
}) {
  return (
    <Row gap={6} wrap>
      {MODE_OPTIONS.map((item) => (
        <span key={item.id}>
          <Pill active={mode === item.id} onClick={() => onChange(item.id)}>
            {item.label}
          </Pill>
        </span>
      ))}
    </Row>
  );
}

const STATUS_LABELS: Readonly<Record<Status, string>> = {
  built: "Built",
  partial: "Partial",
  "feature-gated": "Feature gated",
  approximate: "Approximate",
  rejected: "Rejected",
};

function StatusLabel({ status }: { status: Status }) {
  const theme = useHostTheme();
  const rejected = status === "rejected";
  return (
    <span
      className={`caw-status caw-status-${status}`}
      aria-label={`Implementation status: ${STATUS_LABELS[status]}`}
      style={{
        color: rejected ? theme.diff.stripRemoved : theme.text.secondary,
        background: rejected ? theme.diff.removedLine : theme.fill.tertiary,
        borderColor: rejected ? theme.diff.stripRemoved : theme.stroke.secondary,
      }}
    >
      {STATUS_LABELS[status]}
    </span>
  );
}

function AdmissionLabel({ valid }: { valid: boolean }) {
  const theme = useHostTheme();
  return (
    <span
      className="caw-status"
      aria-label={`Route admission: ${valid ? "Admitted" : "Rejected"}`}
      style={{
        color: valid ? theme.text.secondary : theme.diff.stripRemoved,
        background: valid ? theme.fill.tertiary : theme.diff.removedLine,
        borderColor: valid ? theme.stroke.secondary : theme.diff.stripRemoved,
        borderStyle: valid ? "solid" : "double",
      }}
    >
      {valid ? "Admitted" : "Rejected"}
    </span>
  );
}

function AlgorithmCard({ algorithm }: { algorithm: AlgorithmDefinition }) {
  const theme = useHostTheme();
  const dispatch = useCanvasAction();
  const openSource = () => dispatch({ type: "openFile", path: algorithm.source.path });

  return (
    <Card size="lg">
      <CardHeader
        trailing={
          <Row gap={6} wrap>
            <StatusLabel status={algorithm.status} />
            <Pill size="sm">{algorithm.id}</Pill>
          </Row>
        }
      >
        {algorithm.title}
      </CardHeader>
      <CardBody>
        <Stack gap={14}>
          <Text tone="secondary">{algorithm.summary}</Text>
          <Button variant="secondary" onClick={openSource}>
            {algorithm.source.path}:{algorithm.source.startLine}-{algorithm.source.endLine} ·{" "}
            {algorithm.source.symbol}
          </Button>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: 12,
            }}
          >
            {[
              ["Inputs", algorithm.inputs],
              ["Outputs", algorithm.outputs],
              ["State", algorithm.state.length > 0 ? algorithm.state : ["No mutable state"]],
              ["Gates", algorithm.gates],
              ["Failures", algorithm.failures],
              ["Invariants", algorithm.invariants],
            ].map(([label, values]) => (
              <div key={label as string}>
                <Stack gap={4}>
                  <Text size="small" weight="semibold">
                    {label as string}
                  </Text>
                  {(values as readonly string[]).map((value) => (
                    <div key={value}>
                      <Text size="small" tone="secondary">
                        {value}
                      </Text>
                    </div>
                  ))}
                </Stack>
              </div>
            ))}
          </div>
          <Row gap={8} wrap>
            <Pill size="sm">Time · {algorithm.complexity.time}</Pill>
            <Pill size="sm">Memory · {algorithm.complexity.memory}</Pill>
            {algorithm.routeTags.map((tag) => (
              <span key={tag}>
                <Pill size="sm">{tag}</Pill>
              </span>
            ))}
          </Row>
          <CollapsibleSection title="Pseudocode" defaultOpen>
            <Stack gap={4}>
              {algorithm.pseudocode.map((line) => (
                <div
                  key={line.id}
                  style={{
                    padding: "6px 8px",
                    borderLeft: `2px solid ${theme.stroke.secondary}`,
                    background: theme.bg.editor,
                  }}
                >
                  <Code>
                    {line.id} · {line.text}
                  </Code>
                </div>
              ))}
            </Stack>
          </CollapsibleSection>
          <CollapsibleSection title={`Trace frames · ${algorithm.frames.length}`}>
            <Stack gap={10}>
              {algorithm.frames.map((frame) => (
                <div
                  key={frame.id}
                  style={{
                    padding: 10,
                    border: `1px solid ${theme.stroke.tertiary}`,
                    background: theme.fill.tertiary,
                  }}
                >
                  <Stack gap={5}>
                    <Row gap={6} wrap>
                      <Text size="small" weight="semibold">
                        {frame.label}
                      </Text>
                      <Pill size="sm">{frame.activeLineId}</Pill>
                      {frame.activeActors.map((actor) => (
                        <span key={actor}>
                          <Pill size="sm">{actor}</Pill>
                        </span>
                      ))}
                    </Row>
                    <Text size="small" tone="secondary">
                      Before: {frame.before.join(" · ")}
                    </Text>
                    <Text size="small" tone="secondary">
                      After: {frame.after.join(" · ")}
                    </Text>
                    <Text size="small">
                      Check: {frame.invariantChecks.join(" · ")}
                    </Text>
                  </Stack>
                </div>
              ))}
            </Stack>
          </CollapsibleSection>
          <CollapsibleSection title={`Evidence · ${algorithm.evidence.length}`}>
            <Stack gap={6}>
              {algorithm.evidence.map((evidence) => (
                <div key={`${evidence.path}/${evidence.symbol}`}>
                  <Button
                    variant="secondary"
                    onClick={() => dispatch({ type: "openFile", path: evidence.path })}
                  >
                    {evidence.kind} · {evidence.path} · {evidence.symbol}
                  </Button>
                </div>
              ))}
            </Stack>
          </CollapsibleSection>
          <Text size="small" tone="tertiary">
            Predecessors: {algorithm.predecessors.join(", ") || "entry"} · Successors:{" "}
            {algorithm.successors.join(", ") || "terminal"}
          </Text>
        </Stack>
      </CardBody>
    </Card>
  );
}

function LegacyWorkbookMode({
  chapter,
  search,
}: {
  chapter: ChapterId;
  search: string;
}) {
  const theme = useHostTheme();
  const normalizedSearch = search.trim().toLowerCase();
  const visibleChapters = CHAPTERS.filter(
    (item) =>
      item.id === chapter ||
      normalizedSearch.length > 0 &&
        `${item.label} ${item.thesis}`.toLowerCase().includes(normalizedSearch),
  );
  const visibleAlgorithms = ALGORITHMS.filter((algorithm) => {
    if (algorithm.chapter !== chapter) return false;
    if (normalizedSearch.length === 0) return true;
    return [
      algorithm.id,
      algorithm.title,
      algorithm.summary,
      algorithm.source.path,
      algorithm.source.symbol,
      ...algorithm.routeTags,
    ]
      .join(" ")
      .toLowerCase()
      .includes(normalizedSearch);
  });

  return (
    <Stack gap={20}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          gap: 16,
        }}
      >
        <Stack gap={10}>
          <H2>Algorithm chapters</H2>
          <Text tone="secondary">
            Eight chapters follow a run from admission through durable outputs. Select a chapter to
            establish the reading context for the source-grounded catalog.
          </Text>
          <Stack gap={2}>
            {visibleChapters.map((item, index) => (
              <div
                key={item.id}
                style={{
                  padding: "10px 12px",
                  borderLeft: `2px solid ${
                    item.id === chapter ? theme.accent.primary : theme.stroke.tertiary
                  }`,
                  background: item.id === chapter ? theme.fill.tertiary : theme.bg.editor,
                }}
              >
                <Row gap={10} align="start">
                  <Text size="small" tone="tertiary">
                    {String(index + 1).padStart(2, "0")}
                  </Text>
                  <Stack gap={2}>
                    <Text weight="semibold">{item.label}</Text>
                    <Text size="small" tone="secondary">
                      {item.thesis}
                    </Text>
                  </Stack>
                </Row>
              </div>
            ))}
          </Stack>
        </Stack>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm">Fail closed</Pill>}>Catalog contract</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Text>
                Every algorithm entry must bind executable behavior to a source range, evidence,
                invariants, a failure contract, pseudocode, and at least two trace frames. The
                validator enforces frame count and line-ID integrity; it does not infer whether
                authored frames semantically represent admission, rejection, or another boundary.
              </Text>
              <Divider />
              <CollapsibleSection title="Identity and provenance" defaultOpen>
                <Text size="small" tone="secondary">
                  Stable IDs connect chapters. <Code>SourceRef</Code> anchors implementation and{" "}
                  <Code>EvidenceRef</Code> identifies the proving boundary.
                </Text>
              </CollapsibleSection>
              <CollapsibleSection title="Executable explanation">
                <Text size="small" tone="secondary">
                  Pseudocode lines and trace frames share IDs so a playback cannot point outside the
                  explained algorithm.
                </Text>
              </CollapsibleSection>
              <CollapsibleSection title="Graph integrity">
                <Text size="small" tone="secondary">
                  Predecessors, successors, and validation gates must resolve to known algorithm
                  IDs before any workbook mode renders.
                </Text>
              </CollapsibleSection>
            </Stack>
          </CardBody>
        </Card>
      </div>

      <Stack gap={10}>
        <Row gap={10} align="center" wrap>
          <H2>{CHAPTERS.find((item) => item.id === chapter)?.label}</H2>
          <Pill>{visibleAlgorithms.length} algorithms</Pill>
        </Row>
        <Text tone="secondary">
          {CHAPTERS.find((item) => item.id === chapter)?.thesis}
        </Text>
        {visibleAlgorithms.length > 0 ? (
          <Stack gap={12}>
            {visibleAlgorithms.map((algorithm) => (
              <div key={algorithm.id}>
                <AlgorithmCard algorithm={algorithm} />
              </div>
            ))}
          </Stack>
        ) : (
          <Callout tone="info" title="No catalog entries match">
            Clear the search or select Entry & eligibility or Ownership & budgets.
          </Callout>
        )}
      </Stack>

      <Stack gap={10}>
        <H2>Execution actors</H2>
        <Text tone="secondary">
          The trace vocabulary is fixed now so later frames can illuminate ownership transfers
          without inventing new roles.
        </Text>
        <Row gap={6} align="center" wrap>
          {ACTORS.map((actor, index) => (
            <span key={actor}>
              <Row gap={6} align="center">
                <Pill>{actor}</Pill>
                {index < ACTORS.length - 1 ? (
                  <Text tone="tertiary" as="span">
                    →
                  </Text>
                ) : null}
              </Row>
            </span>
          ))}
        </Row>
      </Stack>
    </Stack>
  );
}

function WorkbookIndex({
  selectedId,
  search,
  chapter,
  favorites,
  onSearchChange,
  onChapterChange,
  onSelect,
}: {
  selectedId: string;
  search: string;
  chapter: ChapterFilter;
  favorites: readonly string[];
  onSearchChange: (value: string) => void;
  onChapterChange: (value: ChapterFilter) => void;
  onSelect: (algorithmId: string) => void;
}) {
  const theme = useHostTheme();
  const algorithms = filterAlgorithms(
    ALGORITHMS,
    search,
    chapter === "all" ? undefined : chapter,
  );

  return (
    <Stack gap={10}>
      <Stack gap={6}>
        <Text size="small" tone="tertiary" weight="semibold">
          WORKBOOK INDEX
        </Text>
        <label>
          <Stack gap={4}>
            <Text as="span" size="small" tone="secondary">
              Search algorithms
            </Text>
            <TextInput
              value={search}
              onChange={onSearchChange}
              placeholder="Source, invariant, failure…"
              type="search"
              style={{ width: "100%" }}
            />
          </Stack>
        </label>
        <label>
          <Stack gap={4}>
            <Text as="span" size="small" tone="secondary">
              Chapter
            </Text>
            <Select
              value={chapter}
              onChange={(value) => onChapterChange(value as ChapterFilter)}
              options={[
                { value: "all", label: "All chapters" },
                ...CHAPTERS.map((item) => ({ value: item.id, label: item.label })),
              ]}
              style={{ width: "100%" }}
            />
          </Stack>
        </label>
      </Stack>
      <Row gap={8} align="center">
        <Text size="small" weight="semibold">
          {algorithms.length} indexed
        </Text>
        <Text size="small" tone="tertiary">
          of {ALGORITHMS.length}
        </Text>
      </Row>
      {favorites.length > 0 ? (
        <CollapsibleSection title={`Favorites · ${favorites.length}`}>
          <Stack gap={4}>
            {favorites.map((id) => {
              const algorithm = ALGORITHMS.find((item) => item.id === id);
              return algorithm ? (
                <div key={id}>
                  <Button variant="ghost" onClick={() => onSelect(id)}>
                    {algorithm.title}
                  </Button>
                </div>
              ) : null;
            })}
          </Stack>
        </CollapsibleSection>
      ) : null}
      {algorithms.length > 0 ? (
        <div
          className="workbook-index-scroll"
          style={{ borderTop: `1px solid ${theme.stroke.tertiary}` }}
        >
          {CHAPTERS.map((chapterDefinition) => {
            const chapterAlgorithms = algorithms.filter(
              (algorithm) => algorithm.chapter === chapterDefinition.id,
            );
            if (chapterAlgorithms.length === 0) return null;
            return (
              <div key={chapterDefinition.id}>
                <div
                  style={{
                    position: "sticky",
                    top: 0,
                    zIndex: 1,
                    padding: "9px 8px 6px",
                    background: theme.bg.editor,
                    borderBottom: `1px solid ${theme.stroke.tertiary}`,
                  }}
                >
                  <Text size="small" tone="tertiary" weight="semibold">
                    {chapterDefinition.label} · {chapterAlgorithms.length}
                  </Text>
                </div>
                {chapterAlgorithms.map((algorithm) => {
                  const active = algorithm.id === selectedId;
                  return (
                    <button
                      type="button"
                      className="workbook-index-row"
                      key={algorithm.id}
                      onClick={() => onSelect(algorithm.id)}
                      aria-pressed={active}
                      aria-label={`${algorithm.title}. ${algorithm.id}`}
                      style={{
                        cursor: "pointer",
                        display: "block",
                        width: "100%",
                        font: "inherit",
                        color: theme.text.primary,
                        textAlign: "left",
                        padding: "10px 9px",
                        borderTop: 0,
                        borderRight: 0,
                        borderBottom: `1px solid ${theme.stroke.tertiary}`,
                        borderLeft: `2px solid ${
                          active ? theme.accent.primary : "transparent"
                        }`,
                        background: active ? theme.fill.secondary : "transparent",
                      }}
                    >
                      <Stack gap={3}>
                        <Text size="small" weight={active ? "semibold" : "normal"}>
                          {algorithm.title}
                        </Text>
                        <Text size="small" tone="tertiary" truncate>
                          {algorithm.id}
                        </Text>
                      </Stack>
                    </button>
                  );
                })}
              </div>
            );
          })}
        </div>
      ) : null}
    </Stack>
  );
}

function actorLinkAliases(from: Actor, to: Actor): readonly string[] {
  return [`${from}->${to}`, `${from}:${to}`, `${from}-${to}`, `${from}/${to}`];
}

function StateTrace({
  algorithm,
  frame,
}: {
  algorithm: AlgorithmDefinition;
  frame: TraceFrame;
}) {
  const theme = useHostTheme();
  const usedActors = new Set(algorithm.frames.flatMap((item) => item.activeActors));
  const usedIndexes = ACTORS.flatMap((actor, index) => (usedActors.has(actor) ? [index] : []));
  const contextActors = new Set<Actor>();
  for (const index of usedIndexes) {
    if (index > 0) contextActors.add(ACTORS[index - 1]);
    if (index < ACTORS.length - 1) contextActors.add(ACTORS[index + 1]);
  }
  const activeActors = new Set(frame.activeActors);
  const activeLinks = new Set(frame.activeLinks);
  const width = 760;
  const top = 34;
  const bottom = 176;
  const x = (index: number) => 62 + index * ((width - 124) / (ACTORS.length - 1));

  return (
    <Stack gap={10}>
      <div
        className="trace-stage caw-trace-scroll"
        style={{
          overflowX: "auto",
          padding: "8px 0 2px",
          borderTop: `1px solid ${theme.stroke.tertiary}`,
          borderBottom: `1px solid ${theme.stroke.tertiary}`,
        }}
      >
        <svg
          className="caw-trace-svg"
          viewBox={`0 0 ${width} 220`}
          role="img"
          aria-label={`${algorithm.title}: ${frame.label}`}
          style={{ width: "100%", display: "block" }}
        >
          {ACTORS.slice(0, -1).map((actor, index) => {
            const to = ACTORS[index + 1];
            const active = actorLinkAliases(actor, to).some((alias) => activeLinks.has(alias));
            return (
              <line
                className={`trace-link${active ? " caw-active-link" : ""}`}
                key={`${actor}-${to}`}
                x1={x(index)}
                y1={top + 20}
                x2={x(index + 1)}
                y2={top + 20}
                stroke={active ? theme.accent.primary : theme.stroke.tertiary}
                strokeWidth={active ? 3 : 1}
                opacity={active ? 1 : 0.34}
              />
            );
          })}
          {ACTORS.map((actor, index) => {
            const active = activeActors.has(actor);
            const contextual = usedActors.has(actor) || contextActors.has(actor);
            return (
              <g
                className="trace-actor"
                key={actor}
                opacity={usedActors.has(actor) ? 1 : contextual ? 0.46 : 0.16}
              >
                <line
                  x1={x(index)}
                  y1={top + 28}
                  x2={x(index)}
                  y2={bottom}
                  stroke={active ? theme.accent.primary : theme.stroke.secondary}
                  strokeWidth={active ? 2 : 1}
                  strokeDasharray={active ? undefined : "3 5"}
                />
                <circle
                  cx={x(index)}
                  cy={top + 20}
                  r={active ? 9 : 6}
                  fill={active ? theme.accent.control : theme.fill.secondary}
                  stroke={active ? theme.accent.primary : theme.stroke.secondary}
                  strokeWidth={1.5}
                />
                <text
                  x={x(index)}
                  y={16}
                  fill={active ? theme.text.primary : theme.text.secondary}
                  fontSize={11}
                  textAnchor="middle"
                >
                  {ACTOR_LABELS[actor]}
                </text>
                <text
                  x={x(index)}
                  y={204}
                  fill={active ? theme.text.primary : theme.text.tertiary}
                  fontSize={10}
                  textAnchor="middle"
                >
                  {active ? "active" : usedActors.has(actor) ? "in path" : "context"}
                </text>
              </g>
            );
          })}
          <text x={14} y={112} fill={theme.text.tertiary} fontSize={10}>
            execution tape
          </text>
        </svg>
      </div>
      <div className="trace-frame">
        <div className="caw-frame-grid">
          <div style={{ padding: 10, background: theme.fill.tertiary }}>
            <Text size="small" tone="tertiary" weight="semibold">
              BEFORE
            </Text>
            {frame.before.map((item) => (
              <div key={item}>
                <Text size="small">{item}</Text>
              </div>
            ))}
          </div>
          <div className="caw-frame-emission" style={{ textAlign: "center" }}>
            <Stack gap={4} style={{ justifyContent: "center" }}>
              <Text size="small" tone="tertiary">
                emits
              </Text>
              <Code>{frame.emitted ?? "state"}</Code>
            </Stack>
          </div>
          <div style={{ padding: 10, background: theme.fill.secondary }}>
            <Text size="small" tone="tertiary" weight="semibold">
              AFTER
            </Text>
            {frame.after.map((item) => (
              <div key={item}>
                <Text size="small">{item}</Text>
              </div>
            ))}
          </div>
        </div>
        <div
          style={{
            marginTop: 8,
            padding: "8px 10px",
            borderLeft: `2px solid ${theme.accent.primary}`,
          }}
        >
          {frame.invariantChecks.map((check) => (
            <div key={check}>
              <Text size="small">
                <Code>check</Code> {check}
              </Text>
            </div>
          ))}
        </div>
      </div>
    </Stack>
  );
}

function PseudocodePanel({
  lines,
  activeLineId,
}: {
  lines: readonly PseudocodeLine[];
  activeLineId: string;
}) {
  const theme = useHostTheme();
  return (
    <Stack gap={0}>
      {lines.map((line) => {
        const active = line.id === activeLineId;
        return (
          <div
            key={line.id}
            style={{
              display: "grid",
              gridTemplateColumns: "58px minmax(0, 1fr)",
              gap: 10,
              background: active ? theme.fill.secondary : undefined,
              color: active ? theme.text.primary : theme.text.secondary,
              borderLeft: `2px solid ${active ? theme.accent.primary : "transparent"}`,
              padding: "7px 10px",
            }}
          >
            <Text size="small" tone={active ? "primary" : "tertiary"}>
              {line.id}
            </Text>
            <Code>{line.text}</Code>
          </div>
        );
      })}
    </Stack>
  );
}

function ContractList({
  label,
  values,
}: {
  label: string;
  values: readonly string[];
}) {
  if (values.length === 0) return null;
  return (
    <Stack gap={4}>
      <Text size="small" tone="tertiary" weight="semibold">
        {label.toUpperCase()}
      </Text>
      {values.map((value) => (
        <div key={value}>
          <Text size="small">{value}</Text>
        </div>
      ))}
    </Stack>
  );
}

function ContractRail({ algorithm }: { algorithm: AlgorithmDefinition }) {
  const dispatch = useCanvasAction();
  const openSource = () =>
    dispatch({
      type: "openFile",
      path: algorithm.source.path,
      selection: {
        startLineNumber: algorithm.source.startLine,
        startColumn: 1,
        endLineNumber: algorithm.source.endLine,
        endColumn: 1,
      },
    });

  return (
    <Stack gap={14}>
      <Stack gap={5}>
        <Text size="small" tone="tertiary" weight="semibold">
          SOURCE CONTRACT
        </Text>
        <Code>{algorithm.source.symbol}</Code>
        <Text size="small" tone="secondary" truncate="start">
          {algorithm.source.path}:{algorithm.source.startLine}-{algorithm.source.endLine}
        </Text>
        <Button variant="secondary" onClick={openSource}>
          Open source range
        </Button>
      </Stack>
      <Divider />
      <ContractList label="Inputs" values={algorithm.inputs} />
      <ContractList label="Outputs" values={algorithm.outputs} />
      <ContractList label="State" values={algorithm.state} />
      <ContractList label="Gates" values={algorithm.gates} />
      <ContractList label="Invariants" values={algorithm.invariants} />
      <ContractList label="Failures" values={algorithm.failures} />
      <Stack gap={4}>
        <Text size="small" tone="tertiary" weight="semibold">
          COMPLEXITY
        </Text>
        <Text size="small">
          <Code>time</Code> {algorithm.complexity.time}
        </Text>
        <Text size="small">
          <Code>memory</Code> {algorithm.complexity.memory}
        </Text>
      </Stack>
      {algorithm.routeTags.length > 0 ? (
        <Row gap={5} wrap>
          {algorithm.routeTags.map((tag) => (
            <span key={tag}>
              <Pill size="sm">{tag}</Pill>
            </span>
          ))}
        </Row>
      ) : null}
      <Divider />
      <Stack gap={6}>
        <Text size="small" tone="tertiary" weight="semibold">
          PROOF BOUNDARIES
        </Text>
        {algorithm.evidence.map((evidence) => (
          <div key={`${evidence.path}/${evidence.symbol}`}>
            <Stack gap={3}>
              <Row gap={6} align="center">
                <Pill size="sm">{evidence.kind}</Pill>
                <Text size="small" weight="semibold">
                  {evidence.symbol}
                </Text>
              </Row>
              <Text size="small" tone="tertiary" truncate="start">
                {evidence.path}
              </Text>
              <Button
                variant="ghost"
                onClick={() => dispatch({ type: "openFile", path: evidence.path })}
              >
                Open proof
              </Button>
            </Stack>
          </div>
        ))}
      </Stack>
    </Stack>
  );
}

function AlgorithmSheet({
  algorithm,
  onSelect,
  favorite,
  onToggleFavorite,
}: {
  algorithm: AlgorithmDefinition;
  onSelect: (algorithmId: string) => void;
  favorite: boolean;
  onToggleFavorite: (algorithmId: string) => void;
}) {
  const theme = useHostTheme();
  const [storedFrameIndex, setStoredFrameIndex] = useCanvasState<number>(
    `cellular.workbook.v1.frame.${algorithm.id}`,
    0,
  );
  const frameIndex = Math.max(0, Math.min(storedFrameIndex, algorithm.frames.length - 1));
  const frame = algorithm.frames[frameIndex];
  const setFrame = (index: number) => {
    cancelWorkbookPlayback();
    setStoredFrameIndex(Math.max(0, Math.min(index, algorithm.frames.length - 1)));
  };
  const play = () => {
    startWorkbookPlayback(
      algorithm.frames.slice(frameIndex + 1).map((_, offset) => frameIndex + offset + 1),
      setStoredFrameIndex,
    );
  };
  keyboardNavigation = keyboardNavigation
    ? {
        ...keyboardNavigation,
        previousFrame: () => setFrame(frameIndex - 1),
        nextFrame: () => setFrame(frameIndex + 1),
      }
    : undefined;

  return (
    <Stack gap={16}>
      <Row gap={12} align="start" wrap>
        <Stack gap={5} style={{ minWidth: 0, flex: 1 }}>
          <Row gap={7} align="center" wrap>
            <StatusLabel status={algorithm.status} />
            <Text size="small" tone="tertiary">
              {CHAPTERS.find((item) => item.id === algorithm.chapter)?.label}
            </Text>
          </Row>
          <H2>{algorithm.title}</H2>
          <Text tone="secondary">{algorithm.summary}</Text>
        </Stack>
        <Stack gap={6}>
          <Code>{algorithm.id}</Code>
          <Button variant="ghost" onClick={() => onToggleFavorite(algorithm.id)}>
            {favorite ? "Remove favorite" : "Save favorite"}
          </Button>
        </Stack>
      </Row>
      <div
        className="workbook-live-region"
        role="status"
        aria-live="polite"
        aria-atomic="true"
      >
        {algorithm.title}. Frame {frameIndex + 1} of {algorithm.frames.length}: {frame.label}.
      </div>

      <div
        className="algorithm-workbench caw-sheet"
        style={{
          alignItems: "start",
        }}
      >
        <div className="caw-main">
          <Stack gap={14}>
          <Row gap={8} align="center" wrap>
            <Text size="small" tone="tertiary" weight="semibold">
              FRAME {frameIndex + 1}/{algorithm.frames.length}
            </Text>
            <Text size="small" weight="semibold">
              {frame.label}
            </Text>
            <Spacer />
            <Button
              variant="ghost"
              disabled={frameIndex === 0}
              onClick={() => setFrame(frameIndex - 1)}
            >
              Back
            </Button>
            <Button
              variant="secondary"
              disabled={frameIndex === algorithm.frames.length - 1}
              onClick={() => setFrame(frameIndex + 1)}
            >
              Step
            </Button>
            <span className="workbook-play">
              <Button
                variant="primary"
                disabled={frameIndex === algorithm.frames.length - 1}
                onClick={play}
              >
                Play
              </Button>
            </span>
            <Button variant="ghost" disabled={frameIndex === 0} onClick={() => setFrame(0)}>
              Reset
            </Button>
            <span className="workbook-play-reduced-note">
              <Text as="span" size="small" tone="tertiary">
                Reduced motion · step controls
              </Text>
            </span>
          </Row>
            <StateTrace algorithm={algorithm} frame={frame} />
          </Stack>
          <Stack gap={7}>
            <Row gap={8} align="center">
              <Text size="small" tone="tertiary" weight="semibold">
                SYNCHRONIZED PSEUDOCODE
              </Text>
              <Spacer />
              <Pill size="sm" active>
                {frame.activeLineId}
              </Pill>
            </Row>
            <div
              style={{
                padding: "6px 0",
                background: theme.bg.elevated,
                border: `1px solid ${theme.stroke.tertiary}`,
              }}
            >
              <PseudocodePanel lines={algorithm.pseudocode} activeLineId={frame.activeLineId} />
            </div>
          </Stack>
        </div>
        <div
          className="caw-contract-rail"
          style={{ paddingLeft: 18, borderLeft: `1px solid ${theme.stroke.secondary}` }}
        >
          <CollapsibleSection title="Contracts and proof" defaultOpen>
            <ContractRail algorithm={algorithm} />
          </CollapsibleSection>
        </div>
      </div>

      <Divider />
      <Row gap={8} align="center" wrap>
        <Text size="small" tone="tertiary" weight="semibold">
          TRAVERSE
        </Text>
        {algorithm.predecessors.map((id) => (
          <span key={`predecessor-${id}`}>
            <Pill onClick={() => onSelect(id)}>
              Back to {ALGORITHMS.find((item) => item.id === id)?.title ?? id}
            </Pill>
          </span>
        ))}
        {algorithm.successors.map((id) => (
          <span key={`successor-${id}`}>
            <Pill active onClick={() => onSelect(id)}>
              Next: {ALGORITHMS.find((item) => item.id === id)?.title ?? id}
            </Pill>
          </span>
        ))}
        {algorithm.predecessors.length === 0 ? (
          <Text size="small" tone="tertiary">
            Entry algorithm
          </Text>
        ) : null}
        {algorithm.successors.length === 0 ? (
          <Text size="small" tone="tertiary">
            Terminal algorithm
          </Text>
        ) : null}
      </Row>
    </Stack>
  );
}

function WorkbookMode({
  selectedId,
  search,
  chapter,
  onSearchChange,
  onChapterChange,
  onSelect,
  favorites,
  onToggleFavorite,
}: {
  selectedId: string;
  search: string;
  chapter: ChapterFilter;
  onSearchChange: (value: string) => void;
  onChapterChange: (value: ChapterFilter) => void;
  onSelect: (algorithmId: string) => void;
  favorites: readonly string[];
  onToggleFavorite: (algorithmId: string) => void;
}) {
  const selected = ALGORITHMS.find((algorithm) => algorithm.id === selectedId) ?? ALGORITHMS[0];

  return (
    <div
      className="workbook-layout"
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(230px, 0.28fr) minmax(0, 1fr)",
        gap: 28,
        alignItems: "start",
      }}
    >
      <WorkbookIndex
        selectedId={selected.id}
        search={search}
        chapter={chapter}
        favorites={favorites}
        onSearchChange={onSearchChange}
        onChapterChange={onChapterChange}
        onSelect={onSelect}
      />
      <AlgorithmSheet
        algorithm={selected}
        onSelect={onSelect}
        favorite={favorites.includes(selected.id)}
        onToggleFavorite={onToggleFavorite}
      />
    </div>
  );
}

function RecipeRail({
  onSelect,
}: {
  onSelect: (selection: SelectorState) => void;
}) {
  const renderRecipes = (kind: RouteRecipe["kind"]) => (
    <div className="recipe-strip" aria-label={`${kind} route recipes`}>
      {ROUTE_RECIPES.filter((recipe) => recipe.kind === kind).map((recipe) => {
        const route = cachedRoute(recipe.selection);
        const outcome = route.valid
          ? `${route.algorithmIds.length} stops`
          : `${GATE_STAGE_LABELS[route.gateStage]} · ${route.rejectedBy}`;
        return (
          <div key={recipe.id} style={{ flex: "0 0 min(250px, 76vw)" }}>
            <Stack gap={4}>
              <Button variant="secondary" onClick={() => onSelect(recipe.selection)}>
                {recipe.title}
              </Button>
              <Text size="small" tone="tertiary">
                {outcome}
              </Text>
            </Stack>
          </div>
        );
      })}
    </div>
  );

  return (
    <Stack gap={6}>
      <CollapsibleSection title={`Canonical recipes · ${ROUTE_RECIPES.filter((recipe) => recipe.kind === "canonical").length}`}>
        {renderRecipes("canonical")}
      </CollapsibleSection>
      <CollapsibleSection title={`First-rejection recipes · ${ROUTE_RECIPES.filter((recipe) => recipe.kind === "rejected").length}`}>
        {renderRecipes("rejected")}
      </CollapsibleSection>
    </Stack>
  );
}

function ComposeMode({
  selection,
  setSelection,
  onOpenAlgorithm,
}: {
  selection: SelectorState;
  setSelection: (selection: SelectorState) => void;
  onOpenAlgorithm: (algorithmId: string) => void;
}) {
  const theme = useHostTheme();
  const route = cachedRoute(selection);
  const update = (key: SelectorKey, value: string) =>
    setSelection({ ...selection, [key]: value } as SelectorState);
  const routeSections = route.valid
    ? ROUTE_ORDER.map((chapter) => ({
        key: chapter,
        sectionId: `route-${chapter}`,
        chapter,
        algorithms: route.algorithmIds
          .map((id) => ALGORITHMS.find((algorithm) => algorithm.id === id))
          .filter(
            (algorithm): algorithm is AlgorithmDefinition =>
              algorithm !== undefined && algorithm.chapter === chapter,
          ),
      })).filter((item) => item.algorithms.length > 0)
    : route.algorithmIds.flatMap((id, index) => {
        const algorithm = ALGORITHMS.find((item) => item.id === id);
        return algorithm
          ? [{
              key: `${index}-${id}`,
              sectionId: `route-stop-${index}-${id}`,
              chapter: algorithm.chapter,
              algorithms: [algorithm],
            }]
          : [];
      });

  return (
    <Stack gap={22}>
      <Row align="start" wrap gap={12}>
        <Stack gap={5} style={{ maxWidth: 760 }}>
          <H2>Compose an execution route</H2>
          <Text tone="secondary">
            Select a run shape to derive the exact admission, ownership, transport, execution,
            capture, merge, and artifact algorithms in runtime order.
          </Text>
        </Stack>
        <Spacer />
        <Button variant="secondary" onClick={() => setSelection(DEFAULT_SELECTION)}>
          Reset selectors
        </Button>
      </Row>
      <RecipeRail onSelect={setSelection} />
      <div
        className="composer-layout caw-composer"
        style={{
          alignItems: "start",
        }}
      >
        <Card size="lg">
          <CardHeader>Route selectors</CardHeader>
          <CardBody>
            <Stack gap={12}>
              {(Object.keys(SELECTOR_OPTIONS) as SelectorKey[]).map((key) => (
                <label key={key}>
                  <Stack gap={5}>
                    <Text size="small" weight="semibold">
                      {SELECTOR_LABELS[key]}
                    </Text>
                    <Select
                      value={selection[key]}
                      onChange={(value) => update(key, value)}
                      options={SELECTOR_OPTIONS[key].map((value) => ({
                        value,
                        label: optionLabel(value),
                      }))}
                    />
                  </Stack>
                </label>
              ))}
            </Stack>
          </CardBody>
        </Card>
        <Stack gap={16}>
          <Callout tone="info" title="Requested → effective runtime settings">
            <Row gap={12} wrap>
              <Text size="small">
                Workload: <Code>{selection.workload}</Code> → <Code>{route.effective.workload}</Code>
              </Text>
              <Text size="small">
                Topology: <Code>{selection.topology}</Code> → <Code>{route.effective.topology}</Code>
              </Text>
              <Text size="small">
                Artifacts: <Code>{selection.artifacts}</Code> → <Code>{route.effective.artifacts}</Code>
              </Text>
              <Text size="small">
                Storage: <Code>{selection.storage}</Code> → <Code>{route.effective.storage}</Code>
              </Text>
            </Row>
          </Callout>
          {route.valid ? (
            <Callout tone="success" title={`${route.algorithmIds.length} ordered algorithm stops`}>
              This selection passes every current cellular gate.
            </Callout>
          ) : (
            <Callout tone="danger" title={`${GATE_STAGE_LABELS[route.gateStage]} · ${route.rejectedBy}`}>
              <Stack gap={4}>
                <Text size="small">{route.reason}</Text>
                <Text size="small" tone="secondary">
                  The ordered route stops at the actual runtime enforcement stage.
                </Text>
              </Stack>
            </Callout>
          )}
          {route.limitations.length > 0 ? (
            <Callout tone="warning" title="Verification boundary">
              <Stack gap={4}>
                {route.limitations.map((limitation) => (
                  <div key={limitation}>
                    <Text size="small">{limitation}</Text>
                  </div>
                ))}
              </Stack>
            </Callout>
          ) : null}
          <Stack gap={8}>
            <Text size="small" weight="semibold">
              Stage-aware composed lifecycle
            </Text>
            <div
              className="composed-route caw-route-scroll"
              aria-label="Composed cellular algorithm route"
            >
              {routeSections.map(({ key, sectionId, chapter, algorithms }) => (
                <section className="composed-route-stop" key={key} aria-labelledby={sectionId}>
                  <Stack gap={7}>
                    <div id={sectionId}>
                      <Text size="small" tone="tertiary" weight="semibold">
                        {CHAPTERS.find((item) => item.id === chapter)?.label}
                      </Text>
                    </div>
                    {algorithms.map((algorithm) => (
                      <details
                        key={algorithm.id}
                        style={{
                          border: `1px solid ${theme.stroke.tertiary}`,
                          background: theme.fill.tertiary,
                          padding: 10,
                        }}
                      >
                        <summary>
                          <Text size="small" weight="semibold">
                            {route.algorithmIds.indexOf(algorithm.id) + 1}. {algorithm.title}
                          </Text>
                        </summary>
                        <Stack gap={8} style={{ paddingTop: 10 }}>
                          <Text size="small" tone="secondary">{algorithm.summary}</Text>
                          <Row gap={6} wrap>
                            <StatusLabel status={algorithm.status} />
                            <Pill size="sm">{algorithm.id}</Pill>
                          </Row>
                          <Button variant="secondary" onClick={() => onOpenAlgorithm(algorithm.id)}>
                            Open workbook sheet
                          </Button>
                        </Stack>
                      </details>
                    ))}
                  </Stack>
                </section>
              ))}
            </div>
          </Stack>
          {route.valid ? (
            <div className="caw-route-facets">
              <Stack gap={4}>
                <Text size="small" weight="semibold">Workload classification</Text>
                <Text size="small" tone="secondary">
                  Requested {optionLabel(selection.workload)} → effective {optionLabel(route.effective.workload)}
                </Text>
              </Stack>
              <Stack gap={4}>
                <Text size="small" weight="semibold">Aggregation topology</Text>
                <Text size="small" tone="secondary">
                  Requested {optionLabel(selection.topology)} → effective {optionLabel(route.effective.topology)}
                </Text>
              </Stack>
              <Stack gap={4}>
                <Text size="small" weight="semibold">Artifact behavior</Text>
                <Text size="small" tone="secondary">
                  Requested {optionLabel(selection.artifacts)} → effective {optionLabel(route.effective.artifacts)}
                </Text>
              </Stack>
              <Stack gap={4}>
                <Text size="small" weight="semibold">Effective memory</Text>
                <Text size="small" tone="secondary">{route.memory}</Text>
              </Stack>
              <Stack gap={4}>
                <Text size="small" weight="semibold">Fidelity</Text>
                <Text size="small" tone="secondary">{route.fidelity}</Text>
              </Stack>
              <Stack gap={4}>
                <Text size="small" weight="semibold">Artifacts</Text>
                <Text size="small" tone="secondary">{route.artifacts}</Text>
              </Stack>
              <Stack gap={4}>
                <Text size="small" weight="semibold">Compile features</Text>
                {route.compileFeatures.map((value) => (
                  <div key={value}><Code>{value}</Code></div>
                ))}
              </Stack>
              <Stack gap={4}>
                <Text size="small" weight="semibold">Environment variables</Text>
                {route.environment.map((value) => (
                  <div key={value}><Code>{value}</Code></div>
                ))}
              </Stack>
            </div>
          ) : null}
          <CollapsibleSection title={`Evidence carried by route · ${route.valid ? route.evidence.length : 0}`}>
            {route.valid ? (
              <Stack gap={5}>
                {route.evidence.map((item) => (
                  <div key={`${item.path}/${item.symbol}`}>
                    <Text size="small" tone="secondary">
                      <Code>{item.kind}</Code> {item.path} · {item.symbol}
                    </Text>
                  </div>
                ))}
              </Stack>
            ) : null}
          </CollapsibleSection>
        </Stack>
      </div>
      <Card size="lg">
        <CardHeader>Storage invariant matrix</CardHeader>
        <CardBody>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))",
              gap: 12,
            }}
          >
            {[
              ["Retain", STORAGE_INVARIANTS.retain, STORAGE_MODE_DETAILS.retain],
              ["Exact fold", STORAGE_INVARIANTS.exactFold, STORAGE_MODE_DETAILS.exactFold],
              ["Sketch", STORAGE_INVARIANTS.sketch, STORAGE_MODE_DETAILS.sketch],
            ].map(([label, assertion, details]) => (
              <div key={label as string}>
                <Stack gap={5}>
                  <H3>{label as string}</H3>
                  <Text size="small"><Code>assertion</Code> {assertion as string}</Text>
                  <Text size="small"><Code>rows</Code> {(details as typeof STORAGE_MODE_DETAILS.retain).rows}</Text>
                  <Text size="small"><Code>memory</Code> {(details as typeof STORAGE_MODE_DETAILS.retain).memory}</Text>
                  <Text size="small"><Code>fidelity</Code> {(details as typeof STORAGE_MODE_DETAILS.retain).fidelity}</Text>
                  <Text size="small"><Code>count</Code> {(details as typeof STORAGE_MODE_DETAILS.retain).count}</Text>
                </Stack>
              </div>
            ))}
          </div>
        </CardBody>
      </Card>
    </Stack>
  );
}

type RouteBands = {
  prefix: readonly string[];
  leftDelta: readonly string[];
  rightDelta: readonly string[];
  suffix: readonly string[];
};

function collapseSharedRoute(
  left: readonly string[],
  right: readonly string[],
): RouteBands {
  let prefixLength = 0;
  while (
    prefixLength < left.length &&
    prefixLength < right.length &&
    left[prefixLength] === right[prefixLength]
  ) {
    prefixLength += 1;
  }
  let suffixLength = 0;
  while (
    suffixLength < left.length - prefixLength &&
    suffixLength < right.length - prefixLength &&
    left[left.length - suffixLength - 1] === right[right.length - suffixLength - 1]
  ) {
    suffixLength += 1;
  }
  const leftRemainder = left.slice(prefixLength, left.length - suffixLength);
  const rightRemainder = right.slice(prefixLength, right.length - suffixLength);
  const leftRemainderIds = new Set(leftRemainder);
  const rightRemainderIds = new Set(rightRemainder);
  return {
    prefix: left.slice(0, prefixLength),
    leftDelta: leftRemainder.filter((id) => !rightRemainderIds.has(id)),
    rightDelta: rightRemainder.filter((id) => !leftRemainderIds.has(id)),
    suffix: suffixLength === 0 ? [] : left.slice(left.length - suffixLength),
  };
}

type DecisionFacet =
  | "effective"
  | "memory"
  | "fidelity"
  | "artifacts"
  | "limitations"
  | "rejection";

const DECISION_FACET_LABELS: Readonly<Record<DecisionFacet, string>> = {
  effective: "Effective settings",
  memory: "Memory",
  fidelity: "Fidelity",
  artifacts: "Artifacts",
  limitations: "Limitations",
  rejection: "Admission",
};

function decisionFacet(route: RouteResult, facet: DecisionFacet): string {
  if (facet === "effective") {
    return `workload=${route.effective.workload}; topology=${route.effective.topology}; storage=${route.effective.storage}; artifacts=${route.effective.artifacts}`;
  }
  if (!route.valid) {
    if (facet === "rejection") {
      return `${GATE_STAGE_LABELS[route.gateStage]} · ${route.rejectedBy}: ${route.reason}`;
    }
    if (facet === "limitations") return route.limitations.join(" ") || "None";
    return "Not reached";
  }
  if (facet === "limitations") return route.limitations.join(" ") || "None";
  if (facet === "rejection") return "Admitted";
  return route[facet];
}

function algorithmTitle(id: string): string {
  return ALGORITHMS.find((algorithm) => algorithm.id === id)?.title ?? id;
}

function SharedRouteBand({
  label,
  algorithmIds,
}: {
  label: string;
  algorithmIds: readonly string[];
}) {
  const theme = useHostTheme();
  if (algorithmIds.length === 0) return null;
  return (
    <div
      style={{
        padding: "8px 10px",
        background: theme.fill.tertiary,
        border: `1px solid ${theme.stroke.tertiary}`,
      }}
    >
      <Text size="small" tone="secondary">
        {label} · {algorithmIds.length} shared stops
      </Text>
    </div>
  );
}

function DecisionSide({
  label,
  selection,
  route,
  delta,
  differingFacets,
  onOpenRoute,
}: {
  label: string;
  selection: SelectorState;
  route: RouteResult;
  delta: readonly string[];
  differingFacets: readonly DecisionFacet[];
  onOpenRoute: (selection: SelectorState) => void;
}) {
  const theme = useHostTheme();
  return (
    <section
      style={{
        padding: "14px 0 0 14px",
        borderTop: `1px solid ${theme.stroke.tertiary}`,
        borderLeft: `2px solid ${route.valid ? theme.stroke.secondary : theme.diff.stripRemoved}`,
      }}
      aria-label={label}
    >
      <Stack gap={12}>
        <Row gap={8} align="center" wrap>
          <H3>{label}</H3>
          <Spacer />
          <AdmissionLabel valid={route.valid} />
        </Row>
        <Button variant="secondary" onClick={() => onOpenRoute(selection)}>
          Open derived route
        </Button>
        <Stack gap={6}>
          <Text size="small" weight="semibold">Distinct algorithm stops</Text>
          {delta.length > 0 ? delta.map((id) => (
            <div key={id}>
              <Text size="small" tone="secondary">
                <Code>{id}</Code> · {algorithmTitle(id)}
              </Text>
            </div>
          )) : (
            <Text size="small" tone="tertiary">All stops are shared.</Text>
          )}
        </Stack>
        {differingFacets.map((facet) => (
          <div key={facet}>
            <Stack gap={3}>
              <Text size="small" weight="semibold">{DECISION_FACET_LABELS[facet]}</Text>
              <Text size="small" tone="secondary">{decisionFacet(route, facet)}</Text>
            </Stack>
          </div>
        ))}
      </Stack>
    </section>
  );
}

function DecisionLaboratory({
  onOpenRoute,
}: {
  onOpenRoute: (selection: SelectorState) => void;
}) {
  const theme = useHostTheme();
  return (
    <Stack gap={14}>
      <Stack gap={5} style={{ maxWidth: 760 }}>
        <H2>Decision laboratory</H2>
        <Text tone="secondary">
          Compare neighboring selector states. Shared lifecycle bands collapse so the branch,
          storage tradeoff, artifact boundary, or first rejecting gate stays visible.
        </Text>
      </Stack>
      <div className="decision-grid">
        {DECISIONS.map((decision, index) => {
          const leftRoute = cachedRoute(decision.left);
          const rightRoute = cachedRoute(decision.right);
          const bands = collapseSharedRoute(leftRoute.algorithmIds, rightRoute.algorithmIds);
          const featured = decision.id === "storage-fidelity";
          const differingFacets = (
            ["memory", "fidelity", "artifacts", "limitations", "rejection"] as const
          ).filter((facet) => decisionFacet(leftRoute, facet) !== decisionFacet(rightRoute, facet));
          return (
            <section
              key={decision.id}
              aria-labelledby={`decision-${decision.id}`}
              style={{
                padding: featured ? 18 : "24px 0",
                margin: featured ? "10px 0" : 0,
                background: featured ? theme.fill.tertiary : undefined,
                border: featured ? `1px solid ${theme.stroke.tertiary}` : undefined,
                borderTop: featured ? undefined : `1px solid ${theme.stroke.secondary}`,
              }}
            >
              <Stack gap={12}>
                <Row gap={14} align="start" wrap>
                  <Text size="small" tone="tertiary" weight="semibold">
                    {String(index + 1).padStart(2, "0")}
                  </Text>
                  <Stack gap={4} style={{ minWidth: 0, flex: 1 }}>
                    <div id={`decision-${decision.id}`}>
                      <H2>{decision.title}</H2>
                    </div>
                    <Text size="small" tone="secondary">{decision.invariant}</Text>
                  </Stack>
                </Row>
                <SharedRouteBand label="Shared prefix" algorithmIds={bands.prefix} />
                <div className="decision-sides">
                  <DecisionSide
                    label={decision.leftLabel}
                    selection={decision.left}
                    route={leftRoute}
                    delta={bands.leftDelta}
                    differingFacets={differingFacets}
                    onOpenRoute={onOpenRoute}
                  />
                  <DecisionSide
                    label={decision.rightLabel}
                    selection={decision.right}
                    route={rightRoute}
                    delta={bands.rightDelta}
                    differingFacets={differingFacets}
                    onOpenRoute={onOpenRoute}
                  />
                </div>
                <SharedRouteBand label="Shared suffix" algorithmIds={bands.suffix} />
              </Stack>
            </section>
          );
        })}
      </div>
    </Stack>
  );
}

export default function CellularAlgorithmWorkbook() {
  installReducedMotionCancellation();
  installKeyboardNavigation();
  const theme = useHostTheme();
  const [persistedMode, setPersistedMode] = useCanvasState<unknown>(
    "cellular.workbook.v1.mode",
    "workbook",
  );
  const [persistedChapter, setPersistedChapter] = useCanvasState<unknown>(
    "cellular.workbook.v1.chapter",
    "all",
  );
  const [persistedQuery, setPersistedQuery] = useCanvasState<unknown>(
    "cellular.workbook.v1.query",
    "",
  );
  const [persistedAlgorithmId, setPersistedAlgorithmId] = useCanvasState<unknown>(
    "cellular.workbook.v1.algorithm",
    ALGORITHMS[0].id,
  );
  const [persistedReducedMotion, setPersistedReducedMotion] = useCanvasState<unknown>(
    "cellular.workbook.v1.reduced-motion",
    false,
  );
  const [persistedFavorites, setFavorites] = useCanvasState<readonly string[]>(
    "cellular.workbook.v1.favorites",
    [],
  );
  const [persistedSelection, setPersistedSelection] = useCanvasState<unknown>(
    "cellular.workbook.v1.selection",
    DEFAULT_SELECTION,
  );
  const mode = normalizeMode(persistedMode);
  const chapter = normalizeChapter(persistedChapter);
  const query = normalizeQuery(persistedQuery);
  const selectedId = normalizeAlgorithmId(persistedAlgorithmId);
  const reducedMotion = normalizeReducedMotion(persistedReducedMotion);
  manualReducedMotion = reducedMotion;
  const favorites = Array.isArray(persistedFavorites)
    ? persistedFavorites.filter(
        (id): id is string =>
          typeof id === "string" && ALGORITHMS.some((algorithm) => algorithm.id === id),
      )
    : [];
  const selection = normalizeSelection(persistedSelection);
  const setSelection = (nextSelection: SelectorState) =>
    setPersistedSelection(normalizeSelection(nextSelection));
  const errors = validateWorkbook(ALGORITHMS, VALIDATION_GATES);
  const selectAlgorithm = (algorithmId: string) => {
    cancelWorkbookPlayback();
    setPersistedAlgorithmId(normalizeAlgorithmId(algorithmId));
  };
  const changeMode = (nextMode: Mode) => {
    cancelWorkbookPlayback();
    setPersistedMode(normalizeMode(nextMode));
  };
  const openAlgorithmFromRoute = (algorithmId: string) => {
    selectAlgorithm(algorithmId);
    changeMode("workbook");
  };
  const openDerivedRoute = (nextSelection: SelectorState) => {
    cancelWorkbookPlayback();
    setSelection(nextSelection);
    setPersistedMode("compose");
  };
  const toggleFavorite = (algorithmId: string) =>
    setFavorites(
      favorites.includes(algorithmId)
        ? favorites.filter((id) => id !== algorithmId)
        : [...favorites, algorithmId],
    );
  const changeReducedMotion = (nextReducedMotion: boolean) => {
    cancelWorkbookPlayback();
    manualReducedMotion = nextReducedMotion;
    setPersistedReducedMotion(normalizeReducedMotion(nextReducedMotion));
  };
  const selectedIndex = Math.max(
    0,
    ALGORITHMS.findIndex((algorithm) => algorithm.id === selectedId),
  );
  keyboardNavigation = {
    previousAlgorithm: () =>
      selectAlgorithm(ALGORITHMS[(selectedIndex - 1 + ALGORITHMS.length) % ALGORITHMS.length].id),
    nextAlgorithm: () =>
      selectAlgorithm(ALGORITHMS[(selectedIndex + 1) % ALGORITHMS.length].id),
  };

  if (errors.length > 0) {
    return (
      <Stack gap={8} style={{ padding: 24, background: theme.bg.editor, minHeight: "100vh" }}>
        <Callout tone="danger" title="Workbook integrity failure">
          <Stack gap={4}>
            {errors.map((error) => (
              <div key={error}>
                <Text size="small">{error}</Text>
              </div>
            ))}
          </Stack>
        </Callout>
      </Stack>
    );
  }

  return (
    <div
      id={WORKBOOK_ROOT_ID}
      className={`caw-shell${reducedMotion ? " caw-reduced" : ""}`}
    >
      <Stack
        gap={22}
        style={{
          padding: "24px clamp(18px, 4vw, 48px) 48px",
          background: theme.bg.editor,
          color: theme.text.primary,
          minHeight: "100vh",
        }}
      >
      <style>{WORKBOOK_CSS}</style>
      <div
        className="caw-header"
        style={{
          padding: "10px 0 12px",
          background: theme.bg.editor,
          borderBottom: `1px solid ${theme.stroke.tertiary}`,
        }}
      >
        <Stack gap={10}>
          <Stack gap={5} style={{ maxWidth: 760 }}>
            <Text size="small" tone="tertiary" weight="semibold">
              CELLULAR EXECUTION · ALGORITHM WORKBOOK
            </Text>
            <H1>Reason from gate to artifact</H1>
            <Text tone="secondary">
              Study how a run is admitted, partitioned, executed, captured, and merged across cells.
            </Text>
          </Stack>
          <Row gap={10} align="center" wrap>
            <ModeTabs mode={mode} onChange={changeMode} />
            <Spacer />
            <Text size="small" tone="tertiary">
              [ ] algorithms · ← → frames
            </Text>
            <label>
              <Row gap={7} align="center">
                <Text as="span" size="small" tone="secondary">
                  Reduced motion
                </Text>
                <Toggle checked={reducedMotion} onChange={changeReducedMotion} />
              </Row>
            </label>
          </Row>
        </Stack>
      </div>

      <div role="status">
        <Row gap={8} align="center" wrap>
          <Pill size="sm" active>
            {ALGORITHMS.length} source-grounded algorithms
          </Pill>
          <Text size="small" tone="tertiary">
            Integrity validated · {favorites.length} favorites
          </Text>
        </Row>
      </div>

      {mode === "workbook" ? (
        <WorkbookMode
          selectedId={selectedId}
          chapter={chapter}
          search={query}
          onSearchChange={(nextQuery) => setPersistedQuery(normalizeQuery(nextQuery))}
          onChapterChange={(nextChapter) =>
            setPersistedChapter(normalizeChapter(nextChapter))
          }
          onSelect={selectAlgorithm}
          favorites={favorites}
          onToggleFavorite={toggleFavorite}
        />
      ) : mode === "compose" ? (
        <ComposeMode
          selection={selection}
          setSelection={setSelection}
          onOpenAlgorithm={openAlgorithmFromRoute}
        />
      ) : (
        <DecisionLaboratory onOpenRoute={openDerivedRoute} />
      )}
      </Stack>
    </div>
  );
}
