import type { SlideDefinition } from "../../core/types";

type StepInput = Omit<SlideDefinition, "narration">;

const STEPS: readonly StepInput[] = [
  {
    eyebrow: "Launch · 1 of 20",
    title: "One run, many cells, one report",
    lede:
      "Cellular scale keeps a single benchmark identity while autonomous processes share the work. Scaling changes placement, not the measurement contract: the authored run, the cells, and the one authoritative report frame everything that follows.",
    term: {
      word: "run_cellular",
      meaning: "Controller entry that partitions one authored run across cell processes and merges their results into one report.",
    },
    points: [
      "Invariant: scaling changes placement, not the measurement contract.",
      "Meet the authored run, an autonomous cell, and the final report.",
      "Evidence: `rust/runtime/src/engine/cellular_controller.rs`.",
    ],
    caption: "authored run → many cells → one report.",
  },
  {
    eyebrow: "Launch · 2 of 20",
    title: "Author the Config v2 run",
    lede:
      "The human-facing command resolves exactly one strict benchmark request before any child process exists. Every cell later derives from that same resolved request, so there is a single authored source of truth.",
    term: {
      word: "ProfileFlags",
      meaning: "Config v2 CLI surface resolved into one BenchmarkRun before cellular partitioning.",
    },
    points: [
      "Every cell derives from the same resolved request.",
      "Resolution happens once, in the parent, before fan-out.",
      "Evidence: `rust/cli/src/flags.rs`.",
    ],
    caption: "one authored Config v2 request · resolved before fan-out.",
  },
  {
    eyebrow: "Launch · 3 of 20",
    title: "Re-exec the unified binary",
    lede:
      "The entry point launches the same aiperf image in a hidden execution mode over stdio. Process isolation is achieved without a second product executable: the protocol-v2 self-exec boundary is the only handoff.",
    term: {
      word: "exec_bin::resolve",
      meaning: "Resolves current_exe() so the parent can spawn itself as the protocol-v2 --execute child.",
    },
    points: [
      "Process isolation survives without a second product executable.",
      "The child receives one protocol-v2 envelope on stdin.",
      "Evidence: `rust/cli/src/exec_bin.rs`, `execute_mode.rs`.",
    ],
    caption: "self-exec --execute · protocol-v2 boundary.",
  },
  {
    eyebrow: "Launch · 4 of 20",
    title: "Promote to controller at cells > 1",
    lede:
      "Single-cell execution stays direct. Multi-cell execution promotes one process into a coordinator that partitions and launches, but never dispatches inference load itself. Native Kubernetes roles reuse the same entry points.",
    term: {
      word: "controller promotion",
      meaning: "At cells > 1 the execute process becomes a coordinator; the controller coordinates, cells dispatch.",
    },
    points: [
      "The controller coordinates but does not dispatch inference load.",
      "Native `controller` / `cell` / `aggregator` roles wire the same model.",
      "Evidence: `rust/runtime/src/engine/cellular_controller.rs`.",
    ],
    caption: "cells > 1 → promote one coordinator.",
  },
  {
    eyebrow: "Launch · 5 of 20",
    title: "Validate and fail closed",
    lede:
      "Eligibility is checked before cells launch. Unsupported transport, budget, storage, and workload combinations are rejected up front so a distributed run never degrades silently into a partial or incorrect result.",
    term: {
      word: "validate_cellular_run_shape",
      meaning: "Pre-launch gate that rejects unsupported cellular combinations instead of degrading silently.",
    },
    points: [
      "Unsupported combinations never degrade silently.",
      "Scheduled duration and unbounded runs fail closed.",
      "Evidence: `rust/runtime/src/engine/cellular_controller.rs`.",
    ],
    caption: "eligibility gate → reject unsupported shapes.",
  },
  {
    eyebrow: "Distribute · 6 of 20",
    title: "Slice the global budget",
    lede:
      "Request and conversation budgets divide across cells without overlap or omission. The union of every cell's owned positions exactly tiles the global budget through a deterministic modulo partition and a stable workload source.",
    term: {
      word: "owned_positions",
      meaning: "The set i where i % cells == cell_id; cell-owned global positions that tile the budget.",
    },
    points: [
      "The union of cell-owned positions exactly tiles the global budget.",
      "`ModuloCellPartition` assigns i % cells == cell_id.",
      "Evidence: `rust/runtime/src/cellular/partition.rs`, `rust/runtime/src/engine/cell_launcher.rs`.",
    ],
    caption: "global budget → disjoint modulo tiles.",
  },
  {
    eyebrow: "Distribute · 7 of 20",
    title: "Register, then release START",
    lede:
      "Children become ready before the controller releases synchronized execution. No synchronized cell dispatches before the registration barrier opens, so cross-cell start skew is bounded without a shared wall clock.",
    term: {
      word: "await_all_registered → trigger",
      meaning: "Default lifecycle gate: the controller triggers the shared start event only once every cell registers.",
    },
    points: [
      "No synchronized cell dispatches before the barrier opens.",
      "This is the default START policy.",
      "Evidence: `rust/runtime/src/engine/cellular_controller.rs`.",
    ],
    caption: "register all → trigger → synchronized dispatch.",
  },
  {
    eyebrow: "Distribute · 8 of 20",
    title: "Choose START policy and timing origin",
    lede:
      "Phaser and barrier-free START are opt-in alternatives to the synchronized barrier. A shared timing origin can zero cell clocks at the barrier. Start policy and timing epoch change coordination, never ownership.",
    term: {
      word: "PhaserServer / PhaserClient",
      meaning: "Velo phaser transport for monotonic distributed START, selected with AIPERF_CELL_PHASER_START.",
    },
    points: [
      "Start policy and timing epoch change coordination, never ownership.",
      "Shared origin zeroes cell clocks at the synchronized barrier.",
      "Evidence: `rust/runtime/src/cellular/transport/phaser_velo.rs`, `engine/cell_origin.rs`.",
    ],
    caption: "synchronized · phaser · barrier-free · shared origin.",
  },
  {
    eyebrow: "Distribute · 9 of 20",
    title: "Distribute datasets by overlay or HTTP",
    lede:
      "The opt-in Velo overlay verifies request fan-out, while Stage G serves cross-host file and graph inputs over HTTP plus zstd. Every cell receives only its owned input identities, independent of the delivery mechanism.",
    term: {
      word: "build_dataset_serve_plan",
      meaning: "Stage G plan that serves file and graph inputs to cross-host cells over HTTP + zstd.",
    },
    points: [
      "Every cell receives only its owned input identities.",
      "Overlay and HTTP serving are both built, without replacing cell-local execution.",
      "Evidence: `rust/runtime/src/cellular/dataset_session.rs`, `engine/cellular_controller.rs`.",
    ],
    caption: "Velo overlay · Stage G HTTP+zstd · owned inputs only.",
  },
  {
    eyebrow: "Distribute · 10 of 20",
    title: "Index stable cell ownership",
    lede:
      "Each cell tracks indexed, in-flight, and completed request identities. A request identity belongs to exactly one cell, so the cell-local ownership index is the ground truth for who runs what.",
    term: {
      word: "DatasetIndex",
      meaning: "Cell-local index of owned request identities across indexed, in-flight, and completed states.",
    },
    points: [
      "A request identity belongs to exactly one cell.",
      "Indexed → InFlight → completed is tracked per cell.",
      "Evidence: `rust/runtime/src/cellular/dataset_session.rs`.",
    ],
    caption: "owned ids only · indexed → in-flight → complete.",
  },
  {
    eyebrow: "Execute · 11 of 20",
    title: "Enter one autonomous cell",
    lede:
      "A cell owns its runtime, transport state, metrics, and local lifecycle. Cells share no hot-path collector lock, so deterministic ownership connects directly to an autonomous process running the ordinary engine.",
    term: {
      word: "fetch_cell_envelope",
      meaning: "Cell startup that fetches its sliced protocol-v2 envelope and runs the ordinary single-run engine.",
    },
    points: [
      "Cells share no hot-path collector lock.",
      "Each cell runs the ordinary engine on its slice.",
      "Evidence: `rust/runtime/src/engine/cellular_cell.rs`.",
    ],
    caption: "sliced envelope → autonomous cell process.",
  },
  {
    eyebrow: "Execute · 12 of 20",
    title: "Partition again across worker shards",
    lede:
      "Inside a cell the same ownership rule applies across thread-per-core workers. Nested ownership stays deterministic and disjoint, so the second partition level never overlaps the first.",
    term: {
      word: "run_sharded_scheduled",
      meaning: "Thread-per-core sharded execution that partitions the cell's owned work across !Send workers.",
    },
    points: [
      "Nested ownership remains deterministic and disjoint.",
      "Workers are thread-per-core and !Send on the hot path.",
      "Evidence: `rust/runtime/src/engine/sharded_scheduled.rs`.",
    ],
    caption: "cell slice → worker shards · second disjoint partition.",
  },
  {
    eyebrow: "Execute · 13 of 20",
    title: "Stamp global ordinals",
    lede:
      "Local worker progress maps back to one stable global request order. The ordinal formula lets every worker carry a globally unique identity so merged output can be reordered exactly.",
    term: {
      word: "global_ordinal",
      meaning: "ordinal = phase_base + within_phase_local × cell_count + cell_id.",
    },
    points: [
      "ordinal = phase_base + local × cell_count + cell_id.",
      "Every worker carries the identity formula.",
      "Evidence: `rust/runtime/src/cellular/issuance.rs`.",
    ],
    caption: "local progress → stable global ordinal.",
  },
  {
    eyebrow: "Execute · 14 of 20",
    title: "Dispatch online; reject unwired simulation",
    lede:
      "Every shard drives the same clock-injected HTTP or gRPC seam that single-cell runs use. Scheduled gRPC cellular is proven; graph plus gRPC cellular is unproven; DynoSim cellular fails closed because its SimClock executor is unwired.",
    term: {
      word: "RequestSink",
      meaning: "Transport-neutral dispatch seam; only the execution child dispatches inference requests.",
    },
    points: [
      "Only the execution child dispatches inference requests.",
      "Scheduled HTTP and gRPC cellular are built; graph+gRPC cellular is unproven.",
      "Evidence: `rust/loadgen-core/src/sink.rs`, `engine/cellular_controller.rs`.",
    ],
    caption: "clock-injected HTTP/gRPC · DynoSim cellular rejected.",
  },
  {
    eyebrow: "Execute · 15 of 20",
    title: "Observe and finalize each record",
    lede:
      "Arrival, admission, token, usage, and terminal observations become one CapturedRecord. Storage mode begins only after the record is finalized, which is where execution crosses into the results plane.",
    term: {
      word: "CapturedRecord",
      meaning: "Finalized per-request observation assembled from arrival through terminal events.",
    },
    points: [
      "Storage mode begins only after the record is finalized.",
      "One record spans arrival → admission → token → usage → terminal.",
      "Evidence: `rust/runtime/src/engine/records.rs`.",
    ],
    caption: "observe events → one CapturedRecord.",
  },
  {
    eyebrow: "Reduce · 16 of 20",
    title: "Retain rows for exact artifacts",
    lede:
      "The retain path preserves per-record data and merges it in a deterministic topology order. Retain costs O(records) but keeps raw record artifacts available, carried home as CellMessage::Partition.",
    term: {
      word: "RecordsShardPartition",
      meaning: "Retain-mode partition of raw CapturedRecords shipped over the cell wire.",
    },
    points: [
      "Retain costs O(records) and keeps raw record artifacts available.",
      "Records restore global dispatch order at merge.",
      "Evidence: `rust/runtime/src/cellular/shard.rs`, `engine/execute.rs`.",
    ],
    caption: "keep rows → ship Partition → global-order merge.",
  },
  {
    eyebrow: "Reduce · 17 of 20",
    title: "Exact-fold into ColumnStore",
    lede:
      "Each completed record folds into an exact mergeable store and the row is dropped. Exact-fold retains exact record-derived aggregates without retaining rows, trading raw artifacts for bounded memory.",
    term: {
      word: "RunCapture::fold_streaming",
      meaning: "Folds each finalized record into an exact ColumnStore, then drops the row.",
    },
    points: [
      "Exact-fold keeps exact aggregates without retaining rows.",
      "Float sums may differ by a ULP; distributions stay exact.",
      "Evidence: `rust/runtime/src/engine/execute.rs`.",
    ],
    caption: "fold each record → exact store · drop row.",
  },
  {
    eyebrow: "Reduce · 18 of 20",
    title: "Sketch into t-digest plus exact moments",
    lede:
      "Finite metric values stream into bounded sketches while counts and moments stay exact. Percentiles become approximate; counts, sums, extrema, mean, standard deviation, and rates remain exact — the k6-style bounded-memory path.",
    term: {
      word: "TagSketch",
      meaning: "t-digest per tag plus exact moments; approximate percentiles, no per-record artifacts.",
    },
    points: [
      "Percentiles approximate; counts, sums, extrema, mean, std, rates exact.",
      "Bounded memory; per-record artifacts are unavailable.",
      "Evidence: `rust/runtime/src/metrics_core/store.rs`, `cellular/sketch.rs`.",
    ],
    caption: "finite values → t-digest + exact moments.",
  },
  {
    eyebrow: "Scale · 19 of 20",
    title: "Merge, publish, and know the boundary",
    lede:
      "Flat star fan-in and hierarchical aggregator trees are both built; external stream-only aggregation remains planned. One authoritative report exists — through controller merge, record-lane artifacts, and cross-host shipping — unless a future external sink explicitly replaces it.",
    term: {
      word: "run_aggregator",
      meaning: "Fold-only tree tier merging bounded child store partitions before the controller merge.",
    },
    points: [
      "One authoritative report exists unless an external sink replaces it.",
      "Flat and tree merges built; scheduled infinite budgets rejected.",
      "Evidence: `rust/runtime/src/engine/cellular_aggregator.rs`, `record_lane.rs`, `artifact_shipping.rs`.",
    ],
    caption: "flat + tree merge → artifacts + report · external planned.",
  },
  {
    eyebrow: "Scale · 20 of 20",
    title: "The full cellular system atlas",
    lede:
      "Every plane assembled: control coordination, data distribution, execution, and results reduction. The complete atlas preserves the measurement contract end to end. For the interactive recipe explorer and ability map, open the source canvas.",
    term: {
      word: "measurement contract",
      meaning: "The invariant that scaling to many cells never changes what a single-cell run would have measured.",
    },
    points: [
      "Control · data · execution · results planes assembled in one view.",
      "The complete atlas preserves the measurement contract.",
      "Full explorer: `docs/canvases/cellular-architecture.canvas.tsx`.",
    ],
    caption: "all planes together · one measurement contract.",
  },
] as const;

const NARRATION = [
  "Cellular scale keeps one benchmark identity while autonomous processes share the work. Scaling changes placement, not the measurement contract: one authored run, many cells, one report.",
  "The human-facing command resolves exactly one strict benchmark request before any child exists. Every cell later derives from that same resolved source of truth.",
  "The entry point launches the same aiperf image in a hidden execution mode over stdio. Process isolation is achieved through the protocol-v2 self-exec boundary, not a second executable.",
  "Single-cell execution stays direct. At more than one cell, one process is promoted to a coordinator that partitions and launches but never dispatches inference load itself.",
  "Eligibility is checked before cells launch. Unsupported transport, budget, storage, and workload combinations are rejected up front so a run never degrades silently.",
  "Request and conversation budgets divide across cells without overlap or omission. The union of every cell's owned positions exactly tiles the global budget by modulo.",
  "Children become ready before the controller releases synchronized execution. No synchronized cell dispatches before the registration barrier opens.",
  "Phaser and barrier-free start are opt-in alternatives, and a shared origin can zero cell clocks at the barrier. Start policy and timing epoch change coordination, never ownership.",
  "The Velo overlay verifies request fan-out while Stage G serves cross-host inputs over HTTP and zstd. Every cell receives only its owned input identities.",
  "Each cell tracks indexed, in-flight, and completed request identities. A request identity belongs to exactly one cell.",
  "A cell owns its runtime, transport state, metrics, and lifecycle, sharing no hot-path collector lock. Deterministic ownership connects directly to an autonomous process.",
  "Inside a cell the same ownership rule applies across thread-per-core workers. Nested ownership stays deterministic and disjoint.",
  "Local worker progress maps back to one global order. The ordinal is phase base plus local index times cell count plus cell id.",
  "Every shard drives the same clock-injected HTTP or gRPC seam. Scheduled gRPC cellular is proven, graph plus gRPC is unproven, and DynoSim cellular fails closed.",
  "Arrival, admission, token, usage, and terminal observations become one captured record. Storage mode begins only after the record is finalized.",
  "The retain path preserves per-record data and merges it in deterministic topology order. It costs order records but keeps raw artifacts available.",
  "Each completed record folds into an exact mergeable store and the row is dropped. Exact-fold keeps exact aggregates without retaining rows.",
  "Finite metric values stream into bounded sketches while counts and moments stay exact. Percentiles are approximate; everything else remains exact.",
  "Flat and hierarchical merges are both built while external stream-only aggregation is planned. One authoritative report exists through merge, artifacts, and shipping.",
  "The full atlas assembles the control, data, execution, and results planes in one view. The complete picture preserves the measurement contract; open the source canvas for the interactive explorer.",
] as const;

export const SLIDES: readonly SlideDefinition[] = STEPS.map((slide, index) => ({
  ...slide,
  narration: NARRATION[index] ?? "",
}));
