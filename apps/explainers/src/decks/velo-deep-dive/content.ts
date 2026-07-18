import type { SlideDefinition } from "../../core/types";

type StepInput = Omit<SlideDefinition, "narration">;

const STEPS: readonly StepInput[] = [
  {
    eyebrow: "1 · Connection",
    title: "Resolve the controller over Velo",
    lede:
      "A cell starts from one known coordinate: a TCP or UDS endpoint. The Velo hello exchange turns that address into routable controller PeerInfo, and the cell registers the peer before any protocol traffic flows.",
    term: {
      word: "register_peer",
      meaning: "Velo call that records a resolved coordinate as an addressable peer so later unary and active-message calls can route to it.",
    },
    points: [
      "SLURM derives the coordinate from `SLURM_*`; Kubernetes from the operator address.",
      "`_hello` establishes the return route before `register_peer(controller)`.",
      "Evidence: `rust/runtime/src/cellular/transport/connect.rs`, `velo_transport.rs`.",
    ],
    caption: "endpoint → _hello → register_peer(controller).",
  },
  {
    eyebrow: "2 · Registration",
    title: "The controller answers with the run envelope",
    lede:
      "Each cell sends a CellRegister carrying its identity. The controller decodes it, registers the cell peer for the return path, looks up the pre-sliced protocol-v2 envelope for that cell id, and returns it inside a RegisterReply alongside the run-wide START handle.",
    term: {
      word: "RegisterReply",
      meaning: "Unary reply carrying the cell's sliced protocol-v2 envelope bytes plus the shared start EventHandle.",
    },
    points: [
      "`register_peer(cell)` fixes the controller→cell return route.",
      "`spec_for(cell_id)` is a pure lookup into the pre-partitioned envelopes.",
      "Reply = envelope bytes + start_event EventHandle; identity is `cell_id: u32`.",
    ],
    caption: "decode CellRegister → spec_for(cell_id) → RegisterReply.",
  },
  {
    eyebrow: "3 · START barrier",
    title: "Asynchronous arrival, one release",
    lede:
      "Cells register at different times, but synchronized START holds them at a barrier. The Nth registration satisfies the controller's all-registered condition; triggering the shared start event wakes every awaiting cell together, so no cell dispatches early.",
    term: {
      word: "all_registered → trigger",
      meaning: "The controller counts registrations and fires the shared start event only when every cell is ready.",
    },
    points: [
      "Default START is synchronized; barrier-free and phaser are opt-in policies.",
      "The barrier bounds cross-cell start skew without a shared wall clock.",
      "Evidence: `rust/runtime/src/engine/cellular_controller.rs`.",
    ],
    caption: "N registrations → trigger → all awaiters go Ready.",
  },
  {
    eyebrow: "4 · MessagePack",
    title: "Typed state becomes raw payload bytes",
    lede:
      "Velo carries opaque bytes, so cellular messages serialize through rmp-serde. MessagePack preserves NaN and infinity that metric aggregates depend on, and the receiving handler reconstructs the exact typed CellMessage from the slice.",
    term: {
      word: "rmp_serde",
      meaning: "MessagePack serde codec: `to_vec` on the sender, `from_slice` on the handler, with finite-and-non-finite floats intact.",
    },
    points: [
      "Heartbeats and partitions both ride the same encode/decode seam.",
      "NaN and ±∞ survive the round trip so extrema anchors stay correct.",
      "Evidence: `rust/runtime/src/cellular/transport/mod.rs` CellMessage codec.",
    ],
    caption: "to_vec → raw payload → from_slice → typed value.",
  },
  {
    eyebrow: "5 · Heartbeat",
    title: "Fire-and-forget liveness and live metrics",
    lede:
      "Cells push MetricsHeartbeat snapshots without waiting for a reply. Each snapshot carries counters, mergeable sketches, and an observation time, so the controller reads live progress and detects a lagging or missing cell before the run ends.",
    term: {
      word: "MetricsHeartbeat",
      meaning: "Periodic cell→controller snapshot of counts and sketches used for a live cross-cell aggregate; enabling the lane forces retain.",
    },
    points: [
      "Fire-and-forget: heartbeats never block the dispatch hot path.",
      "A stalled channel surfaces as rising lag and a missing pulse.",
      "Evidence: `rust/runtime/src/cellular/heartbeat.rs`.",
    ],
    caption: "snapshot counters + sketches → live aggregate · lag visible.",
  },
  {
    eyebrow: "6 · Partition",
    title: "Ship the terminal partition home",
    lede:
      "When a cell finishes, it sends a terminal unary payload containing a fresh shipper PeerInfo plus its records or folded-store partition. The controller registers that shipper peer first, then returns a CellAck, so delivery is confirmed end to end.",
    term: {
      word: "CellPartitionShip",
      meaning: "Terminal payload carrying the shipping instance's own coordinate plus its RecordsShardPartition or ColumnStorePartition.",
    },
    points: [
      "The fresh return route is registered before the controller acknowledges.",
      "Retain ships raw rows; fold ships an exact or sketch store partition.",
      "Evidence: `rust/runtime/src/cellular/transport/mod.rs` CellMessage::Partition.",
    ],
    caption: "ship peer + partition → register_peer(shipper) → CellAck.",
  },
  {
    eyebrow: "7 · Merge",
    title: "Feed the associative reduction center",
    lede:
      "The controller merges every arriving partition. Retained records restore one global dispatch order by ordinal; folded stores append exact algebra and merge approximate t-digests. Both paths collapse many cell inputs into one authoritative result.",
    term: {
      word: "merge_store_partitions",
      meaning: "Associative fold that appends exact aggregates and merges sketches across cell store partitions.",
    },
    points: [
      "Records mode: sort by global ordinal to reconstruct dispatch order.",
      "Store mode: exact counts, sums, and extrema; percentiles via t-digest.",
      "Evidence: `rust/runtime/src/cellular/shard.rs`.",
    ],
    caption: "records → global-order sort · stores → append + merge.",
  },
  {
    eyebrow: "8 · Phaser",
    title: "Replay the prefix, then live push",
    lede:
      "The monotonic phaser broadcasts generation events for opt-in distributed START. When a subscriber attaches, the current generation prefix returns in the unary reply as replay; only later generations arrive by active-message push, so no event is missed or duplicated.",
    term: {
      word: "PhaserServer / PhaserClient",
      meaning: "Velo phaser transport that captures the current generation on attach and streams later generations as live pushes.",
    },
    points: [
      "Attach captures a generation boundary; everything ≤ it is replayed.",
      "Generations strictly increase, so ordering is deterministic across cells.",
      "Evidence: `rust/runtime/src/cellular/transport/phaser_velo.rs`, `phaser.rs`.",
    ],
    caption: "attach → reply replay ≤ g · live push > g.",
  },
  {
    eyebrow: "9 · Dataset",
    title: "Broadcast once, retain by modulo",
    lede:
      "The opt-in dataset overlay publishes input chunks once. MessagePack plus zstd carries the prefix already published at attach time as replay and every later chunk as live push. Each cell keeps only the identities where request_id % cells equals its cell id.",
    term: {
      word: "DatasetIndex",
      meaning: "Cell-local index that accepts replayed and live chunks but retains only its owned request identities.",
    },
    points: [
      "One broadcast fans out to all cells; ownership filters on receipt.",
      "Replay-on-attach plus live tail mirrors the phaser subscription model.",
      "Evidence: `rust/runtime/src/cellular/transport/dataset_velo.rs`, `dataset_session.rs`.",
    ],
    caption: "publish chunk → replay + live · keep id % cells == cell_id.",
  },
  {
    eyebrow: "10 · Aggregators",
    title: "Collapse payload up a fold-only tree",
    lede:
      "Folded stores can reduce through an aggregator tier: cells ship bounded store partitions to aggregators, which merge subtrees before the controller. Retained raw records stay flat so the controller can still restore one global dispatch order.",
    term: {
      word: "run_aggregator",
      meaning: "Fold-only tree tier that merges bounded child store partitions into one partition per subtree.",
    },
    points: [
      "Tree merge requires fold mode; retain partitions cannot enter a subtree.",
      "Flat star fan-in remains the fallback and the retain-mode path.",
      "Evidence: `rust/runtime/src/engine/cellular_aggregator.rs`.",
    ],
    caption: "fold: cells → aggregators → controller · retain stays flat.",
  },
] as const;

const NARRATION = [
  "A cell begins from one known coordinate. The Velo hello exchange resolves it into routable controller PeerInfo, and the cell registers that peer before any protocol traffic.",
  "Each cell sends a CellRegister. The controller decodes its identity, registers the return route, looks up the pre-sliced protocol-v2 envelope, and replies with it plus the shared START handle.",
  "Cells register asynchronously but wait at the synchronized barrier. The Nth registration releases the controller, and triggering the shared start event wakes every cell together.",
  "Velo carries opaque bytes, so messages serialize through MessagePack. NaN and infinity survive the round trip, and the handler reconstructs the exact typed value from the slice.",
  "Cells push fire-and-forget heartbeat snapshots of counters and sketches. The controller reads live progress and sees a lagging or missing cell as rising lag and a dropped pulse.",
  "A finished cell ships a terminal payload with a fresh return peer and its partition. The controller registers that shipper before returning an acknowledgement.",
  "The controller merges every partition. Retained records restore one global dispatch order; folded stores append exact algebra and merge approximate t-digests.",
  "The monotonic phaser broadcasts generations for opt-in distributed start. Attaching replays the current prefix in the reply, and later generations arrive by live push.",
  "The dataset overlay broadcasts input chunks once over MessagePack and zstd. Each cell replays the published prefix, follows the live tail, and keeps only its modulo-owned identities.",
  "Folded stores can reduce through an aggregator tree that merges subtrees before the controller. Retained raw records stay flat so global dispatch order can still be restored.",
] as const;

export const SLIDES: readonly SlideDefinition[] = STEPS.map((slide, index) => ({
  ...slide,
  narration: NARRATION[index] ?? "",
}));
