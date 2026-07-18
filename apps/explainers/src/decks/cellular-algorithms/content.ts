/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SlideDefinition } from "../../core/types";

type StepInput = Omit<SlideDefinition, "narration">;

const STEPS: readonly StepInput[] = [
  {
    eyebrow: "Workbook · 1 of 16",
    title: "Read the algorithm workbook as a route",
    lede:
      "The workbook is not a second architecture atlas. It is an evidence-indexed catalog of executable decisions, ordered from admission through terminal artifacts. This deck follows that route while selecting representative algorithms from the exhaustive catalog.",
    term: {
      word: "algorithm route",
      meaning: "The ordered algorithms selected by one cellular run shape.",
    },
    points: [
      "Eight chapters: eligibility, ownership, control, distribution, execution, capture, merge, artifacts.",
      "Each entry records inputs, outputs, state, invariants, gates, failures, pseudocode, and evidence.",
      "This 16-slide deck is a maintainer map, not the full catalog.",
    ],
    caption: "eight chapters · one derived route.",
  },
  {
    eyebrow: "Workbook · 2 of 16",
    title: "Status and evidence are part of the contract",
    lede:
      "A colored node is never enough to communicate implementation state. Built means the path exists on the ordinary route, feature-gated means code exists behind build or runtime admission, and partial means a known fidelity or lifecycle boundary remains. Every status stays visible as text.",
    term: {
      word: "evidence vocabulary",
      meaning: "Boundary, unit, integration, and end-to-end references attached to an algorithm.",
    },
    points: [
      "Built — executable on the admitted route.",
      "Feature-gated — executable only when its explicit gate is present.",
      "Partial — implemented with a named limitation or omitted fidelity.",
    ],
    caption: "status labels · source-backed evidence.",
  },
  {
    eyebrow: "Eligibility · 3 of 16",
    title: "Fail closed before cells launch",
    lede:
      "Eligibility begins at execution-mode-dispatch, promotes a multi-cell request with controller-promotion, then validates run kind, budgets, storage, and artifacts. velo-feature-admission is explicitly feature-gated: a lean binary reports an unsupported role instead of silently running direct.",
    term: {
      word: "cellular-run-shape-validation",
      meaning: "Pre-launch admission for supported transport, dataset, workload, and ownership combinations.",
    },
    points: [
      "execution-mode-dispatch and controller-promotion are built.",
      "velo-feature-admission is feature-gated.",
      "execution-merge-backstops revalidates the payload actually returned by cells.",
    ],
    caption: "dispatch → promote → validate → admit.",
  },
  {
    eyebrow: "Ownership · 4 of 16",
    title: "Tile work without gaps or collisions",
    lede:
      "modulo-cell-ownership assigns each global position to exactly one cell, while owned-positions-tiling proves the uneven slices still cover the whole budget. conversation-ownership advances a shared deterministic draw sequence and yields only the positions owned by that cell.",
    term: {
      word: "modulo-cell-ownership",
      meaning: "Cell k owns global position i exactly when i modulo cell_count equals k.",
    },
    points: [
      "Request, graph-instance, and conversation ownership remain disjoint.",
      "cellular-issuance-authority stamps interleaved global ordinals.",
      "A conversation remains wholly owned; turns never split across cells.",
    ],
    caption: "one position · exactly one owner.",
  },
  {
    eyebrow: "Control · 5 of 16",
    title: "Coordinate generations and process state",
    lede:
      "broadcast-attach-replay closes the replay-plus-live race, and phaser-generation-advance makes progress monotonic. synchronized-start is the default Velo-gated rendezvous; phaser-start is an opt-in feature-gated path. The gate changes coordination, never ownership.",
    term: {
      word: "broadcast-attach-replay",
      meaning: "Attach under the broadcast seam so concurrent data appears in replay or live delivery, never neither.",
    },
    points: [
      "Replay and live delivery meet at one locked seam.",
      "phaser-await-generation uses a greater-than-or-equal gate.",
      "controller-child-arbitration surfaces hard child failure.",
    ],
    caption: "replay seam · generations · START.",
  },
  {
    eyebrow: "Distribution · 6 of 16",
    title: "Deliver or regenerate only owned work",
    lede:
      "canonical-dataset-regeneration remains the measured path. controller-fanout-generation is a feature-gated verification overlay that publishes chunks and builds owned indexes; it never replaces canonical execution data. distribution-miss counts and surfaces an unindexed issue instead of skipping it.",
    term: {
      word: "distribution-miss",
      meaning: "A counted DispatchDecision::Miss that leaves ownership state recoverable.",
    },
    points: [
      "dataset-velo-replay-live provides compressed replay plus live chunks.",
      "dataset-http-zstd-reconstruct handles cross-host Stage G inputs.",
      "A verification overlay proves delivery without becoming measurement authority.",
    ],
    caption: "canonical path · optional verification overlay.",
  },
  {
    eyebrow: "Execution · 7 of 16",
    title: "Nest cell and worker shards",
    lede:
      "two-level-partition first assigns work to cells and then to thread-per-core workers. scheduled-shard-runtime runs each worker-local sink, while issuance-dispatch-injection carries the global ordinal authority into capture. Graph and scheduled programs branch only after the owned envelope arrives.",
    term: {
      word: "two-level-partition",
      meaning: "Deterministic cell ownership followed by disjoint worker ownership inside each cell.",
    },
    points: [
      "partitioned-scheduled-sampler and partitioned-graph-source preserve workload semantics.",
      "cell-envelope-fetch hands an autonomous cell its resolved slice.",
      "Only execution workers dispatch inference load.",
    ],
    caption: "global run → cell slice → worker shard.",
  },
  {
    eyebrow: "Capture · 8 of 16",
    title: "Choose retain, exact fold, or sketch",
    lede:
      "terminal-record-finalization closes one observation stream. retain-record-capture keeps rows in dispatch order; streaming-exact-fold appends exact aggregates and drops rows; tagged-sketch-merge bounds distribution memory while retaining exact counts and moments. Storage choice determines later artifact authority.",
    term: {
      word: "terminal-record-finalization",
      meaning: "The point where observations become one complete CapturedRecord.",
    },
    points: [
      "Retain preserves row-level artifacts.",
      "Exact fold drops rows but preserves exact record-derived aggregates.",
      "Sketch bounds memory; percentile interiors are approximate.",
    ],
    caption: "finalize once · select storage fidelity.",
  },
  {
    eyebrow: "Merge · 9 of 16",
    title: "Reduce under an explicit merge law",
    lede:
      "scheduled-global-ordinal-merge validates a dense permutation and re-ingests retained rows in global order. Graph retain uses deterministic concatenation and renumbering. exact-fold-store-merge and sketch-tdigest-merge reduce bounded stores, while final-report-assembly remains partial because later artifact failure cannot roll back earlier exports.",
    term: {
      word: "scheduled-global-ordinal-merge",
      meaning: "Validate, sort, and re-ingest retained scheduled records by global request_index.",
    },
    points: [
      "Duplicate, missing, and out-of-range ordinals fail before accumulation.",
      "Flat controller and hierarchical aggregator reductions share explicit topology.",
      "merged-report-fidelity-boundary documents what each storage mode can reproduce.",
    ],
    caption: "partition set → validated reduction → report.",
  },
  {
    eyebrow: "Artifacts · 10 of 16",
    title: "Move bytes safely and choose one outcome",
    lede:
      "artifact-authority-allowlist is feature-gated and admits only configured relative paths. artifact-http-zstd-upload streams Stage E files, partial-file-atomic-replace avoids publishing torn output, and artifact-completion-barrier waits for required uploads. telemetry-drop-warning and cancellation-propagation remain partial boundaries.",
    term: {
      word: "artifact-completion-barrier",
      meaning: "Controller gate that waits for the expected cross-host artifact set.",
    },
    points: [
      "Path authority is allowlisted before filesystem join.",
      "Temporary files become authoritative only through atomic replacement.",
      "child-exit-arbitration, controller-timeout, and terminal-failure-envelope select one terminal result.",
    ],
    caption: "allowlist → upload → atomic publish → barrier.",
  },
  {
    eyebrow: "Composition · 11 of 16",
    title: "Compose the canonical scheduled route",
    lede:
      "A scheduled exact-fold route composes admission, modulo ownership, synchronized control, canonical data regeneration, nested execution, streaming exact fold, store merge, and final reporting. Route composition is ordered; selecting one policy can add or replace algorithms without changing unrelated invariants.",
    term: {
      word: "scheduled-exact-fold",
      meaning: "Canonical bounded-memory recipe with exact aggregate storage.",
    },
    points: [
      "Selectors derive an ordered algorithm route, not a bag of features.",
      "Shared algorithms remain stable across route variants.",
      "Validation gates stop composition at the first unsupported boundary.",
    ],
    caption: "selectors → ordered algorithms → artifacts.",
  },
  {
    eyebrow: "Composition · 12 of 16",
    title: "Gates add explicit route branches",
    lede:
      "Phaser START, dataset fan-out verification, external aggregation, and cross-host artifact shipping are feature-gated route branches. final-report-assembly and telemetry-drop-warning are partial boundaries, not hidden green paths. The status label travels with the algorithm wherever it appears.",
    term: {
      word: "route branch",
      meaning: "Algorithms added or replaced by an admitted selector and its environment.",
    },
    points: [
      "Built: canonical ownership, execution, capture, and core merge.",
      "Feature-gated: Velo roles, phaser, fan-out, and Stage E/G paths.",
      "Partial: report rollback, side-channel aggregation, and cancellation scope.",
    ],
    caption: "built · feature-gated · partial.",
  },
  {
    eyebrow: "Decisions · 13 of 16",
    title: "Dataset shape decides the run kind",
    lede:
      "The decision log compares routes while preserving their shared bands. A graph selector with a synthetic dataset still resolves to scheduled execution, while a WEKA or Dynamo trace resolves to graph execution. Dataset classification, not the authored selector alone, chooses the effective runtime branch.",
    term: {
      word: "run-kind-classification",
      meaning: "Classifies scheduled versus recorded-graph work from the resolved dataset shape.",
    },
    points: [
      "Synthetic and ordinary file/public datasets follow scheduled semantics.",
      "DAG JSONL, WEKA, and Dynamo traces follow graph semantics.",
      "Budget validation follows the effective run kind.",
    ],
    caption: "authored intent → dataset classification → runtime.",
  },
  {
    eyebrow: "Decisions · 14 of 16",
    title: "Storage and conversation choices change fidelity",
    lede:
      "Exact fold and sketch both drop completed rows, but sketch trades percentile interior fidelity for request-count-independent memory. Single-turn work owns requests; multi-turn work owns complete conversations. These choices alter capture and ownership algorithms while preserving the one-owner invariant.",
    term: {
      word: "storage-fidelity decision",
      meaning: "Select exact fold or sketch before capture and artifact validation.",
    },
    points: [
      "Sketch cannot authorize row-dependent raw artifacts.",
      "Multi-turn ownership selects conversation-ownership.",
      "No route may split one conversation across cells.",
    ],
    caption: "storage fidelity · ownership granularity.",
  },
  {
    eyebrow: "Decisions · 15 of 16",
    title: "Placement decides topology and shipping",
    lede:
      "Same-host routes can concatenate local files directly. Cross-host placement adds Stage G input reconstruction and Stage E artifact upload around the same execution route. A local-tree intent falls back flat without operator wiring, while an external tree requires its DNS tier and explicit placement admission.",
    term: {
      word: "aggregation-topology decision",
      meaning: "Choose flat, local tree, or externally placed tree under deployment constraints.",
    },
    points: [
      "Cross-host tree wiring is an admission decision, not best-effort discovery.",
      "Parquet emission depends on a parquet-bearing build and remains text-labeled.",
      "Shipping changes byte placement, not the measurement contract.",
    ],
    caption: "deployment → topology + Stage E/G.",
  },
  {
    eyebrow: "Reference · 16 of 16",
    title: "Use the source canvas for exhaustive lookup",
    lede:
      "This deck preserves the route grammar, representative algorithm names, status semantics, and key decisions. It intentionally does not reproduce the workbook's roughly one hundred algorithm pages, trace frames, recipes, validation fixtures, or selector interactions. Maintainers should use the source canvas for exhaustive evidence lookup.",
    term: {
      word: "source of exhaustive lookup",
      meaning: "docs/canvases/cellular-algorithm-workbook.canvas.tsx",
    },
    points: [
      "Search by algorithm ID, chapter, status, actor, source path, or route tag.",
      "Inspect pseudocode, invariants, failures, and evidence before changing behavior.",
      "Exhaustive catalog: `docs/canvases/cellular-algorithm-workbook.canvas.tsx`.",
    ],
    caption: "maintainer map → exhaustive source canvas.",
  },
] as const;

const NARRATION = [
  "The cellular algorithm workbook is an evidence-indexed catalog, not another architecture atlas. Read it as a route through eight chapters: eligibility, ownership, control, distribution, execution, capture, merge, and artifacts. This deck picks representative algorithms; the source canvas remains the exhaustive lookup.",
  "Status is part of the contract and must be readable without color. Built means the path executes on an admitted route. Feature-gated means code exists behind a build or runtime gate. Partial names a real limitation. Each algorithm also links boundary, unit, integration, or end-to-end evidence.",
  "Eligibility fails closed before any cell launches. execution-mode-dispatch enters the private protocol, controller-promotion recognizes a multi-cell request, and run-shape and budget validators admit only supported combinations. velo-feature-admission is feature-gated, so a lean build reports an error rather than silently degrading.",
  "Ownership tiles the authored budget deterministically. modulo-cell-ownership gives every position exactly one cell, and owned-positions-tiling accounts for uneven slices. conversation-ownership uses the same deterministic draw sequence while returning only owned conversations, so no turn crosses a cell boundary.",
  "The control plane joins replay and live delivery without a race, then advances monotonic generations. synchronized-start is the default Velo-gated rendezvous, while phaser-start is an opt-in feature-gated branch. These choices change coordination timing, never which cell owns the work.",
  "Canonical regeneration remains the measured data path. The feature-gated controller fan-out is a verification overlay that publishes compressed chunks and builds owned indexes. It never becomes measurement authority. If dispatch asks for an unindexed request, distribution-miss counts and surfaces it rather than silently skipping.",
  "Execution nests two deterministic partitions. First the controller assigns a slice to each cell; then thread-per-core workers divide that slice again. The worker-local runtime dispatches through the ordinary sink, while issuance authority carries stable global identity into record capture and later merge.",
  "Capture begins only after terminal record finalization. Retain keeps every row and its artifact authority. Exact fold appends exact record-derived aggregates and drops the row. Sketch also drops rows and bounds memory, while preserving exact counts and moments and approximating percentile interiors.",
  "Merge law follows workload and storage. Scheduled retain validates a dense global ordinal permutation before re-ingestion; graph retain concatenates deterministically and renumbers. Folded and sketch stores reduce without rows. Final report assembly is labeled partial because a later artifact failure cannot undo exports already written.",
  "Artifacts have their own authority and failure protocol. An allowlist admits only configured relative paths, streaming upload writes a partial file, atomic replacement publishes it, and a completion barrier waits for the required set. Side-channel aggregation and cancellation scope remain explicitly partial.",
  "A route composer turns selectors into an ordered algorithm path. The canonical scheduled exact-fold route combines admission, modulo ownership, synchronized start, canonical data, nested execution, exact folding, store merge, and final reporting. Unsupported combinations stop at the first validation gate.",
  "Optional branches keep their status labels. Phaser start, Velo fan-out, external aggregation, and cross-host shipping are feature-gated. Report rollback, side-channel aggregation, and cancellation scope are partial. Canonical ownership, execution, and core reduction are built.",
  "The first decision is effective run kind. Dataset shape wins over the authored workload selector: synthetic data follows scheduled execution, while DAG JSONL, WEKA, and Dynamo traces select graph execution. Budget and merge validation then follow that effective classification.",
  "Storage and ownership choices change different parts of the route. Exact fold and sketch both remove completed rows, but sketch bounds memory by approximating percentile interiors. Single-turn work owns requests, while multi-turn work owns whole conversations. The one-owner invariant remains unchanged.",
  "Placement adds topology and shipping decisions around the execution route. Cross-host runs reconstruct inputs through Stage G and upload artifacts through Stage E. External trees require operator wiring; without it, local-tree intent falls back flat. Shipping changes byte placement, not measurement meaning.",
  "This deck is intentionally a maintainer map, not a one-to-one copy of roughly one hundred workbook pages. For every algorithm, trace frame, validation fixture, route recipe, selector interaction, and source reference, open docs slash canvases slash cellular algorithm workbook dot canvas dot tsx.",
] as const;

export const SLIDES: readonly SlideDefinition[] = STEPS.map((slide, index) => ({
  ...slide,
  narration: NARRATION[index] ?? "",
}));
