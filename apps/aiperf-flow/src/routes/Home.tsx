/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { Link } from "react-router-dom";
import { TopBar } from "../shell/TopBar.js";
import { Grid } from "../layout/Grid.js";
import { Stack } from "../layout/Stack.js";
import { accentClassName, inkClassName, strokeClassName, surfaceClassName } from "../theme/tokens.js";

export type DeckListing = {
  path: string;
  title: string;
  description: string;
};

/** Every browsable deck. Add an entry here whenever a new deck route is wired into `App.tsx`. */
export const DECKS: readonly DeckListing[] = [
  {
    path: "/python-graph-workload",
    title: "Python Graph Workload",
    description:
      "Narrated walkthrough of the Python plane: Dynamo trace ingest into a session forest, the four-stage workload build, idle warp, and the asyncio dataflow executor with its dual fan-out and parked-Future returns.",
  },
  {
    path: "/async-dataflow-engine",
    title: "How the Async Dataflow Engine Works",
    description:
      "Narrated autoplay walkthrough of the native Graph Workload engine: why scheduling and readiness are different mechanisms, the check-then-park pattern that makes it race-free, version-frozen reads, and why simulation cannot run thread-per-core. Press play and it narrates itself.",
  },
  {
    path: "/native-diagram-vocabulary",
    title: "What Changed: the native diagram vocabulary",
    description:
      "Watch one dataset become progressively more expressible. Six requests start as six identical boxes, then each new node type lands in turn — timeline, intervals, blocks, sweep, slices, ragged — re-reading the same data until every claim a box could not make is simply visible. Ends with what adding the next one takes.",
  },
  {
    path: "/metrics-plane",
    title: "The Metrics Plane",
    description:
      "Narrated walkthrough of the three data-shape decisions behind the metrics accumulator: the sweep line that turns intervals into exact curves without scanning time, the flat ragged layout that holds variable-length list metrics, and the uniform slice grid whose trailing bucket is clipped rather than diluted.",
  },
  {
    path: "/segment-pools",
    title: "Segment Pools",
    description:
      "Content-addressed interning, freeze-to-store, and BodyPlan materialization — a six-page walkthrough with two live simulators, ported from a real Cursor Canvas.",
  },
  {
    path: "/rust-aiperf-architecture",
    title: "Rust AIPerf Architecture",
    description:
      "One binary, two roles — eleven pages spanning process boundaries, runtime seams, scheduled/graph workloads, cellular scaling, and extension points.",
  },
  {
    path: "/rust-port-why",
    title: "Why Rust · Executive Overview",
    description:
      "A management-facing pitch for reimplementing AIPerf in Rust: the shared-core case with Dynamo, retiring the GIL-driven multiprocess control plane, and an honest read on where the performance win is real (and where it isn't).",
  },
  {
    path: "/rust-port-flow",
    title: "Rust Port · Request Lifecycle",
    description:
      "One request's life through the Rust port as a single zoomable canvas: drill into any of nine pipeline stages, play an animated request through, and flip the Clock and Transport seams live.",
  },
  {
    path: "/aiperf-graph-engine",
    title: "AIPerf Graph Engine",
    description: "The async-dataflow graph engine: node groups, control flow, and how graph programs execute end to end.",
  },
  {
    path: "/aiperf-metrics-accumulator",
    title: "AIPerf Metrics Accumulator",
    description: "Record, aggregate, derived, and sweep metric computation — how raw observations become a report.",
  },
  {
    path: "/canvas-repo-layout",
    title: "Canvas Repo Layout",
    description: "A tour of the repository's crate and module layout, mapped as a navigable diagram.",
  },
  {
    path: "/cellular-algorithm-workbook",
    title: "Cellular Algorithm Workbook",
    description: "Interactive workbook for the cellular partitioning and merge algorithms.",
  },
  {
    path: "/cellular-architecture",
    title: "Cellular Architecture",
    description: "Controller/cell topology, partitioning, and folded-store merge for cross-process execution.",
  },
  {
    path: "/claude-code-subagent-stepper",
    title: "Claude Code Subagent Stepper",
    description: "A stepped simulation of subagent dispatch, arrival, and completion during agentic development.",
  },
  {
    path: "/dynosim-offline-flow",
    title: "DynoSim Offline Flow",
    description: "Socket-free Dynamo co-simulation: how offline replay drives the runtime without a live server.",
  },
  {
    path: "/graph-fan-in",
    title: "Graph Fan-In",
    description: "Fan-in semantics in the graph engine — how multiple upstream branches join at a single node.",
  },
  {
    path: "/graph-step-emit-strategy",
    title: "Graph Step/Emit Strategy",
    description: "How graph nodes step and emit credits, and the strategies governing dispatch order.",
  },
  {
    path: "/graph-subsystem-overview",
    title: "Graph Subsystem Overview",
    description: "A top-level map of the graph subsystem: compilation, segment storage, policies, and execution.",
  },
  {
    path: "/mocker-clock-inversion",
    title: "Mocker Clock Inversion",
    description: "How the mock server's clock model inverts relative to real-time execution for deterministic replay.",
  },
  {
    path: "/mock-server-architecture",
    title: "Mock Server Architecture",
    description: "aiperf-mock-server's request handling, latency models, error injection, and usage accounting.",
  },
  {
    path: "/offline-cosimulation",
    title: "Offline Co-Simulation",
    description: "Socket-free Dynamo co-simulation architecture, from SimClock to deterministic event ordering.",
  },
  {
    path: "/rust-architecture-internals",
    title: "Rust Architecture Internals",
    description: "A deeper look at the native runtime's internal seams: clock, dispatch, and transport placement.",
  },
  {
    path: "/slurm-architecture",
    title: "SLURM Architecture",
    description: "The native SLURM path: controller/cell topology derived from an srun/sbatch allocation.",
  },
  {
    path: "/slurm-explained-step-by-step",
    title: "SLURM, Step by Step",
    description: "A guided, step-by-step walkthrough of how `aiperf slurm run` launches a cross-host cellular run.",
  },
  {
    path: "/step-dispatch-emit-system",
    title: "Step/Dispatch/Emit System",
    description: "The core dispatch loop shared across graph execution: step, dispatch, and emit in sequence.",
  },
  {
    path: "/upcoming-async-dataflow",
    title: "Upcoming Async Dataflow",
    description: "Forward-looking design for the next generation of the async-dataflow execution model.",
  },
  {
    path: "/velo-in-aiperf",
    title: "Velo in AIPerf",
    description: "How the Velo framework provides cross-host cell transport for cellular execution.",
  },
  {
    path: "/weka-ingest-pipeline",
    title: "Weka Ingest Pipeline",
    description: "How WEKA trace data is ingested, parsed, and resolved into graph programs.",
  },
  {
    path: "/weka-runtime-stepper",
    title: "Weka Runtime Stepper",
    description: "An interactive stepper over WEKA trace execution inside the runtime.",
  },
  {
    path: "/weka-segment-store",
    title: "Weka Segment Store",
    description: "Content-addressed segment storage for WEKA graph inputs, from BLAKE3 identifiers to materialization.",
  },
  {
    path: "/weka-timing-causality",
    title: "Weka Timing Causality",
    description: "How WEKA trace timing establishes causal ordering across dependent requests.",
  },
  {
    path: "/weka-timing-transforms",
    title: "Weka Timing Transforms",
    description: "The transforms applied to WEKA trace timing data before scheduling.",
  },
  {
    path: "/weka-timing-transforms-interactive",
    title: "Weka Timing Transforms (Interactive)",
    description: "An interactive, simulator-driven companion to the WEKA timing transforms walkthrough.",
  },
  {
    path: "/weka-trie-build",
    title: "Weka Trie Build",
    description: "How WEKA trace prefixes are built into a trie for prefix-reuse targeting.",
  },
];

/**
 * Live simulations, kept separate from the decks above.
 *
 * These are running models you steer rather than diagrams you read, and they are exploratory —
 * listing them in the same grid would misrepresent both. Narrated entries drive their picture from
 * the voice; free-running ones give you the controls instead.
 */
export const SIMULATIONS: readonly DeckListing[] = [
  {
    path: "/sweep-line-desk",
    title: "The Sweep-Line Desk (narrated)",
    description:
      "Every stage of the metrics plane on one dataset: columnar storage with NaN sentinels, the ragged CSR for inter-chunk latency, events becoming a curve, the collision tie-break that keeps touching intervals from double-counting, ICL-aware token arrivals, and then steady-state detection by threshold, CUSUM, MSER-5, consensus and a stationarity test that can veto the answer.",
  },
  {
    path: "/spike-exact-vs-sketch",
    title: "What survives being summarized",
    description:
      "A cellular run folds each cell's t-digest instead of pooling records. Counts, sums and extrema come back exactly right; percentiles come back close. Shows where the digest spends its resolution — finest at the tail, coarsest at the median — and the one distribution shape it handles badly.",
  },
  {
    path: "/spike-dispatch-modes",
    title: "The shortfall that only exists in the sum",
    description:
      "sharded gives each worker thread a fixed 1/workers share of the concurrency target; global gives them one shared pool. Every lane obeys its own cap in both, and only the aggregate curve shows that the sharded run holds less concurrency than it was asked for and takes longer to do identical work.",
  },
  {
    path: "/spike-prefix-identity",
    title: "Same bytes, different segment",
    description:
      "A segment's identity is a hash of its content and its parent's identity, so the same turn is a different segment at a different point in a conversation. Three conversations share a prefix, fork, and never rejoin — with the hash input shown field by field.",
  },
  {
    path: "/spike-two-clocks",
    title: "Two clocks, one workload",
    description:
      "The same tasks on RealClock and SimClock side by side. One waits out the gaps between events because real timers must; the other parks sleepers in a heap ordered by (at_ns, seq_no) and jumps straight to the next deadline. The right pane finishes while the left is still on its first sleep, and both report the same requests, tokens and latencies.",
  },
  {
    path: "/spike-check-then-park",
    title: "Check-then-park, and the race it ignores",
    description:
      "A reader checks its arrival count and parks on a notify that only wakes readers already waiting — the classic lost-wakeup setup, safe only because the engine is current-thread. Switch to a hypothetical multi-threaded runtime, open the window between the check and the park, and watch a reader park into a silence no later write will break.",
  },
  {
    path: "/spike-segments-narrated",
    title: "Dynamo trace → segment pool (narrated)",
    description:
      "A live Dynamo trace record with its KV block hashes lighting green as they turn out to be cached, messages interning into a dense arena under prefix-dependent identity, and then workers materializing request bodies back out of it by concatenating stored wires.",
  },
  {
    path: "/spike-segments",
    title: "Dynamo trace → segment pool (free-running)",
    description:
      "The same pipeline with the controls in your hands: step one message at a time, change speed, or reseed the trace.",
  },
  {
    path: "/spike-warp-narrated",
    title: "The idle-gap warp (narrated)",
    description:
      "One recorded session on two clocks at the same scale — the raw clock and the clock a runtime actually issues on — with the playheads separating as dead air is collapsed, and every bar the same width on both because service time is never compressed.",
  },
  {
    path: "/spike-warp",
    title: "The idle-gap warp (free-running)",
    description:
      "The same two clocks, with the idle cap on a slider. Drag it and watch the warped track shorten or grow back to match the recording.",
  },
  {
    path: "/spike-agents",
    title: "Agent session — lanes appear as they spawn",
    description:
      "A session growing in real time: a lane does not exist until something spawns it, bars have no right edge while they stream, and the dead air between turns accumulates in front of you rather than being summarised afterwards.",
  },
  {
    path: "/spike-lifecycle",
    title: "Request lifecycle — live",
    description:
      "Requests are born, contend for an admission gate, stream tokens as pulse rings, and die. Drag concurrency down to watch the queue build; the curve underneath is drawn from the same events moving the dots above it.",
  },
];

function DeckCard({ deck }: { deck: DeckListing }): React.JSX.Element {
  return (
    <Link
      to={deck.path}
      className={`group block rounded-xl border p-6 shadow-sm transition-colors hover:border-accent-primary hover:bg-surface-panel hover:shadow-md ${surfaceClassName("elevated")} ${strokeClassName("primary")}`}
    >
      <Stack gap={8}>
        <h2
          className={`text-lg font-semibold transition-colors group-hover:text-accent-primary ${inkClassName("primary")}`}
        >
          {deck.title}
        </h2>
        <p className={`text-sm ${inkClassName("secondary")}`}>{deck.description}</p>
      </Stack>
    </Link>
  );
}

/** Landing page: browse every deck currently wired into the app. */
export function Home(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Home" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className="mx-auto max-w-6xl px-10 py-12">
          <Stack gap={5} className="mb-12">
            <span className={`text-xs font-semibold uppercase tracking-[0.2em] ${accentClassName("primary")}`}>
              AIPerf
            </span>
            <h1 className={`text-4xl font-extrabold tracking-tight ${inkClassName("primary")}`}>
              Explainer decks
            </h1>
            <p className={`max-w-2xl text-sm ${inkClassName("secondary")}`}>
              Interactive diagrams and walkthroughs of AIPerf subsystems, built as plain React
              components on React Flow, Motion, and Tailwind — no custom DSL.
            </p>
          </Stack>
          <Grid columns={2} gap={20}>
            {DECKS.map((deck) => (
              <DeckCard key={deck.path} deck={deck} />
            ))}
          </Grid>

          <Stack gap={5} className="mt-16 mb-8">
            <span className={`text-xs font-semibold uppercase tracking-[0.2em] ${accentClassName("primary")}`}>
              Spikes
            </span>
            <h2 className={`text-3xl font-extrabold tracking-tight ${inkClassName("primary")}`}>
              Live simulations
            </h2>
            <p className={`max-w-2xl text-sm ${inkClassName("secondary")}`}>
              Running models rather than drawings: requests are born and die, lanes appear as they
              spawn, and curves are generated by the same events that move the objects above them.
              Nothing here is a replay. The narrated ones drive their picture from the voice; the
              rest hand you the controls.
            </p>
          </Stack>
          <Grid columns={2} gap={20}>
            {SIMULATIONS.map((deck) => (
              <DeckCard key={deck.path} deck={deck} />
            ))}
          </Grid>
        </div>
      </div>
    </div>
  );
}
