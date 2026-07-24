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
        </div>
      </div>
    </div>
  );
}
