/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `rust-port-why` — an executive overview: WHY AIPerf is being reimplemented in Rust. This is a
//! static, scrollable keynote page (no React Flow, no interactivity) built from the shared Systems
//! Chalk vocabulary so it reads with the same visual language as `rust-port-flow`.
//!
//! Content is deliberately grounded and non-promotional. Every claim traces to a durable finding
//! from the AIPerf Tech Lead's own notes / live A-B benchmarks (July 2026), NOT to marketing copy:
//!   - Two primary drivers: (1) a SHARED CORE with Dynamo enabling bidirectional code reuse, and
//!     (2) collapsing the ~10-service ZeroMQ multiprocess control plane that exists ONLY to escape
//!     Python's GIL — an accidental burden, not a benchmarking feature.
//!   - Performance is presented HONESTLY: on a normal, server-bound LLM run the two clients are
//!     statistically indistinguishable (250-concurrency live A-B, 2026-07-19); the Rust win is real
//!     only at the throughput ceiling against a fast target (~13.7k vs ~4.5k req/s over 100k
//!     requests, 2026-07-20; ~40k req/s single-process, 2026-07-13).
//!   - It explicitly refuses to overclaim: the single-node ~15-28k connection ceiling is an OS/TCP
//!     ephemeral-port fact, not a language advantage.

import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Grid } from "../../layout/Grid.js";
import { Stat } from "../../prose/Stat.js";
import { Table } from "../../prose/Table.js";
import { Callout } from "../../prose/Callout.js";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { Framed } from "../../prose/Framed.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";
import { HubSpoke, ChalkCard } from "../../chalk/index.js";
import type { ChalkCardProps } from "../../chalk/index.js";
import {
  Diagram,
  NodeChip,
  RoundNode,
  DbNode,
  MiniArrow,
  BiArrow,
  MiniBars,
} from "../../chalk/index.js";

// ── The six reasons, as hub-and-spoke drivers ─────────────────────────────────
// Ordered so the two PRIMARY drivers (strategic reuse, accidental-complexity collapse) come first;
// the remainder are the concrete consequences that follow from them.

const REASONS: ReadonlyArray<ChalkCardProps> = [
  {
    accent: "blue",
    badge: 1,
    title: "Shared core with Dynamo",
    diagram: (
      <Diagram>
        <NodeChip accent>AIPerf</NodeChip>
        <BiArrow />
        <NodeChip>Dynamo</NodeChip>
      </Diagram>
    ),
    children:
      "AIPerf and Dynamo reuse each other's Rust code — bidirectionally. Dynamo maintainers asked for this; the Dynamo mocker was built partly as a stand-in engine for AIPerf. One codebase feeds both.",
  },
  {
    accent: "purple",
    badge: 2,
    title: "One process, not ten services",
    diagram: (
      <Diagram>
        <NodeChip>10× svc</NodeChip>
        <MiniArrow />
        <RoundNode accent>1</RoundNode>
      </Diagram>
    ),
    children:
      "To use ONE machine, Python needs ~10 services, a ZeroMQ bus, ZMQ proxies, and mmap dataset sharing — almost all of it only to work around the GIL. A single-node Rust run is one multithreaded, thread-per-core process. Multi-node scale-out (cellular) stays opt-in and behind a trait boundary — deliberate, not forced on every run.",
  },
  {
    accent: "green",
    badge: 3,
    title: "Headroom at the ceiling",
    diagram: (
      <Diagram>
        <MiniBars heights={[30, 30, 100]} />
        <NodeChip accent>~3×</NodeChip>
      </Diagram>
    ),
    children:
      "Against a fast target one Rust process sustains ~40k req/s and ~3× Python's throughput at the ceiling. Python must spin up multiprocessing + ZMQ just to under-perform a single Rust process.",
  },
  {
    accent: "cyan",
    badge: 4,
    title: "One front-end, three modes",
    diagram: (
      <Diagram>
        <NodeChip accent>front-end</NodeChip>
        <MiniArrow />
        <NodeChip>×3</NodeChip>
      </Diagram>
    ),
    children:
      "One unified front-end drives online-real, online-mock, and offline (virtual-clock) co-simulation over a single {transport, clock} seam. Every feature — arrival patterns, datasets, metrics, exporters — works across all three for free, replacing three fragmented front-ends that exist today.",
  },
  {
    accent: "orange",
    badge: 5,
    title: "Measurement parity, proven",
    diagram: (
      <Diagram>
        <NodeChip>py</NodeChip>
        <BiArrow />
        <NodeChip accent>rs</NodeChip>
      </Diagram>
    ),
    children:
      "The Rust CLI is validated byte-for-byte against Python config projection, and a live A-B at 250 concurrency is statistically indistinguishable on every metric. The rewrite does not change the numbers customers see — that is the point.",
  },
  {
    accent: "yellow",
    badge: 6,
    title: "Simpler to operate",
    diagram: (
      <Diagram>
        <DbNode>ZMQ</DbNode>
        <MiniArrow />
        <NodeChip accent>gone</NodeChip>
      </Diagram>
    ),
    children:
      "No message bus, no per-service CPU tuning, no multi-pod control plane to keep alive for a single-node run. Fewer moving parts means fewer failure modes at high load.",
  },
];

// ── Honest performance rows (recent live A-B, July 2026) ──────────────────────

const PERF_COLUMNS = [
  { key: "scenario", label: "Scenario" },
  { key: "rust", label: "Rust", align: "end" as const },
  { key: "python", label: "Python", align: "end" as const },
  { key: "read", label: "What it means", align: "start" as const },
];

const PERF_ROWS = [
  {
    scenario: "Real server, 250 concurrency — req/s",
    rust: "7.24",
    python: "7.26",
    read: "Tied — the server, not the client, sets the pace.",
    tone: "neutral" as const,
  },
  {
    scenario: "Real server, 250 concurrency — TTFT p50",
    rust: "25,072 ms",
    python: "25,138 ms",
    read: "Within noise. Every latency metric matches.",
    tone: "neutral" as const,
  },
  {
    scenario: "Fast target, 100k requests — throughput",
    rust: "13,746 req/s",
    python: "4,500 req/s",
    read: "≈3× — the win appears only when the client is the limit.",
    tone: "success" as const,
  },
  {
    scenario: "Fast target, single process — ceiling",
    rust: "~40,000 req/s",
    python: "needs multiproc + ZMQ",
    read: "One Rust process beats Python's whole fan-out.",
    tone: "success" as const,
  },
];

/** A compact two-column contrast panel: what Python carries vs what Rust keeps. */
function ContrastColumn({
  eyebrowTone,
  eyebrow,
  title,
  items,
}: {
  eyebrowTone: "red" | "green";
  eyebrow: string;
  title: string;
  items: readonly string[];
}): React.JSX.Element {
  return (
    <Framed surfaceRole="elevated">
      <Stack gap={8}>
        <Eyebrow tone={eyebrowTone}>{eyebrow}</Eyebrow>
        <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>{title}</h3>
        <ul className={`ml-4 list-disc space-y-1 text-sm ${inkClassName("secondary")}`}>
          {items.map((it) => (
            <li key={it}>{it}</li>
          ))}
        </ul>
      </Stack>
    </Framed>
  );
}

/** Section heading with an eyebrow kicker. */
function SectionHead({ kicker, title }: { kicker: string; title: string }): React.JSX.Element {
  return (
    <Stack gap={4}>
      <Eyebrow>{kicker}</Eyebrow>
      <h2 className={`text-xl font-bold tracking-tight ${inkClassName("primary")}`}>{title}</h2>
    </Stack>
  );
}

/**
 * The executive "Why Rust" overview page. Static and scrollable — no interactivity — but built from
 * the same Systems Chalk cards/nodes as the interactive decks so it reads as one product.
 */
export function RustPortWhyDeck(): React.JSX.Element {
  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Why Rust · Executive Overview" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className={`mx-auto min-h-full max-w-6xl px-10 py-10 ${surfaceClassName("page")}`}>
          <Stack gap={40}>
            {/* ── Hero + thesis ─────────────────────────────────────────────── */}
            <Stack gap={14}>
              <Eyebrow tone="cyan">AIPerf · perf_analyzer → GenAI-Perf → AIPerf (Python) → AIPerf (Rust)</Eyebrow>
              <h1 className={`max-w-4xl text-4xl font-extrabold tracking-tight ${inkClassName("primary")}`}>
                The case for porting AIPerf to Rust
              </h1>
              <p className={`max-w-4xl text-base leading-relaxed ${inkClassName("secondary")}`}>
                Porting AIPerf to Rust removes a whole class of accidental complexity — a roughly
                ten-service, ZeroMQ, multiprocess control plane that{" "}
                <span className={inkClassName("primary")}>exists only to work around Python's GIL</span> —
                and unlocks a shared code core with{" "}
                <span className={inkClassName("primary")}>Dynamo</span>. It does this while matching
                Python's measurements exactly on real runs and running several times faster only where
                the client, not the inference server, is the limit.
              </p>
            </Stack>

            {/* ── Headline numbers (recent, honest) ────────────────────────── */}
            <Grid columns={4} gap={16}>
              <Stat label="Ceiling throughput" value="≈3×" trend="13.7k vs 4.5k req/s" tone="positive" />
              <Stat label="Metric drift @ 250 conc." value="±0.3%" trend="indistinguishable" tone="neutral" />
              <Stat label="Processes for a single-node run" value="1" trend="was ~10 services" tone="positive" />
              <Stat label="Execution modes, one front-end" value="3" trend="online · mock · offline" tone="positive" />
            </Grid>

            {/* ── The reasons, hub-and-spoke ───────────────────────────────── */}
            <Stack gap={16}>
              <SectionHead kicker="The case" title="Two primary drivers, four consequences" />
              <HubSpoke
                hub={{
                  kicker: "PORT TO",
                  title: "Rust",
                  body: "One multithreaded process. Shared core with Dynamo.",
                }}
                spokes={REASONS}
                liveWire={0}
              />
            </Stack>

            {/* ── Accidental complexity contrast ───────────────────────────── */}
            <Stack gap={16}>
              <SectionHead
                kicker="Where the complexity went"
                title="Python distributes to USE one box; Rust distributes only to EXCEED one"
              />
              <Grid columns={2} gap={16}>
                <ContrastColumn
                  eyebrowTone="red"
                  eyebrow="Python — mandatory, every run"
                  title="A distributed system just to fill one machine"
                  items={[
                    "~10 services (controller, workers, records manager, timing, …)",
                    "ZeroMQ message bus + ZMQ proxies for routing",
                    "mmap dataset sharing to dodge per-process copies",
                    "multiprocess service managers to fan out past the GIL — on a single host",
                    "per-service CPU tuning at high load (e.g. records-manager starvation > 500k)",
                  ]}
                />
                <ContrastColumn
                  eyebrowTone="green"
                  eyebrow="Rust — deliberate, opt-in"
                  title="One process by default; cells only for true scale-out"
                  items={[
                    "Single-node run: one binary, worker-local state, no cross-process bus",
                    "Shared {transport, clock} seams instead of a service mesh",
                    "Dataset in-process — no mmap, no serialization hop",
                    "Cellular multi-node behind a trait boundary — used to pass the OS connection ceiling, not to use a machine's cores",
                    "You pay for distribution only when you genuinely outgrow one host",
                  ]}
                />
              </Grid>
              <Callout tone="info" title="“But aren't we going distributed too, with cellular?”">
                Yes — and that is the honest distinction, not a contradiction. Python's fan-out is{" "}
                <span className={inkClassName("primary")}>mandatory to use a single machine at all</span>{" "}
                (the GIL forces multiple processes + a bus). Rust's{" "}
                <span className={inkClassName("primary")}>cellular</span> mode is a{" "}
                <span className={inkClassName("primary")}>deliberate, opt-in</span> scale-out that only
                engages when a run genuinely exceeds one host's ~15–28k connection ceiling. The default
                single-node path stays one process. Same word — “distributed” — opposite reason.
              </Callout>
            </Stack>

            {/* ── Honest performance ───────────────────────────────────────── */}
            <Stack gap={16}>
              <SectionHead kicker="Performance, without the spin" title="Parity on real runs; the win is at the ceiling" />
              <Table columns={PERF_COLUMNS} rows={PERF_ROWS} />
              <Callout tone="warning" title="The integrity caveat we lead with">
                On a normal LLM benchmark{" "}
                <span className={inkClassName("primary")}>
                  the inference server is the bottleneck, not the client
                </span>
                . That is exactly why the two implementations are{" "}
                <span className={inkClassName("primary")}>statistically indistinguishable</span> at 250
                concurrency. The ≈3× only shows up against a near-zero-latency target, where AIPerf's own
                dispatch rate is the limit. We do not claim faster customer numbers — we claim the same
                numbers, from a simpler system, with more headroom.
              </Callout>
              <Callout tone="neutral" title="And what we do NOT claim">
                The single-node ~15–28k concurrent-connection limit is an{" "}
                <span className={inkClassName("primary")}>OS/TCP ephemeral-port fact, not a language advantage</span>.
                Rust hits the same wall Python does; it is addressed the same way in both — HTTP/2 stream
                multiplexing or multi-node distribution — so it is not part of the case for Rust.
              </Callout>
            </Stack>

            {/* ── Why now ──────────────────────────────────────────────────── */}
            <Stack gap={16}>
              <SectionHead kicker="Timing" title="Why Rust, why now" />
              <Grid columns={3} gap={14}>
                <ChalkCard
                  accent="gray"
                  badge="A"
                  title="Python was the right first call"
                  diagram={
                    <Diagram>
                      <NodeChip accent>reach</NodeChip>
                    </Diagram>
                  }
                >
                  Python was chosen for the developer and customer landscape — accessibility and
                  contribution speed. That was correct for AIPerf's first generation.
                </ChalkCard>
                <ChalkCard
                  accent="purple"
                  badge="B"
                  title="The language barrier is eroding"
                  diagram={
                    <Diagram>
                      <NodeChip>barrier</NodeChip>
                      <MiniArrow />
                      <NodeChip accent>lower</NodeChip>
                    </Diagram>
                  }
                >
                  AI-assisted, spec-driven development makes Rust approachable to write and review,
                  removing the accessibility reason that favored Python.
                </ChalkCard>
                <ChalkCard
                  accent="green"
                  badge="C"
                  title="Feasibility is not a guess"
                  diagram={
                    <Diagram>
                      <NodeChip>py</NodeChip>
                      <BiArrow />
                      <NodeChip accent>rs</NodeChip>
                    </Diagram>
                  }
                >
                  A working prototype already reaches byte-exact config parity and an indistinguishable
                  live A-B against Python. This is a de-risked engineering investment, not a research
                  bet — with Python as the reference oracle throughout.
                </ChalkCard>
              </Grid>
            </Stack>

            {/* ── Bottom line ──────────────────────────────────────────────── */}
            <Framed surfaceRole="elevated">
              <Stack gap={8}>
                <Eyebrow tone="cyan">The ask</Eyebrow>
                <p className={`text-base leading-relaxed ${inkClassName("secondary")}`}>
                  Greenlight the Rust port. The case rests primarily on{" "}
                  <span className={inkClassName("primary")}>strategic reuse with Dynamo</span> and on{" "}
                  <span className={inkClassName("primary")}>retiring the GIL-driven distributed system</span>{" "}
                  AIPerf never actually wanted — replacing mandatory single-host fan-out with one process
                  and opt-in cellular scale-out. Performance is a real but bounded bonus: parity where the
                  server dominates, ~3× ahead at the client ceiling. Because a prototype already
                  demonstrates parity, this is a low-risk investment, not a research bet.
                </p>
              </Stack>
            </Framed>

            <div className={`border-t pt-3 text-xs ${strokeClassName("secondary")} ${inkClassName("tertiary")}`}>
              Sources: AIPerf Tech Lead design notes and live A-B benchmarks (Jul 2026) — 250-concurrency
              parity run (2026-07-19), 100k-request ceiling run (2026-07-20), single-process ~40k req/s
              microbench (2026-07-13). Figures are honest snapshots, not projections.
            </div>
          </Stack>
        </div>
      </div>
    </div>
  );
}
