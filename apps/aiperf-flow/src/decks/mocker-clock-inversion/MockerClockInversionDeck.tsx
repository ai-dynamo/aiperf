/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { Edge, Node } from "@xyflow/react";
import { ReactFlow, Background, BackgroundVariant } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "../../nodes/nodeTypes.js";
import { edgeTypes } from "../../edges/edgeTypes.js";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Callout, type CalloutTone } from "../../prose/Callout.js";
import { Code } from "../../prose/Code.js";
import { Button } from "../../prose/Button.js";
import {
  categoryBgClassName,
  categoryBgTintClassName,
  categoryClassName,
  strokeClassName,
  surfaceClassName,
  inkClassName,
  type CategoryRole,
} from "../../theme/tokens.js";

// Ported from
// ~/.cursor/projects/home-anthony-nvidia-projects-dynamo-aiperf-native/canvases/mocker-clock-inversion.canvas.tsx
// (a real, hand-authored Cursor Canvas). Single-view canvas: no PageTabs union in the
// source, so this is one component. Explains the outer-loop inversion between AIPerf's
// dynosim_offline runtime and Dynamo Mocker's offline replay engine.

type Winner = "poll" | "engine" | "clock" | "route";

type Frame = {
  title: string;
  now: string;
  clockEvent: string;
  engineEvent: string;
  winner: Winner;
  explanation: string;
};

const FRAMES: Frame[] = [
  {
    title: "AIPerf runs until its futures park",
    now: "0 ms",
    clockEvent: "R2 arrival · 10 ms",
    engineEvent: "R1 work · 0 ms",
    winner: "poll",
    explanation:
      "run_paced_with_backend dispatches R1 through DynosimSink, schedules R2 with Clock::sleep, and parks R1 on its waiter mailbox.",
  },
  {
    title: "The engine is ready before AIPerf’s next timer",
    now: "0 ms",
    clockEvent: "R2 arrival · 10 ms",
    engineEvent: "R1 pass · now",
    winner: "engine",
    explanation:
      "The pump compares SimClock::next_event_time with EngineHost::next_event_ns. Mocker wins this round, but it may not cross 10 ms.",
  },
  {
    title: "AIPerf gives Mocker a bounded slice",
    now: "0 → 4 ms",
    clockEvent: "hard horizon · 10 ms",
    engineEvent: "token · 4 ms",
    winner: "engine",
    explanation:
      "EngineHost calls SteppableReplay::step_until(10 ms). Mocker advances only to its 4 ms token event and returns a StepOutcome.",
  },
  {
    title: "Engine events wake ordinary AIPerf futures",
    now: "4 ms",
    clockEvent: "R2 arrival · 10 ms",
    engineEvent: "next pass · 12 ms",
    winner: "route",
    explanation:
      "EngineHost routes the token into R1’s waiter. DynosimSink resumes and emits RequestObserver callbacks; TTFT is measured by AIPerf’s clock.",
  },
  {
    title: "AIPerf’s timer now wins",
    now: "4 → 10 ms",
    clockEvent: "R2 arrival · 10 ms",
    engineEvent: "R1 pass · 12 ms",
    winner: "clock",
    explanation:
      "The pump advances SimClock to 10 ms first and synchronizes EngineHost. The sleeping rate gate wakes before Mocker may execute the 12 ms pass.",
  },
  {
    title: "The new arrival joins at the exact virtual instant",
    now: "10 ms",
    clockEvent: "ready task · now",
    engineEvent: "R1 pass · 12 ms",
    winner: "poll",
    explanation:
      "AIPerf re-polls to quiescence, dispatches R2, and submits it dynamically at 10 ms. Batch composition is controlled by AIPerf scheduling.",
  },
  {
    title: "Repeat until the workload resolves, then drain",
    now: "10 ms → done",
    clockEvent: "phase / pacing / gates",
    engineEvent: "passes / tokens / terminal",
    winner: "route",
    explanation:
      "The same arbitration repeats. When the paced future becomes Ready, the pump drains Mocker, merges both reports, and verifies parity.",
  },
];

const WINNER_CATEGORY: Record<Winner, CategoryRole> = {
  poll: "gray",
  engine: "blue",
  clock: "yellow",
  route: "green",
};

const WINNER_CALLOUT_TONE: Record<Winner, CalloutTone> = {
  poll: "info",
  engine: "info",
  clock: "warning",
  route: "success",
};

// Legacy (before) chain: Mocker owns its own drain loop end to end.
const legacyNodes: Node[] = [
  {
    id: "legacy-queue",
    type: "panel",
    position: { x: 0, y: 0 },
    data: { title: "Static arrival queue", detail: "Whole workload handed to Mocker up front" },
  },
  {
    id: "legacy-run",
    type: "card",
    position: { x: 0, y: 130 },
    data: {
      title: "run_to_completion()",
      subtitle: "Mocker owns the loop",
      detail: "now_ms, next timestamp, admission, and drain loop",
    },
  },
  {
    id: "legacy-report",
    type: "panel",
    position: { x: 0, y: 280 },
    data: { title: "TraceSimulationReport", detail: "Caller regains control only after drain" },
  },
];

const legacyEdges: Edge[] = [
  { id: "e-legacy-queue-run", source: "legacy-queue", target: "legacy-run", type: "flow" },
  { id: "e-legacy-run-report", source: "legacy-run", target: "legacy-report", type: "flow" },
];

// AIPerf (after) chain: AIPerf owns the clock and outer loop; Mocker steps on request.
const aiperfNodes: Node[] = [
  {
    id: "aiperf-run",
    type: "card",
    position: { x: 0, y: 0 },
    data: {
      title: "run_paced_with_backend",
      subtitle: "ajc/rust",
      detail: "The ordinary AIPerf concurrency / request-rate workload future",
    },
  },
  {
    id: "aiperf-sink",
    type: "panel",
    position: { x: 0, y: 150 },
    data: {
      title: "DynosimSink + EngineHost",
      detail: "Waiter mailboxes bridge async AIPerf tasks to Mocker events",
    },
  },
  {
    id: "aiperf-replay",
    type: "panel",
    position: { x: 0, y: 280 },
    data: {
      title: "SteppableReplay",
      detail: "Mocker owns scheduler state, but no longer owns the outer loop",
    },
  },
];

const aiperfEdges: Edge[] = [
  {
    id: "e-aiperf-run-sink",
    source: "aiperf-run",
    target: "aiperf-sink",
    type: "flow",
    label: "DirectRequest",
  },
  {
    id: "e-aiperf-sink-replay",
    source: "aiperf-sink",
    target: "aiperf-replay",
    type: "flow",
    label: "bounded step",
  },
];

const COMPOSITION_NODES: { title: string; detail: string; active: boolean }[] = [
  { title: "run_paced_offline", detail: "creates SimClock + EngineHost", active: true },
  { title: "run_paced_with_backend", detail: "ordinary AIPerf workload loop", active: false },
  { title: "drive_sim_with_source", detail: "poll + dual-queue arbitration", active: true },
  { title: "EngineHost", detail: "SimEventSource + waiter routing", active: false },
  { title: "SteppableReplay", detail: "Dynamo scheduler + perf model", active: false },
];

const PUMP_STAGES: { id: Winner; title: string; detail: string }[] = [
  { id: "poll", title: "Poll AIPerf", detail: "LocalSet to quiescence" },
  { id: "clock", title: "Advance clock", detail: "wake due sleepers" },
  { id: "engine", title: "Step Mocker", detail: "step_until(clock deadline)" },
  { id: "route", title: "Route events", detail: "wake waiter futures" },
];

const OPEN_FILE_TARGETS = [
  {
    label: "AIPerf sim pump",
    path: "rust/runtime/src/graph/runtime.rs",
    line: 205,
  },
  {
    label: "dynosim_offline",
    path: "rust/runtime/src/dynosim.rs",
    line: 2387,
  },
  {
    label: "Mocker seam",
    path: "lib/mocker/src/loadgen/steppable.rs",
    line: 77,
  },
];

function CategoryPill({ category, label }: { category: CategoryRole; label: string }): React.JSX.Element {
  return (
    <span
      className={`rounded-md px-2 py-0.5 text-xs font-semibold shadow-sm ${categoryBgTintClassName(category)} ${categoryClassName(category)}`}
    >
      {label}
    </span>
  );
}

function PumpStage({ winner }: { winner: Winner }): React.JSX.Element {
  return (
    <Grid columns={4} gap={8}>
      {PUMP_STAGES.map((stage, index) => {
        const active = winner === stage.id;
        return (
          <div key={stage.id} className="min-w-0">
            <div
              className={`mb-2 h-[3px] rounded-full ${active ? categoryBgClassName(WINNER_CATEGORY[stage.id]) : "bg-stroke-tertiary"}`}
            />
            <Row gap={7} align="center">
              <span
                className={`inline-flex h-[19px] w-[19px] items-center justify-center rounded-full text-[10px] font-semibold ${
                  active
                    ? `${categoryBgClassName(WINNER_CATEGORY[stage.id])} text-white`
                    : "bg-surface-panel text-ink-secondary"
                }`}
              >
                {index + 1}
              </span>
              <span className={`text-sm ${active ? "font-semibold" : ""} ${inkClassName("primary")}`}>
                {stage.title}
              </span>
            </Row>
            <p className={`mt-1 text-xs ${inkClassName("quaternary")}`}>{stage.detail}</p>
          </div>
        );
      })}
    </Grid>
  );
}

function DualQueue({ frame }: { frame: Frame }): React.JSX.Element {
  const clockActive = frame.winner === "clock";
  const engineActive = frame.winner === "engine";
  return (
    <Grid columns="1fr 86px 1fr" gap={12} align="center">
      <div
        className={`rounded-lg border px-4 py-3.5 shadow-sm ${
          clockActive ? `border-accent-primary ${surfaceClassName("elevated")}` : `${strokeClassName("tertiary")} ${surfaceClassName("chrome")}`
        }`}
      >
        <Row gap={8} align="center">
          <span
            className={`inline-block h-2 w-2 rounded-full ${clockActive ? "bg-accent-primary" : "bg-ink-quaternary"}`}
          />
          <span className={`text-sm font-semibold ${inkClassName("primary")}`}>AIPerf SimClock</span>
        </Row>
        <p className={`mt-2 text-sm ${inkClassName("secondary")}`}>{frame.clockEvent}</p>
        <p className={`text-xs ${inkClassName("quaternary")}`}>arrivals · pacing · phase gates · backoff</p>
      </div>

      <Stack gap={5} className="items-center text-center">
        <span className={`text-xs ${inkClassName("quaternary")}`}>choose</span>
        <Code inline>min(t)</Code>
        <span className={`text-xs ${inkClassName("quaternary")}`}>clock wins ties</span>
      </Stack>

      <div
        className={`rounded-lg border px-4 py-3.5 shadow-sm ${
          engineActive ? `border-accent-primary ${surfaceClassName("elevated")}` : `${strokeClassName("tertiary")} ${surfaceClassName("chrome")}`
        }`}
      >
        <Row gap={8} align="center">
          <span
            className={`inline-block h-2 w-2 rounded-full ${engineActive ? "bg-accent-primary" : "bg-ink-quaternary"}`}
          />
          <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Mocker event source</span>
        </Row>
        <p className={`mt-2 text-sm ${inkClassName("secondary")}`}>{frame.engineEvent}</p>
        <p className={`text-xs ${inkClassName("quaternary")}`}>passes · transfers · tokens · terminal</p>
      </div>
    </Grid>
  );
}

/**
 * Ports `mocker-clock-inversion.canvas.tsx` — a Cursor Canvas explaining the outer-loop
 * inversion between AIPerf's `dynosim_offline` runtime and Dynamo Mocker's offline replay
 * engine. Single view: before/after loop-ownership diagrams, a `Callout` on what
 * "inversion" means, a real `drive_sim_with_source` arbitration-cycle step simulator, the
 * actual `ajc/rust` composition chain, and a keeps/keeps/adds responsibility split.
 */
export function MockerClockInversionDeck(): React.JSX.Element {
  const sim = useStepSimulator(FRAMES, { autoPlayMs: 1200 });
  const frame = sim.current ?? FRAMES[0];

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 bg-surface-page px-10 py-8">
      <Stack gap={8}>
        <Row gap={9} align="center" wrap>
          <CategoryPill category="green" label="AIPerf ajc/rust" />
          <span className={`text-sm ${inkClassName("tertiary")}`}>
            <Code inline>dynosim_offline</Code> · concrete virtual-clock execution
          </span>
        </Row>
        <h1 className={`text-2xl font-semibold ${inkClassName("primary")}`}>
          AIPerf takes ownership of simulation time
        </h1>
        <p className={`max-w-3xl text-sm ${inkClassName("secondary")}`}>
          The key change is an outer-loop inversion: AIPerf runs its normal workload future and
          owns the authoritative <Code inline>SimClock</Code>; Dynamo Mocker becomes one passive
          event source competing with AIPerf timers.
        </p>
      </Stack>

      <Grid columns={2} gap={14}>
        <Stack gap={8}>
          <Row gap={8} align="center" justify="space-between">
            <h2 className={`text-base font-semibold ${inkClassName("primary")}`}>
              Mocker legacy offline replay
            </h2>
            <CategoryPill category="gray" label="before" />
          </Row>
          <div style={{ height: 380 }}>
            <ReactFlow
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              nodes={legacyNodes}
              edges={legacyEdges}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              proOptions={{ hideAttribution: true }}
            >
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
            </ReactFlow>
          </div>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            Mocker decides <strong>when</strong> arrivals become visible to its scheduler.
          </p>
        </Stack>

        <Stack gap={8}>
          <Row gap={8} align="center" justify="space-between">
            <h2 className={`text-base font-semibold ${inkClassName("primary")}`}>AIPerf dynosim_offline</h2>
            <CategoryPill category="green" label="ajc/rust" />
          </Row>
          <div style={{ height: 380 }}>
            <ReactFlow
              nodeTypes={nodeTypes}
              edgeTypes={edgeTypes}
              nodes={aiperfNodes}
              edges={aiperfEdges}
              fitView
              fitViewOptions={{ padding: 0.2 }}
              proOptions={{ hideAttribution: true }}
            >
              <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--color-stroke-secondary)" />
            </ReactFlow>
          </div>
          <p className={`text-sm ${inkClassName("secondary")}`}>
            AIPerf decides <strong>when</strong> the engine may observe each arrival or gate.
          </p>
        </Stack>
      </Grid>

      <Callout tone="info" title="What inversion means here">
        AIPerf owns the clock object and arbitration policy. Mocker still maintains its local{" "}
        <Code inline>now_ms</Code>, but <Code inline>EngineHost</Code> synchronizes it to AIPerf
        time and bounds every <Code inline>step_until</Code> so the engine cannot skip an AIPerf
        arrival or firing gate.
      </Callout>

      <Stack gap={12}>
        <Row align="end" justify="space-between" wrap>
          <div>
            <h2 className={`text-base font-semibold ${inkClassName("primary")}`}>
              Example: R1 is running; AIPerf schedules R2 for 10 ms
            </h2>
            <p className={`text-sm ${inkClassName("secondary")}`}>
              Step through one real <Code inline>drive_sim_with_source</Code> arbitration cycle.
            </p>
          </div>
          <CategoryPill category="green" label={frame.now} />
        </Row>

        <div className={`rounded-lg border px-5 py-4 shadow-sm ${strokeClassName("secondary")} ${surfaceClassName("elevated")}`}>
          <Row align="center" justify="space-between">
            <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>{frame.title}</h3>
            <span className={`text-xs font-semibold ${inkClassName("tertiary")}`}>
              {sim.index + 1} / {FRAMES.length}
            </span>
          </Row>
          <Stack gap={16} className="mt-4">
            <PumpStage winner={frame.winner} />
            <DualQueue frame={frame} />
            <Callout tone={WINNER_CALLOUT_TONE[frame.winner]}>{frame.explanation}</Callout>
            <Row gap={8} align="center">
              <Button variant="secondary" disabled={sim.isFirst} onClick={sim.back}>
                Back
              </Button>
              <Button variant="primary" disabled={sim.isLast} onClick={sim.next}>
                Next event
              </Button>
              <Button variant="ghost" disabled={sim.isFirst} onClick={sim.reset}>
                Reset
              </Button>
              <span className={`ml-auto text-xs ${inkClassName("quaternary")}`}>
                wall time does not participate
              </span>
            </Row>
          </Stack>
        </div>
      </Stack>

      <Divider />

      <Stack gap={10}>
        <h2 className={`text-base font-semibold ${inkClassName("primary")}`}>The actual `ajc/rust` composition</h2>
        <Grid columns={5} gap={8}>
          {COMPOSITION_NODES.map((node) => (
            <div
              key={node.title}
              className={`rounded-lg border px-3 py-2.5 shadow-sm ${
                node.active ? "border-accent-primary" : strokeClassName("tertiary")
              } ${surfaceClassName("elevated")}`}
            >
              <div className={`text-xs font-semibold ${node.active ? "text-accent-primary" : inkClassName("primary")}`}>
                {node.title}
              </div>
              <p className={`mt-1 text-xs leading-4 ${inkClassName("tertiary")}`}>{node.detail}</p>
            </div>
          ))}
        </Grid>
        <Grid columns={3} gap={12}>
          <Callout tone="info" title="AIPerf keeps">
            Workload policy, pacing, concurrency, phases, cancellation, observers, metrics, and
            report.
          </Callout>
          <Callout tone="info" title="Mocker keeps">
            Batching, scheduler cores, router, KV state, offload, disagg handoff, and performance
            model.
          </Callout>
          <Callout tone="success" title="The seam adds">
            Dynamic submit, next-event visibility, deadline-bounded stepping, and typed token /
            terminal events.
          </Callout>
        </Grid>
      </Stack>

      <Row gap={8} align="center" wrap>
        {OPEN_FILE_TARGETS.map((target) => (
          <span
            key={target.path}
            className={`rounded-md border px-3 py-1.5 text-xs font-mono shadow-sm ${strokeClassName("secondary")} ${surfaceClassName("panel")} ${inkClassName("secondary")}`}
          >
            {target.label} — {target.path}:{target.line}
          </span>
        ))}
        <span className={`text-xs ${inkClassName("quaternary")}`}>
          Sources: AIPerf `ajc/rust` + current Dynamo Mocker branch
        </span>
      </Row>
    </div>
  );
}
