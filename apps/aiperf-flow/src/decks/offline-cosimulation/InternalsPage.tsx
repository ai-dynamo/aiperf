/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import clsx from "clsx";
import type { Edge, Node } from "@xyflow/react";
import { AutoLayoutFlow } from "../../layout/graph/index.js";
import { useStepSimulator } from "../../state/useStepSimulator.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Grid } from "../../layout/Grid.js";
import { Divider } from "../../layout/Divider.js";
import { Callout } from "../../prose/Callout.js";
import { Button } from "../../prose/Button.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";

//! Ported from `offline-cosimulation.canvas.tsx` internals page: the layered `AIPERF -> DYNAMO
//! MOCKER (PASSIVE)` stack (dependency strictly one-way), the Poll/Compare/Advance/Step/Route
//! drive-loop walkthrough over the source canvas's fixed FRAMES, and the Level-B observer
//! pipeline (on_arrival -> on_admit -> on_token -> on_usage -> on_terminal).

// ---- InternalsStackDiagram: LAYERED INTERNALS — DEPENDENCY IS STRICTLY AIPERF -> MOCKER ----

const stackNodes: Node[] = [
  { id: "band-aiperf", type: "header", position: { x: 0, y: 0 }, data: { title: "AIPERF" } },

  {
    id: "scheduledruntime",
    type: "card",
    position: { x: 0, y: 50 },
    data: { title: "ScheduledRuntime", detail: "admit · poll · stop" },
  },
  {
    id: "clock",
    type: "card",
    position: { x: 260, y: 50 },
    data: { title: "Clock", detail: "SimClock | RealClock" },
  },
  {
    id: "drive",
    type: "card",
    position: { x: 520, y: 50 },
    data: { title: "drive_*_with_source", detail: "owns the run loop" },
  },
  {
    id: "observertee",
    type: "card",
    position: { x: 780, y: 50 },
    data: { title: "ObserverTee", detail: "Collector + NativeMetrics (+ Trace)" },
  },

  {
    id: "enginehost",
    type: "card",
    position: { x: 300, y: 190 },
    data: { title: "EngineHost : SimEventSource", detail: "next_event_ns · set_time_ns · step · route" },
  },

  {
    id: "band-mocker",
    type: "header",
    position: { x: 0, y: 320 },
    data: { title: "DYNAMO MOCKER (PASSIVE)" },
  },
  {
    id: "steppableengine",
    type: "card",
    position: { x: 0, y: 370 },
    data: { title: "SteppableEngine", detail: "single worker" },
  },
  {
    id: "steppableagg",
    type: "card",
    position: { x: 260, y: 370 },
    data: { title: "SteppableAgg", detail: "N workers · router" },
  },
  {
    id: "steppabledisagg",
    type: "card",
    position: { x: 520, y: 370 },
    data: { title: "SteppableDisagg", detail: "prefill + decode" },
  },
  {
    id: "scalarnow",
    type: "panel",
    position: { x: 780, y: 370 },
    data: { title: "scalar now", detail: "never Clock" },
  },
];

const stackEdges: Edge[] = [
  { id: "e-drive-enginehost", source: "drive", target: "enginehost", type: "flow" },
  { id: "e-enginehost-observertee", source: "enginehost", target: "observertee", type: "flow", label: "route → on_*" },
  { id: "e-enginehost-engine", source: "enginehost", target: "steppableengine", type: "flow" },
  { id: "e-enginehost-agg", source: "enginehost", target: "steppableagg", type: "flow" },
  { id: "e-enginehost-disagg", source: "enginehost", target: "steppabledisagg", type: "flow" },
];

// ---- Drive-loop walkthrough (FRAMES) ----

type Frame = { stage: number; vt: number; title: string; cap: string };

const FRAMES: Frame[] = [
  { stage: 0, vt: 0, title: "Poll", cap: "Workload futures poll to quiescence. Admitted turns submit token arrays into the steppable engine and park as Pending waiters." },
  { stage: 1, vt: 0, title: "Compare", cap: "The driver compares the next clock sleeper against EngineHost.next_event_ns(). Earliest wins; clock wins ties. Engine cannot overshoot." },
  { stage: 2, vt: 0, title: "Advance", cap: "AIPerf advances SimClock (or waits on RealClock). EngineHost.set_time_ns / advance_now_ms receives a plain scalar — never a Clock object." },
  { stage: 3, vt: 1, title: "Step", cap: "EngineHost.step calls step_until(deadline_ms). The mocker forms a batch / decode tick and returns EngineEvents. Scheduler math is untouched." },
  { stage: 4, vt: 1, title: "Route", cap: "Events wake waiters. First emitted token fires on_token (TTFT), releases the prefill slot, and opens the graph first-token gate." },
  { stage: 1, vt: 2, title: "Compare", cap: "Back to Compare: next engine event is still earliest, so the engine wins again without advancing workload sleepers." },
  { stage: 3, vt: 2, title: "Step", cap: "Another bounded step_until produces decode tokens. per-token events keep streaming into RequestObserver during the run." },
  { stage: 4, vt: 3, title: "Route", cap: "Terminal events fire on_usage then on_terminal. Waiter futures resolve Ready; StopChecker can exit the pump." },
];

const STAGE_NAMES = ["Poll", "Compare", "Advance", "Step", "Route"];
const STAGE_SUBS = ["LocalSet", "clock ≤ engine", "advance_to", "step_until", "wake waiters"];
const TICKS = [
  { x: 70, ms: "0" },
  { x: 210, ms: "1.8" },
  { x: 340, ms: "2.0" },
  { x: 500, ms: "22" },
];

// ---- Level-B observer pipeline ----

type ObserverStep = { key: string; sub: string; emphasize?: boolean };

const OBSERVER_STEPS: ObserverStep[] = [
  { key: "on_arrival", sub: "ScheduledRuntime" },
  { key: "on_admit", sub: "first engine event" },
  { key: "on_token", sub: "TTFT = first", emphasize: true },
  { key: "on_usage", sub: "endpoint usage" },
  { key: "on_terminal", sub: "future Ready" },
];

/**
 * Internals page of the Offline Co-simulation deck.
 *
 * Ports the source canvas's internals view: the layered `AIPERF -> DYNAMO MOCKER (PASSIVE)` stack
 * (dependency strictly one-way), a Poll/Compare/Advance/Step/Route drive-loop walkthrough driven
 * by {@link useStepSimulator} over the source's fixed FRAMES, and the Level-B observer pipeline.
 */
export function InternalsPage(): React.JSX.Element {
  const sim = useStepSimulator(FRAMES, { autoPlayMs: 1400 });
  const frame = sim.current ?? FRAMES[0]!;

  return (
    <Stack gap={16}>
      <div>
        <h2 className={`text-lg font-semibold ${inkClassName("primary")}`}>Inside the boundary</h2>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          Inside the boundary: AIPerf&apos;s driver owns the loop; <strong>EngineHost</strong>{" "}
          adapts the passive Dynamo steppable core as a <strong>SimEventSource</strong>; events
          route through waiters into a shared <strong>RequestObserver</strong> tee.
        </p>
      </div>

      {/* LAYERED INTERNALS — DEPENDENCY IS STRICTLY AIPERF -> MOCKER */}
      <AutoLayoutFlow nodes={stackNodes} edges={stackEdges} layout={{ direction: "DOWN" }} height={500} />
      <p className={`text-xs ${inkClassName("tertiary")}`}>
        step_until(until_ms) · next_event_ms · submit — scheduler math unchanged; only driver loops
        inverted
      </p>

      <Divider />

      {/* Drive loop interactive walkthrough */}
      <div>
        <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>The drive loop</h3>
        <p className={`mt-1 max-w-3xl text-sm ${inkClassName("secondary")}`}>
          Step the Poll → Compare → Advance → Step → Route cycle. The highlighted stage is what&apos;s
          executing; the virtual-time ruler advances only when the loop advances it.
        </p>
      </div>

      <Row gap={8} align="center" wrap>
        <Button variant="secondary" onClick={sim.back} disabled={sim.isFirst}>
          Back
        </Button>
        <Button variant="primary" onClick={sim.next} disabled={sim.isLast}>
          Step
        </Button>
        <Button variant="ghost" onClick={sim.reset} disabled={sim.isFirst}>
          Reset
        </Button>
        <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>
          {sim.index + 1} / {sim.total}
        </span>
      </Row>

      <Grid columns={5} gap={10}>
        {STAGE_NAMES.map((name, idx) => {
          const active = idx === frame.stage;
          return (
            <div
              key={name}
              className={clsx(
                "rounded-lg border px-3 py-4 text-center shadow-sm",
                active
                  ? "border-accent-primary bg-accent-primary text-white"
                  : clsx(strokeClassName("primary"), surfaceClassName("elevated")),
              )}
            >
              <div className="text-sm font-semibold">{name}</div>
              <div className={clsx("mt-1 text-[10px]", active ? "text-white/80" : inkClassName("tertiary"))}>
                {STAGE_SUBS[idx]}
              </div>
            </div>
          );
        })}
      </Grid>

      <Callout tone={frame.stage === 4 ? "success" : "info"} title={frame.title}>
        {frame.cap}
      </Callout>

      <div
        className={clsx("rounded-lg border p-4 shadow-sm", strokeClassName("secondary"), surfaceClassName("elevated"))}
      >
        <div className="mb-2 flex items-center justify-between">
          <span className={`text-sm font-semibold ${inkClassName("primary")}`}>Virtual time</span>
          <span
            className={`rounded-md border px-2 py-0.5 text-[11px] ${strokeClassName("secondary")} ${inkClassName("tertiary")}`}
          >
            SimClock
          </span>
        </div>
        <Row gap={0} align="center">
          {TICKS.map((tick, idx) => {
            const on = idx === frame.vt;
            const isLast = idx === TICKS.length - 1;
            return (
              <div key={tick.ms} className="flex flex-1 flex-col items-center gap-1">
                <div className={clsx("h-2 w-0.5", on ? "bg-accent-primary" : "bg-stroke-secondary")} />
                <span className={clsx("text-[11px]", on ? inkClassName("primary") : inkClassName("tertiary"))}>
                  {isLast ? `${tick.ms} ms` : tick.ms}
                </span>
              </div>
            );
          })}
        </Row>
      </div>

      <Divider />

      <h3 className={`text-base font-semibold ${inkClassName("primary")}`}>Level-B observer contract</h3>

      {/* ObserverPipeline: on_arrival -> on_admit -> on_token (TTFT) -> on_usage -> on_terminal */}
      <Grid columns={5} gap={10}>
        {OBSERVER_STEPS.map((step) => (
          <div
            key={step.key}
            className={clsx(
              "rounded-lg border px-3 py-3 text-center shadow-sm",
              step.emphasize
                ? "border-accent-primary bg-accent-primary text-white"
                : clsx(strokeClassName("primary"), surfaceClassName("elevated")),
            )}
          >
            <div className="text-sm font-semibold">{step.key}</div>
            <div className={clsx("mt-1 text-[10px]", step.emphasize ? "text-white/80" : inkClassName("tertiary"))}>
              {step.sub}
            </div>
          </div>
        ))}
      </Grid>
      <p className={`text-xs ${inkClassName("tertiary")}`}>
        ObserverTee fans each callback to CollectorObserver + NativeMetricsObserver (+ optional
        TraceCollector).
      </p>

      <Grid columns={2} gap={12}>
        <Callout tone="info" title="Engine never sees Clock">
          <strong>step_until</strong> and <strong>next_event_ms</strong> take scalar milliseconds.
          Dependency is one-way: aiperf → mocker.
        </Callout>
        <Callout tone="info" title="Live, not post-hoc">
          Per-token callbacks fire during the run, so adaptive windows, streaming metrics, and
          dashboards work offline.
        </Callout>
      </Grid>
    </Stack>
  );
}
