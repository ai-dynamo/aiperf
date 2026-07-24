/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `rust-port-flow` — an interactive, zoomable request-lifecycle deck for the AIPerf Rust port.
//! This is the deck SHELL: it composes the shared `src/interactive/` primitives (ZoomStage +
//! PipelineCanvas + useFlowPlayer/RequestParticle + SeamToggle) with the AIPerf pipeline content.
//! The level-0 overview is a real React Flow diagram of the 9 spine stages; clicking a stage drills
//! into it. Per-stage detail lives in one `stages/<id>.ts` module each (the `STAGES` registry
//! below), so stage agents extend the deck without editing this shell.

import { useMemo, useState } from "react";
import type { Node } from "@xyflow/react";
import clsx from "clsx";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Button } from "../../prose/Button.js";
import { Callout } from "../../prose/Callout.js";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";
import {
  PipelineCanvas,
  RequestParticle,
  SeamToggle,
  ZoomStage,
  useFlowPlayer,
  type FlowStep,
  type SeamToggleOption,
} from "../../interactive/index.js";
import { OVERVIEW_ID, buildZoomTree, type StageDef, type StageEvidence } from "./stage.js";
import { bigPictureStage } from "./stages/bigPicture.js";
import { runtimeStage } from "./stages/runtime.js";
import { datasetStage } from "./stages/dataset-load.js";
import { datasetShareStage } from "./stages/dataset-share.js";
import { workersStage } from "./stages/workers.js";
import { clockStage } from "./stages/clock.js";
// transport.ts (foundation stub) and transport.tsx (real content) share a basename; import the
// real one by explicit extension so resolution is deterministic (allowImportingTsExtensions is on).
import { transportStage, transportFlowSteps } from "./stages/transport.tsx";
import { hotPathStage, hotPathSteps } from "./stages/hotpath.js";
import { aggregationResultsStage, aggregationResultsSteps } from "./stages/aggregation-results.js";

/**
 * The stage registry the deck maps over. To add real content for a stage, a stage agent edits that
 * stage's own module under `./stages/` (filling in `subgraph`/`leaves`/`evidence` on its exported
 * `StageDef`) — the nine slots below already exist, so the shell is never edited. Order in this
 * array is irrelevant; the overview is laid out and wired by each stage's `order` field.
 */
export const STAGES: readonly StageDef[] = [
  bigPictureStage,
  runtimeStage,
  datasetStage,
  datasetShareStage,
  workersStage,
  clockStage,
  transportStage,
  hotPathStage,
  aggregationResultsStage,
];

type ClockMode = "real" | "sim";
type Transport = "http" | "grpc" | "dry-run" | "dynosim";

const CLOCK_OPTIONS: SeamToggleOption<ClockMode>[] = [
  { value: "real", label: "RealClock", tone: "green" },
  { value: "sim", label: "SimClock", tone: "purple" },
];

const TRANSPORT_OPTIONS: SeamToggleOption<Transport>[] = [
  { value: "http", label: "HTTP", tone: "blue" },
  { value: "grpc", label: "gRPC", tone: "cyan" },
  { value: "dry-run", label: "dry-run", tone: "gray" },
  { value: "dynosim", label: "dynosim", tone: "orange" },
];

// Real sink types per transport target (the two-trait seam swap), named in the play caption.
const TRANSPORT_SINK: Record<Transport, string> = {
  http: "TransportSink (hyper, streaming)",
  grpc: "GrpcTransportSink (Tonic, non-streaming)",
  "dry-run": "the dry-run sink",
  dynosim: "the dynosim SteppableEngine sink",
};

/** The active-node highlight applied to the overview stage the play head is on. Static literals for Tailwind. */
const HIGHLIGHT_CLASS = "ring-2 ring-accent-primary shadow-md";

/** Caption for a stage's play step; Transport/Clock steps reflect the current seam selections. */
function stageStepCaption(stage: StageDef, transport: Transport, clock: ClockMode): string {
  if (stage.id === "transport") {
    return `Dispatcher routes the request through ${TRANSPORT_SINK[transport]} — same upstream, one of the two-trait sinks.`;
  }
  if (stage.id === "clock") {
    return clock === "sim"
      ? "SimClock advances virtual time in discrete integer-nanosecond hops — the simulation driver."
      : "RealClock paces the request against wall-clock time — the real reactor.";
  }
  return stage.caption;
}

/** One overview play step per stage, in spine order; pacing follows the selected clock. */
function buildOverviewSteps(
  stages: readonly StageDef[],
  transport: Transport,
  clock: ClockMode,
): FlowStep[] {
  return [...stages]
    .sort((a, b) => a.order - b.order)
    .map((stage) => ({
      nodeId: stage.id,
      caption: stageStepCaption(stage, transport, clock),
      variant: transport,
      timingMs: clock === "sim" ? 250 : 900,
    }));
}

// The transport target → the real level-2 sink node id whose FlowStep the request is routed through
// (the two-trait seam swap). Keys match the `transport.tsx` leaf ids so the sink hop reroutes live.
const TRANSPORT_SINK_NODE: Record<Transport, string> = {
  http: "transport-http",
  grpc: "transport-grpc",
  "dry-run": "transport-dry-run",
  dynosim: "transport-dynosim",
};

// Human labels for the lifecycle particle's active hop (static literals — no Tailwind involvement).
const LIFECYCLE_LABELS: Record<string, string> = {
  "hp-workload": "RequestRateWorkload",
  "hotpath.admission": "SlotPool admission",
  "hotpath.dispatch": "Rc<dyn Dispatcher>",
  "transport-http": "TransportSink (HTTP · hyper)",
  "transport-grpc": "GrpcTransportSink (gRPC · Tonic)",
  "transport-dry-run": "DryRunTransportFactoryV2",
  "transport-dynosim": "SteppableEngine (dynosim)",
  "hp-reduce": "reduce_parsed_response · TTFT",
  "hp-measure": "measure_dispatch",
  "agg-observer": "NativeMetricsObserver",
  "agg-boundary": "Deterministic boundary merge",
  "agg-registry": "ExporterRegistry",
  "agg-terminal": "RunTerminalV2 · report_path",
};

/** Resolve a real FlowStep fragment by node id; fragments are static, so a miss is a wiring bug. */
function pickStep(fragment: readonly FlowStep[], nodeId: string): FlowStep {
  const step = fragment.find((s) => s.nodeId === nodeId);
  if (step === undefined) {
    throw new Error(`rust-port-flow: no FlowStep "${nodeId}" in fragment`);
  }
  return step;
}

/**
 * The full request-lifecycle FlowStep[] the play-layer particle traverses, assembled from the
 * stages' OWN verified fragments (hot-path, transport, aggregation): issued at the workload →
 * SlotPool admission → `Rc<dyn Dispatcher>` → the chosen transport sink (SSE tokens begin here on
 * HTTP) → TTFT/reduce → measure → worker-local `NativeMetricsObserver` → deterministic boundary
 * merge → `ExporterRegistry` → `RunTerminalV2`. The Transport seam reroutes the sink hop; the Clock
 * seam replaces every hop (discrete virtual-time hops vs wall-paced).
 */
function buildRequestLifecycleSteps(transport: Transport, clock: ClockMode): FlowStep[] {
  const timingMs = clock === "sim" ? 220 : 820;
  const spine: FlowStep[] = [
    pickStep(hotPathSteps, "hp-workload"), // issue
    pickStep(hotPathSteps, "hotpath.admission"), // SlotPool admission
    pickStep(hotPathSteps, "hotpath.dispatch"), // Dispatcher
    pickStep(transportFlowSteps, TRANSPORT_SINK_NODE[transport]), // chosen transport sink
    pickStep(hotPathSteps, "hp-reduce"), // SSE tokens → TTFT latch
    pickStep(hotPathSteps, "hp-measure"), // reduce/measure record
    pickStep(aggregationResultsSteps, "agg-observer"), // NativeMetricsObserver
    pickStep(aggregationResultsSteps, "agg-boundary"), // deterministic merge
    pickStep(aggregationResultsSteps, "agg-registry"), // exporter
    pickStep(aggregationResultsSteps, "agg-terminal"), // terminal report_path
  ];
  return spine.map((step) => ({ ...step, timingMs, variant: transport }));
}

/** Append the highlight class to the node the play head is currently on. */
function applyHighlight(nodes: Node[], activeId: string | undefined): Node[] {
  if (activeId === undefined) {
    return nodes;
  }
  return nodes.map((node) =>
    node.id === activeId
      ? {
          ...node,
          data: {
            ...node.data,
            className: clsx((node.data as { className?: string }).className, HIGHLIGHT_CLASS),
          },
        }
      : node,
  );
}

/** Local source-anchor row (mirrors the architecture deck's `EvidenceRow`; kept deck-local to avoid cross-deck coupling). */
function EvidenceRow({ items }: { items: ReadonlyArray<StageEvidence> }): React.JSX.Element {
  return (
    <div>
      <Eyebrow className="mb-2">Source anchors</Eyebrow>
      <Row gap={8} wrap>
        {items.map((item) => (
          <span
            key={item.path + item.label}
            className={`inline-flex items-center gap-2 rounded-md border px-3 py-1 text-xs shadow-sm ${strokeClassName("secondary")}`}
          >
            <span className={`font-medium ${inkClassName("secondary")}`}>{item.label}</span>
            <code className={inkClassName("tertiary")}>{item.path}</code>
          </span>
        ))}
      </Row>
    </div>
  );
}

/**
 * The `rust-port-flow` deck shell. Composes the shared semantic-zoom + play primitives with the
 * AIPerf request-lifecycle content, driven by the `STAGES` registry.
 */
export function RustPortFlowDeck(): React.JSX.Element {
  const [clock, setClock] = useState<ClockMode>("real");
  const [transport, setTransport] = useState<Transport>("http");

  const tree = useMemo(() => buildZoomTree(STAGES), []);
  const steps = useMemo(() => buildOverviewSteps(STAGES, transport, clock), [transport, clock]);
  const player = useFlowPlayer(steps, { autoPlayMs: clock === "sim" ? 250 : 900 });

  // The finer full-pipeline particle: one request's life through the real hot-path + aggregation
  // fragments, rerouted by the Transport seam and replaced by the Clock seam.
  const lifecycleSteps = useMemo(() => buildRequestLifecycleSteps(transport, clock), [transport, clock]);
  const lifecyclePlayer = useFlowPlayer(lifecycleSteps, { autoPlayMs: clock === "sim" ? 220 : 820 });
  const lifecycleLabel =
    lifecyclePlayer.activeNodeId !== undefined ? LIFECYCLE_LABELS[lifecyclePlayer.activeNodeId] : undefined;

  const activeStage = STAGES.find((stage) => stage.id === player.activeNodeId);

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Rust Port · Request Lifecycle" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className={`mx-auto min-h-full max-w-6xl px-10 py-8 ${surfaceClassName("page")}`}>
          <Stack gap={18}>
            <div>
              <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>
                One request&apos;s life through the AIPerf Rust port
              </h1>
              <p className={`mt-1 max-w-4xl text-sm ${inkClassName("secondary")}`}>
                A single zoomable canvas: start at the big picture, click any of the nine pipeline
                stages to drill in, and press Play to send a request through — flip the Clock and
                Transport seams to watch the same request re-route. Esc or the breadcrumb backs out;
                arrow keys move between sibling stages.
              </p>
            </div>

            <Row gap={24} align="center" wrap>
              <SeamToggle
                label="Clock"
                options={CLOCK_OPTIONS}
                value={clock}
                onChange={setClock}
                ariaLabel="Clock mode"
              />
              <SeamToggle
                label="Transport"
                options={TRANSPORT_OPTIONS}
                value={transport}
                onChange={setTransport}
                ariaLabel="Transport target"
              />
            </Row>

            <Row gap={10} align="center" wrap>
              <Button
                variant="primary"
                onClick={player.togglePlay}
                disabled={player.isLast && !player.isPlaying}
              >
                {player.isPlaying ? "Pause" : "Play"}
              </Button>
              <Button variant="secondary" onClick={player.back} disabled={player.isFirst}>
                Back
              </Button>
              <Button variant="secondary" onClick={player.next} disabled={player.isLast}>
                Next
              </Button>
              <Button variant="ghost" onClick={player.reset}>
                Reset
              </Button>
              <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>
                step {player.index + 1}/{player.total}
              </span>
            </Row>

            <RequestParticle
              step={player.current}
              position={player.index + 1}
              total={player.total}
              nodeLabel={activeStage?.label}
              tone="cyan"
            />

            <ZoomStage tree={tree} rootId={OVERVIEW_ID}>
              {(ctx) => {
                const isOverview = ctx.level === 0;
                const highlightId = isOverview ? player.activeNodeId : undefined;
                const nodes = applyHighlight(ctx.node.nodes, highlightId);
                const stage = ctx.node.data;
                return (
                  <Stack gap={14}>
                    <PipelineCanvas
                      nodes={nodes}
                      edges={ctx.node.edges}
                      height={isOverview ? 470 : 340}
                      onNodeClick={(id) => ctx.drill(id)}
                    />
                    {isOverview ? (
                      <p className={`text-sm ${inkClassName("tertiary")}`}>
                        Click any stage node to zoom in on its subsystem.
                      </p>
                    ) : (
                      stage && (
                        <Stack gap={12}>
                          <Callout tone="info" title={stage.label}>
                            {stage.caption}
                          </Callout>
                          {stage.evidence && stage.evidence.length > 0 && (
                            <EvidenceRow items={stage.evidence} />
                          )}
                        </Stack>
                      )
                    )}
                  </Stack>
                );
              }}
            </ZoomStage>

            <Stack gap={10}>
              <div>
                <Eyebrow>Full request lifecycle</Eyebrow>
                <p className={`mt-1 max-w-4xl text-sm ${inkClassName("secondary")}`}>
                  Watch one request&apos;s life traverse the real hot-path and aggregation spine:
                  issued at <code>RequestRateWorkload</code> → <code>SlotPool</code> admission →{" "}
                  <code>Rc&lt;dyn Dispatcher&gt;</code> → the{" "}
                  <span className={`font-semibold ${inkClassName("primary")}`}>
                    {LIFECYCLE_LABELS[TRANSPORT_SINK_NODE[transport]]}
                  </span>{" "}
                  sink → SSE tokens / TTFT → <code>reduce</code>/<code>measure</code> →{" "}
                  <code>NativeMetricsObserver</code> → deterministic merge → <code>ExporterRegistry</code>.
                  Flip Transport to reroute the sink; flip Clock to change the pacing.
                </p>
              </div>

              <RequestParticle
                step={lifecyclePlayer.current}
                position={lifecyclePlayer.index + 1}
                total={lifecyclePlayer.total}
                nodeLabel={lifecycleLabel}
                tone={clock === "sim" ? "purple" : "green"}
              />

              <Row gap={10} align="center" wrap>
                <Button
                  variant="primary"
                  onClick={lifecyclePlayer.togglePlay}
                  disabled={lifecyclePlayer.isLast && !lifecyclePlayer.isPlaying}
                >
                  {lifecyclePlayer.isPlaying ? "Pause request" : "Play request"}
                </Button>
                <Button variant="secondary" onClick={lifecyclePlayer.back} disabled={lifecyclePlayer.isFirst}>
                  Back token
                </Button>
                <Button variant="secondary" onClick={lifecyclePlayer.next} disabled={lifecyclePlayer.isLast}>
                  Next token
                </Button>
                <Button variant="ghost" onClick={lifecyclePlayer.reset}>
                  Reset request
                </Button>
                <span className={`text-xs font-medium ${inkClassName("tertiary")}`}>
                  hop {lifecyclePlayer.index + 1}/{lifecyclePlayer.total}
                </span>
              </Row>
            </Stack>

            <div className={`border-t pt-3 text-xs ${strokeClassName("secondary")} ${inkClassName("tertiary")}`}>
              Nine stages from the narrative spine. Detail views for each stage are authored in their
              own module under <code>stages/</code>; the overview and interaction shell are shared.
            </div>
          </Stack>
        </div>
      </div>
    </div>
  );
}
