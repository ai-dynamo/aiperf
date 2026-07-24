/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `rust-port-flow` (v2) — the request-lifecycle deck as a SWIMLANE-TIMELINE. This is the deck
//! SHELL: it composes the shared `src/interactive/` primitives with the AIPerf pipeline content.
//! The level-0 OVERVIEW is now a `TimelineTrack` — one request line riding a horizontal time axis
//! through six subsystem swimlanes, grouped inside three nested seam frames — driven entirely by the
//! stages' `lane`/`events` metadata (`buildTimelineModel`). The Clock `SeamToggle` rescales the
//! x-axis (RealClock wall-ms ↔ SimClock virtual ticks); the Transport `SeamToggle` reroutes the
//! dispatch hop; `useFlowPlayer` rides the request line (its active event highlights the owning
//! region). Clicking a region drills into that stage via the existing `ZoomStage` — the v1
//! subgraph/leaves are now the stage's DRILL detail. Per-stage lane/events + drill detail live in one
//! `stages/<id>` module each, so stage/lane agents extend the deck without editing this shell.

import { useCallback, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { TopBar } from "../../shell/TopBar.js";
import { Stack } from "../../layout/Stack.js";
import { Row } from "../../layout/Row.js";
import { Button } from "../../prose/Button.js";
import { Callout } from "../../prose/Callout.js";
import { Legend } from "../../prose/Legend.js";
import { Eyebrow } from "../../prose/Eyebrow.js";
import { inkClassName, strokeClassName, surfaceClassName } from "../../theme/tokens.js";
import {
  PipelineCanvas,
  RequestParticle,
  SeamToggle,
  TimelineTrack,
  ZoomStage,
  useFlowPlayer,
  type FlowStep,
  type SeamToggleOption,
  type TimelineEvent,
  type TimelineScale,
} from "../../interactive/index.js";
import type { ElkOptions } from "../../layout/graph/index.js";

/**
 * Stable ELK options for every stage/leaf drill-down canvas: left→right layered layout with
 * measured node sizes. Module-level so its identity is stable across renders (the hook relayouts
 * when `opts` identity changes). Replaces the stages' now-ignored hand-authored `position` hints.
 */
const STAGE_LAYOUT: ElkOptions = { direction: "RIGHT" };
import {
  OVERVIEW_ID,
  NODE_ROLE_LEGEND,
  buildTimelineModel,
  buildZoomTree,
  type StageDef,
  type StageEvidence,
} from "./stage.js";
import { bigPictureStage } from "./stages/bigPicture.js";
import { runtimeStage } from "./stages/runtime.js";
import { datasetStage } from "./stages/dataset-load.js";
import { datasetShareStage } from "./stages/dataset-share.js";
import { workersStage } from "./stages/workers.js";
import { clockStage } from "./stages/clock.js";
// transport.ts (foundation stub) and transport.tsx (real content) share a basename; import the
// real one by explicit extension so resolution is deterministic (allowImportingTsExtensions is on).
import { transportStage } from "./stages/transport.tsx";
import { hotPathStage } from "./stages/hotpath.js";
import { aggregationResultsStage } from "./stages/aggregation-results.js";

/**
 * The stage registry the deck maps over. To add real content for a stage, a stage/lane agent edits
 * that stage's own module under `./stages/` (its `lane`/`events` for the timeline, plus
 * `subgraph`/`leaves`/`evidence` for the drill detail) — the nine slots below already exist, so the
 * shell is never edited. Order in this array is irrelevant; the timeline is laid out by each stage's
 * `events` (order + wall-ms) and the overview by each stage's `order`.
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

// Real sink types per transport target (the two-trait seam swap), named in the dispatch caption.
const TRANSPORT_SINK: Record<Transport, string> = {
  http: "TransportSink (hyper, streaming)",
  grpc: "GrpcTransportSink (Tonic — unary, server-streaming, or bidi per endpoint)",
  "dry-run": "the dry-run sink (no network round-trip)",
  dynosim: "the dynosim SteppableEngine sink (virtual time)",
};

/** Caption for a request event; the dispatch/clock events reflect the current seam selections. */
function eventCaption(
  stage: StageDef,
  event: TimelineEvent,
  transport: Transport,
  clock: ClockMode,
): string {
  if (event.id === "tp-dispatch") {
    return `Rc<dyn Dispatcher> routes the request through ${TRANSPORT_SINK[transport]} — same upstream, one of the two-trait sinks.`;
  }
  if (event.id === "ck-select") {
    return clock === "sim"
      ? "Clock::is_virtual() → SimClock advances virtual time in discrete integer-nanosecond hops (the simulation driver)."
      : "Clock::is_virtual() → RealClock paces the request against monotonic wall-clock time (the real reactor).";
  }
  return `${event.label} — ${stage.caption}`;
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
 * The `rust-port-flow` v2 deck shell. Composes the swimlane-timeline overview + semantic-zoom drill
 * + play primitives with the AIPerf request-lifecycle content, driven by the `STAGES` registry.
 */
export function RustPortFlowDeck(): React.JSX.Element {
  const [clock, setClock] = useState<ClockMode>("real");
  const [transport, setTransport] = useState<Transport>("http");

  const tree = useMemo(() => buildZoomTree(STAGES), []);
  const model = useMemo(() => buildTimelineModel(STAGES), []);

  // Drill state is mirrored in the `?stage=` URL param so a drilled view is a shareable deep link
  // and browser back/forward walks the drill path. The param holds the post-overview ids joined by
  // "/"; we rebuild the full path from the root, keeping only valid consecutive child steps.
  const [searchParams, setSearchParams] = useSearchParams();
  const stageParam = searchParams.get("stage") ?? "";
  const drillPath = useMemo(() => {
    const path: string[] = [OVERVIEW_ID];
    for (const seg of stageParam.split("/").filter(Boolean)) {
      const parent = tree[path[path.length - 1]!];
      if (parent?.children?.includes(seg) && tree[seg]) {
        path.push(seg);
      } else {
        break;
      }
    }
    return path;
  }, [stageParam, tree]);

  const handleNavigate = useCallback(
    (path: readonly string[]) => {
      const segments = path.slice(1).join("/");
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          if (segments) {
            next.set("stage", segments);
          } else {
            next.delete("stage");
          }
          return next;
        },
        { replace: false },
      );
    },
    [setSearchParams],
  );

  // eventId → its owning stage + event, for captions and the active-event label.
  const eventOwner = useMemo(() => {
    const map = new Map<string, { stage: StageDef; event: TimelineEvent }>();
    for (const stage of STAGES) {
      for (const event of stage.events) {
        map.set(event.id, { stage, event });
      }
    }
    return map;
  }, []);

  // One play step per request event, in path order; captions reflect the current seams.
  const steps: FlowStep[] = useMemo(
    () =>
      model.requestPath.map((id) => {
        const owner = eventOwner.get(id)!;
        return {
          nodeId: id,
          caption: eventCaption(owner.stage, owner.event, transport, clock),
          variant: transport,
          timingMs: clock === "sim" ? 260 : 820,
        };
      }),
    [model, eventOwner, transport, clock],
  );

  const player = useFlowPlayer(steps, { autoPlayMs: clock === "sim" ? 260 : 820 });
  const scale: TimelineScale = clock === "sim" ? "virtual" : "real";
  const activeEventId = player.activeNodeId;
  const activeEventLabel =
    activeEventId !== undefined ? eventOwner.get(activeEventId)?.event.label : undefined;

  return (
    <div className="flex h-screen flex-col bg-surface-chrome">
      <TopBar section="Rust Port · Request Lifecycle" />
      <div className="min-h-0 flex-1 overflow-auto">
        <div className={`mx-auto min-h-full max-w-6xl 2xl:max-w-[1728px] px-10 py-8 ${surfaceClassName("page")}`}>
          <Stack gap={18}>
            <div>
              <h1 className={`text-2xl font-bold ${inkClassName("primary")}`}>
                One request&apos;s life through the AIPerf Rust port
              </h1>
              <p className={`mt-1 max-w-4xl text-sm ${inkClassName("secondary")}`}>
                One request&apos;s nine stages, in order left-to-right across six subsystem swimlanes,
                grouped inside the Clock / Workload / Transport seam frames. Press Play to step the
                request through its events; the Clock seam switches RealClock ↔ SimClock and the
                Transport seam reroutes the dispatch hop. Click any stage to drill into its subsystem —
                every box there is colored by what it is. Esc or the breadcrumb backs out; arrow keys
                move between sibling stages.
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
              nodeLabel={activeEventLabel}
              tone={clock === "sim" ? "purple" : "green"}
            />

            <ZoomStage
              tree={tree}
              rootId={OVERVIEW_ID}
              initialPath={drillPath}
              onNavigate={handleNavigate}
            >
              {(ctx) => {
                const isOverview = ctx.level === 0;
                const stage = ctx.node.data;
                return (
                  <Stack gap={14}>
                    {isOverview ? (
                      <TimelineTrack
                        lanes={model.lanes}
                        regions={model.regions}
                        events={model.events}
                        seamFrames={model.seamFrames}
                        requestPath={model.requestPath}
                        activeEventId={activeEventId}
                        scale={scale}
                        onRegionClick={(id) => ctx.drill(id)}
                      />
                    ) : (
                      <Stack gap={12}>
                        <PipelineCanvas
                          nodes={ctx.node.nodes}
                          edges={ctx.node.edges}
                          heightClass="h-[68vh] min-h-[480px]"
                          onNodeClick={(id) => ctx.drill(id)}
                          layout={STAGE_LAYOUT}
                        />
                        {/* Color key: every box is tinted by what it IS (semantic role). */}
                        <Legend
                          entries={NODE_ROLE_LEGEND.map((r) => ({ color: r.color, label: r.label }))}
                        />
                        {stage && (
                          <Stack gap={12}>
                            <Callout tone="info" title={stage.label}>
                              {stage.caption}
                            </Callout>
                            {stage.evidence && stage.evidence.length > 0 && (
                              <EvidenceRow items={stage.evidence} />
                            )}
                          </Stack>
                        )}
                      </Stack>
                    )}
                  </Stack>
                );
              }}
            </ZoomStage>

            <div className={`border-t pt-3 text-xs ${strokeClassName("secondary")} ${inkClassName("tertiary")}`}>
              Six subsystem swimlanes, nine stages from the narrative spine. Each stage&apos;s timeline
              events and drill detail are authored in its own module under <code>stages/</code>; the
              swimlane-timeline overview and interaction shell are shared.
            </div>
          </Stack>
        </div>
      </div>
    </div>
  );
}
