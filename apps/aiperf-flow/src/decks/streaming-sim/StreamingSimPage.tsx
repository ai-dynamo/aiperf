/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Narrated walkthrough of the streaming Dynamo shadow-replay pipeline.
//!
//! Each beat walks one stage of the pipeline — source acquisition, session coordination,
//! action dispatch, results compaction — with the voice driving the highlighted stage.
//! Grounded in `rust/runtime/src/streaming/` and `rust/runtime/src/engine/streaming_execution.rs`.

import { motion, useReducedMotion } from "motion/react";
import { PresentationShell } from "../../shell/PresentationShell.js";
import { useNarratedDeck } from "../../audio/index.js";
import type { SlideDefinition } from "../../deck/types.js";
import type { BeatAnchor } from "../../spike/useBeatClock.js";
import { useBeatClock } from "../../spike/useBeatClock.js";

// ---------------------------------------------------------------------------
// Beats
// ---------------------------------------------------------------------------

type Beat = BeatAnchor & { title: string; lede: string; caption: string; active: readonly string[] };

const BEATS: Beat[] = [
  {
    endAt: 0.07,
    title: "Trace files land in object store",
    lede: "S3SourceConfig: bucket, prefix, endpoint_url, force_path_style, inventory policy.",
    caption: "rust/runtime/src/streaming/sources/s3.rs — S3SourceConfig, validate_s3_policy, VersionedPrefixSnapshot.",
    active: ["s3", "local"],
    narration:
      "The pipeline begins at a source. Dynamo writes compressed trace files — gzipped newline-delimited JSON — into an S3-compatible bucket. The source component enumerates the prefix, acquires each object exactly once into an immutable snapshot, and seals the inventory. Nothing downstream ever re-reads the object store; the source hands an owned snapshot to the format decoder.",
  },
  {
    endAt: 0.14,
    title: "DynamoFormat decodes the wire",
    lede: "Decompress, parse JSON, validate schema, emit DatasetActionV1 records.",
    caption: "rust/runtime/src/streaming/sources/ — DynamoFormat, dynamo schema, action binding registration.",
    active: ["dynamo_fmt"],
    narration:
      "The format decoder receives the acquired snapshot and parses it. Each gzipped shard is decompressed in a no-follow private read, its JSON lines parsed against a strict schema, and the result emitted as a stream of DatasetActionV1 records — one per trace line. Schema violations are logged and skipped; the pipeline never retries a parse failure.",
  },
  {
    endAt: 0.22,
    title: "ConversationCoordinator joins fragments",
    lede: "Fragment records arriving on separate partition keys are joined by stable session key.",
    caption: "rust/runtime/src/streaming/session/conversation.rs — joins across partitions by stable session key.",
    active: ["conv_coord"],
    narration:
      "Trace files arrive across many S3 prefixes and may contain fragments of the same logical conversation. The conversation coordinator joins those fragments by their stable session key. Authored turns and observed endpoint replies fold into one durable transcript in order. A turn pair closes on exactly one terminal event and on nothing else — whichever disposition the endpoint returned.",
  },
  {
    endAt: 0.30,
    title: "SessionClosurePolicy seals the session",
    lede: "Five checked proofs: authored close, hard watermark, verified finite seal, complete sorted run, exhausted predecessor policy.",
    caption: "rust/runtime/src/streaming/session/closure.rs — exactly five checked proofs, quarantined sessions retire to tombstone map.",
    active: ["closure_pol"],
    narration:
      "A session is only sealed when one of exactly five checked proofs holds: an authored close record, a hard watermark, a verified finite seal, a verified complete sorted run, or an exhausted missing-predecessor policy. Partition exhaustion alone never closes a session. A session that cannot be cleanly closed is quarantined and retired into a budgeted tombstone map keyed by input domain and session identifier — it will not resurface.",
  },
  {
    endAt: 0.37,
    title: "ActionHost emits sequenced actions",
    lede: "ActionInventoryLedger accumulates terminal membership at each dense global sequence position.",
    caption: "rust/runtime/src/streaming/action/host/inventory.rs — ActionInventoryLedger, FrozenActionInventory.",
    active: ["action_host"],
    narration:
      "The action host emits each session's actions in sequence. The inventory ledger accumulates every finalized action's terminal membership at its dense global sequence position. When all outstanding actions are accounted for, the ledger mints a frozen action inventory — the sealed proof the reliability reporter requires. It refuses to freeze if any outstanding action has not yet closed.",
  },
  {
    endAt: 0.45,
    title: "TurnClosureIntake gates each turn",
    lede: "Reserves a terminal slot before dispatch; records a Dropped terminal if the stop condition refuses.",
    caption: "rust/runtime/src/streaming/action/scheduled_request.rs — bounding admission because the runtime does not.",
    active: ["turn_intake"],
    narration:
      "Before each turn is dispatched, the turn closure intake reserves its terminal slot. Admission is bounded here because the underlying runtime does not bound it. If a stop condition refuses the issue, the action records a Dropped terminal rather than a run failure — the pipeline stays healthy and the inventory ledger accounts for the dropped slot normally.",
  },
  {
    endAt: 0.52,
    title: "StreamingPipeline routes actions",
    lede: "shadow_replay workload; admits exactly one profiling phase, requires the reliability policy digest to match the frozen execution plan.",
    caption: "rust/runtime/src/engine/streaming_execution.rs — StreamingPipeline, shadow_replay workload, SynthesisAuthority.",
    active: ["stream_pipe"],
    narration:
      "The streaming pipeline receives admitted actions and routes them. It is registered as the shadow underscore replay workload — its capability agreement admits exactly one profiling phase, refuses a Dynamo source or format composition when canonical content reconstruction has no registered factory, and requires the resolved reliability policy digest to equal the one recorded in the frozen execution plan.",
  },
  {
    endAt: 0.60,
    title: "LocalPlacement or CellularTransport delivers",
    lede: "Same-host: LocalPlacement via shared acquired snapshot. Remote: CellularTransport lands exact acquired set privately before compilation.",
    caption: "rust/runtime/src/cellular/ — LocalPlacement shares the controller snapshot; remote cells land privately.",
    active: ["local_place", "cell_trans"],
    narration:
      "Placement routes each action to where it will execute. Same-host placements share the controller's acquired snapshot. Remote cells receive the exact acquired set privately before compilation — the same files, landed privately, so no compiled graph intermediate ever crosses the cellular wire. Cross-host artifact transfer uses a per-run pinned-TLS channel authenticated after controller registration.",
  },
  {
    endAt: 0.68,
    title: "ScheduledRequestSink dispatches to the endpoint",
    lede: "Issues each aiperf.action.request.v1 through ScheduledRuntime; translates dispatch lifecycle to ActionExecutionEvents.",
    caption: "rust/runtime/src/streaming/action/scheduled_request.rs — the one action binding that reaches an inference endpoint.",
    active: ["sched_sink"],
    narration:
      "The scheduled request sink is the one action binding that reaches an inference endpoint. It issues each aiperf dot action dot request dot v1 action through the scheduled runtime, and translates the dispatch lifecycle back into action execution events at the existing first-token and completion hooks. Admission is bounded by the sink itself before reaching the runtime.",
  },
  {
    endAt: 0.75,
    title: "ActionInventory records each terminal",
    lede: "Dense global sequence; mints FrozenActionInventory only when all outstanding actions are closed.",
    caption: "rust/runtime/src/streaming/action/host/inventory.rs — refuses to freeze past an outstanding action.",
    active: ["action_inv"],
    narration:
      "As each dispatched action reaches its terminal — whether completed, failed, or dropped — the action inventory records its terminal membership at the dense global sequence position. The inventory can only be frozen when no outstanding action remains open. This invariant guarantees that a no-more-actions frontier can never outrun the terminals that close it.",
  },
  {
    endAt: 0.83,
    title: "EpochCoordinator advances the epoch",
    lede: "Ownership epochs and crash-safe route migration; committed transactionally.",
    caption: "rust/runtime/src/streaming/ — epoch coordination, transactional commit of route installation and barrier advancement.",
    active: ["epoch_coord"],
    narration:
      "The epoch coordinator advances the current ownership epoch. Route installation, artifact authorization, and barrier advancement commit transactionally — an epoch does not advance until all participants within it have committed. Crash recovery re-applies the committed epoch state from the durable spill run, whose crash-orphaned peers are reclaimed only after clock-driven owner-lease expiry through a bounded cursor scan.",
  },
  {
    endAt: 0.90,
    title: "ResultCompactor folds records",
    lede: "Per-record JSON, per-session summaries, OTLP/MLflow/W&B exporters; sketch-mode uses mergeable t-digests.",
    caption: "rust/runtime/src/streaming/ — ResultCompactor, delivery checkpoint, compacted result export.",
    active: ["result_comp"],
    narration:
      "The result compactor folds each epoch's records into summary statistics and emits them through the configured exporters — JSON, Parquet, OTLP, MLflow, or W&B. In exact mode every record is retained. In sketch mode counts, sums, extrema, and rate aggregates remain exact while percentiles and standard deviation are streaming t-digest estimates. Per-record outputs are unavailable in sketch mode.",
  },
  {
    endAt: 0.96,
    title: "DeliveryRestart guards resumable delivery",
    lede: "Resume verifies SynthesisAuthority matches the frozen plan before any participant initializes.",
    caption: "rust/runtime/src/streaming/ — DeliveryRestart, reliability reporter, resume guard.",
    active: ["delivery"],
    narration:
      "The delivery restart guard is the boundary between a run and a resume. Before any participant initializes, it verifies that the synthesis authority on the resumed plan matches the one frozen when the run was first authored. A disagreement is refused — not retried. This means a partially-completed run always resumes against exactly the same source shape it started with.",
  },
  {
    endAt: 1.0,
    title: "NativeReport emits the final results",
    lede: "Steady-state window, per-record artifacts, sweep metrics, W&B/MLflow sync.",
    caption: "rust/runtime/src/report/ — NativeReport; --steady-state derives the window from the in-flight concurrency sweep-line curve.",
    active: ["report"],
    narration:
      "Finally, the native report assembles the measurement output. With steady-state enabled, the window is derived from the shared in-flight concurrency sweep-line curve — the first up-crossing and last down-crossing of a fraction of the concurrency target — excluding ramp and drain. The result is a window-scoped summary in the report and in the star-aiperf dot JSON artifact. The full pipeline has run.",
  },
];

const SLIDES: readonly SlideDefinition[] = BEATS.map((b, i) => ({
  id: `streaming-sim-${i}`,
  eyebrow: `${String(i + 1).padStart(2, "0")} · STREAMING SHADOW REPLAY`,
  title: b.title,
  lede: b.lede,
  narration: b.narration,
  caption: b.caption,
  nodes: [],
  edges: [],
}));

// ---------------------------------------------------------------------------
// SVG pipeline diagram
// ---------------------------------------------------------------------------

/** Fixed layout in SVG viewBox units. */
const VW = 860;
const VH = 640;

// Node box: [id, label, cx, cy, w, h, layer, beatIndex]
type NodeDef = {
  id: string;
  label: string;
  cx: number;
  cy: number;
  w: number;
  h: number;
  layer: string;
  beat: number;
};

const NODES: readonly NodeDef[] = [
  { id: "s3",          label: "S3 / Object Store",        cx: 160, cy: 55,  w: 150, h: 32, layer: "src",    beat: 0 },
  { id: "local",       label: "Local Files",              cx: 340, cy: 55,  w: 120, h: 32, layer: "src",    beat: 0 },
  { id: "dynamo_fmt",  label: "DynamoFormat",             cx: 250, cy: 125, w: 150, h: 32, layer: "src",    beat: 1 },
  { id: "conv_coord",  label: "ConvCoordinator",          cx: 120, cy: 210, w: 155, h: 32, layer: "ses",    beat: 2 },
  { id: "closure_pol", label: "SessionClosurePolicy",     cx: 320, cy: 210, w: 165, h: 32, layer: "ses",    beat: 3 },
  { id: "action_host", label: "ActionHost",               cx: 530, cy: 210, w: 130, h: 32, layer: "ses",    beat: 4 },
  { id: "turn_intake", label: "TurnClosureIntake",        cx: 200, cy: 295, w: 155, h: 32, layer: "pipe",   beat: 5 },
  { id: "stream_pipe", label: "StreamingPipeline",        cx: 430, cy: 295, w: 155, h: 32, layer: "pipe",   beat: 6 },
  { id: "local_place", label: "LocalPlacement",           cx: 200, cy: 375, w: 140, h: 32, layer: "pipe",   beat: 7 },
  { id: "cell_trans",  label: "CellularTransport",        cx: 430, cy: 375, w: 150, h: 32, layer: "pipe",   beat: 7 },
  { id: "sched_sink",  label: "ScheduledRequestSink",     cx: 220, cy: 455, w: 170, h: 32, layer: "replay", beat: 8 },
  { id: "action_inv",  label: "ActionInventory",          cx: 460, cy: 455, w: 145, h: 32, layer: "replay", beat: 9 },
  { id: "epoch_coord", label: "EpochCoordinator",         cx: 100, cy: 535, w: 150, h: 32, layer: "res",    beat: 10 },
  { id: "result_comp", label: "ResultCompactor",          cx: 310, cy: 535, w: 145, h: 32, layer: "res",    beat: 11 },
  { id: "delivery",    label: "DeliveryRestart",          cx: 515, cy: 535, w: 140, h: 32, layer: "res",    beat: 12 },
  { id: "report",      label: "NativeReport",             cx: 310, cy: 615, w: 135, h: 32, layer: "eng",    beat: 13 },
];

// Edges: [fromId, toId]
const EDGES: readonly [string, string][] = [
  ["s3",          "dynamo_fmt"],
  ["local",       "dynamo_fmt"],
  ["dynamo_fmt",  "conv_coord"],
  ["conv_coord",  "closure_pol"],
  ["closure_pol", "action_host"],
  ["action_host", "turn_intake"],
  ["turn_intake", "stream_pipe"],
  ["stream_pipe", "local_place"],
  ["stream_pipe", "cell_trans"],
  ["local_place", "sched_sink"],
  ["cell_trans",  "sched_sink"],
  ["sched_sink",  "action_inv"],
  ["action_inv",  "epoch_coord"],
  ["epoch_coord", "result_comp"],
  ["result_comp", "delivery"],
  ["result_comp", "report"],
  ["delivery",    "report"],
];

const LAYER_COLOR: Record<string, string> = {
  src:    "var(--color-category-blue)",
  ses:    "var(--color-category-purple)",
  pipe:   "var(--color-category-cyan)",
  replay: "var(--color-category-green)",
  res:    "var(--color-category-orange)",
  eng:    "var(--color-category-yellow)",
};

const LAYER_LABEL: Record<string, string> = {
  src:    "SOURCE",
  ses:    "SESSION",
  pipe:   "PIPELINE",
  replay: "DISPATCH",
  res:    "RESULTS",
  eng:    "REPORT",
};

// Band background rects (y ranges)
const BANDS: { layer: string; y: number; h: number }[] = [
  { layer: "src",    y: 28,  h: 145 },
  { layer: "ses",    y: 183, h: 72  },
  { layer: "pipe",   y: 265, h: 128 },
  { layer: "replay", y: 427, h: 72  },
  { layer: "res",    y: 507, h: 72  },
  { layer: "eng",    y: 590, h: 56  },
];

function nodeById(id: string): NodeDef | undefined {
  return NODES.find((n) => n.id === id);
}

// Compute edge connection points (nearest box edge)
function edgePts(a: NodeDef, b: NodeDef): { x1: number; y1: number; x2: number; y2: number } {
  const dy = b.cy - a.cy;
  const dx = b.cx - a.cx;
  if (Math.abs(dy) < 16) {
    // horizontal
    const dir = dx > 0 ? 1 : -1;
    return { x1: a.cx + dir * a.w / 2, y1: a.cy, x2: b.cx - dir * b.w / 2, y2: b.cy };
  }
  if (dy > 0) {
    return { x1: a.cx + dx * 0.15, y1: a.cy + a.h / 2, x2: b.cx - dx * 0.15, y2: b.cy - b.h / 2 };
  }
  return { x1: a.cx, y1: a.cy - a.h / 2, x2: b.cx, y2: b.cy + b.h / 2 };
}

interface PipelineSvgProps {
  /** Smooth 0..BEATS.length position driven by useBeatClock */
  position: number;
  /** Current beat index from narrated.index */
  beatIndex: number;
}

function PipelineSvg({ position, beatIndex }: PipelineSvgProps): React.JSX.Element {
  const reducedMotion = useReducedMotion() ?? false;

  // Active node IDs for the current beat
  const activeBeat = BEATS[beatIndex];
  const activeIds = new Set(activeBeat?.active ?? []);

  // Packet position: center of the primary active node (first one listed)
  const primaryActiveId = activeBeat?.active[0];
  const packetNode = primaryActiveId !== undefined ? nodeById(primaryActiveId) : undefined;

  return (
    <svg
      viewBox={`0 0 ${VW} ${VH}`}
      width="100%"
      height="100%"
      style={{ display: "block" }}
    >
      {/* Band backgrounds */}
      {BANDS.map((b) => {
        const color = LAYER_COLOR[b.layer] ?? "transparent";
        const reached = NODES.filter((n) => n.layer === b.layer).some((n) => position >= n.beat - 0.3);
        return (
          <g key={b.layer}>
            <rect
              x={0} y={b.y} width={VW} height={b.h} rx={6}
              fill={color}
              fillOpacity={reached ? 0.06 : 0.02}
            />
            <text
              x={VW - 8} y={b.y + b.h / 2 + 5}
              textAnchor="end"
              fontSize={9} fontWeight="700" letterSpacing="0.12em"
              fill={color}
              fillOpacity={reached ? 0.7 : 0.3}
            >
              {LAYER_LABEL[b.layer]}
            </text>
          </g>
        );
      })}

      {/* Edges */}
      {EDGES.map(([fromId, toId]) => {
        const a = nodeById(fromId);
        const b = nodeById(toId);
        if (a === undefined || b === undefined) return null;
        const { x1, y1, x2, y2 } = edgePts(a, b);
        // Edge lights up when the source node's beat has been reached
        const lit = position >= a.beat + 0.5;
        const color = LAYER_COLOR[a.layer] ?? "var(--color-stroke-secondary)";
        return (
          <line
            key={`${fromId}-${toId}`}
            x1={x1} y1={y1} x2={x2} y2={y2}
            stroke={lit ? color : "var(--color-stroke-tertiary)"}
            strokeOpacity={lit ? 0.6 : 0.25}
            strokeWidth={lit ? 1.5 : 1}
            strokeDasharray={lit ? undefined : "3 3"}
          />
        );
      })}

      {/* Edge arrowheads */}
      <defs>
        {Object.entries(LAYER_COLOR).map(([layer, color]) => (
          <marker
            key={layer}
            id={`arrow-${layer}`}
            markerWidth="6" markerHeight="6"
            refX="5" refY="3"
            orient="auto"
          >
            <path d="M0,0 L6,3 L0,6 Z" fill={color} fillOpacity={0.5} />
          </marker>
        ))}
      </defs>

      {/* Nodes */}
      {NODES.map((node) => {
        const reached = position >= node.beat - 0.2;
        const active = activeIds.has(node.id);
        const color = LAYER_COLOR[node.layer] ?? "var(--color-stroke-secondary)";
        const opacity = reached ? 1 : 0.28;
        const { cx, cy, w, h } = node;

        return (
          <g key={node.id} opacity={opacity}>
            {/* Glow behind active node */}
            {active && !reducedMotion && (
              <motion.rect
                x={cx - w / 2 - 6} y={cy - h / 2 - 6}
                width={w + 12} height={h + 12}
                rx={10}
                fill={color}
                initial={{ fillOpacity: 0 }}
                animate={{ fillOpacity: [0.08, 0.22, 0.08] }}
                transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
              />
            )}

            {/* Node body */}
            <rect
              x={cx - w / 2} y={cy - h / 2}
              width={w} height={h}
              rx={5}
              fill={active ? color : "var(--color-surface-panel)"}
              fillOpacity={active ? 0.18 : 0.9}
              stroke={active ? color : "var(--color-stroke-secondary)"}
              strokeOpacity={active ? 1 : 0.4}
              strokeWidth={active ? 1.5 : 1}
            />

            {/* Label */}
            <text
              x={cx} y={cy + 5}
              textAnchor="middle"
              fontSize={11}
              fontWeight={active ? "700" : "500"}
              fill={active ? color : "var(--color-ink-primary)"}
              fillOpacity={active ? 1 : 0.8}
            >
              {node.label}
            </text>
          </g>
        );
      })}

      {/* Traveling packet */}
      {packetNode !== undefined && (
        <motion.circle
          key={`packet-${beatIndex}`}
          r={5}
          fill="var(--color-category-cyan)"
          stroke="var(--color-surface-page)"
          strokeWidth={2}
          animate={
            reducedMotion
              ? { cx: packetNode.cx, cy: packetNode.cy - packetNode.h / 2 - 8 }
              : {
                  cx: packetNode.cx,
                  cy: packetNode.cy - packetNode.h / 2 - 8,
                  scale: [1, 1.6, 1],
                  opacity: [1, 0.7, 1],
                }
          }
          transition={
            reducedMotion
              ? { duration: 0 }
              : {
                  cx: { type: "spring", stiffness: 120, damping: 18 },
                  cy: { type: "spring", stiffness: 120, damping: 18 },
                  scale: { duration: 1.2, repeat: Infinity, ease: "easeInOut" },
                  opacity: { duration: 1.2, repeat: Infinity, ease: "easeInOut" },
                }
          }
        />
      )}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function StreamingSimPage(): React.JSX.Element {
  const narrated = useNarratedDeck({
    narrations: BEATS.map((b) => b.narration),
    storagePrefix: "streaming-sim",
  });

  const { position } = useBeatClock(BEATS, narrated.index, narrated.activeWordIndex, BEATS.length);

  const beat = BEATS[narrated.index];
  const active = beat?.active ?? [];

  return (
    <PresentationShell
      slides={SLIDES}
      slideIndex={narrated.index}
      onSlideIndexChange={narrated.goTo}
      narrated={narrated}
      title="Streaming shadow replay"
    >
      <div className="flex h-full gap-4 overflow-hidden px-4 pt-3 pb-2">
        {/* SVG pipeline diagram — left column */}
        <div className="min-w-0 flex-1 overflow-hidden rounded-lg border border-white/10 bg-surface-elevated p-2">
          <PipelineSvg position={position} beatIndex={narrated.index} />
        </div>

        {/* Active stage detail — right column */}
        <div className="flex w-64 shrink-0 flex-col gap-3 overflow-auto">
          {/* Now-playing indicator */}
          <div className="rounded-lg border border-white/10 bg-surface-elevated p-3">
            <div className="mb-2 flex items-center gap-2">
              <motion.span
                aria-hidden="true"
                className="inline-block h-2.5 w-2.5 shrink-0 rounded-full bg-category-cyan"
                animate={{ scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }}
                transition={{ duration: 1.1, repeat: Infinity, ease: "easeInOut" }}
              />
              <span className="text-[10px] font-bold uppercase tracking-widest text-ink-tertiary">Now</span>
            </div>
            <div className="mb-1 text-[14px] font-semibold text-ink-primary">
              {beat?.title ?? "—"}
            </div>
            <div className="text-[12px] leading-snug text-ink-secondary">{beat?.lede ?? ""}</div>
          </div>

          {/* Active nodes */}
          <div className="rounded-lg border border-white/10 bg-surface-panel p-3">
            <div className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink-tertiary">
              Active nodes
            </div>
            <div className="flex flex-col gap-1.5">
              {active.map((id) => {
                const node = nodeById(id);
                const color = node !== undefined ? LAYER_COLOR[node.layer] : "var(--color-category-cyan)";
                return (
                  <div
                    key={id}
                    className="flex items-center gap-2 rounded px-2 py-1"
                    style={{
                      background: `color-mix(in srgb, ${color} 12%, var(--color-surface-elevated))`,
                      borderLeft: `3px solid ${color}`,
                    }}
                  >
                    <span className="text-[12px] font-semibold" style={{ color }}>
                      {node?.label ?? id}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Progress through pipeline */}
          <div className="rounded-lg border border-white/10 bg-surface-panel p-3">
            <div className="mb-2 text-[10px] font-bold uppercase tracking-widest text-ink-tertiary">
              Pipeline progress
            </div>
            <div className="h-1.5 overflow-hidden rounded-full bg-surface-elevated">
              <motion.div
                className="h-full rounded-full bg-category-cyan"
                animate={{ width: `${((narrated.index + 1) / BEATS.length) * 100}%` }}
                transition={{ type: "spring", stiffness: 80, damping: 20 }}
              />
            </div>
            <div className="mt-1.5 flex justify-between text-[11px] text-ink-quaternary">
              <span>source</span>
              <span>{narrated.index + 1} / {BEATS.length}</span>
              <span>report</span>
            </div>
          </div>

          {/* Source location */}
          <div className="rounded-lg border border-white/10 bg-surface-panel p-3">
            <div className="mb-1 text-[10px] font-bold uppercase tracking-widest text-ink-tertiary">
              Source
            </div>
            <code className="block whitespace-pre-wrap font-mono text-[10px] leading-relaxed text-ink-secondary">
              {beat?.caption ?? ""}
            </code>
          </div>
        </div>
      </div>
    </PresentationShell>
  );
}
