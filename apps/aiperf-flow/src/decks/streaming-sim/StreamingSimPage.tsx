/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Narrated walkthrough of the streaming Dynamo shadow-replay pipeline.
//!
//! Each beat walks one stage of the pipeline — source acquisition, session coordination,
//! action dispatch, results compaction — with the voice driving the highlighted stage.
//! Grounded in `rust/runtime/src/streaming/` and `rust/runtime/src/engine/streaming_execution.rs`.

import { PresentationShell } from "../../shell/PresentationShell.js";
import { useNarratedDeck } from "../../audio/index.js";
import type { SlideDefinition } from "../../deck/types.js";
import type { BeatAnchor } from "../../spike/useBeatClock.js";
import clsx from "clsx";

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
    caption: "rust/runtime/src/streaming/session/conversation.rs — joins across partitions by stable session key, folds endpoint replies into the same transcript as authored turns.",
    active: ["conv_coord"],
    narration:
      "Trace files arrive across many S3 prefixes and may contain fragments of the same logical conversation. The conversation coordinator joins those fragments by their stable session key. Authored turns and observed endpoint replies fold into one durable transcript in order. A turn pair closes on exactly one terminal event and on nothing else — whichever disposition the endpoint returned.",
  },
  {
    endAt: 0.30,
    title: "SessionClosurePolicy seals the session",
    lede: "Five checked proofs: authored close, hard watermark, verified finite seal, complete sorted run, exhausted predecessor policy.",
    caption: "rust/runtime/src/streaming/session/closure.rs — exactly five checked proofs, partition exhaustion never closes a session, quarantined sessions retire to tombstone map.",
    active: ["closure_pol"],
    narration:
      "A session is only sealed when one of exactly five checked proofs holds: an authored close record, a hard watermark, a verified finite seal, a verified complete sorted run, or an exhausted missing-predecessor policy. Partition exhaustion alone never closes a session. A session that cannot be cleanly closed is quarantined and retired into a budgeted tombstone map keyed by input domain and session identifier — it will not resurface.",
  },
  {
    endAt: 0.37,
    title: "ActionHost emits sequenced actions",
    lede: "ActionInventoryLedger accumulates terminal membership at each dense global sequence position.",
    caption: "rust/runtime/src/streaming/action/host/inventory.rs — ActionInventoryLedger, FrozenActionInventory, refuses to freeze past outstanding actions.",
    active: ["action_host"],
    narration:
      "The action host emits each session's actions in sequence. The inventory ledger accumulates every finalized action's terminal membership at its dense global sequence position. When all outstanding actions are accounted for, the ledger mints a frozen action inventory — the sealed proof the reliability reporter requires. It refuses to freeze if any outstanding action has not yet closed.",
  },
  {
    endAt: 0.45,
    title: "TurnClosureIntake gates each turn",
    lede: "Reserves a terminal slot before dispatch; records a Dropped terminal if the stop condition refuses.",
    caption: "rust/runtime/src/streaming/action/scheduled_request.rs — bounding admission because the runtime does not; Dropped terminal rather than run failure.",
    active: ["turn_intake"],
    narration:
      "Before each turn is dispatched, the turn closure intake reserves its terminal slot. Admission is bounded here because the underlying runtime does not bound it. If a stop condition refuses the issue, the action records a Dropped terminal rather than a run failure — the pipeline stays healthy and the inventory ledger accounts for the dropped slot normally.",
  },
  {
    endAt: 0.52,
    title: "StreamingPipeline routes actions",
    lede: "shadow_replay workload; admits exactly one profiling phase, requires the reliability policy digest to match the frozen execution plan.",
    caption: "rust/runtime/src/engine/streaming_execution.rs — StreamingPipeline, shadow_replay workload, SynthesisAuthority, capability agreement.",
    active: ["stream_pipe"],
    narration:
      "The streaming pipeline receives admitted actions and routes them. It is registered as the shadow underscore replay workload — its capability agreement admits exactly one profiling phase, refuses a Dynamo source or format composition when canonical content reconstruction has no registered factory, and requires the resolved reliability policy digest to equal the one recorded in the frozen execution plan. A resume whose SynthesisAuthority disagrees with the authored plan fails before any participant initializes.",
  },
  {
    endAt: 0.60,
    title: "LocalPlacement or CellularTransport delivers",
    lede: "Same-host: LocalPlacement via shared acquired snapshot. Remote: CellularTransport lands exact acquired set privately before compilation.",
    caption: "rust/runtime/src/cellular/ — LocalPlacement shares the controller snapshot; remote cells land privately; no compiled Graph-IR crosses the cellular wire.",
    active: ["local_place", "cell_trans"],
    narration:
      "Placement routes each action to where it will execute. Same-host placements share the controller's acquired snapshot. Remote cells receive the exact acquired set privately before compilation — the same files, landed privately, so no compiled graph intermediate ever crosses the cellular wire. Cross-host artifact transfer uses a per-run pinned-TLS channel authenticated after controller registration.",
  },
  {
    endAt: 0.68,
    title: "ScheduledRequestSink dispatches to the endpoint",
    lede: "Issues each aiperf.action.request.v1 through ScheduledRuntime; translates dispatch lifecycle to ActionExecutionEvents at first-token and completion hooks.",
    caption: "rust/runtime/src/streaming/action/scheduled_request.rs — the one action binding that reaches an inference endpoint.",
    active: ["sched_sink"],
    narration:
      "The scheduled request sink is the one action binding that reaches an inference endpoint. It issues each aiperf dot action dot request dot v1 action through the scheduled runtime, and translates the dispatch lifecycle back into action execution events at the existing first-token and completion hooks. Admission is bounded by the sink itself before reaching the runtime.",
  },
  {
    endAt: 0.75,
    title: "ActionInventory records each terminal",
    lede: "Dense global sequence; mints FrozenActionInventory only when all outstanding actions are closed.",
    caption: "rust/runtime/src/streaming/action/host/inventory.rs — accumulates at dense global sequence, refuses to freeze past an outstanding action.",
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
    caption: "rust/runtime/src/report/ — NativeReport; --steady-state derives the window from the shared in-flight concurrency sweep-line curve.",
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
// Pipeline diagram
// ---------------------------------------------------------------------------

const LAYER_COLORS: Record<string, string> = {
  src: "var(--color-category-blue)",
  ses: "var(--color-category-purple)",
  pipe: "var(--color-category-cyan)",
  replay: "var(--color-category-green)",
  res: "var(--color-category-orange)",
  eng: "var(--color-category-yellow)",
};

function PipelineNode({
  id,
  label,
  sublabel,
  layer,
  active,
}: {
  id: string;
  label: string;
  sublabel?: string;
  layer: string;
  active: boolean;
}): React.JSX.Element {
  const color = LAYER_COLORS[layer] ?? "var(--color-category-gray)";
  return (
    <div
      data-node={id}
      className={clsx(
        "flex flex-col items-center justify-center rounded-lg border px-3 py-2 text-center transition-all duration-300",
        active ? "border-transparent" : "border-white/10 bg-surface-panel",
      )}
      style={
        active
          ? {
              background: `color-mix(in srgb, ${color} 18%, var(--color-surface-panel))`,
              borderColor: color,
              boxShadow: `0 0 14px 2px color-mix(in srgb, ${color} 40%, transparent)`,
            }
          : undefined
      }
    >
      <span
        className="text-[13px] font-semibold leading-tight"
        style={{ color: active ? color : "var(--color-ink-primary)" }}
      >
        {label}
      </span>
      {sublabel !== undefined && (
        <span className="mt-0.5 text-[11px] leading-tight text-ink-tertiary">{sublabel}</span>
      )}
    </div>
  );
}

function LayerBand({
  label,
  layer,
  children,
}: {
  label: string;
  layer: string;
  children: React.ReactNode;
}): React.JSX.Element {
  const color = LAYER_COLORS[layer] ?? "var(--color-category-gray)";
  return (
    <div className="flex items-center gap-2">
      <div
        className="w-[72px] shrink-0 text-right text-[10px] font-bold uppercase tracking-widest"
        style={{ color }}
      >
        {label}
      </div>
      <div className="flex flex-1 items-center gap-2">{children}</div>
    </div>
  );
}

function VerticalConnector(): React.JSX.Element {
  return (
    <div className="flex" style={{ paddingLeft: 80 }}>
      <div
        className="h-4 w-px"
        style={{ background: "var(--color-stroke-secondary)" }}
      />
    </div>
  );
}

function HConnector(): React.JSX.Element {
  return (
    <div
      className="h-px w-4 shrink-0 self-center"
      style={{ background: "var(--color-stroke-secondary)" }}
    />
  );
}

function PipelineDiagram({ active }: { active: readonly string[] }): React.JSX.Element {
  const isActive = (id: string) => active.includes(id);

  return (
    <div className="flex flex-col gap-1.5 overflow-auto p-3 text-sm">
      {/* SOURCE */}
      <LayerBand label="Source" layer="src">
        <PipelineNode id="s3" label="S3 / Object Store" sublabel="bucket · prefix · policy" layer="src" active={isActive("s3")} />
        <HConnector />
        <PipelineNode id="local" label="Local Files" sublabel="no-follow snapshot" layer="src" active={isActive("local")} />
      </LayerBand>

      <VerticalConnector />

      <LayerBand label="" layer="src">
        <PipelineNode id="dynamo_fmt" label="DynamoFormat" sublabel="decompress · parse · validate" layer="src" active={isActive("dynamo_fmt")} />
      </LayerBand>

      <VerticalConnector />

      {/* SESSION */}
      <LayerBand label="Session" layer="ses">
        <PipelineNode id="conv_coord" label="ConversationCoordinator" sublabel="join by session key" layer="ses" active={isActive("conv_coord")} />
        <HConnector />
        <PipelineNode id="closure_pol" label="SessionClosurePolicy" sublabel="5 checked proofs" layer="ses" active={isActive("closure_pol")} />
        <HConnector />
        <PipelineNode id="action_host" label="ActionHost" sublabel="inventory ledger" layer="ses" active={isActive("action_host")} />
      </LayerBand>

      <VerticalConnector />

      {/* PIPELINE */}
      <LayerBand label="Pipeline" layer="pipe">
        <PipelineNode id="turn_intake" label="TurnClosureIntake" sublabel="gate · admit · dropped terminal" layer="pipe" active={isActive("turn_intake")} />
        <HConnector />
        <PipelineNode id="stream_pipe" label="StreamingPipeline" sublabel="shadow_replay workload" layer="pipe" active={isActive("stream_pipe")} />
      </LayerBand>

      <VerticalConnector />

      {/* PLACEMENT */}
      <LayerBand label="Placement" layer="pipe">
        <PipelineNode id="local_place" label="LocalPlacement" sublabel="shared snapshot" layer="pipe" active={isActive("local_place")} />
        <HConnector />
        <PipelineNode id="cell_trans" label="CellularTransport" sublabel="private landing · pinned TLS" layer="pipe" active={isActive("cell_trans")} />
      </LayerBand>

      <VerticalConnector />

      {/* DISPATCH */}
      <LayerBand label="Dispatch" layer="replay">
        <PipelineNode id="sched_sink" label="ScheduledRequestSink" sublabel="action.request.v1 → endpoint" layer="replay" active={isActive("sched_sink")} />
        <HConnector />
        <PipelineNode id="action_inv" label="ActionInventory" sublabel="dense global sequence" layer="replay" active={isActive("action_inv")} />
      </LayerBand>

      <VerticalConnector />

      {/* RESULTS */}
      <LayerBand label="Results" layer="res">
        <PipelineNode id="epoch_coord" label="EpochCoordinator" sublabel="transactional epoch advance" layer="res" active={isActive("epoch_coord")} />
        <HConnector />
        <PipelineNode id="result_comp" label="ResultCompactor" sublabel="exact · sketch · exporters" layer="res" active={isActive("result_comp")} />
        <HConnector />
        <PipelineNode id="delivery" label="DeliveryRestart" sublabel="SynthesisAuthority guard" layer="res" active={isActive("delivery")} />
      </LayerBand>

      <VerticalConnector />

      {/* REPORT */}
      <LayerBand label="Report" layer="eng">
        <PipelineNode id="report" label="NativeReport" sublabel="steady-state · artifacts · sweep" layer="eng" active={isActive("report")} />
      </LayerBand>
    </div>
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
        {/* Pipeline diagram — left column */}
        <div className="min-w-0 flex-1 overflow-auto rounded-lg border border-white/10 bg-surface-elevated">
          <PipelineDiagram active={active} />
        </div>

        {/* Active stage detail — right column */}
        <div className="flex w-72 shrink-0 flex-col gap-3 overflow-auto">
          <div className="rounded-lg border border-white/10 bg-surface-elevated p-3">
            <div className="mb-1 text-[10px] font-bold uppercase tracking-widest text-ink-tertiary">
              Active stage
            </div>
            <div className="mb-1 text-[15px] font-semibold text-ink-primary">
              {beat?.title ?? "—"}
            </div>
            <div className="text-[13px] leading-snug text-ink-secondary">{beat?.lede ?? ""}</div>
          </div>

          <div className="rounded-lg border border-white/10 bg-surface-panel p-3">
            <div className="mb-1 text-[10px] font-bold uppercase tracking-widest text-ink-tertiary">
              Source location
            </div>
            <code className="block whitespace-pre-wrap font-mono text-[11px] leading-relaxed text-ink-secondary">
              {beat?.caption ?? ""}
            </code>
          </div>

          <div className="rounded-lg border border-white/10 bg-surface-panel p-3">
            <div className="mb-1 text-[10px] font-bold uppercase tracking-widest text-ink-tertiary">
              Highlighted nodes
            </div>
            <div className="flex flex-wrap gap-1.5">
              {active.map((id) => (
                <span
                  key={id}
                  className="rounded px-2 py-0.5 font-mono text-[11px] font-semibold text-black"
                  style={{ background: "var(--color-category-cyan)" }}
                >
                  {id}
                </span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </PresentationShell>
  );
}
