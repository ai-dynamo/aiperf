/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — the sweep-line desk's panels. Each is a pure function of the scenario and a progress
//! fraction, so the narrated and free-running pages draw identical pictures.

import { toSeconds, type Scenario } from "./scenario.js";
import type { StepFn } from "./sweepAlgo.js";

const INK = "var(--color-ink-secondary)";
const DIM = "var(--color-ink-quaternary)";
const GRID = "var(--color-stroke-tertiary)";
const BLUE = "var(--color-category-blue)";
const GREEN = "var(--color-category-green)";
const ORANGE = "var(--color-category-orange)";
const PURPLE = "var(--color-category-purple)";
const RED = "var(--color-category-red)";
const CYAN = "var(--color-category-cyan)";

export function Panel({
  title,
  hint,
  children,
}: {
  title: string;
  hint?: string;
  children: React.ReactNode;
}): React.JSX.Element {
  return (
    <section className="flex min-h-0 flex-col rounded-lg border border-white/10 bg-surface-elevated p-3">
      <div className="mb-2 flex items-baseline gap-3">
        <h2 className="text-[10px] font-bold tracking-widest text-ink-secondary">{title}</h2>
        {hint !== undefined && <span className="text-[10px] text-ink-quaternary">{hint}</span>}
      </div>
      <div className="min-h-0 flex-1">{children}</div>
    </section>
  );
}

/**
 * Columns filling by absolute request index.
 *
 * Drawn with metrics as rows and records as columns: the alignment being demonstrated is *across*
 * metrics at one index, so putting the index on the horizontal axis makes a vertical slice the
 * thing you read — one record, every column at once.
 */
export function ColumnStorePanel({
  scenario,
  rows,
}: {
  scenario: Scenario;
  rows: number;
}): React.JSX.Element {
  const shown = Math.max(0, Math.min(scenario.store.rows, Math.round(rows)));
  const n = scenario.store.rows;
  const W = 1000;
  const labelW = 128;
  const cellW = Math.max(2, (W - labelW - 10) / n);
  const rowH = 16;
  const gap = 4;
  const iclMax = Math.max(1, ...scenario.store.icl.lengths);

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${(scenario.store.columns.length + 1) * (rowH + gap) + 26}`}
      className="block">
      {scenario.store.columns.map((column, ci) => {
        const y = ci * (rowH + gap);
        return (
          <g key={column.name}>
            <text x={labelW - 8} y={y + rowH - 4} textAnchor="end" fontSize={9} fill={INK}>
              {column.name}
            </text>
            {column.values.map((value, row) => {
              const filled = row < shown;
              const absent = Number.isNaN(value);
              return (
                <rect key={row} x={labelW + row * cellW} y={y}
                  width={Math.max(1, cellW - 0.6)} height={rowH - 2}
                  fill={absent ? "none" : BLUE}
                  stroke={absent ? ORANGE : "none"} strokeDasharray={absent ? "2 2" : undefined}
                  opacity={filled ? 0.85 : 0.12} />
              );
            })}
          </g>
        );
      })}

      {(() => {
        const y = scenario.store.columns.length * (rowH + gap);
        return (
          <g>
            <text x={labelW - 8} y={y + rowH - 4} textAnchor="end" fontSize={9} fill={PURPLE}>
              icl (ragged)
            </text>
            {scenario.store.icl.lengths.map((length, row) => {
              const filled = row < shown;
              // Height encodes how many values this row contributed — the raggedness itself.
              const h = Math.max(1.5, (length / iclMax) * (rowH - 2));
              return (
                <rect key={row} x={labelW + row * cellW} y={y + (rowH - 2 - h)}
                  width={Math.max(1, cellW - 0.6)} height={h}
                  fill={length === 0 ? "none" : PURPLE}
                  stroke={length === 0 ? DIM : "none"} strokeDasharray={length === 0 ? "1 2" : undefined}
                  opacity={filled ? 0.9 : 0.12} />
              );
            })}
          </g>
        );
      })()}

      <text x={labelW} y={(scenario.store.columns.length + 1) * (rowH + gap) + 12} fontSize={9} fill={DIM}>
        request index 0 → {n - 1}   ·   a vertical slice is one record   ·   bar height = ICL values in that row
      </text>
    </svg>
  );
}

/** The event list, sorted, with colliding pairs called out. */
export function EventSortPanel({
  scenario,
  upTo,
}: {
  scenario: Scenario;
  upTo: number;
}): React.JSX.Element {
  const n = Math.max(0, Math.min(scenario.sortedEvents.length, Math.round(upTo)));
  // Centre the view on the first collision so the tie-break is on screen when narrated.
  const focus = scenario.collisions[0] ?? 0;
  const from = Math.max(0, Math.min(focus - 6, Math.max(0, n - 14)));
  const slice = scenario.sortedEvents.slice(from, from + 16);

  return (
    <div className="font-mono text-[10px] leading-[1.45]">
      <div className="mb-1 flex gap-3 text-[9px] tracking-widest" style={{ color: DIM }}>
        <span className="w-16">t (s)</span>
        <span className="w-10">delta</span>
        <span className="w-8">rec</span>
        <span>note</span>
      </div>
      {slice.map((event, i) => {
        const index = from + i;
        const revealed = index < n;
        const collided = scenario.collisions.includes(index);
        const partner = scenario.collisions.includes(index + 1);
        return (
          <div
            key={index}
            className="flex gap-3"
            style={{
              opacity: revealed ? 1 : 0.18,
              background: collided || partner ? "rgba(255,255,255,0.06)" : undefined,
            }}
          >
            <span className="w-16" style={{ color: INK }}>
              {toSeconds(event.timestampNs, scenario.runStartNs).toFixed(6)}
            </span>
            <span className="w-10 font-bold" style={{ color: event.delta < 0 ? ORANGE : GREEN }}>
              {event.delta > 0 ? `+${event.delta}` : event.delta}
            </span>
            <span className="w-8" style={{ color: DIM }}>
              {event.record}
            </span>
            <span style={{ color: collided ? RED : DIM }}>
              {collided ? "◄ same timestamp — end sorted first" : ""}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** The running cumulative sum, as a table of steps beside the curve it produces. */
export function CumsumPanel({
  scenario,
  upTo,
}: {
  scenario: Scenario;
  upTo: number;
}): React.JSX.Element {
  const n = Math.max(0, Math.min(scenario.steps.length, Math.round(upTo)));
  const recent = scenario.steps.slice(Math.max(0, n - 9), n);
  return (
    <div className="font-mono text-[10px] leading-[1.5]">
      {recent.length === 0 && <div style={{ color: DIM }}>Waiting for the first event…</div>}
      {recent.map((step, i) => (
        <div key={i} className="flex items-baseline gap-2">
          <span className="w-14" style={{ color: DIM }}>
            {toSeconds(step.event.timestampNs, scenario.runStartNs).toFixed(3)}
          </span>
          <span className="w-8 font-bold" style={{ color: step.event.delta < 0 ? ORANGE : GREEN }}>
            {step.event.delta > 0 ? `+${step.event.delta}` : step.event.delta}
          </span>
          <span style={{ color: DIM }}>→</span>
          <span className="w-8 text-right font-bold" style={{ color: INK }}>
            {step.running}
          </span>
          {step.collided && <span style={{ color: RED }}>tie</span>}
        </div>
      ))}
      <div className="mt-2 border-t border-white/10 pt-1 text-[9px]" style={{ color: DIM }}>
        residuals snapped to zero: <strong style={{ color: INK }}>{scenario.snapped}</strong>
        {"  ·  "}threshold 1e-9 × max
      </div>
    </div>
  );
}

export type CurveOverlay = {
  /** Vertical band, e.g. a detected window. */
  band?: { fromNs: number; toNs: number; color: string; label?: string };
  /** Horizontal threshold line. */
  level?: { value: number; color: string; label?: string };
  /** Second curve drawn behind, for comparison. */
  ghost?: { curve: StepFn; color: string; label?: string };
};

/** A step-function plot with an optional playhead, band, and threshold. */
export function CurvePanel({
  scenario,
  curve,
  headNs,
  height = 150,
  color = BLUE,
  overlay,
  valueLabel,
}: {
  scenario: Scenario;
  curve: StepFn;
  headNs: number | null;
  height?: number;
  color?: string;
  overlay?: CurveOverlay;
  valueLabel?: string;
}): React.JSX.Element {
  const W = 900;
  const L = 44;
  const R = 12;
  const T = 10;
  const B = 22;
  const span = Math.max(1, scenario.runEndNs - scenario.runStartNs);
  const all = [...curve.values, ...(overlay?.ghost?.curve.values ?? []), overlay?.level?.value ?? 0];
  const vMax = Math.max(1, ...all) * 1.12;

  const x = (ns: number) => L + ((ns - scenario.runStartNs) / span) * (W - L - R);
  const y = (v: number) => T + (1 - v / vMax) * (height - T - B);

  const path = (c: StepFn, clipNs: number | null) => {
    if (c.timestampsNs.length === 0) return "";
    let d = `M ${x(c.timestampsNs[0]!).toFixed(1)} ${y(0).toFixed(1)}`;
    let prev = 0;
    for (let i = 0; i < c.timestampsNs.length; i++) {
      const t = c.timestampsNs[i]!;
      if (clipNs !== null && t > clipNs) break;
      d += ` L ${x(t).toFixed(1)} ${y(prev).toFixed(1)} L ${x(t).toFixed(1)} ${y(c.values[i]!).toFixed(1)}`;
      prev = c.values[i]!;
    }
    if (clipNs !== null) d += ` L ${x(Math.min(clipNs, scenario.runEndNs)).toFixed(1)} ${y(prev).toFixed(1)}`;
    return d;
  };

  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => scenario.runStartNs + f * span);

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${height}`} className="block">
      {ticks.map((t, i) => (
        <g key={i}>
          <line x1={x(t)} y1={T} x2={x(t)} y2={height - B} stroke={GRID} strokeWidth={1} opacity={0.4} />
          <text x={x(t)} y={height - 6} textAnchor="middle" fontSize={9} fill={DIM}>
            {toSeconds(t, scenario.runStartNs).toFixed(0)}s
          </text>
        </g>
      ))}

      {overlay?.band !== undefined && (
        <>
          <rect
            x={x(overlay.band.fromNs)}
            y={T}
            width={Math.max(1, x(overlay.band.toNs) - x(overlay.band.fromNs))}
            height={height - T - B}
            fill={overlay.band.color}
            opacity={0.14}
          />
          {overlay.band.label !== undefined && (
            <text x={x(overlay.band.fromNs) + 5} y={T + 11} fontSize={9} fill={overlay.band.color}>
              {overlay.band.label}
            </text>
          )}
        </>
      )}

      {overlay?.level !== undefined && (
        <>
          <line
            x1={L}
            y1={y(overlay.level.value)}
            x2={W - R}
            y2={y(overlay.level.value)}
            stroke={overlay.level.color}
            strokeWidth={1}
            strokeDasharray="4 4"
          />
          <text x={W - R - 4} y={y(overlay.level.value) - 3} textAnchor="end" fontSize={9} fill={overlay.level.color}>
            {overlay.level.label ?? overlay.level.value}
          </text>
        </>
      )}

      {overlay?.ghost !== undefined && (
        <path d={path(overlay.ghost.curve, headNs)} fill="none" stroke={overlay.ghost.color}
          strokeWidth={1.4} strokeDasharray="5 3" opacity={0.85} />
      )}

      <path d={path(curve, headNs)} fill="none" stroke={color} strokeWidth={1.8} />

      {headNs !== null && (
        <line x1={x(headNs)} y1={T} x2={x(headNs)} y2={height - B} stroke={RED} strokeWidth={1.5} />
      )}

      <line x1={L} y1={height - B} x2={W - R} y2={height - B} stroke={INK} strokeWidth={1} />
      <text x={4} y={T + 8} fontSize={9} fill={DIM}>
        {valueLabel ?? ""}
      </text>
      <text x={4} y={height - B - 2} fontSize={9} fill={DIM}>0</text>
      <text x={4} y={T + 18} fontSize={9} fill={DIM}>{Math.round(vMax)}</text>
    </svg>
  );
}

/** CUSUM forward/backward traces, with the chosen turning points. */
export function CusumPanel({ scenario }: { scenario: Scenario }): React.JSX.Element {
  const { cusum } = scenario;
  const W = 900;
  const H = 130;
  const n = cusum.forward.length;
  if (n === 0) return <div className="text-xs text-ink-quaternary">No curve.</div>;
  const min = Math.min(...cusum.forward, ...cusum.backward, 0);
  const max = Math.max(...cusum.forward, ...cusum.backward, 0);
  const x = (i: number) => 40 + (i / Math.max(1, n - 1)) * (W - 52);
  const y = (v: number) => 10 + (1 - (v - min) / Math.max(1e-9, max - min)) * (H - 32);
  const line = (xs: readonly number[]) =>
    xs.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ");

  return (
    <>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
        <line x1={40} y1={y(0)} x2={W - 12} y2={y(0)} stroke={GRID} strokeWidth={1} />
        <path d={line(cusum.forward)} fill="none" stroke={CYAN} strokeWidth={1.6} />
        <path d={line(cusum.backward)} fill="none" stroke={PURPLE} strokeWidth={1.4} strokeDasharray="4 3" />
        <line x1={x(cusum.rampUpIndex)} y1={10} x2={x(cusum.rampUpIndex)} y2={H - 22} stroke={CYAN} strokeWidth={1.5} />
        <text x={x(cusum.rampUpIndex) + 4} y={20} fontSize={9} fill={CYAN}>argmin forward</text>
        <line x1={x(cusum.rampDownIndex)} y1={10} x2={x(cusum.rampDownIndex)} y2={H - 22} stroke={PURPLE} strokeWidth={1.5} />
        <text x={x(cusum.rampDownIndex) + 4} y={32} fontSize={9} fill={PURPLE}>argmin backward</text>
        <text x={4} y={20} fontSize={9} fill={DIM}>cusum</text>
      </svg>
      <div className="mt-1 text-[10px]" style={{ color: DIM }}>
        target (time-weighted p95) <strong style={{ color: INK }}>{cusum.target.toFixed(1)}</strong>
        {"  ·  "}method{" "}
        <strong style={{ color: cusum.method.startsWith("cusum_inverted") ? RED : GREEN }}>
          {cusum.method}
        </strong>
      </div>
    </>
  );
}

/** MSER-5 batch means with the statistic beneath and the chosen truncation marked. */
export function Mser5Panel({
  trace,
  label,
}: {
  trace: Scenario["mser5Latency"];
  label: string;
}): React.JSX.Element {
  const W = 440;
  const H = 118;
  if (trace.batches.length === 0) {
    return <div className="text-[10px] text-ink-quaternary">{label}: too few samples to run.</div>;
  }
  const m = trace.batches.length;
  const bMin = Math.min(...trace.batches);
  const bMax = Math.max(...trace.batches);
  const sMax = Math.max(...trace.mser);
  const x = (i: number) => 30 + (i / Math.max(1, m - 1)) * (W - 42);
  const yB = (v: number) => 12 + (1 - (v - bMin) / Math.max(1e-9, bMax - bMin)) * 42;
  const yS = (v: number) => 68 + (1 - v / Math.max(1e-12, sMax)) * 36;

  return (
    <div>
      <div className="mb-0.5 text-[10px]" style={{ color: DIM }}>
        {label} · d* = <strong style={{ color: INK }}>{trace.dStar}</strong> batches (
        {trace.truncation} samples), max {trace.maxD}
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
        <path d={trace.batches.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${yB(v).toFixed(1)}`).join(" ")}
          fill="none" stroke={BLUE} strokeWidth={1.5} />
        <text x={2} y={16} fontSize={8} fill={DIM}>means</text>
        <path d={trace.mser.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${yS(v).toFixed(1)}`).join(" ")}
          fill="none" stroke={ORANGE} strokeWidth={1.5} />
        <text x={2} y={74} fontSize={8} fill={DIM}>mser</text>
        <line x1={x(trace.maxD)} y1={64} x2={x(trace.maxD)} y2={H - 4} stroke={GRID} strokeDasharray="3 3" />
        <text x={x(trace.maxD) + 3} y={H - 6} fontSize={8} fill={DIM}>half</text>
        <line x1={x(trace.dStar)} y1={8} x2={x(trace.dStar)} y2={H - 4} stroke={GREEN} strokeWidth={1.5} />
        <text x={x(trace.dStar) + 3} y={14} fontSize={8} fill={GREEN}>d*</text>
      </svg>
    </div>
  );
}

/** The four signals and the window they agree on. */
export function ConsensusPanel({ scenario }: { scenario: Scenario }): React.JSX.Element {
  const { consensus, runStartNs, runEndNs, thresholdWindow } = scenario;
  const span = Math.max(1, runEndNs - runStartNs);
  const W = 440;
  const rowH = 20;
  const rows = [
    ...(thresholdWindow !== null
      ? [{ name: "threshold (rust)", window: { startNs: thresholdWindow.startNs, endNs: thresholdWindow.endNs }, color: GREEN }]
      : []),
    ...consensus.signals.map((s, i) => ({
      name: s.name,
      window: s.window,
      color: [CYAN, PURPLE, ORANGE][i % 3]!,
    })),
    { name: "consensus", window: consensus.window, color: RED },
  ];
  const x = (ns: number) => 108 + ((ns - runStartNs) / span) * (W - 120);

  return (
    <div>
      <svg width="100%" viewBox={`0 0 ${W} ${rows.length * rowH + 14}`} className="block">
        {rows.map((row, i) => {
          const y = 6 + i * rowH;
          return (
            <g key={row.name}>
              <text x={104} y={y + 11} textAnchor="end" fontSize={9} fill={row.color}>
                {row.name}
              </text>
              <line x1={108} y1={y + 8} x2={W - 12} y2={y + 8} stroke={GRID} strokeWidth={1} />
              {row.window === null ? (
                <text x={112} y={y + 11} fontSize={9} fill={DIM}>
                  no boundary
                </text>
              ) : (
                <rect x={x(row.window.startNs)} y={y + 2} height={12}
                  width={Math.max(2, x(row.window.endNs) - x(row.window.startNs))}
                  fill={row.color} opacity={row.name === "consensus" ? 0.75 : 0.4} rx={2} />
              )}
            </g>
          );
        })}
      </svg>
      <div className="mt-1 text-[10px]" style={{ color: DIM }}>
        method <strong style={{ color: consensus.method.startsWith("fallback") ? RED : GREEN }}>{consensus.method}</strong>
        {"  ·  "}latest start, earliest end
      </div>
    </div>
  );
}

/** Batch means across the window, with the trend correlation. */
export function StationarityPanel({ scenario }: { scenario: Scenario }): React.JSX.Element {
  const { stationarity } = scenario;
  const W = 440;
  const H = 92;
  if (stationarity.batches.length === 0) {
    return <div className="text-[10px] text-ink-quaternary">Too few in-window records to test.</div>;
  }
  const min = Math.min(...stationarity.batches);
  const max = Math.max(...stationarity.batches);
  const x = (i: number) => 28 + (i / Math.max(1, stationarity.batches.length - 1)) * (W - 40);
  const y = (v: number) => 12 + (1 - (v - min) / Math.max(1e-9, max - min)) * (H - 34);

  return (
    <div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="block">
        <path d={stationarity.batches.map((v, i) => `${i === 0 ? "M" : "L"} ${x(i).toFixed(1)} ${y(v).toFixed(1)}`).join(" ")}
          fill="none" stroke={stationarity.warning ? RED : GREEN} strokeWidth={1.6} />
        {stationarity.batches.map((v, i) => (
          <circle key={i} cx={x(i)} cy={y(v)} r={2.4} fill={stationarity.warning ? RED : GREEN} />
        ))}
        <text x={2} y={16} fontSize={8} fill={DIM}>batch means</text>
      </svg>
      <div className="text-[10px]" style={{ color: DIM }}>
        Spearman ρ vs batch index{" "}
        <strong style={{ color: stationarity.warning ? RED : GREEN }}>{stationarity.rho.toFixed(3)}</strong>
        {"  ·  "}
        {stationarity.warning ? "still trending — window rejected" : "flat — window holds"}
        {"  ·  flag at |ρ| > 0.65"}
      </div>
    </div>
  );
}
