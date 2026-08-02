/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! SPIKE — exact against sketch.
//!
//! Every number in this page is computed by a faithful port of the runtime's own t-digest, so the
//! errors shown are the errors it actually produces rather than a stand-in for them.

import { useMemo, useState } from "react";
import {
  centroidSpans,
  clustered,
  compare,
  foldCells,
  latencySamples,
  splitAcrossCells,
  DEFAULT_COMPRESSION,
  PERCENTILES,
  type Shape,
} from "./sketchSim.js";
import { ControlBar, Legend, LegendItem, Panel, Readout, SourceNote, SpikeHeader, Toggle } from "../ui.js";

const GREEN = "var(--color-category-green)";
const CYAN = "var(--color-category-cyan)";
const ORANGE = "var(--color-category-orange)";
const RED = "var(--color-category-red)";
const DIM = "var(--color-ink-quaternary)";

const CELL_CHOICES = [1, 3, 6] as const;
const SIZE_CHOICES = [2_000, 20_000, 100_000] as const;
const SHAPES: readonly Shape[] = ["lognormal", "bimodal"];

export function SketchFoldSpike(): React.JSX.Element {
  const [cells, setCells] = useState(3);
  const [size, setSize] = useState(20_000);
  const [shape, setShape] = useState<Shape>("lognormal");

  const values = useMemo(() => latencySamples(size, 13, shape), [size, shape]);
  const slices = useMemo(() => splitAcrossCells(values, cells), [values, cells]);
  const folded = useMemo(() => foldCells(slices), [slices]);
  const rows = useMemo(() => compare(values, folded), [values, folded]);
  const spans = useMemo(() => centroidSpans(folded), [folded]);
  const centroids = useMemo(() => clustered(folded), [folded]);

  // What the two paths cost to hold. Exact mode keeps every value; the sketch keeps centroids.
  const exactBytes = values.length * 8;
  const sketchBytes = centroids.length * 16;

  return (
    <div className="min-h-screen bg-surface-page px-8 py-7 text-ink-primary">
      <SpikeHeader title="What survives being summarized">
        <p>
          A cellular run cannot pool every record in one place, so each cell summarizes its own
          slice into a t-digest and the controller folds them. Some figures come back{" "}
          <strong>exactly right</strong>. Others come back <strong>close</strong>. The difference
          is not which ones happen to agree — it is which ones are computed from running totals
          the sketch keeps whole, and which are read off a compressed picture of the distribution.
        </p>
      </SpikeHeader>

      <ControlBar>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">cells</span>
          {CELL_CHOICES.map((c) => (
            <Toggle key={c} active={cells === c} onClick={() => setCells(c)}>{c}</Toggle>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">records</span>
          {SIZE_CHOICES.map((n) => (
            <Toggle key={n} active={size === n} onClick={() => setSize(n)}>
              {n.toLocaleString()}
            </Toggle>
          ))}
        </div>
        <div className="flex items-center gap-1.5">
          <span className="mr-1 text-base text-ink-tertiary">latency shape</span>
          {SHAPES.map((s) => (
            <Toggle key={s} active={shape === s} onClick={() => setShape(s)}
              title={s === "lognormal" ? "What request latency actually looks like"
                : "A hard split between a fast and a slow mode — the digest's worst case"}>
              {s}
            </Toggle>
          ))}
        </div>
        <div className="ml-auto flex items-center gap-6">
          <Readout label="δ" value={DEFAULT_COMPRESSION} />
          <Readout label="centroids" value={centroids.length} color={CYAN} />
        </div>
      </ControlBar>

      <div className="mb-4 grid grid-cols-2 gap-4">
        <Panel label="EXACT" hint="every value retained, sorted, type-7 interpolation">
          <div className="flex items-baseline gap-6">
            <span className="text-[19px] tabular-nums">
              <strong>{values.length.toLocaleString()}</strong>
              <span className="text-ink-tertiary"> values held</span>
            </span>
            <span className="text-[19px] tabular-nums">
              <strong>{(exactBytes / 1024).toFixed(0)} KB</strong>
              <span className="text-ink-tertiary"> to keep them</span>
            </span>
          </div>
          <p className="mt-2 text-[14px] leading-relaxed text-ink-quaternary">
            Answers any percentile you ask for afterwards, and every per-record artifact. Costs
            the whole distribution in memory, and cannot be merged from separate machines without
            shipping every record.
          </p>
        </Panel>

        <Panel label="SKETCH" hint="t-digest, mergeable, bounded">
          <div className="flex items-baseline gap-6">
            <span className="text-[19px] tabular-nums">
              <strong style={{ color: CYAN }}>{centroids.length}</strong>
              <span className="text-ink-tertiary"> centroids</span>
            </span>
            <span className="text-[19px] tabular-nums">
              <strong style={{ color: CYAN }}>{(sketchBytes / 1024).toFixed(1)} KB</strong>
              <span className="text-ink-tertiary">
                {" "}— {(exactBytes / sketchBytes).toFixed(0)}× smaller
              </span>
            </span>
          </div>
          <p className="mt-2 text-[14px] leading-relaxed text-ink-quaternary">
            Bounded by δ regardless of how many records arrive, and mergeable: concatenate
            centroids, compress. That is what lets {cells} {cells === 1 ? "cell" : "cells"} fold
            without a central pass. Per-record outputs are gone.
          </p>
        </Panel>
      </div>

      <Panel label="WHAT EACH ONE REPORTS" className="mb-4">
        <table className="w-full text-base tabular-nums">
          <thead>
            <tr className="text-left text-[14px] text-ink-tertiary">
              <th className="w-56 font-normal" />
              <th className="pb-1 font-normal">exact</th>
              <th className="pb-1 font-normal">sketch, after folding {cells}</th>
              <th className="pb-1 font-normal">difference</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const bad = Math.abs(row.errorPct) > 1;
              return (
                <tr key={row.label} className="border-t border-white/5">
                  <td className="py-1.5 pr-6">
                    <span className="text-[15px] text-ink-secondary">{row.label}</span>
                    {row.guaranteed && (
                      <span className="ml-2 rounded px-1.5 py-0.5 text-[12px] font-bold text-black"
                        style={{ background: GREEN }}>EXACT BY CONSTRUCTION</span>
                    )}
                  </td>
                  <td className="py-1.5 pr-4">{fmt(row.exact)}</td>
                  <td className="py-1.5 pr-4" style={{ color: row.guaranteed ? GREEN : CYAN }}>
                    {fmt(row.sketch)}
                  </td>
                  <td className="py-1.5 pr-4"
                    style={{ color: row.guaranteed ? GREEN : bad ? RED : DIM }}>
                    {row.guaranteed
                      ? "none — same number"
                      : `${row.errorPct >= 0 ? "+" : ""}${row.errorPct.toFixed(3)}%`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Panel>

      <Panel label="WHERE THE RESOLUTION GOES" hint="one bar per centroid — height is how much of the run it answers for"
        className="mb-4">
        <Legend>
          <LegendItem mark="▮" color={CYAN}>a centroid — taller means it summarizes more of the distribution</LegendItem>
          <LegendItem mark="│" color={ORANGE}>a reported percentile</LegendItem>
        </Legend>
        <CentroidBands spans={spans} />
        <p className="mt-2 max-w-5xl text-[14px] leading-relaxed text-ink-quaternary">
          The bars are short at both ends and tall through the middle, and that is deliberate. A
          cluster may span one unit of <code>k(q) = δ·asin(2q−1)/2π</code>, and{" "}
          <code>asin</code> steepens towards the edges — so one unit of budget buys a{" "}
          <em>narrow</em> band of quantiles at the tail and a wide one through the body. The digest
          spends its precision where p99 lives and is coarsest at the median. That is the reverse
          of the usual worry about approximations being worst at the extremes.
        </p>
      </Panel>

      {shape === "bimodal" && (
        <div className="mb-4 rounded-lg border px-5 py-4 text-base leading-relaxed"
          style={{ borderColor: RED, background: "rgba(255,0,0,0.05)" }}>
          <strong style={{ color: RED }}>This is the shape it handles worst.</strong>{" "}
          A hard split between a fast mode and a slow one puts a near-vertical cliff in the
          distribution, and quantiles are read by interpolating <em>linearly</em> between centroid
          means — a step is the one thing that representation cannot express. Look at p90 in the
          table: on the smooth shape it is off by about a tenth of a percent, here by well over
          one. Away from the cliff the same sample is still fine, so it is the shape and not the
          data.
        </div>
      )}

      <SourceNote>
        Ported from <code>rust/runtime/src/cellular/sketch.rs</code> — the merging t-digest
        (Dunning), δ = {DEFAULT_COMPRESSION}, K1 scale, and quantiles anchored on the exact min and
        max. Two scopes use it and they are not the same: a cellular run&apos;s <em>live</em> lane
        always reports sketch-derived percentiles while its final report stays exact from record
        partitions, and <code>--sketch-metrics</code> opts the whole report in. Under either,
        counts, sums, extrema and rate aggregates stay exact while percentiles and standard
        deviation become estimates and per-record outputs are unavailable. One property the page
        does not overstate: <code>merge</code> is deterministic at a <em>fixed topology</em>, so
        folding the same cells in a different order moves p50 and p90 by under a tenth of a
        percent — but moves p99 by around one. The extreme tail is the most sensitive, because its
        centroids carry the least weight each.
      </SourceNote>
    </div>
  );
}

function fmt(value: number): string {
  if (!Number.isFinite(value)) return "—";
  if (Number.isInteger(value)) return value.toLocaleString();
  return value.toFixed(2);
}

const BAND_W = 1400;
const BAND_H = 210;

/**
 * Centroid band width against quantile.
 *
 * An earlier version plotted centroid *value* against quantile, which buried the point: a
 * lognormal CDF is nearly flat until its tail, so every centroid piled onto one line and the
 * claim about resolution was invisible. The claim is about how much of the distribution each
 * centroid is responsible for, so that is what this plots — an arch, tall through the body and
 * short at both ends, which is the K1 scale drawn.
 */
function CentroidBands({
  spans,
}: {
  spans: ReturnType<typeof centroidSpans>;
}): React.JSX.Element {
  const padL = 116;
  const padB = 30;
  const inner = BAND_W - padL - 20;
  const widths = spans.map((s) => s.q1 - s.q0);
  const widest = Math.max(...widths, 1e-9);
  const narrowest = Math.min(...widths);
  const x = (q: number) => padL + q * inner;
  const y = (w: number) => BAND_H - padB - (w / widest) * (BAND_H - padB - 26);

  return (
    <div>
      <svg viewBox={`0 0 ${BAND_W} ${BAND_H}`} width="100%" height={BAND_H}
        role="img" aria-label="quantile span of each centroid, across the quantile axis">
        {[0, 0.25, 0.5, 0.75, 1].map((q) => (
          <text key={q} x={x(q)} y={BAND_H - 8} fontSize={12} textAnchor="middle" fill={DIM}>
            {q === 0 ? "q0" : q === 1 ? "q1" : `p${q * 100}`}
          </text>
        ))}

        {spans.map((span, i) => {
          const w = span.q1 - span.q0;
          return (
            <rect key={i} x={x(span.q0) + 0.4} y={y(w)}
              width={Math.max(1.2, x(span.q1) - x(span.q0) - 0.8)}
              height={BAND_H - padB - y(w)} rx={1}
              fill={CYAN} opacity={0.6} />
          );
        })}

        {PERCENTILES.map((p) => (
          <g key={p}>
            <line x1={x(p / 100)} x2={x(p / 100)} y1={20} y2={BAND_H - padB}
              stroke={ORANGE} strokeWidth={1} opacity={0.6} />
            <text x={x(p / 100)} y={16} fontSize={12} textAnchor="middle" fill={ORANGE}>p{p}</text>
          </g>
        ))}

        <line x1={padL} x2={BAND_W - 20} y1={BAND_H - padB} y2={BAND_H - padB}
          stroke="rgba(255,255,255,0.12)" />
        <text x={padL - 8} y={y(widest) + 4} fontSize={12} textAnchor="end" fill={DIM}>
          {(widest * 100).toFixed(1)}%
        </text>
        <text x={padL - 8} y={BAND_H - padB + 4} fontSize={12} textAnchor="end" fill={DIM}>0</text>
        <text x={padL - 8} y={y(widest) - 12} fontSize={11} textAnchor="end" fill={DIM}>
          share of the run
        </text>
      </svg>

      <div className="mt-1 flex gap-8 text-[15px] tabular-nums">
        <span>
          <span className="text-ink-tertiary">widest centroid covers</span>{" "}
          <strong style={{ color: CYAN }}>{(widest * 100).toFixed(2)}%</strong>
          <span className="text-ink-quaternary"> of the distribution</span>
        </span>
        <span>
          <span className="text-ink-tertiary">narrowest covers</span>{" "}
          <strong style={{ color: CYAN }}>{(narrowest * 100).toFixed(3)}%</strong>
          <span className="text-ink-quaternary">
            {" "}— {(widest / Math.max(narrowest, 1e-12)).toFixed(0)}× finer
          </span>
        </span>
      </div>
    </div>
  );
}
