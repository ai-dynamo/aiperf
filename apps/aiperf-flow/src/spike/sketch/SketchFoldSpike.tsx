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
  count,
  exactPercentile,
  foldCells,
  latencySamples,
  quantile,
  splitAcrossCells,
  DEFAULT_COMPRESSION,
  PERCENTILES,
  type Shape,
  type TDigest,
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

      <Panel label="WHERE THE RESOLUTION GOES" hint="one box per centroid, placed at the quantiles it covers"
        className="mb-4">
        <Legend>
          <LegendItem mark="▭" color={CYAN}>a centroid — width is the quantile band it summarizes</LegendItem>
          <LegendItem mark="│" color={ORANGE}>a reported percentile</LegendItem>
        </Legend>
        <CentroidBands spans={spans} digest={folded} values={values} />
        <p className="mt-2 max-w-5xl text-[14px] leading-relaxed text-ink-quaternary">
          The boxes are narrow at both ends and wide in the middle, and that is deliberate. A
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

const BAND_W = 900;
const BAND_H = 132;

/**
 * Centroids laid out along the quantile axis.
 *
 * Quantile is the x axis rather than value, because the claim being made is about where the
 * digest spends resolution — which is a statement about quantile space, not about milliseconds.
 */
function CentroidBands({
  spans,
  digest,
  values,
}: {
  spans: ReturnType<typeof centroidSpans>;
  digest: TDigest;
  values: readonly number[];
}): React.JSX.Element {
  const pad = 44;
  const inner = BAND_W - pad - 16;
  const x = (q: number) => pad + q * inner;
  const sorted = useMemo(() => [...values].sort((a, b) => a - b), [values]);

  const lo = digest.min;
  const hi = digest.max;
  const y = (v: number) => BAND_H - 30 - ((v - lo) / Math.max(1e-9, hi - lo)) * (BAND_H - 52);

  return (
    <svg viewBox={`0 0 ${BAND_W} ${BAND_H}`} width="100%" height={BAND_H}
      role="img" aria-label="centroids across the quantile axis">
      {[0, 0.25, 0.5, 0.75, 1].map((q) => (
        <text key={q} x={x(q)} y={BAND_H - 6} fontSize={12} textAnchor="middle" fill={DIM}>
          {q === 0 ? "q0" : q === 1 ? "q1" : `p${q * 100}`}
        </text>
      ))}

      {spans.map((span, i) => (
        <rect key={i} x={x(span.q0)} y={y(span.centroid.mean) - 4}
          width={Math.max(0.7, x(span.q1) - x(span.q0))} height={8} rx={1.5}
          fill={CYAN} opacity={0.55} />
      ))}

      {/* The exact distribution behind them, so the fit is visible rather than asserted. */}
      <path
        d={Array.from({ length: 120 }, (_, i) => {
          const q = i / 119;
          return `${i === 0 ? "M" : "L"} ${x(q)} ${y(exactPercentile(sorted, q * 100))}`;
        }).join(" ")}
        fill="none" stroke="var(--color-ink-secondary)" strokeWidth={1.25} opacity={0.75} />

      {PERCENTILES.map((p) => (
        <g key={p}>
          <line x1={x(p / 100)} x2={x(p / 100)} y1={10} y2={BAND_H - 26}
            stroke={ORANGE} strokeWidth={1} opacity={0.55} />
          <text x={x(p / 100)} y={8} fontSize={11} textAnchor="middle" fill={ORANGE}>p{p}</text>
        </g>
      ))}

      <text x={4} y={y(hi) + 4} fontSize={11} fill={DIM}>{hi.toFixed(0)}</text>
      <text x={4} y={y(lo) + 4} fontSize={11} fill={DIM}>{lo.toFixed(0)}</text>
      <text x={4} y={BAND_H - 6} fontSize={11} fill={DIM}>ms</text>
      {quantile(digest, 0.5) !== null && count(digest) > 0 && (
        <title>{`${count(digest)} values in ${spans.length} centroids`}</title>
      )}
    </svg>
  );
}
