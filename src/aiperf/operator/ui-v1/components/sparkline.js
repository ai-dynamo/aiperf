// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Pure-SVG sparkline. No chart lib dependency — lives inline on each KPI
 * tile at ~80×28 px. Renders the recent rolling window of a single stat
 * (``current`` by default) so customers can tell at a glance whether a
 * metric is climbing, steady, or falling.
 */

import { html } from 'htm/preact';

const DEFAULT_WIDTH = 120;
const DEFAULT_HEIGHT = 28;
const PADDING = 2;

/** @param {object} props
 *  @param {Array<{t:number, v:number}>} props.points — chronological order
 *  @param {number} [props.width]
 *  @param {number} [props.height]
 *  @param {string} [props.stroke] — CSS color; default follows the KPI tile
 *  @param {string} [props.fill]   — CSS color for area fill (semi-transparent)
 */
export function Sparkline({
  points,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  stroke = 'var(--accent)',
  fill = 'var(--accent-dim)',
}) {
  if (!points || points.length < 2) {
    // Stable placeholder so the tile reserves the space even when empty.
    return html`<svg class="sparkline" width=${width} height=${height} aria-hidden="true"></svg>`;
  }

  let minV = Infinity, maxV = -Infinity;
  let minT = Infinity, maxT = -Infinity;
  for (const p of points) {
    if (p.v < minV) minV = p.v;
    if (p.v > maxV) maxV = p.v;
    if (p.t < minT) minT = p.t;
    if (p.t > maxT) maxT = p.t;
  }
  // Pad the value range so a perfectly flat line still renders as a band,
  // not a zero-height sliver at the top/bottom.
  const vSpan = (maxV - minV) || Math.max(1, Math.abs(maxV) * 0.05);
  const tSpan = (maxT - minT) || 1;

  const innerW = width - PADDING * 2;
  const innerH = height - PADDING * 2;

  // Map each point to SVG coords; y is inverted.
  const coords = points.map((p) => {
    const x = PADDING + ((p.t - minT) / tSpan) * innerW;
    const y = PADDING + innerH - ((p.v - minV) / vSpan) * innerH;
    return [x, y];
  });

  const linePath = coords
    .map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(1)} ${y.toFixed(1)}`)
    .join(' ');
  const areaPath =
    `${linePath} L${coords[coords.length - 1][0].toFixed(1)} ${(height - PADDING).toFixed(1)} `
    + `L${coords[0][0].toFixed(1)} ${(height - PADDING).toFixed(1)} Z`;

  // Small circle on the latest point — the "you are here" marker.
  const [lx, ly] = coords[coords.length - 1];

  return html`
    <svg class="sparkline" width=${width} height=${height}
         viewBox=${'0 0 ' + width + ' ' + height} preserveAspectRatio="none"
         role="img" aria-label="trend sparkline">
      <path d=${areaPath} fill=${fill} stroke="none"></path>
      <path d=${linePath} fill="none" stroke=${stroke} stroke-width="1.4"
            stroke-linejoin="round" stroke-linecap="round"></path>
      <circle cx=${lx.toFixed(1)} cy=${ly.toFixed(1)} r="1.8" fill=${stroke}></circle>
    </svg>
  `;
}
