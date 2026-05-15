// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Wraps live throughput line chart + latency histogram in one Panel.
 *
 * In `live` mode, both charts read from the rolling 60s window in
 * `liveData.timeseries`.  In `completed` mode, throughput is a whole-run
 * timeline from `results` and the histogram is the final histogram from
 * results.  In `archived` mode, throughput hides; histogram renders if
 * present in `profile_export_aiperf.json`.
 *
 * The data-shaping logic (chartData, options) stays the responsibility of
 * the calling page (job-detail) — this component is a layout wrapper that
 * positions the two ChartWrapper instances and controls visibility based on mode.
 */

import { html } from 'htm/preact';
import { Panel } from './panel.js';
import { ChartWrapper } from './chart-wrapper.js';

export function LiveChartsPanel({
  mode,                     // 'live' | 'completed' | 'archived'
  throughputChartData,
  throughputChartOptions,
  histogramChartData,
  histogramChartOptions,
  windowLabel,              // e.g. 'last 60s · auto' for live, 'whole run' for completed
}) {
  const showThroughput = mode !== 'archived' && throughputChartData;
  const showHistogram = histogramChartData;
  if (!showThroughput && !showHistogram) return null;

  return html`
    <${Panel} title="live charts" badge=${windowLabel} testId="panel-live-charts">
      ${showThroughput && html`
        <div class="live-charts-section">
          <div class="live-charts-section-label">throughput</div>
          <${ChartWrapper} type="line"
                           data=${throughputChartData}
                           options=${throughputChartOptions}
                           height=${200} />
        </div>
      `}
      ${showHistogram && html`
        <div class="live-charts-section" style="margin-top:6px">
          <div class="live-charts-section-label">latency · histogram</div>
          <${ChartWrapper} type="bar"
                           data=${histogramChartData}
                           options=${histogramChartOptions}
                           height=${200} />
        </div>
      `}
    <//>
  `;
}
