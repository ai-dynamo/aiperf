// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';

const METRICS = [
  { value: 'request_throughput', label: 'Request Throughput', help: 'Requests completed per second.' },
  { value: 'request_latency', label: 'Request Latency', help: 'End-to-end request duration.' },
  { value: 'time_to_first_token', label: 'Time to First Token', help: 'Delay from request send to first streamed token.' },
  { value: 'inter_token_latency', label: 'Inter-Token Latency', help: 'Average gap between consecutive output tokens.' },
  { value: 'output_token_throughput', label: 'Output Token Throughput', help: 'Generated output tokens per second.' },
];

const STATS = [
  { value: 'avg', label: 'Average' },
  { value: 'p50', label: 'P50' },
  { value: 'p99', label: 'P99' },
  { value: 'min', label: 'Min' },
  { value: 'max', label: 'Max' },
];

/**
 * Metric + stat selector.
 * @param {{ value?: {metric: string, stat: string}, onSelect: (v: {metric: string, stat: string}) => void }} props
 */
export function MetricSelector({ value, onSelect }) {
  const metric = value?.metric ?? 'request_throughput';
  const stat = value?.stat ?? 'avg';
  const metricHelp = METRICS.find((m) => m.value === metric)?.help ?? '';

  function handleMetricChange(e) {
    const selectedMetric = METRICS.find((m) => m.value === e.target.value);
    if (!selectedMetric) return;
    onSelect({ metric: selectedMetric.value, stat });
  }

  function handleStatChange(e) {
    const selectedStat = STATS.find((s) => s.value === e.target.value);
    if (!selectedStat) return;
    onSelect({ metric, stat: selectedStat.value });
  }

  return html`
    <div class="metric-selector" data-testid="metric-selector">
      <label class="metric-selector-label" for="metric-select" title="Which performance metric to plot">Metric</label>
      <select
        id="metric-select"
        class="metric-selector-select"
        value=${metric}
        onchange=${handleMetricChange}
        title=${metricHelp}
      >
        ${METRICS.map(
          (m) => html`
            <option key=${m.value} value=${m.value} selected=${m.value === metric}>
              ${m.label}
            </option>
          `,
        )}
      </select>

      <label class="metric-selector-label" for="stat-select" title="Aggregation across requests (e.g. average vs p99 tail)">Stat</label>
      <select
        id="stat-select"
        class="metric-selector-select"
        value=${stat}
        onchange=${handleStatChange}
        title="Aggregation across requests (e.g. average vs p99 tail)"
      >
        ${STATS.map(
          (s) => html`
            <option key=${s.value} value=${s.value} selected=${s.value === stat}>
              ${s.label}
            </option>
          `,
        )}
      </select>
    </div>
  `;
}
