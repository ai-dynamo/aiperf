import { html } from 'htm/preact';

const METRICS = [
  { value: 'request_throughput', label: 'Request Throughput' },
  { value: 'request_latency', label: 'Request Latency' },
  { value: 'time_to_first_token', label: 'Time to First Token' },
  { value: 'inter_token_latency', label: 'Inter-Token Latency' },
  { value: 'output_token_throughput', label: 'Output Token Throughput' },
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

  function handleMetricChange(e) {
    onSelect({ metric: e.target.value, stat });
  }

  function handleStatChange(e) {
    onSelect({ metric, stat: e.target.value });
  }

  return html`
    <div class="metric-selector" data-testid="metric-selector">
      <label class="metric-selector-label" for="metric-select">Metric</label>
      <select
        id="metric-select"
        class="metric-selector-select"
        value=${metric}
        onchange=${handleMetricChange}
      >
        ${METRICS.map(
          (m) => html`
            <option key=${m.value} value=${m.value} selected=${m.value === metric}>
              ${m.label}
            </option>
          `,
        )}
      </select>

      <label class="metric-selector-label" for="stat-select">Stat</label>
      <select
        id="stat-select"
        class="metric-selector-select"
        value=${stat}
        onchange=${handleStatChange}
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
