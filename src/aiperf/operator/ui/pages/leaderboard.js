import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { api } from '../lib/api.js';
import { palette } from '../lib/theme.js';
import { MetricSelector } from '../components/metric-selector.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { fmtNumber } from '../lib/format.js';
import { applyChartTheme, PALETTE } from '../lib/chart-theme.js';

const CHART_COLORS = PALETTE;

function formatDate(iso) {
  if (!iso) return '---';
  return new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric', year: '2-digit' });
}

function formatValue(value, unit) {
  if (value == null) return '---';
  const formatted = typeof value === 'number' ? fmtNumber(value, 2) : value;
  return unit ? `${formatted} ${unit}` : String(formatted);
}

export function Leaderboard() {
  const [selected, setSelected] = useState({ metric: 'request_throughput', stat: 'avg' });
  const [model, setModel] = useState('');
  const [endpoint, setEndpoint] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);

    api
      .getLeaderboard(selected.metric, selected.stat)
      .then((resp) => {
        if (!cancelled) setData(resp);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selected.metric, selected.stat]);

  const entries = data?.entries ?? [];

  const filtered = entries.filter((e) => {
    if (model && !(e.model ?? '').toLowerCase().includes(model.toLowerCase())) return false;
    if (endpoint && !(e.endpoint ?? '').toLowerCase().includes(endpoint.toLowerCase())) return false;
    return true;
  });

  const unit = filtered[0]?.unit ?? '';

  const top10 = filtered.slice(0, 10);
  const chartData = {
    labels: top10.map((e) => e.job_id ?? ''),
    datasets: [
      {
        label: selected.metric,
        data: top10.map((e) => e.value ?? 0),
        backgroundColor: top10.map((_, i) => CHART_COLORS[i % CHART_COLORS.length] + 'cc'),
        borderColor: top10.map((_, i) => CHART_COLORS[i % CHART_COLORS.length]),
        borderWidth: 1,
      },
    ],
  };

  const chartOptions = applyChartTheme({
    indexAxis: 'y',
    plugins: {
      legend: { display: false },
    },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: 11 } },
        grid: { color: palette.surface0 + '40' },
        title: {
          display: true,
          text: unit || selected.metric,
          color: palette.overlay1,
          font: { size: 11 },
        },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 11 } },
        grid: { color: palette.surface0 + '40' },
      },
    },
  });

  return html`
    <div class="leaderboard" data-testid="page-leaderboard">
      <div class="section-header" style="margin-bottom: var(--space-4)">
        <span class="section-title">Leaderboard</span>
      </div>

      <!-- Controls -->
      <div class="card" style="margin-bottom: var(--space-4); display: flex; align-items: center; gap: var(--space-6); flex-wrap: wrap">
        <${MetricSelector} value=${selected} onSelect=${setSelected} />
        <div style="display: flex; gap: var(--space-3); align-items: center; flex-wrap: wrap">
          <div style="display: flex; align-items: center; gap: var(--space-2)">
            <label class="metric-selector-label">Model</label>
            <input
              class="metric-selector-select"
              type="text"
              placeholder="Filter by model..."
              value=${model}
              oninput=${(e) => setModel(e.target.value)}
              style="min-width: 160px"
            />
          </div>
          <div style="display: flex; align-items: center; gap: var(--space-2)">
            <label class="metric-selector-label">Endpoint</label>
            <input
              class="metric-selector-select"
              type="text"
              placeholder="Filter by endpoint..."
              value=${endpoint}
              oninput=${(e) => setEndpoint(e.target.value)}
              style="min-width: 160px"
            />
          </div>
        </div>
      </div>

      ${error && html`
        <div class="card" style="border-color: var(--error); color: var(--error); margin-bottom: var(--space-4)">
          Failed to load leaderboard: ${error}
        </div>
      `}

      ${loading && html`
        <div class="card" style="text-align: center; padding: var(--space-8); margin-bottom: var(--space-4)">
          <span class="text-dim">Loading...</span>
        </div>
      `}

      ${!loading && !error && filtered.length === 0 && html`
        <div class="card empty-state" style="margin-bottom: var(--space-4)">
          <p class="text-dim">No results found. Complete some benchmarks and try again.</p>
        </div>
      `}

      ${!loading && filtered.length > 0 && html`
        <!-- Bar chart -->
        <div class="card" style="margin-bottom: var(--space-4)">
          <div class="card-title">Top ${top10.length} -- ${selected.metric} (${selected.stat})</div>
          <${ChartWrapper} type="bar" data=${chartData} options=${chartOptions} height=${Math.max(200, top10.length * 32)} />
        </div>

        <!-- Ranked table -->
        <div class="card">
          <div class="card-title">All Results</div>
          <div style="overflow-x: auto">
            <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm)">
              <thead>
                <tr style="color: var(--subtext0); border-bottom: 1px solid var(--surface1)">
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">#</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Job</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Namespace</th>
                  <th style="text-align: right; padding: var(--space-2) var(--space-3)">Value</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Model</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Endpoint</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Date</th>
                </tr>
              </thead>
              <tbody>
                ${filtered.map((entry, idx) => {
                  const rank = idx + 1;
                  const isTop3 = rank <= 3;
                  const rowColor = rank === 1
                    ? palette.yellow
                    : rank === 2
                    ? palette.subtext1
                    : rank === 3
                    ? palette.peach
                    : null;

                  return html`
                    <tr
                      key=${entry.job_id}
                      style=${'border-bottom: 1px solid var(--surface0);' + (isTop3 ? ' background: ' + rowColor + '0a;' : '')}
                    >
                      <td style=${'padding: var(--space-2) var(--space-3); font-weight: 600;' + (isTop3 ? ' color: ' + rowColor : ' color: var(--overlay0)')}>
                        ${rank}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); font-family: var(--font-mono); font-size: var(--font-size-xs)">
                        ${entry.job_id ?? '---'}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); color: var(--subtext0)">
                        ${entry.namespace ?? '---'}
                      </td>
                      <td style=${'padding: var(--space-2) var(--space-3); text-align: right; font-weight: 600;' + (isTop3 ? ' color: ' + rowColor : '')}>
                        ${formatValue(entry.value, entry.unit)}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); color: var(--subtext0)">
                        ${entry.model ?? '---'}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); color: var(--subtext0); font-size: var(--font-size-xs); max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap">
                        ${entry.endpoint ?? '---'}
                      </td>
                      <td style="padding: var(--space-2) var(--space-3); color: var(--overlay0)">
                        ${formatDate(entry.start_time)}
                      </td>
                    </tr>
                  `;
                })}
              </tbody>
            </table>
          </div>
        </div>
      `}
    </div>
  `;
}
