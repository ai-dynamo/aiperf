import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { api } from '../lib/api.js';
import { palette } from '../lib/theme.js';
import { navigate } from '../lib/router.js';
import { MetricSelector } from '../components/metric-selector.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { fmtNumber } from '../lib/format.js';

function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleDateString([], {
    month: 'short',
    day: 'numeric',
    year: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatDateShort(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

function formatNum(v) {
  if (v == null) return '\u2014';
  return typeof v === 'number' ? fmtNumber(v, 3) : String(v);
}

export function History() {
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
      .getHistory(selected.metric, selected.stat)
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

  const chartData = {
    labels: filtered.map((e) => formatDateShort(e.start_time)),
    datasets: [
      {
        label: `${selected.metric} (${selected.stat})${unit ? ' [' + unit + ']' : ''}`,
        data: filtered.map((e) => e.value ?? null),
        borderColor: palette.blue,
        backgroundColor: palette.blue + '22',
        fill: true,
        tension: 0.3,
        pointRadius: 4,
        pointHoverRadius: 6,
        pointBackgroundColor: palette.blue,
        borderWidth: 2,
      },
    ],
  };

  const chartOptions = {
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          title: (items) => {
            const idx = items[0]?.dataIndex;
            if (idx == null) return '';
            const e = filtered[idx];
            return e ? `${e.job_id ?? ''} — ${formatDate(e.start_time)}` : '';
          },
          label: (item) => {
            const e = filtered[item.dataIndex];
            const val = item.raw;
            return `${formatNum(val)}${unit ? ' ' + unit : ''}`;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: 11 }, maxTicksLimit: 12 },
        grid: { color: palette.surface0 + '40' },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 11 } },
        grid: { color: palette.surface0 + '40' },
        title: {
          display: true,
          text: unit || selected.metric,
          color: palette.overlay1,
          font: { size: 11 },
        },
      },
    },
  };

  return html`
    <div class="history-page" data-testid="page-history">
      <div class="section-header" style="margin-bottom: var(--space-4)">
        <span class="section-title">History</span>
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
              placeholder="Filter by model…"
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
              placeholder="Filter by endpoint…"
              value=${endpoint}
              oninput=${(e) => setEndpoint(e.target.value)}
              style="min-width: 160px"
            />
          </div>
        </div>
      </div>

      ${error && html`
        <div class="card" style="border-color: var(--error); color: var(--error); margin-bottom: var(--space-4)">
          Failed to load history: ${error}
        </div>
      `}

      ${loading && html`
        <div class="card" style="text-align: center; padding: var(--space-8); margin-bottom: var(--space-4)">
          <span class="text-dim">Loading…</span>
        </div>
      `}

      ${!loading && !error && filtered.length === 0 && html`
        <div class="card empty-state" style="margin-bottom: var(--space-4)">
          <p class="text-dim">No history found. Complete some benchmarks and try again.</p>
        </div>
      `}

      ${!loading && filtered.length > 0 && html`
        <!-- Line chart -->
        <div class="card" style="margin-bottom: var(--space-4)">
          <div class="card-title">${selected.metric} (${selected.stat}) over time</div>
          <${ChartWrapper} type="line" data=${chartData} options=${chartOptions} height=${300} />
        </div>

        <!-- Data table -->
        <div class="card">
          <div class="card-title">Data Points</div>
          <div style="overflow-x: auto">
            <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm)">
              <thead>
                <tr style="color: var(--subtext0); border-bottom: 1px solid var(--surface1)">
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Job</th>
                  <th style="text-align: right; padding: var(--space-2) var(--space-3)">Value</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Model</th>
                  <th style="text-align: left; padding: var(--space-2) var(--space-3)">Date</th>
                </tr>
              </thead>
              <tbody>
                ${filtered.map((entry) => html`
                  <tr key=${entry.job_id + entry.start_time} style="border-bottom: 1px solid var(--surface0)">
                    <td style="padding: var(--space-2) var(--space-3)">
                      <span
                        onclick=${() => navigate('/jobs/' + encodeURIComponent(entry.namespace ?? 'default') + '/' + encodeURIComponent(entry.job_id ?? ''))}
                        style="color: var(--blue); cursor: pointer; font-family: var(--font-mono); font-size: var(--font-size-xs)"
                      >
                        ${entry.job_id ?? '—'}
                      </span>
                    </td>
                    <td style="text-align: right; padding: var(--space-2) var(--space-3); font-weight: 600">
                      ${formatNum(entry.value)}${entry.unit ? html` <span style="color: var(--overlay0); font-weight: normal">${entry.unit}</span>` : ''}
                    </td>
                    <td style="padding: var(--space-2) var(--space-3); color: var(--subtext0)">
                      ${entry.model ?? '—'}
                    </td>
                    <td style="padding: var(--space-2) var(--space-3); color: var(--overlay0)">
                      ${formatDate(entry.start_time)}
                    </td>
                  </tr>
                `)}
              </tbody>
            </table>
          </div>
        </div>
      `}
    </div>
  `;
}
