import { html } from 'htm/preact';
import { useState, useEffect } from 'preact/hooks';
import { api } from '../lib/api.js';
import { palette } from '../lib/theme.js';
import { ChartWrapper } from '../components/chart-wrapper.js';
import { fmtNumber } from '../lib/format.js';

// Metrics where lower is better
const LOWER_IS_BETTER = new Set([
  'request_latency',
  'time_to_first_token',
  'inter_token_latency',
]);

const JOB_COLORS = [
  palette.mauve,
  palette.blue,
  palette.green,
  palette.peach,
  palette.pink,
  palette.teal,
  palette.sapphire,
  palette.yellow,
];

function isBetter(metric, a, b) {
  if (a == null || b == null) return false;
  return LOWER_IS_BETTER.has(metric) ? a < b : a > b;
}

function bestValue(metric, values) {
  const vals = Object.values(values).filter((v) => v != null);
  if (vals.length === 0) return null;
  return LOWER_IS_BETTER.has(metric) ? Math.min(...vals) : Math.max(...vals);
}

function formatNum(v) {
  if (v == null) return '\u2014';
  return typeof v === 'number' ? fmtNumber(v, 3) : String(v);
}

export function Compare() {
  const [storedJobs, setStoredJobs] = useState([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [jobsError, setJobsError] = useState(null);

  const [search, setSearch] = useState('');
  // selectedKeys are composite "<namespace>/<job_id>" strings — this is the key
  // format used by the backend's compare response (_pivot_compare_rows).
  const [selectedKeys, setSelectedKeys] = useState([]);

  const [compareData, setCompareData] = useState(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState(null);

  useEffect(() => {
    api
      .listResults()
      .then((resp) => {
        setStoredJobs(resp?.jobs ?? resp ?? []);
        setJobsLoading(false);
      })
      .catch((err) => {
        setJobsError(err.message);
        setJobsLoading(false);
      });
  }, []);

  function compositeKey(job) {
    const id = job.job_id ?? '';
    const ns = job.namespace ?? '';
    return ns ? `${ns}/${id}` : id;
  }

  function splitKey(key) {
    const idx = key.indexOf('/');
    return idx < 0 ? { ns: '', jobId: key } : { ns: key.slice(0, idx), jobId: key.slice(idx + 1) };
  }

  function toggleJob(key) {
    setSelectedKeys((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  }

  function clearSelection() {
    setSelectedKeys([]);
    setCompareData(null);
    setCompareError(null);
  }

  async function handleCompare() {
    if (selectedKeys.length < 2) return;
    setComparing(true);
    setCompareError(null);
    setCompareData(null);
    try {
      // Backend filters by bare job_id; strip the namespace prefix.
      const bareJobIds = selectedKeys.map((k) => splitKey(k).jobId);
      const resp = await api.compareJobs(bareJobIds);
      setCompareData(resp);
    } catch (err) {
      setCompareError(err.message);
    } finally {
      setComparing(false);
    }
  }

  const filtered = storedJobs.filter((job) => {
    const id = job.job_id ?? '';
    const ns = job.namespace ?? '';
    const q = search.toLowerCase();
    return id.toLowerCase().includes(q) || ns.toLowerCase().includes(q);
  });

  const entries = compareData?.entries ?? [];
  // Display keys match composite values-map keys from the backend.
  const displayKeys = selectedKeys;

  // Build chart data: grouped bars per metric, one dataset per job
  const chartData = (() => {
    if (entries.length === 0) return null;
    const metrics = entries.map((e) => e.metric + (e.stat ? ' (' + e.stat + ')' : ''));
    const datasets = displayKeys.map((key, idx) => ({
      label: splitKey(key).jobId,
      data: entries.map((e) => e.values?.[key] ?? null),
      backgroundColor: JOB_COLORS[idx % JOB_COLORS.length] + 'cc',
      borderColor: JOB_COLORS[idx % JOB_COLORS.length],
      borderWidth: 1,
    }));
    return { labels: metrics, datasets };
  })();

  const chartOptions = {
    plugins: {
      legend: {
        display: true,
        labels: { color: palette.overlay0, font: { size: 11 } },
      },
    },
    scales: {
      x: {
        ticks: { color: palette.overlay0, font: { size: 10 }, maxRotation: 30 },
        grid: { color: palette.surface0 + '40' },
      },
      y: {
        ticks: { color: palette.overlay0, font: { size: 11 } },
        grid: { color: palette.surface0 + '40' },
      },
    },
  };

  return html`
    <div class="compare-page">
      <div class="section-header" style="margin-bottom: var(--space-4)">
        <span class="section-title">Compare Jobs</span>
      </div>

      <div style="display: grid; grid-template-columns: 320px 1fr; gap: var(--space-4); align-items: start">

        <!-- Left: Job selector -->
        <div class="card">
          <div class="card-title" style="margin-bottom: var(--space-3)">Select Jobs</div>

          <input
            type="text"
            class="metric-selector-select"
            placeholder="Search jobs…"
            value=${search}
            oninput=${(e) => setSearch(e.target.value)}
            style="width: 100%; margin-bottom: var(--space-3)"
          />

          ${jobsLoading && html`
            <div class="text-dim" style="padding: var(--space-4); text-align: center">Loading…</div>
          `}

          ${jobsError && html`
            <div style="color: var(--error); font-size: var(--font-size-sm)">${jobsError}</div>
          `}

          ${!jobsLoading && filtered.length === 0 && html`
            <div class="text-dim" style="padding: var(--space-3); text-align: center; font-size: var(--font-size-sm)">
              No completed jobs found.
            </div>
          `}

          <div style="max-height: 320px; overflow-y: auto">
            ${filtered.map((job) => {
              const jobId = job.job_id ?? '';
              const ns = job.namespace ?? '';
              const key = compositeKey(job);
              const isChecked = selectedKeys.includes(key);
              return html`
                <label
                  key=${key}
                  style=${'display: flex; align-items: flex-start; gap: var(--space-2); padding: var(--space-2) var(--space-1); cursor: pointer; border-radius: var(--radius-sm);' + (isChecked ? ' background: var(--surface0);' : '')}
                >
                  <input
                    type="checkbox"
                    checked=${isChecked}
                    onchange=${() => toggleJob(key)}
                    style="margin-top: 2px; accent-color: var(--mauve)"
                  />
                  <div>
                    <div style="font-size: var(--font-size-sm); font-family: var(--font-mono)">${jobId}</div>
                    <div style="font-size: var(--font-size-xs); color: var(--overlay0)">${ns}</div>
                  </div>
                </label>
              `;
            })}
          </div>

          <div style="margin-top: var(--space-3); display: flex; gap: var(--space-2)">
            <button
              onclick=${handleCompare}
              disabled=${selectedKeys.length < 2 || comparing}
              style=${'flex: 1; padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid; font-size: var(--font-size-sm); cursor: pointer;'
                + (selectedKeys.length >= 2 && !comparing
                  ? ' background: var(--mauve); color: var(--base); border-color: var(--mauve); font-weight: 600;'
                  : ' background: var(--surface0); color: var(--overlay0); border-color: var(--surface1); cursor: not-allowed;')}
            >
              ${comparing ? 'Comparing…' : `Compare (${selectedKeys.length})`}
            </button>
            ${selectedKeys.length > 0 && html`
              <button
                onclick=${clearSelection}
                style="padding: var(--space-2) var(--space-3); border-radius: var(--radius-sm); border: 1px solid var(--surface1); background: transparent; color: var(--subtext0); font-size: var(--font-size-sm); cursor: pointer"
              >
                Clear
              </button>
            `}
          </div>
        </div>

        <!-- Right: Results -->
        <div>
          ${selectedKeys.length > 0 && html`
            <div style="display: flex; flex-wrap: wrap; gap: var(--space-2); margin-bottom: var(--space-4)">
              ${selectedKeys.map((key, idx) => html`
                <span
                  key=${key}
                  style=${'display: inline-flex; align-items: center; gap: var(--space-1); padding: var(--space-1) var(--space-2); border-radius: 999px; font-size: var(--font-size-xs); font-family: var(--font-mono);'
                    + ' background: ' + JOB_COLORS[idx % JOB_COLORS.length] + '22;'
                    + ' color: ' + JOB_COLORS[idx % JOB_COLORS.length] + ';'
                    + ' border: 1px solid ' + JOB_COLORS[idx % JOB_COLORS.length] + '44;'}
                >
                  ${splitKey(key).jobId}
                  <span
                    onclick=${() => toggleJob(key)}
                    style="cursor: pointer; opacity: 0.7; font-size: var(--font-size-xs)"
                  >✕</span>
                </span>
              `)}
            </div>
          `}

          ${compareError && html`
            <div class="card" style="border-color: var(--error); color: var(--error); margin-bottom: var(--space-4)">
              Compare failed: ${compareError}
            </div>
          `}

          ${!compareData && !comparing && selectedKeys.length < 2 && html`
            <div class="card empty-state">
              <p class="text-dim">Select 2 or more jobs from the list to compare them.</p>
            </div>
          `}

          ${comparing && html`
            <div class="card" style="text-align: center; padding: var(--space-8)">
              <span class="text-dim">Running comparison…</span>
            </div>
          `}

          ${compareData && !comparing && html`
            <!-- Metrics table -->
            <div class="card" style="margin-bottom: var(--space-4)">
              <div class="card-title">Metric Comparison</div>
              <div style="overflow-x: auto">
                <table style="width: 100%; border-collapse: collapse; font-size: var(--font-size-sm)">
                  <thead>
                    <tr style="color: var(--subtext0); border-bottom: 1px solid var(--surface1)">
                      <th style="text-align: left; padding: var(--space-2) var(--space-3)">Metric</th>
                      ${displayKeys.map((key, idx) => html`
                        <th
                          key=${key}
                          style=${'text-align: right; padding: var(--space-2) var(--space-3); color: ' + JOB_COLORS[idx % JOB_COLORS.length]}
                        >
                          ${splitKey(key).jobId}
                        </th>
                      `)}
                    </tr>
                  </thead>
                  <tbody>
                    ${entries.map((entry) => {
                      const best = bestValue(entry.metric, entry.values ?? {});
                      return html`
                        <tr key=${entry.metric + entry.stat} style="border-bottom: 1px solid var(--surface0)">
                          <td style="padding: var(--space-2) var(--space-3)">
                            <div>${entry.metric}</div>
                            ${entry.stat && html`<div style="font-size: var(--font-size-xs); color: var(--overlay0)">${entry.stat}${entry.unit ? ' · ' + entry.unit : ''}</div>`}
                          </td>
                          ${displayKeys.map((key) => {
                            const val = entry.values?.[key] ?? null;
                            const isBest = val != null && val === best;
                            return html`
                              <td
                                key=${key}
                                style=${'text-align: right; padding: var(--space-2) var(--space-3); font-weight: 600;'
                                  + (isBest ? ' color: ' + palette.green : ' color: var(--text)')}
                              >
                                ${formatNum(val)}
                              </td>
                            `;
                          })}
                        </tr>
                      `;
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            <!-- Bar chart -->
            ${chartData && html`
              <div class="card">
                <div class="card-title">Visual Comparison</div>
                <${ChartWrapper} type="bar" data=${chartData} options=${chartOptions} height=${300} />
              </div>
            `}
          `}
        </div>
      </div>
    </div>
  `;
}
