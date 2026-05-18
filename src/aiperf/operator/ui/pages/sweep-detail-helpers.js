const RUNNING_PHASES = new Set(['pending', 'running', 'aggregating']);

const DEFAULT_HEADLINE_METRICS = [
  { key: 'request_throughput', stat: 'avg', label: 'Req throughput', unit: 'req/s' },
  { key: 'output_token_throughput', stat: 'avg', label: 'Output tok/s', unit: 'tok/s' },
  { key: 'total_token_throughput', stat: 'avg', label: 'Total tok/s', unit: 'tok/s' },
  { key: 'request_latency', stat: 'p50', label: 'Req latency p50', unit: 'ms' },
  { key: 'request_latency', stat: 'p99', label: 'Req latency p99', unit: 'ms' },
  { key: 'time_to_first_token', stat: 'p50', label: 'TTFT p50', unit: 'ms' },
  { key: 'time_to_first_token', stat: 'p99', label: 'TTFT p99', unit: 'ms' },
  { key: 'inter_token_latency', stat: 'avg', label: 'ITL avg', unit: 'ms' },
];

function pick(obj, keys) {
  for (const key of keys) {
    if (obj?.[key] != null) return obj[key];
  }
  return null;
}

function meanStd(values) {
  const filtered = values.filter(v => typeof v === 'number' && Number.isFinite(v));
  if (filtered.length === 0) return null;
  const n = filtered.length;
  const mean = filtered.reduce((a, b) => a + b, 0) / n;
  if (n < 2) return { mean, std: 0, cv: null, n };
  const variance = filtered.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const cv = mean !== 0 ? std / Math.abs(mean) : null;
  return { mean, std, cv, n };
}

function metricValue(summary, metric) {
  const value = summary?.[metric.key]?.[metric.stat];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function indexCells(cells) {
  const out = new Map();
  for (const cell of cells?.cells ?? []) {
    const idx = pick(cell, ['variation_index', 'variationIndex']);
    if (idx == null) continue;
    out.set(Number(idx), cell);
  }
  return out;
}

export function buildSweepVariations({
  manifest,
  childSummaries,
  cells,
  headlineMetrics = DEFAULT_HEADLINE_METRICS,
}) {
  if (!manifest || manifest.length === 0) return [];
  const cellsByIndex = indexCells(cells);
  const groups = new Map();
  for (const c of manifest) {
    const idx = Number(pick(c, ['variation_index', 'variationIndex']) ?? 0);
    if (!groups.has(idx)) {
      groups.set(idx, {
        variation_index: idx,
        label: pick(c, ['variation_label', 'variationLabel']) ?? '',
        n_total: 0,
        summaries: [],
      });
    }
    const group = groups.get(idx);
    group.n_total += 1;
    const summary = childSummaries?.[c.name]?.summary ?? null;
    if (summary) group.summaries.push(summary);
  }
  for (const group of groups.values()) {
    if (group.summaries.length === 0) {
      const cell = cellsByIndex.get(group.variation_index);
      if (cell?.metrics) group.summaries.push(cell.metrics);
    }
  }
  return [...groups.values()]
    .sort((a, b) => a.variation_index - b.variation_index)
    .map(group => {
      const perMetric = {};
      for (const metric of headlineMetrics) {
        const values = group.summaries
          .map(summary => metricValue(summary, metric))
          .filter(value => value != null);
        perMetric[metric.key + '.' + metric.stat] = meanStd(values) ?? { mean: null, std: null, cv: null, n: 0 };
      }
      return {
        variation_index: group.variation_index,
        label: group.label,
        n_trials: group.summaries.length,
        n_total: group.n_total,
        perMetric,
      };
    });
}

export function shouldShowSweepDiagnostics(phase) {
  return RUNNING_PHASES.has((phase ?? '').toLowerCase());
}
