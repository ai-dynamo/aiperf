// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

function detectBackends(metrics) {
  const out = { dynamoFrontend: false, dynamoComponent: false, vllm: false, sglang: false, trtllm: false, kvbm: false };
  if (!metrics || typeof metrics !== 'object') return out;
  for (const name of Object.keys(metrics)) {
    if (name.startsWith('dynamo_frontend_')) out.dynamoFrontend = true;
    else if (name.startsWith('dynamo_component_kvstats_')) out.dynamoComponent = true;
    else if (name.startsWith('kvbm_')) out.kvbm = true;
    else if (name.startsWith('vllm:')) out.vllm = true;
    else if (name.startsWith('sglang:')) out.sglang = true;
    else if (name.startsWith('trtllm:')) out.trtllm = true;
  }
  return out;
}

function backendMetric(backendsPresent, metrics, capability) {
  const has = name => metrics && metrics[name] != null;
  const out = [];
  const push = (name, type, statField, role) => {
    if (has(name)) out.push({ name, type, statField, role });
  };

  switch (capability) {
    case 'kvCachePct':
      if (has('dynamo_component_kvstats_gpu_cache_usage_percent')) push('dynamo_component_kvstats_gpu_cache_usage_percent', 'gauge', 'max', 'primary');
      if (has('vllm:kv_cache_usage_perc')) push('vllm:kv_cache_usage_perc', 'gauge', 'max', backendsPresent.dynamoComponent ? 'backend' : 'primary');
      if (has('sglang:token_usage')) push('sglang:token_usage', 'gauge', 'max', (backendsPresent.dynamoComponent || backendsPresent.vllm) ? 'backend' : 'primary');
      break;
    case 'requestsWaiting':
      if (has('dynamo_frontend_queued_requests')) push('dynamo_frontend_queued_requests', 'gauge', 'avg', 'primary');
      if (has('vllm:num_requests_waiting')) push('vllm:num_requests_waiting', 'gauge', 'avg', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      if (has('sglang:num_queue_reqs')) push('sglang:num_queue_reqs', 'gauge', 'avg', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      break;
    case 'reqRate':
      if (has('dynamo_frontend_requests')) push('dynamo_frontend_requests', 'counter', 'rate', 'primary');
      if (has('vllm:request_success')) push('vllm:request_success', 'counter', 'rate', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      if (has('trtllm:request_success')) push('trtllm:request_success', 'counter', 'rate', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      break;
    case 'genTokRate':
      if (has('dynamo_frontend_output_tokens')) push('dynamo_frontend_output_tokens', 'counter', 'rate', 'primary');
      if (has('vllm:generation_tokens')) push('vllm:generation_tokens', 'counter', 'rate', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      if (has('sglang:gen_throughput')) push('sglang:gen_throughput', 'gauge', 'avg', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      break;
    case 'e2eLatency':
      if (has('dynamo_frontend_request_duration_seconds')) push('dynamo_frontend_request_duration_seconds', 'histogram', 'p99_estimate', 'primary');
      if (has('vllm:e2e_request_latency_seconds')) push('vllm:e2e_request_latency_seconds', 'histogram', 'p99_estimate', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      if (has('trtllm:e2e_request_latency_seconds')) push('trtllm:e2e_request_latency_seconds', 'histogram', 'p99_estimate', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      if (has('sglang:e2e_request_latency_seconds')) push('sglang:e2e_request_latency_seconds', 'histogram', 'p99_estimate', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      break;
    case 'ttft':
      if (has('dynamo_frontend_time_to_first_token_seconds')) push('dynamo_frontend_time_to_first_token_seconds', 'histogram', 'p99_estimate', 'primary');
      if (has('vllm:time_to_first_token_seconds')) push('vllm:time_to_first_token_seconds', 'histogram', 'p99_estimate', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      if (has('trtllm:time_to_first_token_seconds')) push('trtllm:time_to_first_token_seconds', 'histogram', 'p99_estimate', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      if (has('sglang:time_to_first_token_seconds')) push('sglang:time_to_first_token_seconds', 'histogram', 'p99_estimate', backendsPresent.dynamoFrontend ? 'backend' : 'primary');
      break;
    default:
      break;
  }
  return out;
}

function extractStatPerSeries(metric, field) {
  const series = metric?.series;
  if (!Array.isArray(series) || series.length === 0) return [];
  const out = [];
  for (const s of series) {
    const v = s?.stats?.[field];
    if (v != null) out.push({ series: s, value: v });
  }
  return out;
}

function sumOf(metric, field) {
  const rows = extractStatPerSeries(metric, field);
  if (rows.length === 0) return null;
  return rows.reduce((acc, row) => acc + row.value, 0);
}

function avgOf(metric, field) {
  const rows = extractStatPerSeries(metric, field);
  if (rows.length === 0) return null;
  return rows.reduce((acc, row) => acc + row.value, 0) / rows.length;
}

function maxOf(metric, field) {
  const rows = extractStatPerSeries(metric, field);
  if (rows.length === 0) return null;
  const max = rows.reduce((acc, row) => Math.max(acc, row.value), -Infinity);
  return isFinite(max) ? max : null;
}

function histogramStat(metric, field) {
  const rows = extractStatPerSeries(metric, field);
  if (rows.length === 0) return null;
  let weighted = 0;
  let weight = 0;
  let plain = 0;
  for (const row of rows) {
    const count = row.series?.stats?.count;
    if (typeof count === 'number' && count > 0) {
      weighted += row.value * count;
      weight += count;
    }
    plain += row.value;
  }
  return weight > 0 ? weighted / weight : plain / rows.length;
}

function normalizePercent(value) {
  if (value == null) return null;
  return value <= 1 ? value * 100 : value;
}

function aggregateForHit(metrics, hit) {
  const metric = metrics?.[hit?.name];
  if (!metric) return null;
  if (hit.type === 'histogram') return histogramStat(metric, hit.statField);
  if (hit.type === 'counter') return sumOf(metric, hit.statField);
  if (hit.statField === 'max') return maxOf(metric, 'max');
  if (hit.statField === 'avg') return avgOf(metric, 'avg');
  return null;
}

function pickBestMetricHit(metrics, backendsPresent, capability) {
  const hits = backendMetric(backendsPresent, metrics, capability);
  if (hits.length <= 1) return hits[0] || null;
  const usable = hits.map(hit => ({ hit, value: aggregateForHit(metrics, hit) }));
  if (capability === 'kvCachePct') {
    return usable.find(item => {
      const avg = avgOf(metrics[item.hit.name], 'avg');
      return (typeof item.value === 'number' && item.value > 0) || (typeof avg === 'number' && avg > 0);
    })?.hit
      || usable.find(item => item.value != null || avgOf(metrics[item.hit.name], 'avg') != null)?.hit
      || hits[0];
  }
  return usable.find(item => typeof item.value === 'number' && item.value > 0)?.hit
    || usable.find(item => item.value != null)?.hit
    || hits[0];
}

function shortHostFromUrl(url) {
  if (!url || typeof url !== 'string') return '';
  const m = url.match(/^(?:[a-z]+:\/\/)?([^/:]+)/i);
  if (!m) return url;
  const host = m[1];
  const dot = host.indexOf('.');
  return dot > 0 ? host.slice(0, dot) : host;
}

function seriesLabel(series) {
  const labels = series?.labels || {};
  if (labels.dynamo_component) return String(labels.dynamo_component);
  if (labels.tp_rank != null) return labels.pp_rank != null ? `tp${labels.tp_rank}/pp${labels.pp_rank}` : `tp${labels.tp_rank}`;
  if (labels.engine != null) return `engine-${labels.engine}`;
  return shortHostFromUrl(series?.endpoint_url) || 'series';
}

function sortSeries(series) {
  if (!Array.isArray(series)) return [];
  return series.slice().sort((a, b) => {
    const au = a?.endpoint_url || '';
    const bu = b?.endpoint_url || '';
    if (au < bu) return -1;
    if (au > bu) return 1;
    return seriesLabel(a).localeCompare(seriesLabel(b));
  });
}

function labelsKey(series) {
  const labels = series?.labels || {};
  return JSON.stringify(Object.keys(labels).sort().map(key => [key, labels[key]]));
}

function seriesKey(series) {
  return `${series?.endpoint_url || ''}|${seriesLabel(series)}|${labelsKey(series)}`;
}

function statForSeries(metric, series, field) {
  const targetKey = seriesKey(series);
  for (const item of metric?.series || []) {
    if (seriesKey(item) === targetKey) {
      const value = item?.stats?.[field];
      return value == null ? null : value;
    }
  }
  return null;
}

function detectedBackendLabels(backendsPresent) {
  const labels = [];
  if (backendsPresent.dynamoFrontend) labels.push('Dynamo Frontend');
  if (backendsPresent.dynamoComponent) labels.push('Dynamo Components');
  if (backendsPresent.vllm) labels.push('vLLM');
  if (backendsPresent.sglang) labels.push('SGLang');
  if (backendsPresent.trtllm) labels.push('TensorRT-LLM');
  if (backendsPresent.kvbm) labels.push('KVBM');
  return labels;
}

function parseTimeMs(value) {
  const ms = Date.parse(value);
  return Number.isNaN(ms) ? null : ms;
}

function buildSummary(serverMetrics, backendsPresent) {
  const summary = serverMetrics?.summary || {};
  const configured = Array.isArray(summary.endpoints_configured) ? summary.endpoints_configured.length : null;
  const successful = Array.isArray(summary.endpoints_successful) ? summary.endpoints_successful.length : null;
  const startMs = parseTimeMs(summary.start_time);
  const endMs = parseTimeMs(summary.end_time);
  const durationSeconds = startMs != null && endMs != null && endMs >= startMs ? (endMs - startMs) / 1000 : null;
  return {
    endpointsConfigured: configured,
    endpointsSuccessful: successful,
    backends: detectedBackendLabels(backendsPresent),
    durationSeconds,
  };
}

function makeKpi({ id, label, value, unit, source, stat, icon, tone = 'accent', progress = null, sub = null, points = null }) {
  if (value == null) return null;
  return { id, label, value, unit, source, stat, icon, tone, progress, sub, points };
}

function collectSeries(metricHits, metrics) {
  const rows = new Map();
  for (const hit of metricHits) {
    if (!hit) continue;
    for (const series of metrics[hit.name]?.series || []) {
      const key = seriesKey(series);
      if (!rows.has(key)) rows.set(key, series);
    }
  }
  return sortSeries([...rows.values()]);
}

export function normalizeServerMetrics(serverMetrics) {
  if (!serverMetrics) return null;
  if (!serverMetrics.endpoint_summaries || serverMetrics.metrics) return serverMetrics;

  const endpointEntries = Object.entries(serverMetrics.endpoint_summaries);
  const metrics = {};
  for (const [endpointKey, endpointSummary] of endpointEntries) {
    const endpointUrl = endpointSummary?.endpoint_url || endpointKey;
    for (const [metricName, metric] of Object.entries(endpointSummary?.metrics || {})) {
      if (!metrics[metricName]) {
        metrics[metricName] = { ...metric, series: [] };
      }
      const series = Array.isArray(metric?.series) ? metric.series : [];
      metrics[metricName].series.push(...series.map(item => ({ ...item, endpoint_url: item.endpoint_url || endpointUrl })));
    }
  }

  const endpoints = endpointEntries.map(([endpointKey, endpointSummary]) => endpointSummary?.endpoint_url || endpointKey);
  return {
    ...serverMetrics,
    summary: {
      ...(serverMetrics.summary || {}),
      endpoints_configured: serverMetrics.summary?.endpoints_configured || endpoints,
      endpoints_successful: serverMetrics.summary?.endpoints_successful || endpoints,
    },
    metrics,
  };
}

export function curateServerMetrics(serverMetrics, sparklines = null) {
  if (!serverMetrics) return null;
  const metrics = serverMetrics.metrics ?? {};
  const backendsPresent = detectBackends(metrics);
  const summary = buildSummary(serverMetrics, backendsPresent);

  const reqHit = pickBestMetricHit(metrics, backendsPresent, 'reqRate');
  const genHit = pickBestMetricHit(metrics, backendsPresent, 'genTokRate');
  const kvHit = pickBestMetricHit(metrics, backendsPresent, 'kvCachePct');
  const ttftHit = pickBestMetricHit(metrics, backendsPresent, 'ttft');
  const e2eHit = pickBestMetricHit(metrics, backendsPresent, 'e2eLatency');
  const waitHit = pickBestMetricHit(metrics, backendsPresent, 'requestsWaiting');
  const latencyHit = ttftHit || e2eHit;

  const reqRate = reqHit ? sumOf(metrics[reqHit.name], reqHit.statField) : null;
  const genRate = genHit ? aggregateForHit(metrics, genHit) : null;
  const kvPeak = kvHit ? normalizePercent(maxOf(metrics[kvHit.name], 'max')) : null;
  const kvAvg = kvHit ? normalizePercent(avgOf(metrics[kvHit.name], 'avg')) : null;
  const latencyP99 = latencyHit ? histogramStat(metrics[latencyHit.name], 'p99_estimate') : null;
  const waitingAvg = waitHit ? avgOf(metrics[waitHit.name], 'avg') : null;
  const waitingPeak = waitHit ? maxOf(metrics[waitHit.name], 'max') : null;

  const waitingPoints = sparklines?.['requests-waiting'] ?? null;
  const waitingBufHasNonzero = Array.isArray(waitingPoints)
    && waitingPoints.some(p => typeof p?.v === 'number' && p.v > 0);

  const kpis = [
    makeKpi({ id: 'request-rate', label: 'Request rate', value: reqRate, unit: 'req/s', source: reqHit?.name, stat: reqHit?.statField, icon: 'speed',
      points: sparklines?.['request-rate'] ?? null }),
    makeKpi({ id: 'generation-token-rate', label: 'Generation token rate', value: genRate, unit: 'tok/s', source: genHit?.name, stat: genHit?.statField, icon: 'tokens',
      points: sparklines?.['generation-token-rate'] ?? null }),
    makeKpi({
      id: 'kv-cache-pressure',
      label: 'KV/cache pressure',
      value: kvPeak,
      unit: '%',
      source: kvHit?.name,
      stat: 'max',
      icon: 'goodput',
      tone: kvPeak != null && kvPeak >= 90 ? 'warn' : 'accent',
      progress: kvPeak,
      sub: kvAvg != null ? `avg ${kvAvg.toFixed(1)}%` : null,
      points: sparklines?.['kv-cache-pressure'] ?? null,
    }),
    makeKpi({
      id: latencyHit === ttftHit ? 'p99-ttft' : 'p99-e2e-latency',
      label: latencyHit === ttftHit ? 'p99 TTFT' : 'p99 e2e latency',
      value: latencyP99 == null ? null : latencyP99 * 1000,
      unit: 'ms',
      source: latencyHit?.name,
      stat: 'p99_estimate',
      icon: 'timer',
      points: (() => {
        if (!latencyHit) return null;
        const latId = latencyHit === ttftHit ? 'p99-ttft' : 'p99-e2e-latency';
        const raw = sparklines?.[latId];
        if (!Array.isArray(raw)) return null;
        return raw.map(p => ({ t: p.t, v: p.v * 1000 }));
      })(),
    }),
    waitingAvg != null && (((waitingPeak ?? waitingAvg) > 0) || waitingBufHasNonzero) ? makeKpi({
      id: 'requests-waiting',
      label: 'Requests waiting',
      value: waitingAvg,
      unit: '',
      source: waitHit?.name,
      stat: 'avg',
      icon: 'clock',
      tone: 'warn',
      sub: waitingPeak != null ? `peak ${waitingPeak.toFixed(0)}` : null,
      points: waitingPoints,
    }) : null,
  ].filter(Boolean);

  const metricHits = [reqHit, genHit, kvHit, latencyHit, waitHit].filter(Boolean);
  const detailRows = collectSeries(metricHits, metrics).map(series => ({
    endpoint: series.endpoint_url || '',
    backend: seriesLabel(series),
    reqRate: reqHit ? statForSeries(metrics[reqHit.name], series, reqHit.statField) : null,
    genRate: genHit ? statForSeries(metrics[genHit.name], series, genHit.statField) : null,
    kvPressure: kvHit ? normalizePercent(statForSeries(metrics[kvHit.name], series, 'max')) : null,
    waiting: waitHit ? statForSeries(metrics[waitHit.name], series, 'avg') : null,
    latencyP99Ms: latencyHit ? (statForSeries(metrics[latencyHit.name], series, 'p99_estimate') ?? null) : null,
  })).map(row => ({ ...row, latencyP99Ms: row.latencyP99Ms == null ? null : row.latencyP99Ms * 1000 }));

  const sources = [...new Set(metricHits.map(hit => `${hit.name} (${hit.statField})`))];
  if (kpis.length === 0 && detailRows.length === 0 && summary.backends.length === 0) return null;
  return { summary, kpis, detailRows, sources };
}

/**
 * Collapse a single normalized server-metrics snapshot into one number per
 * KPI id whose underlying metric resolved to a finite value (including
 * zero). Used by the per-job WS layer to push one sample per scrape into a
 * per-KPI rolling buffer.
 *
 * This is intentionally more permissive than `curateServerMetrics`:
 * `curateServerMetrics` may suppress some tiles when the current snapshot
 * is all-zero (specifically `requests-waiting`, gated on
 * `(waitingPeak ?? waitingAvg) > 0`), but the aggregator emits the
 * zero-sample so the rolling buffer stays continuous. A queue going
 * 5 -> 3 -> 1 -> 0 -> 0 -> 2 must keep its zero entries in the buffer so
 * the sparkline renders the full trend; Task 2 extends the curator's tile
 * gate to use the buffer instead of just the current snapshot.
 *
 * The latency tile id flips between `'p99-ttft'` and `'p99-e2e-latency'`
 * depending on which histogram is present in the snapshot; the resolved id
 * is returned alongside the values so the WS layer can key its buffer
 * consistently.
 *
 * Note: the latency value is returned in seconds (the raw histogram unit);
 * the curator multiplies by 1000 for display. Keep the buffer in seconds
 * and let the rendering layer scale on read.
 *
 * @param {object} normalizedServerMetrics - shape produced by
 *   `normalizeServerMetrics`
 * @returns {{ values: Object<string, number>, latencyKpiId: string|null }}
 */
export function aggregateSparklineSnapshot(normalizedServerMetrics) {
  const out = { values: {}, latencyKpiId: null };
  if (!normalizedServerMetrics) return out;
  const metrics = normalizedServerMetrics.metrics ?? {};
  const backendsPresent = detectBackends(metrics);

  const reqHit = pickBestMetricHit(metrics, backendsPresent, 'reqRate');
  const genHit = pickBestMetricHit(metrics, backendsPresent, 'genTokRate');
  const kvHit = pickBestMetricHit(metrics, backendsPresent, 'kvCachePct');
  const ttftHit = pickBestMetricHit(metrics, backendsPresent, 'ttft');
  const e2eHit = pickBestMetricHit(metrics, backendsPresent, 'e2eLatency');
  const waitHit = pickBestMetricHit(metrics, backendsPresent, 'requestsWaiting');
  const latencyHit = ttftHit || e2eHit;

  const reqRate = reqHit ? sumOf(metrics[reqHit.name], reqHit.statField) : null;
  const genRate = genHit ? aggregateForHit(metrics, genHit) : null;
  const kvPeak = kvHit ? normalizePercent(maxOf(metrics[kvHit.name], 'max')) : null;
  const latencyP99 = latencyHit ? histogramStat(metrics[latencyHit.name], 'p99_estimate') : null;
  const waitingAvg = waitHit ? avgOf(metrics[waitHit.name], 'avg') : null;

  const ifNum = (v) => (typeof v === 'number' && isFinite(v) ? v : null);

  if (ifNum(reqRate) != null) out.values['request-rate'] = reqRate;
  if (ifNum(genRate) != null) out.values['generation-token-rate'] = genRate;
  if (ifNum(kvPeak) != null) out.values['kv-cache-pressure'] = kvPeak;
  if (ifNum(waitingAvg) != null) out.values['requests-waiting'] = waitingAvg;
  if (ifNum(latencyP99) != null) {
    out.latencyKpiId = latencyHit === ttftHit ? 'p99-ttft' : 'p99-e2e-latency';
    out.values[out.latencyKpiId] = latencyP99;
  }
  return out;
}
