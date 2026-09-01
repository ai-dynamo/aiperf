// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * GPU Telemetry grid.
 *
 * Consumes ``telemetryMetrics`` (list of ``MetricResult`` pushed by the
 * ``realtime_telemetry_metrics`` WebSocket message) and groups them into
 * one card per ``(endpoint, gpu_index)`` parsed from the metric header
 * (format: ``"<Name> | <endpoint> | GPU <index> | <model>"``).
 *
 * Each card shows four canonical hero tiles, plus a compact table of any
 * other telemetry values reported for that GPU.
 */

import { html } from 'htm/preact';
import { telemetryMetrics } from '../lib/state.js';
import { fmtNumber, fmtInt } from '../lib/format.js';

/**
 * Hero tiles, keyed by the field names emitted in ``MetricResult.tag``.
 *
 * The wire tag is ``"<field>_dcgm_<source>_gpu<idx>_<uuid>"`` (see
 * ``GPUTelemetryAccumulator.generate_metric_results``), so ``baseName`` below
 * recovers ``<field>`` exactly as it appears in ``GPU_TELEMETRY_METRICS_CONFIG``
 * (``src/aiperf/gpu_telemetry/constants.py``). Each tile therefore lists its
 * per-vendor field names rather than one hardcoded prefix; NVIDIA and AMD name
 * the same physical signal differently (``nvidia_power_usage`` vs ``amd_power``),
 * so a single suffix cannot cover both.
 *
 * ``tests/unit/api/test_gpu_telemetry_tiles.py`` asserts every alias here still
 * exists in ``GPU_TELEMETRY_METRICS_CONFIG``, so the next backend rename fails
 * loudly in CI instead of silently blanking these tiles.
 */
export const PRIMARY_TAGS = [
  { label: 'Power',       aliases: ['nvidia_power_usage', 'amd_power'] },
  { label: 'Utilization', aliases: ['nvidia_gpu_utilization', 'amd_gfx_activity'] },
  { label: 'Temp',        aliases: ['nvidia_temperature', 'amd_temperature'] },
  { label: 'Memory',      aliases: ['nvidia_memory_used', 'amd_memory_used'] },
];

/** Extract (endpoint, gpuIndex, model) from a MetricResult header like
 *  ``"GPU Power Usage | localhost:9401 | GPU 0 | NVIDIA RTX 6000 Ada Generation"``.
 *  Older telemetry payloads omit the model suffix, so only the first three
 *  fields are required for grouping. */
function parseHeader(header) {
  if (!header || typeof header !== 'string') return null;
  const parts = header.split(' | ').map(s => s.trim());
  if (parts.length < 3) return null;
  const [metricName, endpoint, gpuText, ...modelParts] = parts;
  const gpuMatch = /GPU\s+(\d+)/i.exec(gpuText);
  const gpuIndex = gpuMatch ? parseInt(gpuMatch[1], 10) : 0;
  return { metricName, endpoint, gpuIndex, model: modelParts.join(' | ') };
}

/** The canonical short metric name — strip the DCGM-URL/GPU suffix off the tag. */
export function baseName(tag) {
  if (!tag) return '';
  const cut = tag.indexOf('_dcgm_');
  return cut > 0 ? tag.slice(0, cut) : tag;
}

/** Drop the leading vendor segment so a re-prefixed field still lands on its
 *  tile (``nvidia_temperature`` / ``amd_temperature`` / a future
 *  ``intel_temperature`` all reduce to ``temperature``). */
function vendorlessName(name) {
  const cut = name.indexOf('_');
  return cut > 0 ? name.slice(cut + 1) : name;
}

/** Single source of truth for "does this metric belong to this tile?".
 *  Both the tile lookup and the "other metrics" table go through
 *  ``partitionGpuMetrics`` so the two can never drift apart. */
export function metricMatchesTile(metric, tile, exactOnly = false) {
  const base = metric?.baseName ?? baseName(metric?.tag);
  if (!base) return false;
  if (tile.aliases.includes(base)) return true;
  if (exactOnly) return false;
  const suffix = vendorlessName(base);
  return tile.aliases.some(alias => vendorlessName(alias) === suffix);
}

/** Split a GPU's metrics into hero tiles and the leftover "other" rows.
 *  A metric claimed by a tile is removed from ``others``, so nothing can
 *  render twice; duplicates beyond the first claim stay in the table. */
export function partitionGpuMetrics(metrics) {
  const claimed = new Set();
  const tiles = PRIMARY_TAGS.map((tile) => {
    const metric =
      metrics.find(m => !claimed.has(m) && metricMatchesTile(m, tile, true)) ??
      metrics.find(m => !claimed.has(m) && metricMatchesTile(m, tile));
    if (metric) claimed.add(metric);
    return { label: tile.label, metric: metric ?? null };
  });
  return { tiles, others: metrics.filter(m => !claimed.has(m)) };
}

function groupByGpu(metrics) {
  const groups = new Map();
  for (const r of metrics ?? []) {
    const info = parseHeader(r.header);
    if (!info) continue;
    const key = `${info.endpoint}::${info.gpuIndex}`;
    if (!groups.has(key)) {
      groups.set(key, {
        endpoint: info.endpoint,
        gpuIndex: info.gpuIndex,
        model: info.model,
        metrics: [],
      });
    }
    groups.get(key).metrics.push({ ...r, baseName: baseName(r.tag), shortHeader: info.metricName });
  }
  // Sort: by endpoint, then GPU index.
  return [...groups.values()].sort(
    (a, b) => a.endpoint.localeCompare(b.endpoint) || a.gpuIndex - b.gpuIndex,
  );
}

function formatValueUnit(metric) {
  const v = metric?.current ?? metric?.avg ?? null;
  if (v == null || typeof v !== 'number' || !isFinite(v)) return ['---', ''];
  const body = Math.abs(v) >= 1000 ? fmtInt(Math.round(v)) : fmtNumber(v, 1);
  return [body, metric.unit ?? ''];
}

export function GpuTelemetryCard() {
  const gpus = groupByGpu(telemetryMetrics.value);
  if (gpus.length === 0) return null;

  return html`
    <div>
      <div class="card-title" style="padding-left: 4px; margin-bottom: 8px">
        GPU Telemetry <span class="text-dim" style="margin-left: 6px; font-weight: 400">(${gpus.length} GPU${gpus.length === 1 ? '' : 's'})</span>
      </div>
      <div class="gpu-grid">
        ${gpus.map((gpu) => {
          const headerText = `${gpu.endpoint} | GPU ${gpu.gpuIndex}${gpu.model ? ' | ' + gpu.model : ''}`;
          const { tiles, others } = partitionGpuMetrics(gpu.metrics);
          return html`
            <div class="gpu-card" key=${gpu.endpoint + '::' + gpu.gpuIndex}>
              <div class="gpu-header">${headerText}</div>
              <div class="gpu-primary">
                ${tiles.map((tile) => {
                  const [body, unit] = formatValueUnit(tile.metric);
                  return html`
                    <div class="gpu-tile" key=${tile.label}>
                      <div class="gpu-tile-label">${tile.label}</div>
                      <div class="gpu-tile-val">${body}${unit && html`<span class="gpu-tile-unit"> ${unit}</span>`}</div>
                    </div>
                  `;
                })}
              </div>
              ${others.length > 0 && html`
                <table class="gpu-extra">
                  <tbody>
                    ${others.map((m) => {
                      const [body, unit] = formatValueUnit(m);
                      return html`
                        <tr key=${m.tag}>
                          <td>${m.shortHeader ?? m.baseName}</td>
                          <td style="text-align: right">${body}${unit ? ' ' + unit : ''}</td>
                        </tr>
                      `;
                    })}
                  </tbody>
                </table>
              `}
            </div>
          `;
        })}
      </div>
    </div>
  `;
}
