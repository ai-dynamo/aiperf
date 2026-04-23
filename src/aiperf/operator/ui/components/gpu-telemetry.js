// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * GPU Telemetry grid — operator-UI port of ``static-v2/components/gpu-telemetry.js``.
 *
 * Prop-driven: accepts a flat ``metrics`` list of ``MetricResult``-like objects
 * (``{header, tag, current, avg, unit}``) and groups them into one card per
 * ``(endpoint, gpu_index)`` parsed from the metric header
 * (format: ``"<Name> | <endpoint> | GPU <index> | <model>"``).
 *
 * Each card shows the four canonical DCGM metrics as tiles plus a compact
 * table of any other telemetry values reported for that GPU. Returns ``null``
 * when no headers match the GPU format so the host page can hide the card.
 */

import { html } from 'htm/preact';
import { fmtNumber, fmtInt } from '../lib/format.js';

const PRIMARY_TAGS = [
  { match: 'gpu_power_usage',  label: 'Power' },
  { match: 'gpu_utilization',  label: 'Utilization' },
  { match: 'gpu_temperature',  label: 'Temp' },
  { match: 'gpu_memory_used',  label: 'Memory' },
];

/** Extract (endpoint, gpuIndex, model) from a MetricResult header like
 *  ``"GPU Power Usage | localhost:9401 | GPU 0 | NVIDIA RTX 6000 Ada Generation"``. */
function parseHeader(header) {
  if (!header || typeof header !== 'string') return null;
  const parts = header.split(' | ').map(s => s.trim());
  if (parts.length < 4) return null;
  const [metricName, endpoint, gpuText, ...modelParts] = parts;
  const gpuMatch = /GPU\s+(\d+)/i.exec(gpuText);
  const gpuIndex = gpuMatch ? parseInt(gpuMatch[1], 10) : 0;
  return { metricName, endpoint, gpuIndex, model: modelParts.join(' | ') };
}

/** The canonical short metric name — strip the DCGM-URL/GPU suffix off the tag. */
function baseName(tag) {
  if (!tag) return '';
  const cut = tag.indexOf('_dcgm_');
  return cut > 0 ? tag.slice(0, cut) : tag;
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

function findPrimary(gpu, match) {
  return gpu.metrics.find(m => m.baseName === match || m.tag?.startsWith(match + '_'));
}

function formatValueUnit(metric) {
  const v = metric?.current ?? metric?.avg ?? null;
  if (v == null || typeof v !== 'number' || !isFinite(v)) return ['---', ''];
  const body = Math.abs(v) >= 1000 ? fmtInt(Math.round(v)) : fmtNumber(v, 1);
  return [body, metric.unit ?? ''];
}

export function GpuTelemetryCard({ metrics }) {
  const gpus = groupByGpu(metrics);
  if (gpus.length === 0) return null;

  return html`
    <div data-testid="gpu-telemetry">
      <div class="card-title" style="padding-left: 4px; margin-bottom: 8px">
        GPU Telemetry <span class="text-dim" style="margin-left: 6px; font-weight: 400">(${gpus.length} GPU${gpus.length === 1 ? '' : 's'})</span>
      </div>
      <div class="gpu-grid">
        ${gpus.map((gpu) => {
          const headerText = `${gpu.endpoint} | GPU ${gpu.gpuIndex}${gpu.model ? ' | ' + gpu.model : ''}`;
          const otherMetrics = gpu.metrics.filter(
            m => !PRIMARY_TAGS.some(p => m.baseName === p.match || m.tag?.startsWith(p.match + '_')),
          );
          return html`
            <div class="gpu-card" key=${gpu.endpoint + '::' + gpu.gpuIndex}>
              <div class="gpu-header">${headerText}</div>
              <div class="gpu-primary">
                ${PRIMARY_TAGS.map((p) => {
                  const m = findPrimary(gpu, p.match);
                  const [body, unit] = formatValueUnit(m);
                  return html`
                    <div class="gpu-tile" key=${p.match}>
                      <div class="gpu-tile-label">${p.label}</div>
                      <div class="gpu-tile-val">${body}${unit && html`<span class="gpu-tile-unit"> ${unit}</span>`}</div>
                    </div>
                  `;
                })}
              </div>
              ${otherMetrics.length > 0 && html`
                <table class="gpu-extra">
                  <tbody>
                    ${otherMetrics.map((m) => {
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
