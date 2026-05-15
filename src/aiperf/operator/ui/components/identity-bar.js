// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Sticky identity bar for the job-detail page. Replaces the legacy header
 * card. Renders the run name + actions row, then a typed key/value strip
 * (phase / ns / model / run / elapsed [/ sweep]) and the endpoint URL.
 *
 * Click-to-filter on ns/model is preserved by rendering the original
 * ``NsPill``/``ModelPill`` in the actions row (with their existing
 * ``job-detail-ns-pill``/``job-detail-model-pill`` testids); the KV
 * strip carries duplicate values purely for visual scanning. This keeps
 * existing e2e tests that click the pills working unchanged.
 *
 * @param {object} props
 * @param {string} props.name - Run name.
 * @param {string} props.namespace - Run namespace.
 * @param {string} props.phase - Current phase string.
 * @param {string} [props.model] - Model identifier (or "---").
 * @param {string|number} [props.runLabel] - Pinned run epoch / "live".
 * @param {string} [props.elapsed] - Pre-formatted elapsed string.
 * @param {string} [props.endpointUrl] - Inference server URL.
 * @param {{sweepName?: string, variationLabel?: string}} [props.info]
 * @param {preact.ComponentChildren} props.actions - Right-side action row
 *   (run picker, similar-runs link, live indicator, cancel/relaunch buttons).
 * @param {preact.ComponentChildren} [props.beforeKv] - Optional content
 *   rendered between row1 and the KV strip (e.g. sweep deeplink line).
 */
import { html } from 'htm/preact';
import { KV, KvSep } from './kv.js';

export function IdentityBar({
  name,
  namespace,
  phase,
  model,
  runLabel,
  elapsed,
  endpointUrl,
  info,
  actions,
  beforeKv,
}) {
  return html`
    <header class="job-detail__id" data-testid="job-detail-id">
      <div class="job-detail__id-row1">
        <h2 class="job-detail__id-name">${name}</h2>
        <div class="job-detail__id-actions">${actions}</div>
      </div>
      ${beforeKv}
      <div class="job-detail__id-kv">
        <${KV} k="phase" v=${phase} accent />
        <${KvSep} />
        <${KV} k="ns" v=${namespace} />
        ${model && model !== '---' && html`
          <${KvSep} />
          <${KV} k="model" v=${model} />
        `}
        ${runLabel != null && html`
          <${KvSep} />
          <${KV} k="run" v=${runLabel} testId="kv-run" />
        `}
        ${elapsed && html`
          <${KvSep} />
          <${KV} k="elapsed" v=${elapsed} />
        `}
        ${info?.sweepName && html`
          <${KvSep} />
          <${KV} k="sweep" v=${info.sweepName} />
        `}
      </div>
      ${endpointUrl && html`
        <div class="job-detail__id-endpoint" data-testid="job-detail-endpoint" title=${endpointUrl}>
          ${endpointUrl}
        </div>
      `}
    </header>
  `;
}
