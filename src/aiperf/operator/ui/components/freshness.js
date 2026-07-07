// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { html } from 'htm/preact';
import { freshnessSources } from '../lib/state.js';

const LABELS = {
  idle: 'Idle',
  loading: 'Loading',
  fresh: 'Live',
  stale: 'Stale',
  retrying: 'Retrying',
  stopped: 'Static',
};

function sourceLabel(source) {
  return String(source ?? '')
    .replace(/-/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function secondsAgo(at) {
  if (at == null) return null;
  return Math.max(0, Math.round((Date.now() - at) / 1000));
}

function freshnessTitle(source) {
  if (!source) return 'No live source information yet';
  const bits = [];
  if (source.lastSuccessAt != null) {
    bits.push(`last successful update ${new Date(source.lastSuccessAt).toLocaleTimeString()}`);
  }
  if (source.lastAttemptAt != null) {
    bits.push(`last attempt ${new Date(source.lastAttemptAt).toLocaleTimeString()}`);
  }
  if (source.lastError) bits.push(`last error: ${source.lastError}`);
  if (source.reason) bits.push(`reason: ${source.reason}`);
  return bits.length > 0 ? bits.join(' · ') : 'Waiting for first refresh';
}

export function FreshnessPill({ source, compact = false }) {
  if (!source) return null;
  const status = source.status ?? 'idle';
  const ago = secondsAgo(source.lastSuccessAt);
  const label = LABELS[status] ?? status;
  const age = ago == null ? '' : compact ? ` ${ago}s` : ` · ${ago}s ago`;
  return html`
    <span
      class=${`freshness-pill freshness-pill--${status}`}
      title=${freshnessTitle(source)}
      data-testid="freshness-pill"
    >
      <span class="freshness-dot" aria-hidden="true"></span>
      <span>${compact ? sourceLabel(source.source) + ' ' : sourceLabel(source.source) + ': '}${label}${age}</span>
    </span>
  `;
}

export function FreshnessStrip() {
  const sources = freshnessSources.value;
  if (sources.length === 0) return null;
  return html`
    <div class="freshness-strip" role="status" aria-live="polite" data-testid="freshness-strip">
      <span class="freshness-strip-label">Live status</span>
      <div class="freshness-strip-sources">
        ${sources.map((source) => html`<${FreshnessPill} key=${source.source} source=${source} compact=${true} />`)}
      </div>
    </div>
  `;
}

export function StaleBanner({ source, label }) {
  if (!source || !['stale', 'retrying'].includes(source.status)) return null;
  const lastSuccess = source.lastSuccessAt == null
    ? 'no successful refresh yet'
    : `last successful update ${Math.max(0, Math.round((Date.now() - source.lastSuccessAt) / 1000))}s ago`;
  return html`
    <div class="stale-banner" role="status" data-testid="stale-banner">
      <strong>${label ?? sourceLabel(source.source)} is ${source.status === 'retrying' ? 'retrying' : 'stale'}.</strong>
      <span>${lastSuccess}; showing last-known data.</span>
      ${source.lastError && html`<span class="stale-banner-error">${source.lastError}</span>`}
    </div>
  `;
}
