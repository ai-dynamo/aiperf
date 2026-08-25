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
  failed: 'Failed',
};

// Statuses the StaleBanner speaks up for. ``failed`` is here because a source
// that never loaded is the case a reader is least equipped to notice on their
// own: there is no old timestamp drifting, just an empty page.
const DEGRADED_STATUSES = ['stale', 'retrying', 'failed'];

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
  if (!source || !DEGRADED_STATUSES.includes(source.status)) return null;
  const name = label ?? sourceLabel(source.source);
  const neverLoaded = source.lastSuccessAt == null;
  // A never-loaded source has no last-known data to be showing, so it gets
  // its own sentence. Claiming otherwise sends the reader looking for stale
  // numbers on a screen that has none.
  const headline = neverLoaded
    ? `${name} could not be loaded.`
    : `${name} is ${source.status === 'retrying' ? 'retrying' : 'stale'}.`;
  const detail = neverLoaded
    ? 'No successful refresh yet — nothing on screen for this source.'
    : `last successful update ${Math.max(0, Math.round((Date.now() - source.lastSuccessAt) / 1000))}s ago; showing last-known data.`;
  return html`
    <div
      class=${'stale-banner' + (neverLoaded ? ' stale-banner--failed' : '')}
      role="status"
      data-testid="stale-banner"
    >
      <strong>${headline}</strong>
      <span>${detail}</span>
      ${source.lastError && html`<span class="stale-banner-error">${source.lastError}</span>`}
    </div>
  `;
}
