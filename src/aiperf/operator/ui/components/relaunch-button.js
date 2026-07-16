// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Re-launch button — copies a finished AIPerfJob's config into the Launch
 * editor and navigates there. Ported from ``operator/ui/views/run.js``
 * (``serializeYaml`` / ``suggestRetryName`` / ``RelaunchButton``).
 *
 * The bridge between job-detail and the Launch page is a sessionStorage
 * key ``aiperf.launch.prefill`` (60s TTL on the consumer side). We
 * serialize the run's spec back to YAML, write it under that key, and
 * navigate to ``/launch`` — the editor reads + clears it on mount.
 */

import { html } from 'htm/preact';
import { navigate } from '../lib/router.js';
import { palette } from '../lib/theme.js';

const SENSITIVE_CONFIG_KEYS = [
  'api_key',
  'apiKey',
  'authorization',
  'bearerToken',
  'client_secret',
  'password',
  'secret',
  'secretRef',
  'token',
];

function isSensitiveConfigKey(key) {
  const normalized = String(key).toLowerCase().replace(/[^a-z0-9]/g, '');
  return SENSITIVE_CONFIG_KEYS.some(example => {
    const needle = example.toLowerCase().replace(/[^a-z0-9]/g, '');
    return normalized === needle || normalized.includes(needle);
  });
}

export function redactConfigForYaml(value) {
  if (Array.isArray(value)) return value.map(item => redactConfigForYaml(item));
  if (value === null || typeof value !== 'object') return value;
  return Object.fromEntries(Object.entries(value).map(([key, item]) => [
    key,
    isSensitiveConfigKey(key) ? '[REDACTED]' : redactConfigForYaml(item),
  ]));
}

/**
 * Minimal YAML serializer — AIPerfJob specs only. Handles strings, numbers,
 * bools, null, lists, objects. Quotes strings that contain YAML-significant
 * characters. Not a full emitter.
 *
 * Note on the bare-string regex: the legacy version included ``:`` in the
 * allowed character set, which meant URL strings like ``http://x:8000``
 * matched and were emitted unquoted. The Launch editor's hand-rolled
 * ``parseYaml`` then mis-split them on the first ``:``. We drop ``:`` from
 * the bare class so URLs always go through the quoted branch and round-trip
 * cleanly.
 */
export function serializeYaml(obj, indent = 0) {
  const pad = ' '.repeat(indent);
  if (obj === null || obj === undefined) return 'null';
  if (typeof obj === 'boolean') return obj ? 'true' : 'false';
  if (typeof obj === 'number') return String(obj);
  if (typeof obj === 'string') {
    if (obj === '') return "''";
    if (obj.includes('\n')) {
      const blockPad = ' '.repeat(indent + 2);
      return `|\n${obj.split('\n').map(line => blockPad + line).join('\n')}`;
    }
    if (/^[\w./@\-+]+$/.test(obj) && !/^(true|false|null|~)$/i.test(obj) && !/^-?\d+(\.\d+)?$/.test(obj)) {
      return obj;
    }
    return "'" + obj.replace(/'/g, "''") + "'";
  }
  if (Array.isArray(obj)) {
    if (obj.length === 0) return '[]';
    return obj.map(item => {
      if (item !== null && typeof item === 'object' && !Array.isArray(item)) {
        const body = serializeYaml(item, indent + 2);
        // first line gets the dash, subsequent lines stay indented by 2
        const lines = body.split('\n');
        const first = lines[0].trimStart();
        const rest = lines.slice(1).join('\n');
        return `${pad}- ${first}${rest ? '\n' + rest : ''}`;
      }
      return `${pad}- ${serializeYaml(item, indent + 2).trimStart()}`;
    }).join('\n');
  }
  if (typeof obj === 'object') {
    const keys = Object.keys(obj);
    if (keys.length === 0) return '{}';
    return keys.map(k => {
      const v = obj[k];
      if (v !== null && typeof v === 'object') {
        const isEmpty = Array.isArray(v) ? v.length === 0 : Object.keys(v).length === 0;
        if (isEmpty) return `${pad}${k}: ${Array.isArray(v) ? '[]' : '{}'}`;
        return `${pad}${k}:\n${serializeYaml(v, indent + 2)}`;
      }
      return `${pad}${k}: ${serializeYaml(v, indent + 2)}`;
    }).join('\n');
  }
  return String(obj);
}

/**
 * Build a ``<base>-retry-YYMMDD-HHMM`` name. Strips any prior
 * ``-retry-YYMMDD-HHMM`` suffix so repeat relaunches don't stack.
 */
export function suggestRetryName(orig) {
  if (!orig) return 'run-retry';
  const d = new Date();
  const pad = n => String(n).padStart(2, '0');
  const stamp = `${String(d.getFullYear()).slice(2)}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`;
  const base = orig.replace(/-retry-\d{6}-\d{4}(?:-\d+)?$/, '');
  const retryKey = `${base}:${stamp}`;
  const retrySequence = retryKey === suggestRetryName.lastRetryKey ? suggestRetryName.retrySequence + 1 : 0;
  suggestRetryName.lastRetryKey = retryKey;
  suggestRetryName.retrySequence = retrySequence;
  return `${base}-retry-${stamp}${retrySequence ? '-' + (retrySequence + 1) : ''}`;
}

/**
 * Re-launch button. Renders nothing if the run has no spec to copy.
 * Writes the serialized manifest to ``sessionStorage`` under
 * ``aiperf.launch.prefill`` and navigates to ``/launch``.
 */
export function RelaunchButton({ namespace, name, config }) {
  const spec = config?.spec;
  if (!spec || Object.keys(spec).length === 0 || !namespace || !name) return null;
  return html`
    <button type="button"
      class="btn btn--primary"
      onclick=${() => {
        const manifest = {
          apiVersion: config.apiVersion ?? 'aiperf.nvidia.com/v1alpha1',
          kind: config.kind ?? 'AIPerfJob',
          // Relaunch drops server-owned metadata: creationTimestamp, generation, managedFields, resourceVersion, selfLink, uid.
          metadata: {
            name: suggestRetryName(name),
            namespace,
          },
          spec: redactConfigForYaml(spec),
        };
        const yaml = serializeYaml(manifest) + '\n';
        try {
          sessionStorage.setItem('aiperf.launch.prefill', JSON.stringify({
            yaml,
            sourceNs: namespace,
            sourceName: name,
            at: Date.now(),
          }));
        } catch (err) {
          console.warn('Unable to prepare launch prefill', err);
          return;
        }
        navigate('/launch');
      }}
      style=${'background: ' + palette.green + '22; color: ' + palette.green + '; border: 1px solid ' + palette.green + '44; padding: var(--space-2) var(--space-4); border-radius: var(--radius-md); cursor: pointer; font-size: var(--font-size-sm)'}
      data-testid="run-relaunch"
      title="Copy this run's config into the Launch editor"
    >
      Re-launch
    </button>
  `;
}
