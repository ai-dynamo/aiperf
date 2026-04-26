// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Extract the top-level ``namespace:`` field from an AIPerfJob YAML body.
 *
 * Used by the launch view's divergence check — full YAML parsing is
 * overkill (and would pull a parser into the bundle) for one field that
 * is, by spec, top-level on the AIPerfJob CR. Indented ``namespace:``
 * (e.g. ``metadata.namespace``) is intentionally ignored: the AIPerfJob
 * shape places ``namespace`` at the top of the spec body the launch
 * editor produces.
 *
 * @param {string} yamlText raw editor contents
 * @returns {string|null} the unquoted value, or null if absent / empty
 */
export function extractNamespaceField(yamlText) {
  if (!yamlText) return null;
  const lines = yamlText.split('\n');
  for (const raw of lines) {
    if (raw.length === 0) continue;
    if (raw[0] === ' ' || raw[0] === '\t') continue;
    if (raw[0] === '#') continue;
    const m = /^namespace:\s*(.*)$/.exec(raw);
    if (!m) continue;
    let v = m[1].trim();
    if (!v) return null;
    if (v[0] !== '"' && v[0] !== "'") {
      const hashIdx = v.indexOf('#');
      if (hashIdx >= 0) v = v.slice(0, hashIdx).trim();
    }
    if (!v) return null;
    if ((v[0] === '"' && v[v.length - 1] === '"') ||
        (v[0] === "'" && v[v.length - 1] === "'")) {
      v = v.slice(1, -1);
    }
    return v || null;
  }
  return null;
}
