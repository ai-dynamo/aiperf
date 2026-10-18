// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * Resolve app-relative paths against the page's own location so the SPA keeps
 * working when a gateway serves it under a path prefix
 * (e.g. ``https://gw/internal/req/ns/svc/``). Root-absolute URLs like
 * ``/api/v1`` would drop that prefix and hit the gateway itself.
 */

/**
 * @param {string} path app-relative path, with or without a leading slash
 * @returns {string} absolute path including the serving prefix
 */
export function appPath(path) {
  return new URL(path.replace(/^\/+/, ''), document.baseURI).pathname;
}

/** API root, e.g. ``/internal/req/ns/svc/api/v1`` behind a gateway. */
export const API_BASE = appPath('api/v1');
