/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared source-oriented formatting for Flow diagnostics.
//! Eager-safe: no dependency on developer tools or runtime evaluation.

import type { Diagnostic } from "./schema/index.js";

/** Formats a diagnostic for browser-visible compiler and registry errors. */
export function formatDiagnostic(diagnostic: Diagnostic): string {
  const { source, start } = diagnostic.range;
  const repair =
    diagnostic.repair === undefined ? "" : ` (${diagnostic.repair})`;
  return `${source}:${start.line}:${start.column}: ${diagnostic.severity} ${diagnostic.code}: ${diagnostic.message}${repair}`;
}
