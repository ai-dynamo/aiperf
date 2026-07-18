/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import type { SourceRange } from "./source.js";

export type DiagnosticSeverity = "error" | "warning" | "info";

export type Diagnostic = Readonly<{
  code: string;
  severity: DiagnosticSeverity;
  message: string;
  range: SourceRange;
  repair?: string;
}>;

export type Result<T> =
  | Readonly<{ ok: true; value: T; diagnostics: readonly Diagnostic[] }>
  | Readonly<{ ok: false; diagnostics: readonly Diagnostic[] }>;

/** Constructs an immutable diagnostic value. */
export function diagnostic(
  code: string,
  severity: DiagnosticSeverity,
  message: string,
  range: SourceRange,
  repair?: string,
): Diagnostic {
  return repair === undefined
    ? { code, severity, message, range }
    : { code, severity, message, range, repair };
}

/** Reports whether a diagnostic collection contains an error. */
export function hasErrors(diagnostics: readonly Diagnostic[]): boolean {
  return diagnostics.some(({ severity }) => severity === "error");
}
