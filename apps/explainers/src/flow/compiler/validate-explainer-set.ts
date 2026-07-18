/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-deck uniqueness validation for flow-backed explainer packages.

import {
  diagnostic,
  hasErrors,
  type DeckPackage,
  type Diagnostic,
  type Result,
  type SourceRange,
} from "../schema/index.js";

/**
 * Optional per-package source paths aligned with the `packages` array index.
 * Index-keyed metadata preserves both the first and current `.flow` paths when
 * duplicate ids/routes collide (id/route maps would overwrite the prior path).
 */
export type ValidateExplainerSetOptions = Readonly<{
  /** Source path for `packages[i]`, when known. */
  sourcePaths?: readonly (string | undefined)[];
}>;

function pathAt(
  index: number,
  options: ValidateExplainerSetOptions | undefined,
): string | undefined {
  const path = options?.sourcePaths?.[index];
  return path !== undefined && path.length > 0 ? path : undefined;
}

function rangeForIndex(
  index: number,
  options: ValidateExplainerSetOptions | undefined,
): SourceRange {
  return {
    source: pathAt(index, options) ?? "<unknown>",
    start: { offset: 0, line: 1, column: 1 },
    end: { offset: 0, line: 1, column: 1 },
  };
}

function pathLabel(
  index: number,
  pkg: DeckPackage,
  options: ValidateExplainerSetOptions | undefined,
): string {
  return pathAt(index, options) ?? `package "${pkg.id}" (index ${index})`;
}

/** Validates that every deck in a build set has a unique `id` and `route`. */
export function validateExplainerSet(
  packages: readonly DeckPackage[],
  options: ValidateExplainerSetOptions = {},
): Result<readonly DeckPackage[]> {
  const diagnostics: Diagnostic[] = [];
  const seenIds = new Map<string, number>();
  const seenRoutes = new Map<string, number>();

  for (const [index, pkg] of packages.entries()) {
    const priorIdIndex = seenIds.get(pkg.id);
    if (priorIdIndex !== undefined) {
      const prior = packages[priorIdIndex]!;
      diagnostics.push(
        diagnostic(
          "EXPLAINER_DUPLICATE_ID",
          "error",
          `Duplicate explainer id "${pkg.id}" in ${pathLabel(index, pkg, options)} (also used by ${pathLabel(priorIdIndex, prior, options)}).`,
          rangeForIndex(index, options),
          "Give each explainer deck a unique `id`.",
        ),
      );
    } else {
      seenIds.set(pkg.id, index);
    }

    const priorRouteIndex = seenRoutes.get(pkg.route);
    if (priorRouteIndex !== undefined) {
      const prior = packages[priorRouteIndex]!;
      diagnostics.push(
        diagnostic(
          "EXPLAINER_DUPLICATE_ROUTE",
          "error",
          `Duplicate explainer route "${pkg.route}" in ${pathLabel(index, pkg, options)} (also used by ${pathLabel(priorRouteIndex, prior, options)}).`,
          rangeForIndex(index, options),
          "Give each explainer deck a unique `route`.",
        ),
      );
    } else {
      seenRoutes.set(pkg.route, index);
    }
  }

  if (hasErrors(diagnostics)) {
    return { ok: false, diagnostics };
  }

  return { ok: true, value: packages, diagnostics: [] };
}
