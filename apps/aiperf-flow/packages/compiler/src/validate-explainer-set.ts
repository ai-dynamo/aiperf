/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Cross-deck uniqueness validation for flow-backed explainer packages.
//!
//! A multi-file explainer build must not emit two decks that share an `id` or
//! a `route`. This check runs after individual packages are lowered and before
//! they are registered or packed.

import {
  diagnostic,
  hasErrors,
  type DeckPackage,
  type Diagnostic,
  type Result,
  type SourceRange,
} from "@aiperf/flow-schema";

const unknownRange: SourceRange = {
  source: "<unknown>",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 0, line: 1, column: 1 },
};

/**
 * Validates that every deck in a build set has a unique `id` and `route`.
 *
 * On success, returns the same package list. On failure, returns one diagnostic
 * per colliding field (duplicate id and/or duplicate route).
 */
export function validateExplainerSet(
  packages: readonly DeckPackage[],
): Result<readonly DeckPackage[]> {
  const diagnostics: Diagnostic[] = [];
  const seenIds = new Map<string, number>();
  const seenRoutes = new Map<string, number>();

  for (const [index, pkg] of packages.entries()) {
    const priorId = seenIds.get(pkg.id);
    if (priorId !== undefined) {
      diagnostics.push(
        diagnostic(
          "EXPLAINER_DUPLICATE_ID",
          "error",
          `Duplicate explainer id "${pkg.id}" (also used by package at index ${priorId}).`,
          unknownRange,
          "Give each explainer deck a unique `id`.",
        ),
      );
    } else {
      seenIds.set(pkg.id, index);
    }

    const priorRoute = seenRoutes.get(pkg.route);
    if (priorRoute !== undefined) {
      diagnostics.push(
        diagnostic(
          "EXPLAINER_DUPLICATE_ROUTE",
          "error",
          `Duplicate explainer route "${pkg.route}" (also used by package at index ${priorRoute}).`,
          unknownRange,
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
