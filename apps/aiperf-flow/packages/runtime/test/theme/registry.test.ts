/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import {
  type FlowThemeIr,
  type SourceRange,
  type ThemeRole,
  type ThemeValueIr,
} from "@aiperf/flow-schema";
import { describe, expect, test } from "vitest";

import {
  BUNDLED_ROOT_BASE,
  DuplicateThemeIdError,
  IncompleteThemeError,
  LEGACY_VISUAL_FALLBACKS,
  ReservedThemeIdError,
  SYSTEMS_CHALK,
  ThemeContrastError,
  ThemeInheritanceCycleError,
  ThemeRegistry,
  ThemeRoleKindError,
  UnknownThemeIdError,
  createBootstrapThemeRegistry,
  selectActiveThemeId,
} from "../../src/theme/index.js";

const sourceMap: SourceRange = {
  source: "registry.test.flow",
  start: { offset: 0, line: 1, column: 1 },
  end: { offset: 1, line: 1, column: 2 },
};

function theme(
  id: string,
  extendsId: string,
  values: Readonly<Partial<Record<ThemeRole, ThemeValueIr>>> = {},
): FlowThemeIr {
  return { id, extends: extendsId, values, sourceMap };
}

describe("ThemeRegistry", () => {
  test("registers document theme batches atomically", () => {
    const registry = createBootstrapThemeRegistry();
    registry.registerDocumentThemes([
      theme("first", "systems_chalk"),
      theme("second", "first"),
    ]);

    expect(registry.freeze().ids()).toEqual([
      "first",
      "second",
      "systems_chalk",
    ]);
  });

  test("does not mutate the registry when a document batch has a duplicate", () => {
    const registry = createBootstrapThemeRegistry();

    expect(() =>
      registry.registerDocumentThemes([
        theme("first", "systems_chalk"),
        theme("first", "systems_chalk"),
      ]),
    ).toThrow(DuplicateThemeIdError);
    expect(registry.freeze().has("first")).toBe(false);
  });

  test("rejects bundled IDs and the bundled-root sentinel from documents", () => {
    const registry = createBootstrapThemeRegistry();

    expect(() =>
      registry.registerDocumentThemes([
        theme("systems_chalk", "systems_chalk"),
      ]),
    ).toThrow(ReservedThemeIdError);
    expect(() =>
      registry.registerDocumentThemes([
        theme("invalid_root", BUNDLED_ROOT_BASE),
      ]),
    ).toThrow(ReservedThemeIdError);
  });

  test("resolves inheritance with child overrides and caches the result", () => {
    const registry = createBootstrapThemeRegistry();
    registry.registerDocumentThemes([
      theme("custom", "systems_chalk", {
        "accent.control": { kind: "color", value: "#8BE8E0" },
      }),
    ]);
    const frozen = registry.freeze();

    const first = frozen.resolve("custom");
    const second = frozen.resolve("custom");

    expect(first).toBe(second);
    expect(first.values["accent.control"]).toEqual({
      kind: "color",
      value: "#8BE8E0",
    });
    expect(first.values["surface.canvas"]).toEqual({
      kind: "color",
      value: "#232526",
    });
  });

  test("deep-freezes resolved themes", () => {
    const resolved = createBootstrapThemeRegistry()
      .freeze()
      .resolve("systems_chalk");
    const bodyFont = resolved.values["font.body"];

    expect(Object.isFrozen(resolved)).toBe(true);
    expect(Object.isFrozen(resolved.values)).toBe(true);
    expect(Object.isFrozen(bodyFont)).toBe(true);
    expect(bodyFont.kind).toBe("font");
    if (bodyFont.kind === "font") {
      expect(Object.isFrozen(bodyFont.value)).toBe(true);
    }
  });

  test("reports the exact inheritance cycle", () => {
    const registry = createBootstrapThemeRegistry();
    registry.registerDocumentThemes([
      theme("cycle_a", "cycle_b"),
      theme("cycle_b", "cycle_a"),
    ]);

    expect(() => registry.freeze().resolve("cycle_a")).toThrowError(
      new ThemeInheritanceCycleError(
        "Theme inheritance cycle: cycle_a -> cycle_b -> cycle_a",
      ),
    );
  });

  test("rejects unknown theme IDs", () => {
    const frozen = createBootstrapThemeRegistry().freeze();

    expect(() => frozen.resolve("missing")).toThrowError(
      new UnknownThemeIdError('Unknown theme ID "missing"'),
    );
  });

  test("rejects incomplete bundled roots", () => {
    const registry = new ThemeRegistry();
    registry.registerBundled([
      theme("incomplete", BUNDLED_ROOT_BASE, {
        "surface.canvas": { kind: "color", value: "#232526" },
      }),
    ]);

    expect(() => registry.freeze().resolve("incomplete")).toThrow(
      IncompleteThemeError,
    );
  });

  test("revalidates role value kinds", () => {
    const registry = createBootstrapThemeRegistry();
    registry.registerDocumentThemes([
      theme("wrong_kind", "systems_chalk", {
        "stroke.standard": {
          kind: "color",
          value: "#FFFFFF",
        } as ThemeValueIr,
      }),
    ]);

    expect(() => registry.freeze().resolve("wrong_kind")).toThrow(
      ThemeRoleKindError,
    );
  });

  test("rejects resolved themes with insufficient required contrast", () => {
    const registry = createBootstrapThemeRegistry();
    registry.registerDocumentThemes([
      theme("low_contrast", "systems_chalk", {
        "ink.primary": { kind: "color", value: "#232526" },
      }),
    ]);

    expect(() => registry.freeze().resolve("low_contrast")).toThrow(
      ThemeContrastError,
    );
  });

  test("exposes sorted frozen registry IDs", () => {
    const registry = createBootstrapThemeRegistry();
    registry.registerDocumentThemes([
      theme("zebra", "systems_chalk"),
      theme("amber", "systems_chalk"),
    ]);

    const ids = registry.freeze().ids();

    expect(ids).toEqual(["amber", "systems_chalk", "zebra"]);
    expect(Object.isFrozen(ids)).toBe(true);
  });

  test("registers Systems Chalk as the bundled root", () => {
    const frozen = createBootstrapThemeRegistry().freeze();

    expect(SYSTEMS_CHALK.extends).toBe(BUNDLED_ROOT_BASE);
    expect(frozen.has("systems_chalk")).toBe(true);
  });
});

describe("selectActiveThemeId", () => {
  test.each([
    [{ overrideId: "host", documentDefault: "document" }, "host"],
    [{ documentDefault: "document", legacyId: "legacy" }, "document"],
    [{ legacyId: "legacy" }, "legacy"],
    [{}, undefined],
  ] as const)("uses override, document, then legacy precedence", (input, expected) => {
    expect(selectActiveThemeId(input)).toBe(expected);
  });
});

test("preserves the complete legacy visual fallback contract", () => {
  expect(LEGACY_VISUAL_FALLBACKS).toEqual({
    queueLane: "#111827",
    queueWaiting: "#64748b",
    queueServing: "#22c55e",
    waterfallPoint: "#7dcfff",
    waterfallInterval: "#38bdf8",
    waterfallText: "#f8fafc",
    waterfallPlayhead: "#fbbf24",
    segmentFill: "#334155",
    segmentText: "#f8fafc",
    segmentContinuation: "#38bdf8",
    spanUncovered: "#ef4444",
    spanCovered: "#94a3b8",
    spanEdge: "#38bdf8",
    glyphFill: "#f8fafc",
    morphFill: "#38bdf8",
  });
  expect(Object.isFrozen(LEGACY_VISUAL_FALLBACKS)).toBe(true);
});
