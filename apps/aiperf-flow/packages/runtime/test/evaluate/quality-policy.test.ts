// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, test } from "vitest";

import { buildDisplayList } from "../../src/display-list.js";
import {
  applyQualityPolicy,
  qualityPolicyProfile,
  type QualityAnnotatedCommand,
  type QualityAnnotatedHitRegion,
  type QualityDisplayList,
} from "../../src/evaluate/quality-policy.js";

const bounds = { x: 0, y: 0, width: 100, height: 40 } as const;

function pathCommand(
  id: string,
  order: number,
  extras: Partial<QualityAnnotatedCommand> = {},
): QualityAnnotatedCommand {
  return {
    kind: "path",
    id,
    order,
    path: `M ${order} 0 H ${order + 10} V 10 H ${order} Z`,
    fill: "#76b900",
    paintBounds: { x: order, y: 0, width: 10, height: 10 },
    damageBounds: { x: order, y: 0, width: 10, height: 10 },
    qualityClass: "required-semantic",
    ...extras,
  } as QualityAnnotatedCommand;
}

function textCommand(
  id: string,
  order: number,
  text: string,
  extras: Partial<QualityAnnotatedCommand> = {},
): QualityAnnotatedCommand {
  return {
    kind: "text",
    id,
    order,
    text,
    origin: { x: 0, y: 16 },
    font: { family: "sans-serif", sizePx: 12 },
    fill: "#ffffff",
    paintBounds: bounds,
    damageBounds: bounds,
    qualityClass: "required-semantic",
    ...extras,
  } as QualityAnnotatedCommand;
}

function hitRegion(
  id: string,
  order: number,
  extras: Partial<QualityAnnotatedHitRegion> = {},
): QualityAnnotatedHitRegion {
  return {
    id,
    semanticId: id.replace(/-hit$/, ""),
    order,
    bounds,
    ...extras,
  };
}

function foundationList(): QualityDisplayList {
  return buildDisplayList({
    commands: [
      pathCommand("entity-bounds", 0, {
        qualityClass: "required-semantic",
        semanticEntityId: "request-a",
      }),
      textCommand("entity-label", 1, "Request A", {
        qualityClass: "required-semantic",
        semanticEntityId: "request-a",
      }),
      textCommand("narration-cue", 2, "Admission begins.", {
        qualityClass: "required-semantic",
        narrationCueMarker: true,
      }),
      pathCommand("particle-field", 3, {
        qualityClass: "decorative",
        decorativeFamily: "particles",
        motion: { progress: 0.42, pathId: "orbit-a" },
      }),
      pathCommand("blur-halo", 4, {
        qualityClass: "decorative",
        decorativeFamily: "blur",
        motion: { progress: 0.8, pathId: "halo-pulse" },
      }),
      pathCommand("shadow-cast", 5, {
        qualityClass: "decorative",
        decorativeFamily: "shadow",
        motion: { progress: 0.1, pathId: "shadow-drift" },
      }),
      pathCommand("glow-rim", 6, {
        qualityClass: "decorative",
        decorativeFamily: "glow",
      }),
    ],
    hitRegions: [
      hitRegion("request-a-hit", 0, {
        role: "select",
        qualityClass: "required-semantic",
      }),
      hitRegion("request-a-inspect", 1, {
        role: "inspect",
        qualityClass: "required-semantic",
      }),
      hitRegion("request-a-focus", 2, {
        role: "focus",
        qualityClass: "required-semantic",
      }),
      hitRegion("playhead-scrub", 3, {
        role: "scrub",
        qualityClass: "decorative",
        decorativeFamily: "particles",
      }),
    ],
    paintBounds: bounds,
    damageBounds: bounds,
  }) as QualityDisplayList;
}

describe("applyQualityPolicy", () => {
  test("reference profile preserves every command and hit region", () => {
    const list = foundationList();
    const { list: next, report } = applyQualityPolicy(
      list,
      qualityPolicyProfile("reference"),
    );

    expect(next.commands.map((command) => command.id)).toEqual(
      list.commands.map((command) => command.id),
    );
    expect(next.hitRegions.map((region) => region.id)).toEqual(
      list.hitRegions.map((region) => region.id),
    );
    expect(report.suppressedCommandIndices).toEqual([]);
    expect(report.suppressedFamilies).toEqual([]);
    expect(report.tier).toBe("reference");
  });

  test("degraded profile removes decorative particle/blur/shadow/glow while preserving semantics", () => {
    const list = foundationList();
    const { list: next, report } = applyQualityPolicy(
      list,
      qualityPolicyProfile("degraded"),
    );

    expect(next.commands.map((command) => command.id)).toEqual([
      "entity-bounds",
      "entity-label",
      "narration-cue",
    ]);
    expect(
      next.commands.filter((command) => command.kind === "text").map((c) => {
        if (c.kind !== "text") {
          throw new Error("expected text");
        }
        return c.text;
      }),
    ).toEqual(["Request A", "Admission begins."]);
    expect(
      next.commands.some(
        (command) =>
          "semanticEntityId" in command &&
          command.semanticEntityId === "request-a",
      ),
    ).toBe(true);
    expect(
      next.commands.some(
        (command) =>
          "narrationCueMarker" in command && command.narrationCueMarker === true,
      ),
    ).toBe(true);

    expect(next.hitRegions.map((region) => region.id)).toEqual([
      "request-a-hit",
      "request-a-inspect",
      "request-a-focus",
    ]);
    expect(report.suppressedCommandIndices).toEqual([3, 4, 5, 6]);
    expect(report.suppressedFamilies).toEqual([
      "blur",
      "glow",
      "particles",
      "shadow",
    ]);
    expect(report.tier).toBe("degraded");
  });

  test("never drops required-semantic commands even when decorative families are off", () => {
    const list = buildDisplayList({
      commands: [
        pathCommand("required-glow", 0, {
          qualityClass: "required-semantic",
          decorativeFamily: "glow",
        }),
        pathCommand("optional-glow", 1, {
          qualityClass: "decorative",
          decorativeFamily: "glow",
        }),
      ],
      hitRegions: [],
      paintBounds: bounds,
      damageBounds: bounds,
    }) as QualityDisplayList;

    const { list: next, report } = applyQualityPolicy(
      list,
      qualityPolicyProfile("degraded"),
    );

    expect(next.commands.map((command) => command.id)).toEqual([
      "required-glow",
    ]);
    expect(report.suppressedCommandIndices).toEqual([1]);
    expect(report.suppressedFamilies).toEqual(["glow"]);
  });

  test("never removes select, inspect, or focus hit regions", () => {
    const list = buildDisplayList({
      commands: [
        pathCommand("chrome", 0, {
          qualityClass: "decorative",
          decorativeFamily: "particles",
        }),
      ],
      hitRegions: [
        hitRegion("keep-select", 0, {
          role: "select",
          qualityClass: "decorative",
          decorativeFamily: "particles",
        }),
        hitRegion("keep-inspect", 1, {
          role: "inspect",
          qualityClass: "decorative",
          decorativeFamily: "blur",
        }),
        hitRegion("keep-focus", 2, {
          role: "focus",
          qualityClass: "decorative",
          decorativeFamily: "shadow",
        }),
        hitRegion("drop-scrub", 3, {
          role: "scrub",
          qualityClass: "decorative",
          decorativeFamily: "particles",
        }),
      ],
      paintBounds: bounds,
      damageBounds: bounds,
    }) as QualityDisplayList;

    const { list: next } = applyQualityPolicy(
      list,
      qualityPolicyProfile("degraded"),
    );

    expect(next.hitRegions.map((region) => region.id)).toEqual([
      "keep-select",
      "keep-inspect",
      "keep-focus",
    ]);
  });

  test("reduced-motion zeroes motion metadata on remaining commands", () => {
    const list = foundationList();
    const { list: next, report } = applyQualityPolicy(
      list,
      qualityPolicyProfile("reference", { motion: "reduced" }),
    );

    expect(next.commands).toHaveLength(list.commands.length);
    for (const command of next.commands) {
      if ("motion" in command && command.motion !== undefined) {
        expect(command.motion).toEqual({ progress: 0, pathId: undefined });
      }
    }
    expect(report.motionReduced).toBe(true);
    expect(report.suppressedCommandIndices).toEqual([]);
  });

  test("reduced-motion still suppresses decorative families under degraded tier", () => {
    const list = foundationList();
    const { list: next, report } = applyQualityPolicy(
      list,
      qualityPolicyProfile("degraded", { motion: "reduced" }),
    );

    expect(next.commands.map((command) => command.id)).toEqual([
      "entity-bounds",
      "entity-label",
      "narration-cue",
    ]);
    for (const command of next.commands) {
      expect(
        "motion" in command ? command.motion : undefined,
      ).toBeUndefined();
    }
    expect(report.motionReduced).toBe(true);
    expect(report.suppressedFamilies).toContain("particles");
  });

  test("suppresses nested decorative children deterministically", () => {
    const list = buildDisplayList({
      commands: [
        {
          kind: "group",
          id: "root",
          order: 0,
          paintBounds: bounds,
          damageBounds: bounds,
          qualityClass: "required-semantic",
          children: [
            pathCommand("keep", 0, { qualityClass: "required-semantic" }),
            pathCommand("drop-blur", 1, {
              qualityClass: "decorative",
              decorativeFamily: "blur",
            }),
            {
              kind: "layer",
              id: "fx",
              order: 2,
              paintBounds: bounds,
              damageBounds: bounds,
              qualityClass: "decorative",
              decorativeFamily: "particles",
              children: [
                pathCommand("drop-particle", 0, {
                  qualityClass: "decorative",
                  decorativeFamily: "particles",
                }),
              ],
            },
          ],
        } as QualityAnnotatedCommand,
      ],
      hitRegions: [],
      paintBounds: bounds,
      damageBounds: bounds,
    }) as QualityDisplayList;

    const { list: next, report } = applyQualityPolicy(
      list,
      qualityPolicyProfile("degraded"),
    );

    expect(next.commands).toHaveLength(1);
    const root = next.commands[0];
    expect(root?.kind).toBe("group");
    if (root?.kind !== "group") {
      throw new Error("expected group");
    }
    expect(root.children.map((child) => child.id)).toEqual(["keep"]);
    expect(report.suppressedFamilies).toEqual(["blur", "particles"]);
  });

  test("hoists required-semantic children out of suppressed decorative parents", () => {
    const list = buildDisplayList({
      commands: [
        {
          kind: "layer",
          id: "glow-shell",
          order: 0,
          paintBounds: bounds,
          damageBounds: bounds,
          qualityClass: "decorative",
          decorativeFamily: "glow",
          children: [
            textCommand("label", 0, "Keep me", {
              qualityClass: "required-semantic",
            }),
            pathCommand("spark", 1, {
              qualityClass: "decorative",
              decorativeFamily: "particles",
            }),
          ],
        } as QualityAnnotatedCommand,
      ],
      hitRegions: [
        hitRegion("label-select", 0, {
          role: "select",
          qualityClass: "required-semantic",
        }),
      ],
      paintBounds: bounds,
      damageBounds: bounds,
    }) as QualityDisplayList;

    const { list: next, report } = applyQualityPolicy(
      list,
      qualityPolicyProfile("degraded"),
    );

    expect(next.commands.map((command) => command.id)).toEqual(["label"]);
    expect(next.hitRegions.map((region) => region.id)).toEqual([
      "label-select",
    ]);
    expect(report.suppressedFamilies).toEqual(["glow", "particles"]);
  });

  test("returns immutable display lists and degradation reports", () => {
    const list = foundationList();
    const { list: next, report } = applyQualityPolicy(
      list,
      qualityPolicyProfile("degraded"),
    );

    expect(Object.isFrozen(next)).toBe(true);
    expect(Object.isFrozen(next.commands)).toBe(true);
    expect(Object.isFrozen(report)).toBe(true);
    expect(Object.isFrozen(report.suppressedCommandIndices)).toBe(true);
    expect(Object.isFrozen(report.suppressedFamilies)).toBe(true);
  });

  test("applies the same degraded output for identical inputs", () => {
    const list = foundationList();
    const profile = qualityPolicyProfile("degraded", { motion: "reduced" });

    const first = applyQualityPolicy(list, profile);
    const second = applyQualityPolicy(list, profile);

    expect(first).toEqual(second);
    expect(JSON.stringify(first)).toEqual(JSON.stringify(second));
  });
});
