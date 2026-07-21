/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { describe, expect, it } from "vitest";
import type { RenderNodeIr } from "../schema/index.js";
import {
  CHIP_PAD_X,
  estimateTextWidth,
} from "../../core/diagram/text-metrics.js";
import { resolvedSceneSnapshot } from "../../core/diagram/resolution/serialize.js";
import type { ResolvedSceneSnapshot } from "../../core/diagram/resolution/types.js";
// @ts-expect-error The plain-Node verifier intentionally has no TS declarations.
import { verifyPackageIr } from "../../../scripts/flow-verifier/ir.mjs";
import {
  indexResolvedWorldGeometry,
  resolveSceneForGeometryVerification,
} from "./verify-geometry.js";

describe("verifier geometry layout parity", () => {
  it("exposes the canonical resolver result to browser verification", () => {
    const scene = {
      viewport: { width: 700, height: 400 },
      roots: [
        {
          id: "box",
          kind: "rect",
          geometry: { x: 12, y: 18, width: 40, height: 20 },
          children: [],
        },
      ],
      timeline: [],
    };

    const resolved = resolveSceneForGeometryVerification(scene);

    expect(resolved.worldGeometryById.get("box")).toEqual({
      x: 12,
      y: 18,
      width: 40,
      height: 20,
    });
    expect(resolved.scene).toBe(scene);
  });

  it("indexes intrinsic children after rail reflow in world space", () => {
    const roots = [
      {
        id: "rail",
        kind: "group",
        capabilityId: "layout.rail",
        geometry: { x: 20, y: 30, width: 160, height: 22 },
        style: { direction: "row", gap: 8 },
        accessibility: { label: "rail" },
        children: [
          {
            id: "long",
            kind: "component",
            capabilityId: "core.chip",
            geometry: { x: 0, y: 0, width: 84, height: 26 },
            style: {},
            props: { label: "authoritative" },
            accessibility: { label: "authoritative" },
            children: [],
          },
          {
            id: "short",
            kind: "component",
            capabilityId: "core.chip",
            geometry: { x: 0, y: 0, width: 84, height: 26 },
            style: {},
            props: { label: "ok" },
            accessibility: { label: "ok" },
            children: [],
          },
        ],
      },
    ] as unknown as readonly RenderNodeIr[];

    const index = indexResolvedWorldGeometry(roots);
    const long = index.get("long");
    const short = index.get("short");
    const longWidth =
      estimateTextWidth("authoritative", 11, "bold") + CHIP_PAD_X;

    expect(index.get("rail")?.width).toBe(longWidth + 8 + 84);
    expect(long).toEqual({
      x: 20,
      y: 30,
      width: longWidth,
      height: 70.2,
    });
    expect(short?.x).toBe(20 + long!.width + 8);
  });

  it("indexes a semantic circle after stack reflow in world space", () => {
    const roots = [
      {
        id: "stack",
        kind: "group",
        capabilityId: "layout.stack",
        geometry: { x: 100, y: 60, width: 0, height: 0 },
        style: { direction: "column", gap: 7 },
        accessibility: { label: "stack" },
        children: [
          {
            id: "first",
            kind: "rect",
            capabilityId: "core.rect",
            geometry: { x: 99, y: 99, width: 40, height: 20 },
            style: {},
            accessibility: { label: "first" },
          },
          {
            id: "circle",
            kind: "component",
            capabilityId: "core.circle",
            geometry: { x: 0, y: 0, width: 0, height: 0 },
            style: { r: 12 },
            props: { center: { x: 300, y: 200 }, r: 12 },
            accessibility: { label: "circle" },
            children: [],
          },
        ],
      },
    ] as unknown as readonly RenderNodeIr[];

    const index = indexResolvedWorldGeometry(roots);

    expect(index.get("stack")).toEqual({
      x: 100,
      y: 60,
      width: 40,
      height: 51,
    });
    expect(index.get("first")).toEqual({
      x: 100,
      y: 60,
      width: 40,
      height: 20,
    });
    expect(index.get("circle")).toEqual({
      x: 100,
      y: 87,
      width: 24,
      height: 24,
    });
  });

  it("feeds resolved stack and circle bounds to the Node verifier", () => {
    const roots = [
      {
        id: "stack",
        kind: "group",
        capabilityId: "layout.stack",
        geometry: { x: 100, y: 60, width: 0, height: 0 },
        style: { direction: "column", gap: 7 },
        accessibility: { label: "stack" },
        children: [
          {
            id: "first",
            kind: "rect",
            capabilityId: "core.rect",
            geometry: { x: 0, y: 0, width: 40, height: 20 },
            style: {},
            accessibility: { label: "first" },
          },
          {
            id: "circle",
            kind: "component",
            capabilityId: "core.circle",
            geometry: { x: 0, y: 0, width: 0, height: 0 },
            style: { r: 12 },
            props: { center: { x: 300, y: 200 }, r: 12 },
            accessibility: { label: "circle" },
            children: [],
          },
        ],
      },
    ] as unknown as readonly RenderNodeIr[];
    const scene = {
      roots,
      timeline: [
        {
          id: "enter-stack",
          at: 0,
          duration: 100,
          action: "enter",
          target: "stack",
        },
      ],
    };
    const snapshot = resolvedSceneSnapshot(
      resolveSceneForGeometryVerification(scene),
    );

    const findings = verifyPackageIr(
      {
        id: "layout-parity",
        slides: [
          {
            id: "stack-circle",
            render: {
              scene,
            },
          },
        ],
      },
      { snapshots: [{ slideId: "stack-circle", snapshot }] },
    );

    expect(
      findings.filter(
        ({ code }: { code: string }) =>
          code === "resolved-snapshot-missing" ||
          code === "zero-area-box" ||
          code === "missing-geometry" ||
          code === "out-of-viewport",
      ),
    ).toEqual([]);
  });

  it("forwards canonical resolver diagnostics with stable source locations", () => {
    const snapshot: ResolvedSceneSnapshot = {
      sceneId: "resolved",
      viewport: { width: 700, height: 400 },
      nodes: [],
      generatedParts: [],
      connectors: [],
      fans: [],
      diagnostics: [
        {
          code: "SCENE_VIEWPORT_ESCAPE",
          severity: "warning",
          message: 'Node "escaped" exceeds the scene viewport.',
          range: {
            source: "fixture.flow",
            start: { offset: 40, line: 3, column: 5 },
            end: { offset: 47, line: 3, column: 12 },
          },
          nodeIds: ["escaped"],
        },
      ],
    };

    const findings = verifyPackageIr(
      {
        id: "snapshot-diagnostics",
        slides: [
          {
            id: "resolved",
            render: {
              scene: {
                roots: [
                  {
                    id: "escaped",
                    kind: "rect",
                    geometry: { x: 0, y: 0, width: 10, height: 10 },
                    style: {},
                    accessibility: { label: "escaped" },
                  },
                ],
                timeline: [
                  {
                    id: "enter",
                    at: 0,
                    duration: 100,
                    action: "enter",
                    target: "escaped",
                  },
                ],
              },
            },
          },
        ],
      },
      { snapshot },
    );

    expect(findings).toContainEqual(
      expect.objectContaining({
        severity: "warn",
        code: "SCENE_VIEWPORT_ESCAPE",
        source: "fixture.flow",
        line: 3,
        column: 5,
      }),
    );
  });
});
